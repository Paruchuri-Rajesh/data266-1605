"""Shared code for HW6 (STL-10: supervised vs. rotation SSL vs. SimCLR).

Kept in a module (not the notebook) because macOS DataLoader workers are spawned
processes and can only use classes/functions importable from a file.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2

STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer",
                 "dog", "horse", "monkey", "ship", "truck"]
MEAN = torch.tensor([0.4467, 0.4398, 0.4066]).view(1, 3, 1, 1)
STD = torch.tensor([0.2603, 0.2566, 0.2713]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- data
def stl10_images(bin_dir, split):
    """Memory-mapped uint8 array (N, 3, 96, 96) for split in {train, test, unlabeled}.

    The STL-10 binaries store each channel column-major, hence the (H, W) transpose.
    Returned as a view on a memmap, so nothing is loaded until it is indexed.
    """
    raw = np.memmap(os.path.join(bin_dir, f"{split}_X.bin"), dtype=np.uint8, mode="r")
    return raw.reshape(-1, 3, 96, 96).transpose(0, 1, 3, 2)


def stl10_labels(bin_dir, split):
    """Labels 0..9 (the files store 1..10)."""
    return np.fromfile(os.path.join(bin_dir, f"{split}_y.bin"), dtype=np.uint8).astype(np.int64) - 1


def to_input(x_uint8, device):
    """uint8 (B,3,H,W) numpy/tensor -> normalized float tensor on device."""
    x = torch.as_tensor(np.ascontiguousarray(x_uint8)).to(device).float().div_(255)
    return (x - MEAN.to(device)) / STD.to(device)


def gpu_flip_translate(x, max_shift=0.125):
    """Per-sample random horizontal flip + random translation (reflect padding), on device.

    Light augmentation used for Part A training, the rotation pretext task and the
    linear probes. No rotation/scale, so it cannot leak the rotation label.
    """
    b = x.shape[0]
    flip = torch.where(torch.rand(b, device=x.device) < 0.5, -1.0, 1.0)
    shift = (torch.rand(b, 2, device=x.device) * 2 - 1) * (2 * max_shift)  # grid units span [-1, 1]
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = flip
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def rotate_batch(x):
    """Rotate each image by a balanced random multiple of 90 deg; returns (images, labels 0..3)."""
    b = x.shape[0]
    labels = torch.arange(b, device=x.device) % 4
    labels = labels[torch.randperm(b, device=x.device)]
    out = torch.empty_like(x)
    for k in range(4):
        idx = (labels == k).nonzero(as_tuple=True)[0]
        out[idx] = torch.rot90(x[idx], k, dims=(2, 3))
    return out, labels


class SimCLRPairs(torch.utils.data.Dataset):
    """Two independently augmented views of each image, read lazily from a .npy memmap."""

    def __init__(self, npy_path):
        self.npy_path = npy_path
        self.n = np.load(npy_path, mmap_mode="r").shape[0]
        self.arr = None  # opened per worker
        # The four SimCLR augmentations (as in the class demo / Chen et al. 2020):
        # random resized crop, horizontal flip, colour jitter, random grayscale.
        self.aug = v2.Compose([
            v2.RandomResizedCrop(96, scale=(0.2, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
        ])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.arr is None:
            self.arr = np.load(self.npy_path, mmap_mode="r")
        img = torch.from_numpy(np.array(self.arr[i]))  # uint8 (3,96,96), copied out of the read-only memmap
        return self.aug(img), self.aug(img)


# ----------------------------------------------------------------------------- models
def make_encoder():
    """ResNet-18, random init, classifier removed -> 512-d embedding. Same backbone for all parts."""
    net = torchvision.models.resnet18(weights=None)
    net.fc = nn.Identity()
    return net


class EncoderWithHead(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, x):
        return self.head(self.encoder(x))


class Standardize(nn.Module):
    """Fixed (non-trainable) per-feature standardization; stats are buffers, not parameters."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone() + 1e-6)

    def forward(self, x):
        return (x - self.mean) / self.std


def projection_head(dim_in=512, dim_hidden=512, dim_out=128):
    return nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU(inplace=True), nn.Linear(dim_hidden, dim_out))


def nt_xent(z1, z2, tau):
    """SimCLR NT-Xent loss with cosine similarity and temperature tau."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.t() / tau  # cosine similarity / tau
    n = z1.shape[0]
    sim.fill_diagonal_(float("-inf"))  # an embedding is not its own positive
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def cosine_lr(step, total, base_lr, warmup=0):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))


# ----------------------------------------------------------------------------- utils
class Logger:
    """Prints and appends to a progress file (the notebook runs headless via nbconvert)."""

    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


@torch.no_grad()
def embed(encoder, images_uint8, device, batch=256):
    """512-d embeddings for a uint8 image array, encoder in eval mode."""
    encoder.eval()
    out = []
    for i in range(0, len(images_uint8), batch):
        out.append(encoder(to_input(images_uint8[i:i + batch], device)).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def accuracy(model, images_uint8, labels, device, batch=256):
    model.eval()
    correct = 0
    for i in range(0, len(images_uint8), batch):
        pred = model(to_input(images_uint8[i:i + batch], device)).argmax(1).cpu()
        correct += (pred == torch.as_tensor(labels[i:i + batch])).sum().item()
    return correct / len(images_uint8)
