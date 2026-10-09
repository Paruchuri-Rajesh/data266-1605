# HW6 Metrics

SID4 = 1605, SEED = 1605. All numbers below are read from the final executed run of
`stl10_ssl.ipynb` (0 cell errors; see `RUN_LOG.txt`) and its `outputs/all_outputs.json`. None were
typed in by hand.

## Setup

| Item | Value |
|---|---|
| Dataset | STL-10 (official binary release, MD5 `91f7769df0f17e558f3565bffb0c7dfb`), native 96×96 |
| Labeled subset (Parts A, B, C) | 500 train images = 10%, stratified 50/class, seed 1605 (`outputs/labeled_subset_indices.npy`) |
| Test set | all 8,000 STL-10 test images |
| Backbone (all parts) | `torchvision.models.resnet18(weights=None)`, `fc` removed → 512-d embedding |
| Hardware | Apple M4 (MPS), 16 GB unified memory |
| Software | Python 3.13.7, torch 2.11.0, torchvision 0.26.0, numpy 2.3.3, matplotlib 3.10.6 |

| Part | Training configuration |
|---|---|
| A supervised | 500 labels, 15 epochs, batch 32, Adam lr 1e-3 (cosine), wd 5e-4, flip + translate aug |
| B rotation pretext | all 100,000 unlabeled, 15 epochs × 781 steps, batch 128, one balanced random rotation (0/90/180/270°) per image, Adam lr 1e-3 (500-step warmup + cosine), wd 5e-4 |
| C SimCLR pretext | 20,000 unlabeled (seeded subset), 20 epochs × 156 steps, batch 128 images = 256 views, augmentations: RandomResizedCrop(0.2–1) + HorizontalFlip + ColorJitter(0.4,0.4,0.4,0.1, p=0.8) + RandomGrayscale(0.2); MLP head 512→512→128; NT-Xent, cosine similarity, τ = 0.2; Adam lr 1e-3 (1-epoch warmup + cosine), wd 1e-6 |
| B, C linear probe | encoder frozen (eval mode, `requires_grad=False`, state-dict hash checked) → fixed per-feature standardization (train-set stats) → `Linear(512,10)`; 20 epochs, batch 25 (400 steps), Adam lr 3e-3 (cosine), wd 1e-4, flip + translate aug |

## Test accuracy (main result)

| Model | Test accuracy |
|---|---|
| A — Supervised ResNet-18, end-to-end on 500 labels | 47.84% |
| B — Rotation SSL encoder (frozen) + linear | 35.99% |
| C — SimCLR encoder (frozen) + linear | **56.00%** |

## Training diagnostics

| Model | First epoch | Final epoch | Train time |
|---|---|---|---|
| A supervised | loss 2.2635, train acc 16.4% | loss 0.6374, train acc 78.0% | 0.5 min |
| B rotation pretext | loss 1.0355, rotation acc 55.3% | loss 0.3708, rotation acc 86.0% | 280.0 min wall-clock (~85 min active; the laptop slept, see `RUN_LOG.txt`) |
| B rotation pretext, **held-out** | — | rotation acc on all 4 rotations of the 8,000 test images: **89.85%** | — |
| C SimCLR pretext | NT-Xent 4.2184, positive-pair top-1 18.3% | NT-Xent 2.0655, positive-pair top-1 86.5% | 44.7 min |
| B linear probe | — | loss 1.6714, train acc 39.0% | < 1 min |
| C linear probe | — | loss 0.6326, train acc 78.8% | < 1 min |

Encoder freezing was verified: the SHA-256 of each encoder's `state_dict` is identical before and after
its probe (rotation `117915bfb9f1…`, SimCLR `c5d7a5fc1fc3…`).

## Per-class test accuracy (%)

| Class | A supervised | B rotation | C SimCLR |
|---|---|---|---|
| airplane | **73.1** | 49.1 | 67.0 |
| bird | 34.4 | 31.9 | **52.1** |
| car | **71.5** | 51.8 | 68.6 |
| cat | 37.4 | 40.9 | **44.6** |
| deer | 43.9 | 31.8 | **57.6** |
| dog | **29.6** | 8.5 | 27.0 |
| horse | 55.9 | 40.6 | **61.5** |
| monkey | 34.4 | 29.4 | **42.4** |
| ship | 52.3 | 44.9 | **71.9** |
| truck | 46.0 | 31.1 | **67.3** |

## Part D — nearest neighbours (cosine similarity on L2-normalized 512-d embeddings, test set)

Over all 8,000 test images as queries (query itself excluded):

| Encoder | Precision@5 | 5-NN accuracy (leave-one-out) | Mean top-5 cosine |
|---|---|---|---|
| Supervised | 40.93% | 47.35% | 0.967 |
| Rotation SSL | 24.31% | 28.90% | 0.996 |
| SimCLR | **49.71%** | **57.89%** | 0.917 |

Example queries: 4 test images from 4 classes, chosen at random with seed 1605 and shared across all
three encoders. The table shows neighbours of the same class, out of 5:

| Query | Supervised | Rotation SSL | SimCLR | Figure |
|---|---|---|---|---|
| #6352 cat | 1 | 0 | 1 | `outputs/partD_query1_cat.png` |
| #6138 horse | 4 | 1 | 2 | `outputs/partD_query2_horse.png` |
| #2465 deer | 1 | 1 | 5 | `outputs/partD_query3_deer.png` |
| #7554 monkey | 2 | 2 | 3 | `outputs/partD_query4_monkey.png` |

## Superseded numbers (first run, before the probe fix — kept for transparency)

The first execution used a linear probe on raw (unstandardized) features with 200 steps. It
underfit, so its numbers are not reported as results. See `AI_USE.md` and
`outputs/run1_all_outputs_unstandardized_probe.json`.

| Model | Run 1 (raw-feature probe) | Final (standardized probe) |
|---|---|---|
| B rotation | 24.60% (train acc 25.0%) | 35.99% |
| C SimCLR | 55.75% | 56.00% |

Part A and all Part D numbers don't involve the probe and are identical in both runs.
