# HW9 Metrics

**SID4 = 1605, SEED = 1605.** Every random choice in the notebook derives from SEED: torch/numpy = SEED,
forward-diffusion ε = SEED+1, the 64-image z_T grid = SEED+2, the 500-image sweep z_T = SEED+3, DDPM's
per-step noise = SEED+4, and the held-out loss-per-t noise = SEED+6. SLICE, HP_ID, CLS_A and CLS_B are
not used by HW9. Per the assignment, no second hyperparameter configuration was trained: the pretrained
VAE and the diffusion U-Net are the two models.

All numbers below are read from the final executed run of `Latent_Diffusion_FashionMNIST.ipynb` (0 cell
errors; see `RUN_LOG.txt`). None were typed in by hand.

## Setup

| Item | Value |
|---|---|
| Dataset | FashionMNIST (torchvision; raw-file MD5s match torchvision's published checksums), 60k train / 10k test |
| VAE | `stabilityai/sd-vae-ft-mse` (83.7M params, frozen). 28×28 → bilinear 128×128 → 3 channels → [-1,1] → posterior **mean** |
| Latents | 4×16×16, scaled by **s = 0.18215** |
| Diffusion | T = 1000; β_t linear 1e-4 → 0.02. Index 0 = clean latent; ᾱ_1000 = 4.0e-5 |
| ε-network | LatentUNet: in/out 4 ch, base 64, mults (1, 2, 4) → 64@16², 128@8², 256@4²; 2 ResBlocks/level (GroupNorm + SiLU); sinusoidal t-emb + MLP; label emb (10 + null) added to t-emb; self-attn + cross-attn (4 label tokens) at 16×16 and bottleneck. **14,822,916 params**, 17 ResBlocks, 6 attention blocks |
| CFG | label → null (10) with p = 0.1 during training; ε = (1+w) ε_cond − w ε_uncond, **w = 3** |
| Training | AdamW, constant **lr 2e-5**, **50 epochs**, batch 128 (23,450 steps), grad clip 1.0; EMA (decay warm-up to 0.999) used for sampling |
| Hardware | Apple M4 (MPS), 16 GB; Python 3.13.7, torch 2.11.0, torchvision 0.26.0, diffusers 0.42.1 |
| Train time | **≈ 8.5 h of compute** (median awake epoch 601 s). Notebook prints 2,755 min because sleep pauses are included (see `RUN_LOG.txt`) |

## 1. VAE reconstruction (2,000 test images, pixel MSE on [0,1])

| Comparison | MSE | PSNR |
|---|---|---|
| Decoded 128×128 vs VAE input (VAE alone) | **0.000196** | 37.09 dB |
| Decoded → 28×28 vs original 28×28 | 0.003461 | 24.61 dB |
| Resize round trip only, no VAE | 0.003171 | – |

Per-class MSE at 28×28 ranges from 0.0028 (Sneaker) to 0.0041 (Pullover).
Scaled training latents: mean −0.039, **std 1.265**; per-channel mean [−0.122, −1.271, 0.657, 0.580],
std [1.149, 1.356, 0.585, 0.714]; unscaled std 6.946.

## 2. Forward diffusion

| t | ᾱ_t | √ᾱ_t (signal) | √(1−ᾱ_t) (noise) |
|---|---|---|---|
| 0 | 1.00000 | 1.0000 | 0.0000 |
| 200 | 0.65904 | 0.8118 | 0.5839 |
| 500 | 0.07859 | 0.2803 | 0.9599 |
| 800 | 0.00153 | 0.0391 | 0.9992 |
| 1000 | 0.00004 | 0.0064 | 1.0000 |

## 4. Training

| Epoch | 1 | 2 | 3 | 5 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| Mean loss | 0.6873 | 0.2730 | 0.1628 | 0.1306 | 0.1162 | 0.1060 | 0.1012 | 0.0976 | **0.0955** |

Held-out noise-prediction MSE (EMA model, 1,000 test latents):

| t | 1 | 10 | 100 | 300 | 500 | 600 | 700 | 800 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| conditional | 0.9506 | 0.7658 | 0.2960 | 0.0755 | 0.0182 | 0.0081 | 0.0034 | 0.0016 | 0.0011 |
| unconditional | 0.9541 | 0.7697 | 0.2978 | 0.0765 | 0.0191 | 0.0093 | 0.0043 | 0.0019 | 0.0010 |

## 5. Sampling

Judge: a separately trained FashionMNIST CNN (5 epochs). Generated images are decoded and area-downsampled to 28×28.

| Set | Judge accuracy vs intended label | Mean confidence |
|---|---|---|
| Real test images (10,000) | 0.9129 | 0.927 |
| VAE reconstructions (2,000 test) | 0.9160 | 0.912 |
| **64-image grid, CFG w = 3, DDPM 1000 steps** | **1.0000 (64/64)** | **0.985** |

Grid sampling took 377.1 s (1000 steps × 2 passes) plus 14.4 s of VAE decoding. Generated latents: std 1.274 (training 1.265).

### Guidance sweep: DDIM 50 steps (η = 0), 500 samples (50 per class), same z_T

| Setting | Accuracy | Mean conf | Intra-class diversity (mean pairwise L2, 28×28) | Time (s) |
|---|---|---|---|---|
| w = 0 | 0.824 | 0.863 | 7.964 | 189 |
| w = 1 | 0.960 | 0.944 | 7.038 | 297 |
| w = 3 | 0.988 | 0.975 | 6.487 | 298 |
| w = 5 | **0.996** | **0.981** | 6.358 | 407 |
| Real test images | 0.913 | 0.927 | **8.659** | – |

Unconditional (null label): judge confidence 0.843; predicted-class entropy **3.284 bits** (max 3.322).
Counts: T-shirt 45, Trouser 47, Pullover 50, Dress 50, Coat 79, Sandal 52, Shirt 33, Sneaker 49, Bag 58, Ankle boot 37.

Figures: `figures/1_vae_reconstruction.png`, `2_forward_diffusion_decoded.png`, `2_schedule.png`,
`4_training_loss.png`, `4_loss_per_t.png`, `5_cfg_samples_grid64.png`, `5_reverse_trajectory.png`,
`5_guidance_sweep_metrics.png`, `5_guidance_sweep_samples.png`.
