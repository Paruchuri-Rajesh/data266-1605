# HW8 Metrics

**SID4 = 1605, SEED = 1605.** Every random choice in the notebook derives from SEED:
torch/numpy = SEED, forward-diffusion ε = SEED+1, the 64-image x_T grid = SEED+2, the 1,000-image
evaluation x_T = SEED+3, DDPM's per-step noise = SEED+4, and the speed-benchmark batch = SEED+5.
SLICE, HP_ID, CLS_A and CLS_B are not used by HW8.

All numbers below are read from the final executed run of `Mini_Diffusion_DDPM_DDIM.ipynb` (0 cell
errors; see `RUN_LOG.txt`). None were typed in by hand.

## Setup

| Item | Value |
|---|---|
| Dataset | MNIST (torchvision; raw-file MD5s match torchvision's published checksums), scaled to [-1, 1] |
| Diffusion | T = 100; β_t linear 1e-4 → 0.02 (assignment). Index 0 = clean image (β_0 = 0, ᾱ_0 = 1) |
| ᾱ_T | **0.3636** → x_100 = 0.603·x_0 + 0.798·ε (not pure noise) |
| ε-network | MiniUNet: 28→14→7→14→28, ResBlocks (conv-GroupNorm-ReLU), sinusoidal t-embedding + MLP added per block; **18 conv layers**, 940,193 params, output 1×28×28 |
| Loss | MSE(ε, ε_θ(x_t, t)), t ~ U{1..100} |
| Training | AdamW lr 1e-3, cosine decay, batch 128, 50 epochs (23,450 steps); EMA of weights (decay warm-up to 0.999), EMA used for sampling |
| Hardware | Apple M4 (MPS), 16 GB; Python 3.13.7, torch 2.11.0, torchvision 0.26.0 |
| Train time | 85.5 min (assignment schedule), 83.2 min (scaled schedule) |

## (b) Training loss

| Model | Epoch 1 | Epoch 50 | Held-out MSE averaged over t (EMA) |
|---|---|---|---|
| Assignment schedule (β_T = 0.02) | 0.1260 | **0.0521** | 0.0516 |
| Scaled schedule (β_T = 0.2, extra) | 0.0624 | 0.0210 | — |

The two losses are not directly comparable, because each t has a different noise level under each schedule.

## (c)/(d) Sample quality: 1,000 samples per sampler, same x_T for all samplers

Judge: a separately trained MNIST CNN (3 epochs, **test acc 0.9907**). "Conf" = mean max-softmax.
"Class H" = entropy of the predicted-class histogram (max log2 10 = 3.32 bits). "Pixel KL" = KL
divergence of the 20-bin pixel-intensity histogram from that of 1,000 real test images.

### Assignment schedule (parts c, d)

| Sampler | NFE | Mean conf ↑ | % conf > 0.9 ↑ | Class H (bits) ↑ | Pixel KL ↓ |
|---|---|---|---|---|---|
| DDPM | 100 | **0.806** | **47.4%** | **2.891** | **0.0689** |
| DDIM (η = 0) | 100 | 0.776 | 41.0% | 2.661 | 0.1859 |
| DDIM (η = 0) | 50 | 0.776 | 40.8% | 2.657 | 0.1858 |
| DDIM (η = 0) | 20 | 0.775 | 40.6% | 2.654 | 0.1945 |
| DDIM (η = 0) | 10 | 0.774 | 39.9% | 2.653 | 0.2203 |
| Real MNIST (test) | – | 0.988 | 96.7% | 3.309 | 0 |

Predicted-class counts (0–9): DDPM `[57, 7, 230, 95, 79, 77, 58, 84, 278, 35]`;
DDIM-50 `[28, 7, 279, 88, 88, 51, 38, 67, 326, 28]`. Both are biased toward "8" and "2", and "1" is almost absent.

### Scaled schedule, β_t linear 1e-3 → 0.2, ᾱ_T = 2.04e-5 (extra section, same everything else)

| Sampler | NFE | Mean conf ↑ | % conf > 0.9 ↑ | Class H (bits) ↑ | Pixel KL ↓ |
|---|---|---|---|---|---|
| DDPM | 100 | **0.942** | **83.2%** | 3.313 | 0.0013 |
| DDIM (η = 0) | 100 | 0.932 | 80.3% | 3.317 | **0.0009** |
| DDIM (η = 0) | 50 | 0.931 | 80.2% | **3.318** | **0.0009** |
| DDIM (η = 0) | 20 | 0.926 | 78.7% | 3.316 | 0.0020 |
| DDIM (η = 0) | 10 | 0.915 | 76.2% | 3.312 | 0.0070 |

## Sampling speed (batch 256, min over 5 interleaved rounds, assignment-schedule model)

| Sampler | NFE | s / 256 images | ms / image | Speed-up vs DDPM |
|---|---|---|---|---|
| DDPM | 100 | 16.77 | 65.50 | 1.0× |
| DDIM | 100 | 16.59 | 64.79 | 1.0× |
| DDIM | 50 | 8.23 | 32.16 | **2.0×** |
| DDIM | 20 | 3.30 | 12.87 | **5.1×** |
| DDIM | 10 | 1.65 | 6.46 | 10.1× |

The single-pass times in the quality tables agree: DDPM 67.07 s vs DDIM-100 67.09 s for 1,000 images.

## Determinism / consistency (10 columns, same x_T, assignment-schedule model)

| Check | Result |
|---|---|
| DDIM-20 run twice, max abs pixel difference | **0.0** |
| DDIM-50 / -20 / -10 predicted digit = DDIM-100 predicted digit | 100% / 100% / 100% |
| DDPM run 2 predicted digit = DDPM run 1 predicted digit | 30% |

## x_T mismatch (assignment schedule)

The per-image pixel std of x_100 ~ q(x_100 | x_0) is about 0.87 (training), against about 1.0 for the
N(0, I) sampling start. The distributions barely overlap (figure `figures/d_xT_mismatch.png`).

Figures: `figures/a_forward_diffusion.png`, `a_schedule.png`, `b_training_loss.png`, `b_loss_per_t.png`,
`c_ddpm_samples.png`, `c_ddpm_trajectory.png`, `d_ddpm_vs_ddim.png`, `d_ddim20_trajectory.png`,
`d_metrics.png`, `d_consistency.png`, `d_xT_mismatch.png`, `x_training_loss_both.png`,
`x_ddpm_vs_ddim_scaled.png`.
