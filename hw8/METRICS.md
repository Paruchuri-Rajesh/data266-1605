# HW8 Metrics

**SID4 = 1605.** The HW8 handout defines no personal parameters. The notebook uses `SEED = 0` for
torch/numpy. The sampling noise uses fixed generators: seed 123 for the 64-image grids, 2024 for the
1,000-image evaluation set, and 7 for DDPM's per-step noise. SEED = 1605 is **not** used in HW8.

All numbers below are read from the executed `Mini_Diffusion_DDPM_DDIM.ipynb` (0 cell errors; see
`RUN_LOG.txt`). None were typed in by hand.

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

## (b) Training loss

| Model | Epoch 1 | Epoch 50 | Held-out MSE averaged over t (EMA) |
|---|---|---|---|
| Assignment schedule (β_T = 0.02) | 0.1236 | **0.0518** | 0.0517 |
| Scaled schedule (β_T = 0.2, extra) | 0.0622 | 0.0210 | — |

The two losses are not directly comparable, because each t has a different noise level under each schedule.

## (c)/(d) Sample quality: 1,000 samples per sampler, same x_T for all samplers

Judge: a separately trained MNIST CNN (3 epochs, **test acc 0.9883**). "Conf" = mean max-softmax.
"Class H" = entropy of the predicted-class histogram (max log2 10 = 3.32 bits). "Pixel KL" = KL
divergence of the 20-bin pixel-intensity histogram from that of 1,000 real test images.

### Assignment schedule (parts c, d)

| Sampler | NFE | Mean conf ↑ | % conf > 0.9 ↑ | Class H (bits) ↑ | Pixel KL ↓ |
|---|---|---|---|---|---|
| DDPM | 100 | **0.808** | **46.4%** | **2.660** | **0.0654** |
| DDIM (η = 0) | 100 | 0.785 | 41.7% | 2.271 | 0.1554 |
| DDIM (η = 0) | 50 | 0.785 | 41.4% | 2.273 | 0.1553 |
| DDIM (η = 0) | 20 | 0.784 | 41.2% | 2.255 | 0.1588 |
| DDIM (η = 0) | 10 | 0.783 | 41.3% | 2.237 | 0.1690 |
| Real MNIST (test) | – | 0.986 | 95.8% | 3.310 | 0 |

Predicted-class counts (0–9): DDPM `[85, 6, 199, 37, 110, 88, 19, 21, 369, 66]`;
DDIM-50 `[31, 5, 177, 34, 105, 63, 7, 20, 507, 51]`. Both are heavily biased toward "8".

### Scaled schedule, β_t linear 1e-3 → 0.2, ᾱ_T = 2.04e-5 (extra section, same everything else)

| Sampler | NFE | Mean conf ↑ | % conf > 0.9 ↑ | Class H (bits) ↑ | Pixel KL ↓ |
|---|---|---|---|---|---|
| DDPM | 100 | **0.936** | **82.1%** | **3.304** | 0.0015 |
| DDIM (η = 0) | 100 | 0.928 | 79.2% | 3.303 | **0.0008** |
| DDIM (η = 0) | 50 | 0.925 | 78.7% | 3.301 | **0.0008** |
| DDIM (η = 0) | 20 | 0.920 | 78.1% | 3.295 | 0.0018 |
| DDIM (η = 0) | 10 | 0.912 | 74.3% | 3.282 | 0.0067 |

## Sampling speed (clean re-measurement, batch 256, min over 5 interleaved rounds)

| Sampler | NFE | s / 256 images | ms / image | Speed-up vs DDPM |
|---|---|---|---|---|
| DDPM | 100 | 25.83 | 100.91 | 1.0× |
| DDIM | 100 | 24.47 | 95.59 | 1.1× |
| DDIM | 50 | 12.19 | 47.60 | **2.1×** |
| DDIM | 20 | 4.56 | 17.82 | **5.7×** |
| DDIM | 10 | 1.66 | 6.50 | 15.5× |

The "time" column of the two quality tables in the notebook is **not** a valid speed measurement.
The machine slept or throttled during the run (see `RUN_LOG.txt`), so use this table for speed.

## Determinism / consistency (10 columns, same x_T, assignment-schedule model)

| Check | Result |
|---|---|
| DDIM-20 run twice, max abs pixel difference | **0.0** |
| DDIM-50 / -20 / -10 predicted digit = DDIM-100 predicted digit | 100% / 100% / 100% |
| DDPM run 2 predicted digit = DDPM run 1 predicted digit | 60% |

## x_T mismatch (assignment schedule)

The per-image pixel std of x_100 ~ q(x_100 | x_0) is about 0.87 (training), against about 1.0 for the
N(0, I) sampling start. The distributions barely overlap (figure `figures/d_xT_mismatch.png`).

Figures: `figures/a_forward_diffusion.png`, `a_schedule.png`, `b_training_loss.png`, `b_loss_per_t.png`,
`c_ddpm_samples.png`, `c_ddpm_trajectory.png`, `d_ddpm_vs_ddim.png`, `d_ddim20_trajectory.png`,
`d_metrics.png`, `d_consistency.png`, `d_xT_mismatch.png`, `x_training_loss_both.png`,
`x_ddpm_vs_ddim_scaled.png`.
