# METRICS — HW1

**SID4 = 1605, SEED = 1605, HP_ID = 3 (Learning-rate-low), SLICE = 605, CLS_A = 5, CLS_B = 3**
(SLICE/CLS_A/CLS_B computed per Section 0.1 but not referenced by any HW1 task.)

## 1. Diabetes dataset — data split

Split: 70% train / 15% val / 15% test, stratified on `Outcome`, `random_state = SEED = 1605`.

| Split | n | positive rate |
|---|---|---|
| train | 537 | 0.348 |
| val   | 115 | ~0.348 |
| test  | 116 | ~0.345 |

## 2. Model configurations

| Model | Framework | Hidden layers | Learning rate | Epochs |
|---|---|---|---|---|
| Baseline | PyTorch / TensorFlow | [64, 32] | 0.001 | 30 |
| Modified (HP_ID=3) | PyTorch / TensorFlow | [64, 32] | 0.0003 | 30 |

## 3. Multi-seed test accuracy (training seeds 1605, 1606, 1607; split fixed at SEED=1605)

| Framework | Model | Test acc (mean ± std) |
|---|---|---|
| PyTorch | baseline | 0.7529 ± 0.0050 |
| PyTorch | modified (HP_ID=3) | 0.7270 ± 0.0132 |
| TensorFlow | baseline | 0.7414 ± 0.0172 |
| TensorFlow | modified (HP_ID=3) | 0.7213 ± 0.0199 |

Raw per-seed values: `multiseed_raw_accuracies.csv`. Full summary: `multiseed_summary.csv`.

## 4. Seed-run (SEED=1605) final-epoch loss (epoch 30)

| Framework | Model | Train loss | Val loss |
|---|---|---|---|
| PyTorch | baseline | 0.382 | 0.419 |
| PyTorch | modified (HP_ID=3) | 0.428 | 0.404 |
| TensorFlow | baseline | 0.363 | 0.428 |
| TensorFlow | modified (HP_ID=3) | 0.421 | 0.408 |

Loss-curve figures: `figures/pytorch_loss_curves.png`, `figures/tensorflow_loss_curves.png`.

## 5. CUDA matrix multiplication (Part 3)

**Status: pending — run `cuda.ipynb` on Google Colab (GPU runtime), then paste the `results`
table and `nsys profile --stats=true` output here, replacing this table.**

| Matrix size | CPU (ms) | GPU kernel (ms) | H2D+D2H (ms) | Speedup |
|---|---|---|---|---|
| 256  | TBD (run cuda.ipynb) | TBD | TBD | TBD |
| 1024 | TBD (run cuda.ipynb) | TBD | TBD | TBD |
| 4096 | TBD (run cuda.ipynb) | TBD | TBD | TBD |

GPU used: TBD (from `nvidia-smi` output in Colab, e.g. Tesla T4).
Profiler used: Nsight Systems (`nsys profile --stats=true`) — `nvprof` is not present in modern
CUDA toolkits (removed in CUDA 11+), so `nsys` is the profiler actually available in Colab.
