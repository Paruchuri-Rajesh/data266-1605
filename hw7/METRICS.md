# HW7 Metrics

**SID4 = 1605, SEED = 1605, CLS_A = 5 (Sandal), CLS_B = 3 (Dress)**. SLICE = 605 is not used by
HW7. HP_ID = 3 is not used: the assignment waives the second hyper-parameter model for HW7.

All numbers below are read from the final executed run of `autoencoders_fashionmnist.ipynb` (0 cell
errors; see `RUN_LOG.txt`) and its `outputs/all_outputs.json` / `outputs/test_metrics_per_seed.csv`.
None were typed in by hand. "±" is the std over training seeds 1605, 1606 and 1607.

## Setup

| Item | Value |
|---|---|
| Dataset | FashionMNIST (torchvision; raw-file MD5s match torchvision's published checksums) |
| Split | 54,000 train / 6,000 val (seeded with SEED = 1605) / 10,000 official test |
| FC-AE | 784-512-256-**32**-256-512-784, ReLU, logits out; 1,083,696 params |
| VAE | same layers; encoder heads μ, log σ² ∈ ℝ³²; reparameterization; 1,091,920 params |
| Loss | FC-AE: BCE summed over pixels; VAE: BCE + KL to N(0, I) (= −ELBO) |
| Training | Adam lr 1e-3, batch 128, 30 epochs (12,660 steps), seeds 1605/1606/1607 |
| Hardware | Apple M4 (MPS); Python 3.13.7, torch 2.11.0, torchvision 0.26.0 |
| Train time per run | FC-AE 1.12 ± 0.02 min, VAE 1.43 ± 0.01 min |

## Test reconstruction (10,000 test images; VAE reconstructs from μ)

| Metric | FC-AE | VAE |
|---|---|---|
| MSE / pixel | **0.00807 ± 0.00007** | 0.01484 ± 0.00006 |
| PSNR (dB) | **22.12 ± 0.03** | 19.17 ± 0.02 |
| SSIM (11×11 Gaussian, σ = 1.5) | **0.748 ± 0.001** | 0.642 ± 0.002 |
| BCE (nats / image) | **209.78 ± 0.15** | 224.71 ± 0.10 |
| KL (nats / image) | — | 12.19 ± 0.32 |
| −ELBO (nats / image, 1-sample estimate) | — | 239.84 ± 0.10 |

Per seed:

| Model | Seed | MSE | PSNR | SSIM | BCE |
|---|---|---|---|---|---|
| FC-AE | 1605 | 0.00813 | 22.088 | 0.7460 | 209.91 |
| FC-AE | 1606 | 0.00800 | 22.155 | 0.7489 | 209.62 |
| FC-AE | 1607 | 0.00809 | 22.104 | 0.7477 | 209.81 |
| VAE | 1605 | 0.01488 | 19.157 | 0.6418 | 224.76 |
| VAE | 1606 | 0.01487 | 19.168 | 0.6434 | 224.79 |
| VAE | 1607 | 0.01478 | 19.193 | 0.6399 | 224.60 |

Per-class test MSE (mean over 3 seeds):

| Class | FC-AE | VAE | VAE / FC-AE |
|---|---|---|---|
| T-shirt/top | 0.00747 | 0.01436 | 1.92 |
| Trouser | 0.00358 | 0.00754 | 2.11 |
| Pullover | 0.00728 | 0.01456 | 2.00 |
| Dress (CLS_B) | 0.00735 | 0.01388 | 1.89 |
| Coat | 0.00674 | 0.01351 | 2.00 |
| Sandal (CLS_A) | 0.01255 | 0.02190 | 1.75 |
| Shirt | 0.00840 | 0.01472 | 1.75 |
| Sneaker | 0.00597 | 0.01147 | 1.92 |
| Bag | 0.01356 | 0.02181 | 1.61 |
| Ankle boot | 0.00783 | 0.01468 | 1.88 |

## Latent space

| Probe (logistic regression; fit on 54k train codes, scored on 10k test codes) | Test acc |
|---|---|
| FC-AE 32-d code | **0.8304 ± 0.0013** |
| VAE 32-d μ | 0.8089 ± 0.0062 |
| Raw pixels, 784-d (reference) | 0.8332 |

VAE (seed 1605) active latent dims, i.e. mean KL > 0.01 nats: **6 / 32**. Per-dim KL of the active
dims: 2.72, 2.52, 2.00, 1.85, 1.61, 1.16 nats; all others ≤ 0.001.

## Generation (10,000 decoded random codes per model and seed)

Scored with a separately trained FashionMNIST CNN (`checkpoints/eval_classifier_cnn.pt`, 8 epochs,
val acc 0.9162, **test acc 0.9092**). C-FID is the Fréchet distance between 128-d penultimate CNN
features of the samples and of the 10,000 test images.

| Setting | C-FID ↓ | IS-style ↑ | Mean max-conf ↑ | Class entropy (norm.) ↑ |
|---|---|---|---|---|
| Real train images (reference, 10k) | 0.50 | 8.378 | 0.935 | 0.999 |
| VAE, z ~ N(0, I) | 111.59 ± 2.53 | **5.536 ± 0.042** | **0.804 ± 0.004** | 0.965 ± 0.001 |
| FC-AE, z ~ N(0, I) | 223.38 ± 9.21 | 3.009 ± 0.123 | 0.636 ± 0.022 | 0.883 ± 0.021 |
| FC-AE, z ~ Gaussian fit to train codes | **92.77 ± 5.13** | 4.814 ± 0.073 | 0.752 ± 0.005 | **0.967 ± 0.002** |

Predicted-class distribution of samples (seed 1605):

| Class | VAE prior | FC-AE N(0,I) | FC-AE fit | Real |
|---|---|---|---|---|
| T-shirt/top | 0.095 | 0.132 | 0.061 | 0.092 |
| Trouser | 0.084 | 0.022 | 0.065 | 0.096 |
| Pullover | 0.151 | 0.054 | 0.131 | 0.110 |
| Dress | 0.144 | 0.081 | 0.134 | 0.107 |
| Coat | 0.040 | 0.024 | 0.061 | 0.092 |
| Sandal | 0.042 | 0.050 | 0.054 | 0.097 |
| Shirt | 0.078 | 0.198 | 0.122 | 0.103 |
| Sneaker | 0.093 | 0.067 | 0.085 | 0.096 |
| Bag | 0.138 | 0.266 | 0.176 | 0.100 |
| Ankle boot | 0.137 | 0.106 | 0.112 | 0.109 |

## Interpolation Sandal (CLS_A) → Dress (CLS_B), seed 1605, 10 steps

| Path | Mean CNN confidence | CNN predictions along the path |
|---|---|---|
| Pixel blend | 0.977 | Sandal ×2, Dress ×8 |
| FC-AE latent | 0.890 | Sandal ×4, Bag, Dress ×5 |
| VAE latent | 0.894 | Sandal ×2, Dress, Trouser ×3, Dress ×4 |

Figures: `figures/training_curves.png`, `reconstructions.png`, `per_class_mse.png`, `latent_tsne.png`,
`generated_samples.png`, `generated_class_hist.png`, `interpolation_sandal_to_dress.png`,
`dataset_samples.png`.
