# HW11 Metrics

CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), CIFAR-10 test split, SEED = 1605 (SID4), device MPS. Repeated executions produced identical numbers.

## Part A — caption "a photo of a dog" vs. 1,000 images (100 per class)

| Metric | Value |
|---|---|
| Cosine similarity range (min / mean / max) | 0.1628 / 0.2203 / 0.2854 |
| Dogs in top 15 | 14 / 15 (a deer at rank 10) |
| Precision@100 (dogs in top 100) | 0.77 |
| Top-100 composition | dog 77, deer 10, cat 8, bird 4, horse 1 |
| ROC-AUC, dog vs. rest | 0.979 |

Mean cosine similarity per class: dog 0.2616, cat 0.2389, deer 0.2326, frog 0.2307, bird 0.2258, horse 0.2219, automobile 0.2015, airplane 0.2013, truck 0.1959, ship 0.1929.

Other captions (top-100 composition):

| Caption | Best class (mean) | Top 100 |
|---|---|---|
| a photo of a cat | cat | cat 73, dog 13, deer 6, frog 4 |
| a photo of a ship on the water | ship | ship 92, airplane 3, truck 2, deer 2, horse 1 |
| a vehicle used for transportation | truck | truck 66, automobile 34 |
| an animal that can fly | bird (0.238; airplane 0.231, deer 0.231) | bird 35, airplane 29, deer 20, frog 6, … |
| a blurry low-resolution photo | bird (0.258) | all classes 0.234–0.258 |

## Part B — zero-shot classification, 10,000 test images

| Prompt strategy | Top-1 | Top-2 |
|---|---|---|
| bare class name `"{c}"` | 87.39% | 95.28% |
| **single template `"a photo of a {c}."`** (main) | **88.33%** | 95.53% |
| 8-template ensemble | 88.94% | 95.74% |

Per-class accuracy (recall), 1,000 images per class:

| Class | bare name | single template | ensemble | precision (single) |
|---|---|---|---|---|
| horse | 97.1% | 97.1% | 97.4% | 83.1% |
| ship | 96.4% | 95.9% | 96.6% | 92.7% |
| truck | 94.2% | 95.0% | 95.4% | 92.4% |
| automobile | 89.9% | 93.3% | 93.4% | 93.7% |
| bird | 91.7% | 89.8% | 87.7% | 78.0% |
| dog | 85.2% | 89.1% | 89.2% | 81.7% |
| airplane | 81.1% | 89.0% | 90.3% | 95.4% |
| cat | 79.1% | 85.7% | 85.8% | 82.0% |
| deer | 82.1% | 76.9% | 78.8% | 93.3% |
| frog | 77.1% | 71.5% | 74.8% | 98.1% |

Top confusions (single template): deer→horse 137, frog→bird 104, frog→cat 85, cat→dog 83, dog→cat 63, automobile→truck 61, airplane→bird 60, frog→dog 57, truck→automobile 37, deer→bird 36.
Errors that cross between animals and vehicles: 124 of 1,167 (10.6%), 70 of them airplane ↔ bird.

Part A with the earlier SEED = 0 sample (superseded): precision@100 = 0.73, ROC-AUC 0.966, 15/15 dogs in top 15. Part B does not depend on the seed.
Zero-shot accuracy on the Part A subset (SEED = 1605): 87.60%, 100% agreement with the Part B predictions.
