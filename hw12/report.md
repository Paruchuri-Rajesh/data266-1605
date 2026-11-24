% DATA 266 — Homework 12

# DATA 266 — Homework 12

**SID4 = 1605 | SEED = 1605**

Notebook: `HW12_GPTQ_Distillation.ipynb` (single notebook, both parts, executed on a Google Colab Tesla T4).
The two parts are kept separate because they use different models and tasks. Full tables: `METRICS.md`;
environment, timings and fixes: `RUN_LOG.txt`.

---

## Part 1 — LLM optimisation with GPTQ (5 points)

**Model:** `Qwen/Qwen3-0.6B` @ revision `c1899de289a04d12100db370d81485cdf75e47ca` (the model used in the course-linked
Hugging Face GPTQ guide). The same base checkpoint was used for both runs.
**Quantization:** `transformers.GPTQConfig` (GPT-QModel backend), 4-bit, group size 128, symmetric, `desc_act=False`.
Calibration used 128 passages from the WikiText-2 **train** split; WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1`)
was used only for calibration and evaluation.
**Evaluation:** sliding-window perplexity on the WikiText-2 **test** split (299,078 tokens, context 1024, stride 512).
Latency is greedy generation of exactly 64 new tokens at batch 1, 30 timed runs after 3 warm-ups. Peak GPU memory covers
loading, evaluation and generation. Size on disk is the saved weight files. Both models ran on the same T4.

**Table HW11.1 — GPTQ comparison on the same LLM/task**

| Configuration | Quality metric (WikiText-2 PPL ↓) | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Base LLM (fp16) | 19.89 | 2641.2 / 3333.2 | 2.679 | 1136.9 |
| GPTQ-quantized LLM (4-bit, g128) | 21.14 | 3422.3 / 4186.7 | 2.073 | 516.3 |

**Findings.** 4-bit GPTQ cost 6.3% in perplexity (19.89 → 21.14). In exchange it cut the checkpoint 2.2× (1137 → 516 MB)
and peak GPU memory by 23% (2.68 → 2.07 GB). Memory falls less than disk size because activations, the KV cache, the CUDA
context and the fp16 embeddings / `lm_head` are not quantized. Latency got **worse**: p50 rose 30%, from 41 to 53 ms per
token. GPTQ is weight-only quantization, and the ExLlamaV2 kernel de-quantizes weights inside every matmul. For a 0.6B model
at batch 1 on a T4, that overhead outweighs the memory-bandwidth saving. The faster Marlin kernel needs an Ampere-class GPU.

---

## Part 2 — Knowledge distillation (7 points)

**Teacher:** the instructor's `teacher_model` (BERT-base fine-tuned on AG News, 109.5M parameters), loaded from local
files. Its logits were extracted for **all 120,000** AG News training samples (221.8 s).
**Student:** `distilbert/distilbert-base-uncased` (67.0M parameters).
**Loss:** `loss = α·CE(student, labels) + (1−α)·T²·KL(softmax(teacher/T) || softmax(student/T))`. The T² factor keeps the
soft-target gradients on the same scale across temperatures; at T = 1 it is exactly α·CE + (1−α)·KLD.
**Training (identical for every run):** the student is re-initialised from the same pretrained checkpoint with `SEED = 1605`.
Same data order, 1 epoch over 115,000 examples, batch 32, AdamW lr 5e-5, 6% linear warm-up, max length 128, fp16 autocast.
T and α were chosen on a 5,000-example validation split held out from the training set, not on the test set.

**Sample training log** (best run, T = 5, α = 0.25; running means over 200 steps; the full log is in the notebook):

| step | loss | CE | KLD (×T²) |
|---|---|---|---|
| 200 | 3.974 | 0.705 | 5.063 |
| 1000 | 0.756 | 0.318 | 0.903 |
| 2000 | 0.567 | 0.256 | 0.670 |
| 3000 | 0.493 | 0.231 | 0.580 |
| 3594 | 0.497 | 0.242 | 0.581 |

The loss falls steadily in all six runs. The loss curves for every run are plotted in the notebook. Final losses and
accuracies for all runs are in `METRICS.md`.

**Table HW11.2 — the separate distillation comparison on AG News** (same 7,600-example test set, same T4; latency at batch 1)

| Configuration | Accuracy | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Teacher (BERT-base) | 94.43% | 8.42 / 9.94 | 1.290 | 417.7 |
| Best distilled student (DistilBERT, T = 5, α = 0.25) | 94.25% | 4.33 / 7.09 | 1.001 | 255.4 |

**Effectiveness.** The student keeps **99.8% of the teacher's accuracy**: 94.25% vs. 94.43%, a gap of 0.18 points.
It has 39% fewer parameters, is 1.6× smaller on disk, uses 22% less peak memory, and is 1.9× faster per request
(p50 8.42 → 4.33 ms). The full test set also runs 1.9× faster in batches (14.0 → 7.5 s).

---

## Measurement and analysis (8 points)

### 1. Separate comparisons
GPTQ is compared only against its own base LLM on WikiText-2 (Table HW11.1). Distillation is compared only against its own
BERT teacher on AG News (Table HW11.2). Each comparison uses the same data and the same T4 GPU for both models.

### 2. Distillation temperature sweep (α = 0.5)

| T | val acc | test acc |
|---|---|---|
| 1 | 0.9412 | 0.9434 |
| 2 | 0.9410 | 0.9434 |
| **5** | **0.9426** | 0.9439 |
| 10 | 0.9420 | 0.9441 |

Best temperature by validation accuracy: **T = 5**. The four temperatures are within 0.16 validation points and 0.07 test
points of each other. Softening helps slightly over T = 1, but the effect is small at this accuracy level.

### 3. Loss-weighting sweep (T = 5)

| α | val acc | test acc |
|---|---|---|
| **0.25** | **0.9434** | 0.9425 |
| 0.5 | 0.9426 | 0.9439 |
| 0.75 | 0.9402 | 0.9441 |

Best by validation: **α = 0.25**, which puts the most weight on the teacher's soft targets. The test-set ordering is the
opposite (0.75 is highest), and all three runs lie within 0.16 points. An earlier run with SEED = 42 picked T = 10 and
α = 0.5. So with one seed these differences are within noise, and the weighting matters far less than distilling at all.

### 4. Where the compression hurts

| Class | Teacher | Student | Drop (pts) |
|---|---|---|---|
| World | 0.9489 | 0.9547 | −0.58 |
| Sports | 0.9884 | 0.9895 | −0.11 |
| **Business** | 0.9137 | 0.9000 | **+1.37** |
| Sci/Tech | 0.9263 | 0.9258 | +0.05 |

**Business** degrades most. Five examples the teacher gets right and the student gets wrong:

| Text (shortened) | Student prediction |
|---|---|
| Bush Pledges Strong-Dollar Policy — Bush meets Italian PM Berlusconi in the Oval Office | World |
| Discovery Channel Tests Power of Licensing — 700+ products under the *American Chopper* brand | Sci/Tech |
| PeopleSoft Plays Defense — business software maker inks a deal with IBM, unlikely to dissuade Oracle | Sci/Tech |
| Aussies battle EU over cheese, champagne — WTO interim ruling on protection of regional goods | World |
| Diesels, Hybrids Fated to Wed — joining hybrid motors with diesel engines | Sci/Tech |

**What they have in common:** all five are Business stories whose **surface topic belongs to another class**. Two are about
politics and trade (Bush, Berlusconi, WTO, EU) and the student called them World. Three are about companies whose product is
software, media or engines, and the student called them Sci/Tech. Their real subject is money: currency policy, trade
disputes, corporate deals, licensing revenue. The same pattern holds for all 59 such failures: 39 went to Sci/Tech and 16 to
World. The largest student confusion is Business → Sci/Tech (140 articles). The teacher was also less sure on these
(confidence 0.920 vs. 0.990 on its other correct Business predictions), so they sit near a class boundary. The smaller student
leans on strong topic words and misses the teacher's finer cue that the article is about the economic event.

### 5. Quantization vs. distillation
Quantization and distillation compress in different ways. GPTQ keeps Qwen3-0.6B's architecture and stores its weights in
4 bits without retraining. On the T4 that cut disk size 2.2× and peak memory 23%, at a 6% perplexity cost, but made generation
30% slower because de-quantization overhead dominates at this size and batch 1. Distillation trains a new model with half the
transformer layers. That cut disk size 1.6× and peak memory 22%, and roughly halved latency (8.42 → 4.33 ms), because it
removes real computation. Its costs are a teacher, teacher logits for the full training set and about 35 GPU-minutes of
training for the sweeps, against about 4 minutes of GPTQ calibration. For a **latency-bound** deployment I would try
**distillation first**. For a **memory-bound** deployment I would try **GPTQ first**, since it gives the largest
weight-memory reduction with no retraining. The two can also be stacked.
