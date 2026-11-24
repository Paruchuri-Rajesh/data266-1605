# HW12 Metrics

Google Colab, Tesla T4 (16 GB), Python 3.13.16, torch 2.14.1+cu130, transformers 5.24.1. **SEED = 1605 (SID4)**.
Notebook: `HW12_GPTQ_Distillation.ipynb` (executed, outputs included).

## Part 1 — GPTQ on `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`

GPTQ 4-bit, group size 128, `desc_act=False`, symmetric; 128 calibration passages from the WikiText-2 (`Salesforce/wikitext`,
`wikitext-2-raw-v1`) **train** split. Quality = sliding-window perplexity on the WikiText-2 **test** split (299,078 tokens,
context 1024, stride 512). Latency = greedy generation of exactly 64 new tokens, batch 1, 3 warm-up + 30 timed runs.
The quantized model was reloaded from disk with the ExLlamaV2 kernel (Marlin, the default, needs an Ampere+ GPU).

**Table HW11.1 — GPTQ comparison on the same LLM/task**

| Configuration | Quality metric (WikiText-2 PPL ↓) | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Base LLM (fp16) | 19.89 (loss 2.9900) | 2641.2 / 3333.2 | 2.679 | 1136.9 |
| GPTQ-quantized LLM (4-bit, g128) | 21.14 (loss 3.0512) | 3422.3 / 4186.7 | 2.073 | 516.3 |

Perplexity +6.3%; disk 2.20× smaller; peak memory −22.6%; p50 latency +29.6% (53.5 vs. 41.3 ms per token).
Quantization itself: 257.5 s, 1.55 GB peak.

## Part 2 — Knowledge distillation on AG News (`fancyzhx/ag_news`)

Teacher: instructor `teacher_model` (BERT-base, 109.5M params), loaded locally; teacher logits extracted for all 120,000
training samples (221.8 s; teacher train accuracy 98.95%). Student: `distilbert/distilbert-base-uncased` (67.0M), re-initialised
from the same checkpoint with SEED for every run. 115,000 train / 5,000 stratified validation (used to choose T and α) / 7,600 test.
1 epoch, batch 32, AdamW lr 5e-5, 6% linear warm-up, max length 128, fp16 autocast.
Loss = α·CE + (1−α)·T²·KL(teacher_T ‖ student_T).

### Measurement 2 — temperature sweep (α = 0.5)

| T | val acc | test acc | train time (min) |
|---|---|---|---|
| 1 | 0.9412 | 0.9434 | 5.8 |
| 2 | 0.9410 | 0.9434 | 5.9 |
| **5** | **0.9426** | 0.9439 | 6.0 |
| 10 | 0.9420 | 0.9441 | 5.8 |

Best temperature (by validation accuracy): **T = 5**.

### Measurement 3 — loss-weighting sweep (T = 5)

| α | val acc | test acc |
|---|---|---|
| **0.25** | **0.9434** | 0.9425 |
| 0.5 | 0.9426 | 0.9439 |
| 0.75 | 0.9402 | 0.9441 |

Best (by validation accuracy): **α = 0.25**. All six runs lie within 0.16 test points (94.25–94.41%), and within
0.32 validation points.

### Summary of all runs (final 200-step running means)

| run | final loss | final CE | final KLD (×T²) | val acc | test acc |
|---|---|---|---|---|---|
| T1_a0.5 | 0.1561 | 0.1746 | 0.1377 | 0.9412 | 0.9434 |
| T2_a0.5 | 0.2834 | 0.2026 | 0.3643 | 0.9410 | 0.9434 |
| T5_a0.5 | 0.4043 | 0.2307 | 0.5779 | 0.9426 | 0.9439 |
| T10_a0.5 | 0.3990 | 0.2317 | 0.5663 | 0.9420 | 0.9441 |
| T5_a0.25 | 0.4965 | 0.2418 | 0.5813 | 0.9434 | 0.9425 |
| T5_a0.75 | 0.3067 | 0.2135 | 0.5863 | 0.9402 | 0.9441 |

Loss falls steadily in every run (best run T5_a0.25: 3.97 at step 200 → 0.50 at step 3,594; CE 0.70 → 0.24, KLD 5.06 → 0.58).

**Table HW11.2 — the separate distillation comparison on AG News** (same test set, same T4; latency = batch 1, 500 texts)

| Configuration | Accuracy | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) | Params (M) | Full test set, batched (s) |
|---|---|---|---|---|---|---|
| Teacher (BERT-base) | 94.43% | 8.42 / 9.94 | 1.290 | 417.7 | 109.5 | 14.03 |
| Best distilled student (DistilBERT, T=5, α=0.25) | 94.25% | 4.33 / 7.09 | 1.001 | 255.4 | 67.0 | 7.53 |

### Measurement 4 — per-class accuracy

| Class | Teacher | Student | Drop (pts) |
|---|---|---|---|
| World | 0.9489 | 0.9547 | −0.58 |
| Sports | 0.9884 | 0.9895 | −0.11 |
| **Business** | 0.9137 | 0.9000 | **+1.37** |
| Sci/Tech | 0.9263 | 0.9258 | +0.05 |

Most-degraded class: Business. 59 Business test articles are right for the teacher and wrong for the student: 39 → Sci/Tech,
16 → World, 4 → Sports. Teacher confidence on them 0.920 vs. 0.990 on all teacher-correct Business articles.
Largest student confusion: Business → Sci/Tech (140).
