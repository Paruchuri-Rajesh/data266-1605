# HW12 Metrics

Google Colab, Tesla T4 (16 GB), Python 3.13.16, torch 2.14.1+cu130, transformers 5.24.1. **SEED = 42** for this run.
The course rule is SEED = 1605 (SID4); see `AI_USE.md`. Notebook: `HW12_GPTQ_Distillation.ipynb` (executed, outputs included).

## Part 1 — GPTQ on `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`

GPTQ 4-bit, group size 128, `desc_act=False`, symmetric; 128 calibration passages from the WikiText-2 (`Salesforce/wikitext`,
`wikitext-2-raw-v1`) **train** split. Quality = sliding-window perplexity on the WikiText-2 **test** split (299,078 tokens,
context 1024, stride 512). Latency = greedy generation of exactly 64 new tokens, batch 1, 3 warm-up + 30 timed runs.
The quantized model was reloaded from disk with the ExLlamaV2 kernel (Marlin, the default, needs an Ampere+ GPU).

**Table HW11.1 — GPTQ comparison on the same LLM/task**

| Configuration | Quality metric (WikiText-2 PPL ↓) | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Base LLM (fp16) | 19.89 (loss 2.9900) | 3195.9 / 3623.9 | 2.679 | 1136.9 |
| GPTQ-quantized LLM (4-bit, g128) | 21.11 (loss 3.0499) | 3968.9 / 4765.7 | 2.578 | 516.3 |

Perplexity +6.2%; disk 2.20× smaller; peak memory −3.8%; p50 latency +24.2% (ExLlamaV2 de-quantization overhead outweighs
the bandwidth saving for a 0.6B model at batch 1). Quantization itself: 305.6 s, 1.55 GB peak.

## Part 2 — Knowledge distillation on AG News (`fancyzhx/ag_news`)

Teacher: instructor `teacher_model` (BERT-base, 109.5M params), loaded locally; teacher logits extracted for all 120,000
training samples (251.9 s; teacher train accuracy 98.95%). Student: `distilbert/distilbert-base-uncased` (67.0M), re-initialised
from the same checkpoint with SEED for every run. 115,000 train / 5,000 stratified validation (used to choose T and α) / 7,600 test.
1 epoch, batch 32, AdamW lr 5e-5, 6% linear warm-up, max length 128, fp16 autocast.
Loss = α·CE + (1−α)·T²·KL(teacher_T ‖ student_T).

### Temperature sweep (α = 0.5)

| T | val acc | test acc | train time (min) |
|---|---|---|---|
| 1 | 0.9438 | 0.9424 | 6.1 |
| 2 | 0.9432 | 0.9439 | 6.0 |
| 5 | 0.9440 | 0.9434 | 6.0 |
| **10** | **0.9456** | **0.9462** | 6.0 |

Best temperature: **T = 10**.

### Loss-weighting sweep (T = 10)

| α | val acc | test acc |
|---|---|---|
| 0.25 | 0.9440 | 0.9417 |
| **0.5** | **0.9456** | **0.9462** |
| 0.75 | 0.9446 | 0.9437 |

Best: **α = 0.5**. All six runs lie within 0.45 test points of each other.

### Summary of all runs (final 200-step running means)

| run | final loss | final CE | final KLD (×T²) | val acc | test acc |
|---|---|---|---|---|---|
| T1_a0.5 | 0.1495 | 0.1687 | 0.1302 | 0.9438 | 0.9424 |
| T2_a0.5 | 0.2671 | 0.1903 | 0.3438 | 0.9432 | 0.9439 |
| T5_a0.5 | 0.3782 | 0.2133 | 0.5431 | 0.9440 | 0.9434 |
| T10_a0.5 | 0.3789 | 0.2198 | 0.5380 | 0.9456 | 0.9462 |
| T10_a0.25 | 0.4656 | 0.2339 | 0.5428 | 0.9440 | 0.9417 |
| T10_a0.75 | 0.2954 | 0.2045 | 0.5680 | 0.9446 | 0.9437 |

Loss falls steadily in every run (e.g. T10_a0.5: 2.93 at step 200 → 0.38 at step 3,594).

**Table HW11.2 — the separate distillation comparison on AG News** (same test set, same T4; latency = batch 1, 500 texts)

| Configuration | Accuracy | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) | Params (M) | Full test set, batched (s) |
|---|---|---|---|---|---|---|
| Teacher (BERT-base) | 94.43% | 10.09 / 16.41 | 1.794 | 417.7 | 109.5 | 15.26 |
| Best distilled student (DistilBERT, T=10, α=0.5) | 94.62% | 4.96 / 6.80 | 1.505 | 255.4 | 67.0 | 9.07 |

### Per-class accuracy (Measurement 4)

| Class | Teacher | Student | Drop (pts) |
|---|---|---|---|
| World | 0.9489 | 0.9505 | −0.16 |
| Sports | 0.9884 | 0.9884 | 0.00 |
| **Business** | 0.9137 | 0.9116 | **+0.21** |
| Sci/Tech | 0.9263 | 0.9342 | −0.79 |

Most-degraded class: Business. 40 Business test articles are right for the teacher and wrong for the student: 28 → Sci/Tech,
9 → World, 3 → Sports. Teacher confidence on them 0.882 vs. 0.990 on all teacher-correct Business articles.
Largest student confusion: Business → Sci/Tech (128).
