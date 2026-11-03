# HW4.a + HW4.5 — GPU Assignment II: Training Throughput and Memory Engineering

## HW4.5 clarifications (TA announcement, Oct 23) and how this repo follows them

| Clarification | What the code does |
|---|---|
| 1. Part C (two GPUs at once) is optional | Part C is not part of `run_card.sh`. If `results/*/partC/` is absent, `make_report.py` writes **"Part C skipped (optional)"** in METRICS.md. `run_ddp.sh` is still there for a Kaggle 2× T4 run. |
| 2. Same data as HW4.a: same Twitter dataset, same rows (≤ 5,000 consecutive from SLICE), same 70/15/15 split with the same SEED; state dataset name and exact row range | SLICE = 605 and SEED = 1605 (from SID4 = 1605) are the defaults in `run_card.sh` / `run_ddp.sh`. The HW4.a run writes `data/hw4a_data_lock.json` (file sha256, row range, labels, SEED, split hash). Every HW4.5 run (Parts A, B, C) re-derives the split and **exits with an error if anything differs**. METRICS.md opens with the dataset name, exact row range (0-based rows and CSV lines), and a same-data check over every result file. |
| 3. Any one GPU (lab RTX 5090/4090, Colab, Kaggle, own); record which with a screenshot or `nvidia-smi -L` | `capture_env.sh` saves `nvidia-smi -L` (+ full `nvidia-smi`, versions) to `env/uuid_<card>.txt`, and METRICS.md quotes the `nvidia-smi -L` lines. Part A's two-card comparison (ratio of max batch vs VRAM) is reported as not applicable with one card. |

All measurements are produced by the scripts below and written to `results/<card>/…json`;
`make_report.py` turns them into `figures/` and `METRICS.md`. No number is typed by hand.

## Requirement → where it is done

| Requirement | Script / output |
|---|---|
| 4.a.1 BERT classifier (BertTokenizer → TensorDataset → BertForSequenceClassification, train, eval, save, reload, custom inference) | `train_bert.py` + `bert_common.py` → `results/<card>/hw4a.json`, `models/` |
| 4.a.2 vLLM inference, different HF model, own prompt | `serve_bench.py demo` (default model `Qwen/Qwen2.5-0.5B-Instruct`) |
| 4.a.3.1 5,000 rows from SLICE, exact row range, 70/15/15 split with SEED | `bert_common.load_slice`, `split_70_15_15` (range printed in METRICS.md) |
| 4.a.3.2 per-epoch losses/acc, epoch where val stopped improving, cost of training past it | `early_stop_summary` |
| 4.a.3.3 confusion matrix, per-class P/R, worst class + 5 misclassified | `test_report` (**you write the 5 one-line diagnoses**) |
| 4.a.3.4 peak memory + wall-clock (HW4.5 baseline) | `hw4a.json → metrics` |
| 4.a.3.5 Table HW4.1: HF pipeline b1 vs vLLM b1/8/32, tok/s, p50, p95 | `serve_bench.py hf`, `serve_bench.py vllm` |
| Part A batch scaling to OOM, plot, knee, max-batch vs VRAM ratio | `sweep.py partA` → `figures/partA_throughput_vs_batch.png` |
| Part B AMP / grad ckpt / grad accum / 8-bit AdamW / best combo + ranking | `sweep.py partB` → Table HW4.5.1 + ranking |
| Part C DDP (**optional**): topology, scaling efficiency, comm overhead + bandwidth, break-even batch | `ddp_bert.py` via `run_ddp.sh`; skipped → "Part C skipped (optional)" |
| Part D vLLM on your card, KV cache vs context, longest context at batch 8 | `serve_bench.py vllm` + `serve_bench.py kv` |
| Part E GPU-hours, chosen config, HW4.5 summary | generated in `METRICS.md` (Part E) |
| UUID captures | `capture_env.sh` → `env/uuid_<card>.txt`, `env/topo_<card>.txt` |

## Setup (each machine)

```bash
pip install -r requirements.txt          # RTX 5090: install torch from the cu128 index first (see file)
huggingface-cli login                    # only if you choose a gated model
python get_data.py                       # Twitter US Airline Sentiment -> data/Tweets.csv (prints sha256)
# Defaults: SLICE=605, SEED=1605. Only if your HW4.a used other values: export SLICE=… SEED=…
# Another Twitter file: pass DATA=… and --text-col/--label-col (it must be the one HW4.a used).
```

## Run

```bash
./run_card.sh rtx4090                  # env, hw4a, serve, partA, partB, partD on one card (one GPU is enough)
./run_card.sh rtx4090 partB            # or any subset: env hw4a serve partA partB partD (run hw4a first)
./run_ddp.sh kaggle_t4x2               # OPTIONAL Part C, only with two GPUs (1-GPU baseline, then 2 GPUs)
python make_report.py                  # figures/ + METRICS.md from everything under results/
```

Kaggle / Colab: zip this folder as `hw45.zip`, upload it, and run `HW4_5_run.ipynb` top to bottom (one cell per part).
Kaggle needs *Internet on*. Kaggle's "GPU T4 x2" is a homogeneous single-node PCIe pair if you choose to do the optional Part C.
T4s have no bf16, so `--amp auto` falls back to fp16 with a GradScaler, and vLLM serves in float16. If vLLM will not start on a
T4 (Turing, compute 7.5), run `serve`/`partD` on a Colab L4/A100 or a lab card; the training parts are unaffected.

Rough cost on an RTX 4090: HW4.a ≈ 2 min, Part A ≈ 15–25 min (batch 1 is a full 3,500-step epoch), Part B ≈ 20–30 min,
Part D ≈ 15 min (most of it loading the 7B model for the KV test). Record actual hours in `GPU_HOURS.md`.

## Measurement definitions

- **Throughput (samples/s)**: training samples ÷ pure training time (validation excluded). For Part A batches larger
  than the 3,500-sample train split, and for the Part B max-batch search, a 10-step probe on batches drawn with
  replacement is used. The first 3 steps are excluded, and time/epoch is then *estimated* as train size ÷ throughput (marked "est.").
- **Peak memory**: `torch.cuda.max_memory_allocated()` (tensors). Reserved memory (what the caching allocator holds and is close to
  what `nvidia-smi` shows, minus the CUDA context) is reported in Part A too.
- **Max batch**: largest power of two that completes; each batch size runs in its own process so an OOM can't affect the next.
- **Gradient accumulation**: physical 8 × 16 steps = effective 128, compared against a direct batch-128 run with the same number
  of optimiser steps (`--accum-physical`, `--accum-steps` in `sweep.py`).
- **Serving latency**: a batch of B requests is submitted together and finishes together, so each request's latency is the
  batch's wall time. p50/p95 are taken over 64 requests. Every request generates exactly 128 tokens (greedy, `ignore_eos`).
- **KV cache (Part D.2)**: bytes/token = 2 × layers × KV heads × head_dim × dtype bytes, from the model config. Capacity is read from
  vLLM's `GPU KV cache size: N tokens` log line, longest context at batch 8 = min(capacity ÷ 8, model max length), and that length is then
  verified by a real batch-8 run. The default `KV_MODEL` is `Qwen/Qwen2.5-7B-Instruct`, because the 0.5B model's KV cache is too small to
  hit the memory limit on a 24/32 GB card. On a card under 20 GB (e.g. a 16 GB T4) `run_card.sh` uses `Qwen/Qwen2.5-1.5B-Instruct` instead. Override with `KV_MODEL=… ./run_card.sh <card> partD`.
- **DDP comm overhead**: (DDP step time) − (the same step inside `model.no_sync()`), i.e. communication that backward does not hide.
  The slower rank gates the step, and the per-rank no-sync compute time is reported as straggler wait for heterogeneous pairs.
  A raw `all_reduce` of the 438 MB gradient gives the bus bandwidth (NCCL busbw convention).

## Deliverables checklist

- [ ] `git init && git add -A && git commit -m "HW4.5" && git tag hw4-5` (push to your course repo)
- [ ] `env/uuid_<card>.txt` (`nvidia-smi -L`) for the GPU you used (and the DDP pair only if you did Part C); a screenshot is also accepted
- [ ] `data/hw4a_data_lock.json` committed; METRICS.md "Same-data check" shows N of N
- [ ] `figures/` and `METRICS.md` regenerated after the last run. Written answers and diagnoses live in `ANSWERS.md` (one `## key` section per TODO); `make_report.py` pulls them in, so they survive regeneration.
- [ ] If Part C was skipped, METRICS.md says **Part C skipped (optional)** (automatic)
- [ ] `RUN_LOG.txt` (appended automatically by the run scripts)
- [ ] `AI_USE.md`, `RESERVATION.md`, `GPU_HOURS.md` completed
