# GPU-hour record

GPU-hours = wall-clock hours the GPU was held × number of GPUs. Start/end times are in RUN_LOG.txt (`=== [time] [card] start/end` lines),
in each container's clock (UTC). One GPU at a time (Kaggle T4, later a Colab T4 for the fp16 rerun); Part C skipped (optional).

| Date | Card | # GPUs | Start | End | Hours | GPU-hours | Work |
|---|---|---|---|---|---|---|---|
| 2026-11-03 | Kaggle Tesla T4 (GPU-8cd406d2-0d44-2d5f-f41e-0f7b6d2d317a) | 1 | 07:54:59 | 08:06:04 | 0.18 | 0.18 | First HW4.a run (on transformers 5, superseded) and failed vLLM start-ups before the stack was pinned |
| 2026-11-03 | Kaggle Tesla T4 (same GPU) | 1 | 08:06:04 | 10:59:58 | 2.90 | 2.90 | env, HW4.a, serve, Part A, Part B, Part D (includes ~25 min idle between Part B and Part D) |
| 2026-11-03 | Google Colab Tesla T4 (GPU-e3bcdb6e-864d-471e-7372-88021ba40a10) | 1 | 13:50:49 | 14:11:21 | 0.34 | 0.34 | Part B fp16 rerun: baseline, mixed precision, best combination, best + grad ckpt |
| 2026-11-03 | Kaggle Tesla T4 (GPU-82d48319-8dec-9858-1ea9-8962c5cd7219, first session) | 1 | 05:06:10 | 05:17:08 | 0.18 | 0.18 | First serving test (HF + vLLM batch 1/8/32); its results were lost when Kaggle restarted the session and were re-measured later. Times are from the vLLM log of that session; install/debug time before it was not logged |
| **Total** | | | | | | **3.60** | |

The Part E estimate for one full fine-tune (computed from measured throughput) is in METRICS.md.
