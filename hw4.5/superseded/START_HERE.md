# Superseded results (kept for the record, not used in METRICS.md)

`kaggle_t4_partB_emulated_bf16/` holds the first Part B runs that used `--amp auto` (mixed precision, best combination,
best combination + grad ckpt) on the Kaggle T4, 2026-11-03 08:47–10:19 UTC. `--amp auto` picked bf16 because
`torch.cuda.is_bf16_supported()` is True on a T4 even though compute 7.5 only emulates bf16, so these runs were 28% slower than fp32.
They were rerun with fp16 + GradScaler on a Google Colab T4 (`results/colab_t4/partB/`) together with a same-GPU fp32 baseline.
See AI_USE.md.
