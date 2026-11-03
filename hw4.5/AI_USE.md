# AI Use Appendix — HW4.a + HW4.5

SID4 = 1605 | SLICE = 605 | SEED = 1605

## What I used an assistant for

I used Claude Code (Claude Opus 5.5, Anthropic) for three things:

- **Code:** it generated the experiment scripts (`bert_common.py`, `train_bert.py`, `sweep.py`, `serve_bench.py`,
  `make_report.py`, the run scripts and the notebook). It also updated them for the TA's Oct 23 clarifications:
  Part C optional, the same-data lock, and `nvidia-smi -L` capture.
- **Troubleshooting during the runs:** fixing errors on Kaggle and Colab. These were package versions (vLLM 0.11.0 +
  `transformers<5`, a mismatched torchaudio), a vLLM crash on the T4 (switched to the Triton attention backend), and the
  mixed-precision bug described below.
- **Write-up:** it drafted the written analysis in `ANSWERS.md` (diagnoses, Q6, Parts B, D and E) from my measured
  results, and filled `GPU_HOURS.md` / `RESERVATION.md` from RUN_LOG.txt.

The assistant ran no GPU experiment and produced no measured values. Every number in METRICS.md comes from my runs
(see RUN_LOG.txt and `results/`).

## What I did myself

- Ran every experiment myself on free cloud GPUs, one at a time:
  - Kaggle Tesla T4: HW4.a, serving, Parts A, B and D;
  - Google Colab Tesla T4: the Part B fp16 rerun.

  The GPUs are recorded in `env/uuid_kaggle_t4.txt` and `env/uuid_colab_t4.txt`, and the full output is in RUN_LOG.txt.
- Set up the Kaggle account (phone verification), uploaded the project and dataset, and chose the GPU runtimes.
- Ran the notebook cells and dealt with session restarts by reinstalling and rerunning the lost parts.
- Reported each error back to the assistant (pip conflicts, torchaudio CUDA mismatch, tokenizer error, vLLM crash).
- Downloaded the results and backups after each part.

## Things the assistant produced that were wrong, how they were found, and the fixes

**Mixed precision on the T4.** `--amp auto` chose bf16 whenever `torch.cuda.is_bf16_supported()` was True. On torch 2.8
that is also True on a T4 (compute 7.5), where bf16 is only emulated. Part B's first AMP run was therefore 28% *slower*
than fp32 and used more memory, which is the opposite of what fp16 tensor cores should do. That result is how the bug
was found. `--amp auto` now picks bf16 only on compute capability ≥ 8.0 (fp16 + GradScaler otherwise), and each result
records the dtype it used (`amp_dtype`). The AMP runs were rerun on a Google Colab T4 with `sweep.py partB --only baseline amp best best_ckpt`. fp16 then ran 3.3×
faster than fp32 on the same GPU (171.9 vs 52.4 samples/s). The invalid runs are kept in `superseded/`.

**Part E epochs.** The Part E GPU-hour estimate in `make_report.py` multiplied by a hard-coded `--full-epochs 4` default instead of the epoch count of
the HW4 run. It was found when the report was rendered from a CPU smoke test that used 2 epochs. The default is now 0, which means
"use the HW4 baseline's `epochs`", so the estimate always matches the settings that were measured.

## Verification performed (on a Mac, no NVIDIA GPU)

A CPU smoke test with `prajjwal1/bert-tiny` checked the following. These are code checks, not results.

- The HW4.a run writes the lock, and a later run with the same SLICE/SEED passes.
- A run with SLICE = 700, or with SEED = 42, exits with an error naming the mismatched fields.
- `make_report.py` renders the Data/GPU section, the single-GPU Part A note, Part B, "Part C skipped (optional)" and Part E.
  Part B and Part E were rendered from synthetic copies of the smoke result, which are not submitted.
