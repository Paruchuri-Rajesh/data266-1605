# AI Use Appendix — HW12

SID4 = 1605 | SEED = 1605

> **Before submitting:** this file describes what actually happened in this AI-assisted session.
> Read it, check it against your own understanding, and rewrite it in your own words. Be ready to
> explain any cell in the notebook yourself.

## What the assistant did

I used Claude Code (Claude Opus 5.5, Anthropic) for this assignment:

- writing `HW12_GPTQ_Distillation.ipynb`: the install/config/helper cells, Part 1 (GPTQ quantization through
  `transformers.GPTQConfig`, sliding-window WikiText-2 perplexity, p50/p95 generation latency, peak GPU memory, size on disk),
  Part 2 (teacher-logit extraction, the α·CE + (1−α)·T²·KL training loop, the temperature and α sweeps, evaluation),
  the measurement tables and the auto-generated findings;
- debugging the Colab environment step by step from screenshots (see `RUN_LOG.txt`);
- writing the Measurement 4 answer from the printed outputs, and `report.md`, `METRICS.md`, `RUN_LOG.txt` and this file.

I executed the notebook in Google Colab on a T4 GPU. The assistant had no GPU and did not run it.

## Design decisions (and why)

- **Qwen/Qwen3-0.6B** as the base LLM: it is the model used in the Hugging Face GPTQ guide linked from the assignment.
  The exact revision is recorded in the notebook.
- **Calibration on WikiText-2 train, evaluation on WikiText-2 test**, so no evaluation text is seen during quantization.
- **T² scaling of the KL term** (Hinton et al.), so α means the same thing at every temperature; at T = 1 the loss is exactly
  α·CE + (1−α)·KLD.
- **T and α chosen on a 5,000-example validation split** carved from AG News train, not on the test set; test accuracy is
  reported for every run.
- **Re-saving both classifiers with `save_pretrained`** before measuring disk size, so the format is identical.

## Things that went wrong, and how they were caught

1. **CUDA mismatch** between an upgraded torch and Colab's torchaudio broke `import transformers`; fixed by pinning torch and
   removing torchaudio.
2. **Marlin kernel on a T4.** GPTQModel picked a kernel that needs an Ampere GPU; the reload now tries ExLlamaV2 / Triton / Torch
   and records the one used (ExLlamaV2).
3. **teacher_model access.** The Drive folder was not public and the Drive mount failed, so it was uploaded as a zip.
4. **Wrong seed.** The first complete run used SEED = 42. Before submitting, the earlier homeworks in the repo showed the
   standing rule SEED = 1605 (SID4), so the whole notebook was re-run with it and every number and written answer was
   rewritten from the new outputs. The seed changed the chosen T (10 → 5) and α (0.5 → 0.25) and the Measurement 4 examples.
5. **A notebook with a changed seed but old outputs.** One downloaded copy had `SEED = 1605` typed into the config cell but
   still showed the SEED = 42 outputs; the assistant caught this by diffing outputs against the earlier run and did not
   commit it as a 1605 run.

## What to check yourself

- Read the five Measurement 4 examples and decide whether you agree with the explanation (Business stories whose surface topic
  is politics/trade or technology). It is an interpretation, not a measured fact.
- The temperature/α differences are smaller than the seed-to-seed variation (best T was 10 with SEED 42 and 5 with SEED 1605);
  "best T" from one seed is weak evidence.
- GPTQ was slower than fp16 here; that is specific to a small model at batch 1 with the ExLlamaV2 kernel on a T4.
