# AI Use Appendix — HW12

SID4 = 1605 | SEED used in the executed run = **42** (see "Seed" below)

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
- drafting the Measurement 4 answer from the printed outputs, and writing `METRICS.md`, `RUN_LOG.txt` and this file.

The notebook was executed by me in Google Colab on a T4 GPU; the assistant had no GPU and did not run it.

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

## Seed

The executed run used **SEED = 42**. The standing course rule (used in HW11) is SEED = 1605 (SID4). This was noticed after the
run finished; a re-run with 1605 had not been completed at submission time. The config cell in the committed notebook shows
42 so that the code matches its outputs. Changing the seed changes the GPTQ calibration sample, the validation split, the data
order and the student initialisation, so the exact numbers (and possibly the chosen T) could change slightly with 1605.

## What to check yourself

- Read the five Measurement 4 examples and decide whether you agree with the explanation (Business stories about technology
  companies confused with Sci/Tech). It is an interpretation, not a measured fact.
- The temperature/α differences are small (all six runs within 0.45 test points); with one seed, "best T = 10" is weak evidence.
- GPTQ was slower than fp16 here; that is specific to a small model at batch 1 with the ExLlamaV2 kernel on a T4.
