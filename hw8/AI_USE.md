# AI Use Appendix — HW8

SID4 = 1605

> **Before submitting:** this file describes what actually happened in this AI-assisted session.
> Read it, check it against your own understanding, and rewrite it in your own words. You may be
> asked to explain any two cells on camera (Section 0.6), and a generic or unowned AI_USE.md earns
> zero (Section 0.5).

## What I used an assistant for vs. wrote myself

I used Claude Code (Claude Opus 5.5, Anthropic) for the whole assignment:

- the notebook `Mini_Diffusion_DDPM_DDIM.ipynb`: forward diffusion `q_sample`, the time-conditioned
  MiniUNet ε-network, the training loop with EMA, the DDPM ancestral sampler, the DDIM sampler
  (η = 0, arbitrary timestep sub-sequence), the judge classifier and metrics, the consistency
  checks, the speed benchmark, and the discussion;
- headless execution with `nbconvert`, and the standing artifacts (`METRICS.md`, `RUN_LOG.txt`, this file).

What I directed rather than delegated:

- **Keeping the assignment configuration.** Parts (a)–(d) use exactly T = 100, β linear
  1e-4 → 0.02, batch 128 and 50 epochs. The scaled-β model is a separately labelled extra
  section, not a replacement.
- **A fair comparison.** All samplers use the same trained network and start from the same x_T,
  and the extra model differs from the main one only in β.
- **Reviewing results.** I looked at every figure before accepting the numbers.

## One specific thing the assistant produced that was wrong

The first version of the training loop kept an EMA of the weights with a fixed decay of 0.999 and
sampled from the EMA. In the 1-epoch smoke test, **every sampler produced pure noise**. The EMA's
held-out noise-prediction MSE was 0.35, even though the raw model's training loss was 0.126.

## How I found it

The gap between those two numbers was the giveaway. A model whose training loss is 0.126 should
not score 0.35 on held-out data from the same distribution. After 469 updates with decay 0.999,
the EMA still puts 0.999^469 ≈ 63% of its weight on the **random initial weights**. Sampling the
same checkpoint with the raw (non-EMA) weights produced stroke-like images instead of noise, which
confirmed that the samplers were fine and the EMA was the problem.

## What I changed and why it's right now

The EMA decay now warms up: decay = min(0.999, (1 + step) / (10 + step)), the usual schedule. Early
in training the EMA tracks the model closely, and the decay reaches 0.999 only once there are
enough steps to average over. In the full run (23,450 steps) the EMA's held-out MSE (0.0517)
matches the final training loss (0.0518).

## Other issues hit along the way

- **x_T is not pure noise under the given schedule.** ᾱ_100 = 0.3636, so the network never sees
  N(0, I) at t = 100, yet both samplers start there. This is why the samples from the required
  model look like scribbles more than digits. It is explained in the notebook (including a plot of
  the std mismatch) and checked with the scaled-β model, where every metric improves sharply.
- **Wrong layer count in the text.** The markdown first said 15 conv layers; the code counts 18.
- **Invalid timings.** The laptop slept or throttled during the 22-hour run, which made wall-clock
  times meaningless (e.g., DDPM "3459 s"). Two re-measurements still came out with DDPM ≫ DDIM-100
  at equal NFE. Those were rejected, because the same number of network calls must cost the same
  time. The final measurement takes the minimum over 5 interleaved rounds, and DDPM ≈ DDIM-100 (25.8 s vs 24.5 s).
- **Tooling bug.** Running hidden setup cells through `nbclient.execute_cell(cell, 0)` overwrote
  the notebook's title cell. It was restored from a backup, and all other cells were checked
  against the backup.
- **MNIST download failed** with an SSL certificate error (python.org Python has no CA bundle). It
  was downloaded once using certifi's bundle; the MD5s match torchvision's published checksums.

## Verification performed

- The notebook executed top to bottom with `nbconvert` with 0 cell errors. Every number in
  `METRICS.md` and in the discussion cell was read from the cell outputs.
- DDIM determinism was checked directly: two runs from the same x_T give a max abs difference of 0.0.
- The judge classifier gets 98.8% on real test digits and scores real test images at 0.986 mean
  confidence and 3.310 bits class entropy. These give an upper reference for the sample metrics.
- The sanity check that equal NFE means equal time was used to accept or reject timing runs.
