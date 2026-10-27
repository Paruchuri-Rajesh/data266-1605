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

- **Personal parameters.** SEED = 1605. The first full run used SEED = 0; I had it re-run from
  scratch with SEED = 1605, with every other seed in the notebook derived from it.
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
enough steps to average over. In the final run (23,450 steps) the EMA's held-out MSE (0.0516)
matches the final training loss (0.0521).

## Other issues hit along the way

- **x_T is not pure noise under the given schedule.** ᾱ_100 = 0.3636, so the network never sees
  N(0, I) at t = 100, yet both samplers start there. This is why the samples from the required
  model look like scribbles more than digits. It is explained in the notebook (including a plot of
  the std mismatch) and checked with the scaled-β model, where confidence rises from about 0.78 to
  about 0.93 and pixel KL drops by 50–200×.
- **Wrong layer count in the text.** The markdown first said 15 conv layers; the code counts 18.
- **Invalid timings in the first (SEED = 0) run.** The laptop slept or throttled on battery, so
  that run took 22 hours and its wall-clock times were meaningless. Re-measurements with medians
  still had DDPM 3.6–6× slower than DDIM-100 at equal NFE. Those were rejected, because the same
  number of network calls must cost the same time. The notebook now times samplers interleaved over
  5 rounds and takes the minimum. In the final run, DDPM and DDIM-100 agree (16.77 s vs 16.59 s).
- **Tooling bug.** While patching that first run, `nbclient.execute_cell(cell, 0)` overwrote the
  notebook's title cell. It was restored from a backup. The final notebook comes from one clean
  top-to-bottom execution, so this does not affect it.
- **MNIST download failed** with an SSL certificate error (python.org Python has no CA bundle). It
  was downloaded once using certifi's bundle; the MD5s match torchvision's published checksums.

## Verification performed

- The final notebook executed top to bottom once with `nbconvert` (3 h 02 min, 0 cell errors,
  execution counts in order). Every number in `METRICS.md` and in the discussion cell was read
  from the cell outputs.
- DDIM determinism was checked directly: two runs from the same x_T give a max abs difference of 0.0.
- The judge classifier gets 99.07% on real test digits and scores real test images at 0.988 mean
  confidence and 3.309 bits class entropy. These give an upper reference for the sample metrics.
- The sanity check that equal NFE means equal time holds in the final run, both in the benchmark
  (16.77 s vs 16.59 s) and in the single-pass evaluation (67.07 s vs 67.09 s).
