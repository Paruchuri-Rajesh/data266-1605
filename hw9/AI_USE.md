# AI Use Appendix — HW9

SID4 = 1605

> **Before submitting:** this file describes what actually happened in this AI-assisted session.
> Read it, check it against your own understanding, and rewrite it in your own words. You may be
> asked to explain any two cells on camera (Section 0.6), and a generic or unowned AI_USE.md earns
> zero (Section 0.5).

## What I used an assistant for vs. wrote myself

I used Claude Code (Claude Opus 5.5, Anthropic) for the whole assignment:

- the notebook `Latent_Diffusion_FashionMNIST.ipynb`: VAE loading, encoding and decoding with the
  0.18215 scaling, the reconstruction-MSE check, the forward diffusion and its decoded visualization,
  the conditional latent U-Net (ResBlocks, self- and cross-attention, label embeddings with a null
  label), training with label dropout and EMA, the classifier-free-guidance DDPM sampler, the DDIM
  guidance sweep, the judge classifier and metrics, and the discussion;
- environment setup (a venv with diffusers and a Jupyter kernel), headless execution with
  `nbconvert`, and the standing artifacts (`METRICS.md`, `RUN_LOG.txt`, this file).

What I directed rather than delegated:

- **Personal parameters.** SEED = 1605, with every other seed in the notebook derived from it.
- **Keeping the assignment configuration.** s = 0.18215, T = 1000, linear β 1e-4 → 0.02, base 64 with
  multipliers (1, 2, 4), label dropout 0.1, 50 epochs at lr 2e-5, w = 3, and a 64-image grid with labels 0–9 repeated.
- **Model depth vs. run time.** Measured on this Mac, 50 epochs would take about 5 h with 1 ResBlock per
  level and about 7.5 h with 2. I chose the standard 2 ResBlocks per level rather than cutting epochs.
- **Reviewing results.** The assistant inspected every figure and cross-checked the write-up against the
  cell outputs. *(Edit this line once you have reviewed the figures and numbers yourself.)*

## Interpretation choices in the spec

- "Input: 1-channel image" conflicts with "U-Net operating on latents". The U-Net takes the VAE's
  4-channel 16×16 latent. The 1-channel image is the input to the VAE: upsampled to 128×128 and copied
  to 3 channels, because the SD VAE expects RGB. 128 px was chosen so that the latent is 16×16 and
  "self-attention at 16×16" is meaningful.
- The overview asks for cross-attention conditioning, while Part 3 asks for label embeddings combined with
  the time embedding. The model does both. Cross-attention uses 4 label tokens, because with a single key
  the softmax is always 1 and cross-attention reduces to adding a constant vector.

## One specific thing the assistant produced that was wrong

The first notebook version downsampled decoded 128×128 images to 28×28 with
`F.interpolate(x, size=28, mode="area")` on the MPS device. On MPS that call uses adaptive average
pooling, which only supports sizes where the input is a multiple of the output (128 is not a multiple
of 28), so the smoke test crashed in the reconstruction cell. The same smoke test also caught a second bug:
`loss_per_t` sliced latents and labels with the same `n`, but in smoke mode there were only 128
test latents, so the label tensor had a different length.

## How I found it

Both came from the end-to-end smoke run (`LDM_SMOKE=1`, small data subset, 1 epoch) done before the
long run. The first was a RuntimeError naming the MPS adaptive-pool limitation. The second was a
shape-mismatch RuntimeError (128 vs 1000) in `loss_per_t`.

## What I changed and why it's right now

`to28()` now moves the tensor to the CPU before the area resize. Every caller already used the result
on the CPU (for the judge input, metrics or plots), so nothing else changed. `loss_per_t` slices the
labels to `len(z0)`. The fourth smoke attempt ran with 0 errors, and so did the full run.

## Other issues hit along the way

- **Wrong Jupyter kernel.** The notebook first ran on the system `python3` kernel, which has no diffusers.
  A dedicated `hw9` kernel for the venv fixed it.
- **Dependency conflict.** Installing diffusers pulled huggingface-hub 2.x, which the system
  `transformers` (imported by diffusers) rejects. Pinned to <2.0 inside the venv.
- **Slow hardware.** The SD VAE and U-Net are compute-bound on this GPU. fp16 autocast did not help.
  Encoding the 60k training images took 90 min, so the latents are cached.
- **The laptop slept during the run.** `caffeinate` was active, but with the lid closed or on battery the
  Mac still slept. Training took 48.7 h of wall time for about 8.5 h of compute. This doesn't affect
  results, but the notebook's printed training time includes the sleep. `METRICS.md` and
  `RUN_LOG.txt` give the corrected figure.
- **The scaling factor isn't calibrated for this data.** With s = 0.18215 the FashionMNIST latents have
  std 1.27, not 1. This is kept because the assignment fixes s, and it is discussed in the notebook. The
  generated latents match the training std (1.274 vs 1.265).

## Verification performed

- The final notebook executed top to bottom once with `nbconvert` (exit 0, 0 cell errors, execution
  counts in order). Every number in `METRICS.md` and in the Discussion cell was read from the cell outputs.
- Reconstruction MSE is separated into the VAE's own error and the resize round trip, which is measured on its own.
- The judge classifier gets 91.3% on real test images and 91.6% on VAE reconstructions. These are
  reference points for the sample metrics.
- The 64-image grid was inspected visually, and the judge agreed with all 64 intended labels.
