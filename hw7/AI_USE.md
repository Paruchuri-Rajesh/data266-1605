# AI Use Appendix — HW7

SID4 = 1605

> **Before submitting:** this file describes what actually happened in this AI-assisted session.
> Read it, check it against your own understanding, and rewrite it in your own words. You may be
> asked to explain any two cells on camera (Section 0.6), and a generic or unowned AI_USE.md earns
> zero (Section 0.5).

## What I used an assistant for vs. wrote myself

I used Claude Code (Claude Opus 5.5, Anthropic) for the whole assignment:

- the written answers in `report.md` (Parts 1–3: GAN architecture, mode collapse, GAN variants);
- the notebook `autoencoders_fashionmnist.ipynb` (data split, FC-AE and VAE models, training
  loop, SSIM/PSNR metrics, linear probe, t-SNE, evaluation CNN, C-FID / Inception-style scoring,
  latent interpolation);
- environment setup (an isolated `.venv/` registered as the `hw7` kernel) and headless execution
  with `nbconvert`;
- the standing artifacts (`METRICS.md`, `RUN_LOG.txt`, this file).

What I directed rather than delegated:

- **Personal parameters.** SEED = 1605 for the split and the 3-seed repeats. CLS_A = 5 (Sandal) and
  CLS_B = 3 (Dress) define the interpolation experiment. Per the assignment, no HP_ID second model
  was trained.
- **A fair comparison.** Both models have identical layers, latent size (32) and reconstruction
  loss, so the only difference is the VAE's KL term.
- **Reviewing results.** I looked at every figure before accepting the numbers.

## One specific thing the assistant produced that was wrong

The training-curve figure compared the VAE's **train** and **validation** reconstruction BCE as if
they measured the same thing. They don't. The training loop logs BCE for a reconstruction decoded
from a **sampled** $z=\mu+\sigma\varepsilon$, which is what the VAE is trained on. The validation
`evaluate()` decodes the deterministic **posterior mean $\mu$**. The first version labelled the
plot simply "solid = val, dashed = train". Its numbers look plausible but are not like-for-like
(seed 1605, epoch 30, from `outputs/histories.json`):

```
vae_seed1605  epoch 30  train_bce 225.115   val_bce 222.657
fcae_seed1605 epoch 30  train_bce 207.402   val_bce 207.875
```

Read naively, the VAE has *lower* validation loss than training loss by ~2.5 nats, i.e. it looks
under-fit, or the validation set looks "easier". The FC-AE shows the normal pattern (val slightly
above train).

## How I found it

The asymmetry was the giveaway. Both models train on the same 54k images and validate on the same
6k images, so an "easier" validation set would also show up for the FC-AE, and it doesn't. Checking
the code showed that `train_model` uses `model(x)`, which samples $z$, while `evaluate` uses
`model.decode(model.encode(x))`, which uses $\mu$. Decoding a noisy $z$ is expected to reconstruct
worse than decoding $\mu$, so most of the gap is the sampling noise, not under-fitting.

## What I changed and why it's right now

Reconstructions are still reported from $\mu$. That is the standard way to measure a VAE's
reconstruction quality, and it is deterministic, so it is a fair comparison with the FC-AE. The
figure title now states the difference explicitly: "solid = val (VAE decodes μ), dashed = train
(VAE: sampled z)". The discussion says the gap should not be read as under-fitting. The −ELBO
(which needs a sample) is reported separately and labelled as a one-sample estimate. This was a
labelling and interpretation fix, so no model was retrained; the notebook was re-executed from the
saved checkpoints.

## Other issues hit along the way

- **Crash in the smoke test:** `RuntimeError: Can't call numpy() on Tensor that requires grad`.
  The latent-statistics cell called `VAE.encode_dist()` outside `torch.no_grad()`. Fixed by
  wrapping it in `no_grad`.
- **Singular covariance in C-FID.** scipy warned `LinAlgWarning: Matrix is singular`, because 49
  of the evaluation CNN's 128 penultimate ReLU features are always zero. I added a 1e-6·I ridge to
  both covariances (the same fix pytorch-fid uses) and an assertion that the matrix square root is
  finite. I then checked on the final models that the fix leaves the values unchanged: VAE 108.980
  → 108.980, FC-AE 216.690 → 216.689. So the earlier numbers were not actually corrupted; the
  warning is gone.
- **A misleading smoke-test number.** After 1 epoch the VAE had 31/32 active latent dims. After 30
  epochs it has only 6/32. If the smoke run had been reported, the conclusion about posterior
  collapse would have been the opposite. Only full-run numbers are reported.

## Verification performed

- The notebook was executed top to bottom twice with `nbconvert`, with 0 cell errors. The final
  run reloaded the checkpoints; their mtimes were unchanged, so nothing was retrained. Every
  number in `METRICS.md`, `report.md` and the discussion cell was read from those outputs or from
  `outputs/all_outputs.json`.
- The SSIM implementation asserts SSIM(x, x) = 1. The real-vs-real C-FID reference is 0.50, so
  the metric is near zero for matching distributions.
- The FashionMNIST MD5s match torchvision's published checksums.
- The 3-seed std is small relative to every reported gap. For example, MSE is 0.00807 ± 0.00007
  vs 0.01484 ± 0.00006, so the conclusions do not depend on one lucky seed.
