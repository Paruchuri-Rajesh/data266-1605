# AI Use Appendix — HW6

SID4 = 1605

## What I used an assistant for vs. wrote myself

I used Claude Code (Claude Opus 5.5, Anthropic) to implement the assignment end to end: the notebook
`stl10_ssl.ipynb` and the helper module `hw6_lib.py`. That covers STL-10 loading, the stratified
500-image labeled subset, Part A supervised training, Part B rotation pretext training, Part C SimCLR
training, the frozen-encoder linear probes, and Part D nearest-neighbour retrieval. It also covers
environment setup (an isolated `.venv/` registered as the `hw6` Jupyter kernel) and executing the
notebook headless with `nbconvert`.

What I directed rather than delegated:

- **Compute budget.** A throughput benchmark showed ResNet-18 at 96×96 trains at only ~330 img/s on
  this laptop's MPS GPU, so the full run takes hours. I chose to keep STL-10's native resolution
  rather than downsample to 64×64.
- **Reusing trained weights after a bug fix.** After the linear-probe bug below was found, I chose to
  reload the three trained encoders and re-execute everything downstream of them, instead of
  retraining all three models (~3 h). See `RUN_LOG.txt`.
- **Reviewing results.** I looked at the figures and training curves before accepting any number.

## One specific thing the assistant produced that was wrong

The first version of the linear probe, used for Parts B and C, fed the frozen encoder's raw 512-d
features to `nn.Linear(512, 10)`: Adam, lr 3e-3, batch 50, 20 epochs = 200 steps. The run completed
with no errors and printed a plausible-looking number:

```
[B rotation probe] epoch 20/20 loss 2.0633 train_acc 0.250
[B rotation probe] encoder state_dict hash unchanged (117915bfb9f1...) -> frozen. TEST ACC = 0.2460
```

24.6% looks like a believable "rotation features are bad" result, and rotation features from the last
ResNet layer *are* known to be weak. That made it easy to accept.

## How I discovered the failure

The **training** accuracy in the log was the giveaway. A linear classifier with 5,130 parameters on
500 images should be able to fit nearly all of its own training data. Here it reached only 25%, with
the loss stuck at 2.06 (ln 10 = 2.30 is chance). That means the probe was failing to optimize; it
wasn't measuring the representation. Checking the frozen features directly (on CPU, train images only)
showed why:

| Encoder | overall feature std | median per-dimension std |
|---|---|---|
| Supervised (Part A) | 0.728 | (not computed) |
| Rotation (Part B) | 0.213 | **0.0095** (41% of activations exactly 0) |

With features that small, 200 Adam steps at lr 3e-3 barely move the logits.

## What I changed, and why the fix works

The probe now standardizes each feature with a **fixed** mean/std computed once from the 500
un-augmented training images (`hw6_lib.Standardize`, stored as buffers, not trainable parameters).
It then trains the linear layer with batch 25 (400 steps over the same 20 epochs). A fixed affine
transform followed by a linear layer is still exactly one linear classifier, and the encoder stays
frozen: the notebook still asserts that the encoder's `state_dict` hash is unchanged.

I chose the recipe by looking only at **training-set** convergence, so the test set was never used to
tune it. On the same 500 training images:

| Probe (rotation features) | train loss | train acc |
|---|---|---|
| raw features, lr 3e-3, 200 steps (original) | 2.011 | 0.278 |
| raw features, lr 3e-2, 200 steps | 1.849 | 0.368 |
| standardized, lr 3e-3, 200 steps | 1.551 | 0.462 |
| standardized, lr 3e-3, 400 steps (**chosen**) | 1.380 | 0.512 |
| *same chosen recipe on supervised features* | *0.076* | *0.992* |

The last row shows the recipe itself can fit a good representation almost perfectly. The rotation
encoder's ~51% train accuracy is therefore a real property of its features, not of the probe. The same
recipe is applied identically to Parts B and C.

Effect in the final run (same frozen encoders, only the probe changed): Part B went from 24.60% to
**35.99%** test accuracy, and Part C from 55.75% to **56.00%**. SimCLR's features were already well
scaled, so it barely moved. The ranking of the three models didn't change. What changed is the
conclusion about how far behind the rotation encoder is: the old probe made it look no better than a
weak guess, and part of that gap was the probe, not the representation.

## Other issues hit along the way (briefly)

- **The laptop slept mid-run.** On battery the Mac is set to sleep after 1 minute idle. The rotation
  pretraining (normally ~5.5 min/epoch) slowed to 30–90 min/epoch between 19:36 and 22:20, advancing
  only during brief maintenance wakes. Running on AC power under `caffeinate` fixed it. As a result,
  the "wall-clock" pretraining time recorded for Part B (280 min) includes the sleep; the active
  compute was ~85 min.
- **macOS DataLoader workers can't pickle notebook-defined classes**, since workers are spawned, not
  forked. The SimCLR dataset and augmentations therefore live in `hw6_lib.py`. Workers memory-map the
  20,000-image subset from a `.npy` file instead of each receiving a pickled 550 MB copy, which would
  not fit in 16 GB alongside everything else.
- **`channels_last` crashed in backward on MPS** ("view size is not compatible…") in the throughput
  benchmark, so it isn't used.

## Verification performed

- The notebook was first executed end-to-end in a smoke configuration (`HW6_SMOKE=1`: tiny subsets, 1
  epoch per stage). I visually checked the rendered figures: correct image orientation from the
  column-major STL-10 binaries, rotation labels matching the rotated images, and SimCLR views that look
  like SimCLR augmentations.
- The STL-10 archive's MD5 matches torchvision's published checksum
  (`91f7769df0f17e558f3565bffb0c7dfb`).
- Encoder freezing is asserted, not assumed: a SHA-256 of each encoder's full `state_dict` (weights
  and BatchNorm buffers) is compared before and after its linear probe.
