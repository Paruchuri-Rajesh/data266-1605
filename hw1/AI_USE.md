# AI_USE.md — HW1

> **Before submitting:** this file describes what actually happened during this AI-assisted
> session. Read it, verify it against your own understanding, and rewrite anything below in your
> own words before you submit — you may be asked to explain any two cells on camera (Section 0.6),
> and a generic or unowned AI_USE.md earns zero per Section 0.5.

## Which parts did you use an assistant for, and which did you write yourself?

I used Claude (via Claude Code) to write the full implementation for this assignment: the data
preprocessing pipeline, the PyTorch and TensorFlow models (baseline and HP_ID=3 modified), the
multi-seed measurement loop, the CUDA tiled-matmul kernel, and the supporting notebooks. I
directed it with my actual assignment parameters (SID4 = 1605, so SEED = 1605, HP_ID = 3), the
exact HP_ID=3 configuration from the assignment spec (hidden layers [64, 32], lr = 0.0003, 30
epochs), and the standing-requirements deliverable list (RUN_LOG.txt, METRICS.md, checkpoints,
etc.). I reviewed the generated code and the executed notebook outputs rather than writing the
training loops or the CUDA kernel from scratch myself.

## Give one specific thing it produced that was wrong. Paste the wrong output.

The first version of the PyTorch/TensorFlow training loops used **full-batch gradient descent**
(one optimizer step per epoch, over the entire 537-row training set at once) to satisfy the "30
epochs" requirement literally. That runs without any error and produces a plausible-looking
number:

```
torch baseline seed-run acc 0.681034505367279 final train_loss 0.5644972920417786 final val_loss 0.5552362203598022
torch modified seed-run acc 0.6206896305084229 final train_loss 0.6677919030189514 final val_loss 0.6674894690513611
```

This is a "plausible-looking metric computed incorrectly" in the sense the assignment warns
about: it satisfied the letter of "30 epochs" while only performing **30 total gradient updates**
for the whole model — nowhere near what "30 epochs" normally means for a mini-batch-trained
feedforward network, and it produced noticeably worse, less-converged accuracy (68.1%/62.1%) than
a properly mini-batched run.

## How did you find out? What did the failure look like?

There was no crash or exception — the code ran cleanly and printed a normal-looking accuracy
number, which is what makes this the kind of error the assignment specifically calls out (a metric
that is computed and looks fine but is methodologically wrong). It was caught by inspecting the
training loop before treating the result as final: with `batch_size = len(X_train)` there is
exactly one `opt.step()` call per epoch, i.e. 30 gradient updates total for the entire run — far
too few for Adam to converge on this dataset, and not what a grader would expect "30 epochs" of
training to mean.

## What did you change, and why does your version work?

The loop was rewritten to use mini-batch training (`DataLoader` with `batch_size=32` in PyTorch,
`batch_size=32` in `model.fit(...)` for TensorFlow), so one epoch means one full pass over all
mini-batches (~17 gradient updates per epoch, ~510 updates over 30 epochs) rather than one update
total per epoch. Re-running with the same seeds produced materially different, higher, and more
stable accuracy (baseline 75.0%/74.1% PyTorch/TensorFlow at SEED=1605 vs. 68.1%/− before), and the
loss curves now show the expected shape (steadily decreasing, baseline nearing convergence by
epoch 30) instead of a nearly-flat 30-point curve. This is the version in the submitted
`neural_networks.ipynb`.

**Secondary issue (environment, not code logic):** `import tensorflow` alongside an already-
installed PyTorch crashed the Python process outright (`libc++abi: terminating due to uncaught
exception ... mutex lock failed`) when both were installed into the machine's global
site-packages — a known OpenMP-runtime collision between the two frameworks' bundled `libomp`
copies on macOS. Found by simply trying to import both in one process and hitting the crash
immediately. Fixed by creating an isolated project virtual environment (`.venv/`) and installing
both frameworks there instead of globally — no code change was needed, just environment isolation.

## Verification performed

- The full `neural_networks.ipynb` was executed top-to-bottom via `jupyter nbconvert --execute`
  with no errors (see `RUN_LOG.txt`); accuracy/loss numbers quoted in the notebook's conclusion
  and in `METRICS.md` were read directly from that executed output, not invented.
- `matmul.cu` and `cuda.ipynb` could not be executed on this machine (Apple Silicon, no NVIDIA
  GPU/`nvcc`) — I have not yet run or verified the CUDA portion myself. This still needs to happen
  on Colab before submission; the profiler output and timing table are placeholders until then.
