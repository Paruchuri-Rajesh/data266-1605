# AI Use Appendix — HW4

SID4 = 1605

## What I used an assistant for vs. wrote myself

I used Claude Code (Claude Opus 5.5, Anthropic) to implement this assignment end to end: the
notebook `MiniGPT_Shakespeare.ipynb` (character tokenizer, sliding-window dataset, manual
multi-head masked self-attention, decoder block, GPT model, Adam training loop, loss plot,
greedy / temperature / top-k decoders, and the diversity/coherence metrics) and the findings
report `MiniGPT_Findings.pdf`. This also covered environment setup (an isolated `.venv/`, running
the notebook with `jupyter nbconvert --execute`) and pushing to this repo.

What I directed rather than delegated: the assignment's course copy of `Shakespeare.txt` was not
in my working folder, so the public Tiny Shakespeare file was used (same 1,115,394-character
text). I had the final run redone with SEED = 1605 to match the standing requirement from
earlier homeworks. I reviewed the executed outputs (loss curve, generated samples, metrics
table) before accepting the write-up.

## One specific thing the assistant produced that was wrong

The first draft of the Part 5 analysis was written **before** the metrics existed and made two
claims the data did not support:
1. that **greedy decoding** gives the most coherent output, and
2. that top-k with **k = 30** "reduces the worst misspellings" compared with τ = 1.0 sampling.

## How I discovered the failure

The notebook's quantitative table (Section 4.4) contradicted both claims. Greedy got stuck
repeating a single line (94% repeated word 4-grams, Distinct-2 ≈ 0.1–0.15), which is not coherent
as a whole passage. And in the first run, k = 30 had a *lower* real-word rate than τ = 1.0
(0.858 vs. 0.871), so it was not reducing misspellings at all.

## What I changed, and why the fix works

The analysis (notebook Part 5 and the PDF) was rewritten from the executed numbers. Most coherent
is now low-temperature τ = 0.3 (99.5% real words, no loop), with top-k k = 3 a close second.
Greedy is described as locally fluent but degenerate. The k = 30 discussion now explains *why*
it behaves like τ = 1.0: with only 65 characters, the top 30 hold almost all of the probability
mass, so the cutoff rarely removes anything. In the SEED = 1605 run the two 400-character samples
are literally identical. Every number quoted in the text is copied from `results.json`
written by the final run, not estimated.

## A secondary example: a lost 47-minute run

The first full execution (stride 16, ~7,800 steps) ended after 47 minutes with
`ModuleNotFoundError: No module named 'stack_data'`. The environment lacked an IPython
dependency, so when a cell raised, the kernel could not even format the traceback. nbconvert
aborted without saving, and the original error was lost. Fix: rebuilt the environment with
`ipython` installed explicitly, ran all notebook code as a plain Python script first (2-epoch smoke
test) to prove every cell executes, and ran the final notebook with `--allow-errors` so a failing
cell would be saved into the notebook instead of discarded. I also doubled the window stride (32)
to cut the run to ~23 minutes of training. The final run finished with 0 cell errors.

## Verification performed

- `hw4/MiniGPT_Shakespeare.ipynb` executed top-to-bottom via `jupyter nbconvert --execute` with
  0 cell errors (20/20 code cells, see `RUN_LOG.txt`).
- Causal masking verified numerically, not just visually: perturbing tokens 7–9 leaves the attention
  output at positions 0–6 unchanged (`torch.allclose` → True), and all attention weights above the
  diagonal are exactly 0.
- Initial loss 4.243 ≈ ln(65) = 4.174 confirms a correct, uniform-at-init setup. The final 1.402 / 1.545
  train/val loss with a small gap shows the model learned without overfitting.
- The forbidden modules (`nn.Transformer`, `nn.MultiheadAttention`, HuggingFace) are not used anywhere.
