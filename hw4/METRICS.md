# HW4 Metrics

SID4 = 1605, SEED = 1605. All numbers below are read directly from the final executed run of
`MiniGPT_Shakespeare.ipynb` (0 cell errors, see `RUN_LOG.txt`).

## Part 1: Data

| Item | Value |
|---|---|
| Corpus | `Shakespeare.txt`, 1,115,394 characters |
| Vocabulary size (character-level) | **65** |
| Train / val split | 90% / 10% (1,003,854 / 111,540 tokens) |
| Sequence length | 128 (input `[t..t+127]`, target `[t+1..t+128]`) |
| Sliding-window stride | 32 → 31,367 training windows; val uses non-overlapping windows (871) |

## Part 2: Model

| Item | Value |
|---|---|
| Total parameters | **3,225,153** |
| Hidden dimension | 256 |
| Attention heads | 8 (head dim 32), manual multi-head masked self-attention |
| Decoder layers | 4 (post-LN: `LN(x + MHA(x))`, `LN(x + FFN(x))`, FFN = Linear→GELU→Linear, 4× expansion) |
| Embeddings | token embedding + learnable positional embedding (`nn.Parameter`) |
| Causal mask check | future-token perturbation leaves earlier outputs unchanged: **True**; upper-triangle attention = 0: **True** |

## Part 3: Training

Adam, lr = 3e-4, batch size 64, dropout 0.1, grad-clip 1.0, 8 epochs × 490 steps = 3,920 steps, Apple MPS, ~23 min.
Initial val loss 4.243 (uniform guess would be ln 65 = 4.174).

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Train loss | 2.358 | 1.920 | 1.720 | 1.604 | 1.529 | 1.476 | 1.435 | **1.402** |
| Val loss | 2.067 | 1.864 | 1.741 | 1.665 | 1.622 | 1.586 | 1.559 | **1.545** |

Plot: `training_loss.png`.

## Part 4: Decoding comparison (prompt `"ROMEO:"`)

1,000 generated characters per decoder; stochastic decoders averaged over seeds 1605, 1606, 1607.

| Decoder | Distinct-2 (↑ diverse) | Repeated 4-grams (↑ looping) | Real-word rate (↑ coherent) | Model NLL/char |
|---|---|---|---|---|
| Greedy | 0.154 | 0.939 | 1.000 | 0.660 |
| Temperature τ=0.3 | 0.744 | 0.065 | 0.995 | 0.829 |
| Temperature τ=1.0 | 0.993 | 0.000 | 0.878 | 1.330 |
| Temperature τ=1.8 | 1.000 | 0.000 | 0.477 | 2.608 |
| Top-k k=3 | 0.936 | 0.000 | 0.958 | 1.018 |
| Top-k k=30 | 0.986 | 0.000 | 0.911 | 1.295 |

- **Most coherent:** temperature τ = 0.3 (99.5% real words, no loop), with top-k k = 3 a close second (no repetition at all).
  Greedy spells perfectly but loops ("the present of the princess of the prince...").
- **Most diverse:** temperature τ = 1.8 (Distinct-2 = 1.00, but only 48% real words).
- With the same seed, the 400-character τ = 1.0 and k = 30 samples are identical: the top 30 of 65 characters hold
  nearly all of the probability mass, so the k = 30 cutoff rarely changes the sampled character.

Full samples and discussion: `MiniGPT_Shakespeare.ipynb` (Parts 4-5) and `MiniGPT_Findings.pdf`.
