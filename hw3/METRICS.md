# HW3 Metrics

SID4 = 1605, SEED = 1605. All numbers below are read directly from the final executed run of
`attention_and_prompting.ipynb` (0 cell errors — see `RUN_LOG.txt`).

## Part A — Prompt Engineering (LangChain + local `llama3.1:8b` via Ollama)

Correct math answer (apples problem) is **28**. Correct logic answer (books problem) is **6**.

| Technique | Ex1 (math) output | Ex1 correct? | Ex2 (logic/text) output | Ex2 correct? | Notes |
|---|---|---|---|---|---|
| Zero-Shot | 32 | **No** | Yes | Yes | Math wrong with no visible reasoning to audit. |
| Few-Shot | 21 | **No** | Yes | Yes | Answered the *first exemplar's own question*, not the asked one — exemplar leakage. |
| Chain-of-Thought (few-shot) | 28 (`40-12=28`) | Yes | 6 | Yes | Correctly transferred the "all but N" idiom from the sheep exemplar. |
| Zero-Shot CoT | 28 (`40-12=28`) | Yes | 24 | **No** | Logic run reasoned "6 remain" correctly, then self-contradicted and computed `30-6=24`. |
| Meta-Prompting | 28 | Yes | Mixed | Yes | Same system template reused verbatim across math and sentiment-classification tasks. |
| Tree of Thoughts (3 branches + eval) | no valid expression found | **No** | cat / Boston | **No** (city wrong) | 24-game has a real solution, `(7-3)*(11-5)=24`, never found in 3 branches + evaluation. Logic puzzle: all 3 branches were near-identical (no diversity despite temperature 0.8); correct pet (cat) but wrong city (said Boston, which is actually the dog-owner Ben's city; correct is Chicago). |

Full transcripts: `RUN_LOG.txt` / the executed notebook cells. Discussion: see
`attention_and_prompting.ipynb` "Comparing the Techniques — Findings" and `HW3_Report.pdf`.

## Part B — Self-Attention and Causal Masking

Dataset: the assignment's fixed 5-sentence paragraph. Word-level tokens: **51** (vocab size
**41**). `d_model = 32`, single-head scaled dot-product attention, Adam lr=3e-3, 300 epochs,
autoregressive next-token cross-entropy. Two independently trained models (unmasked vs.
causal-masked), both seeded identically at init (SEED=1605).

| Model | Mask | Final loss (epoch 300) | Verification | Heatmap |
|---|---|---|---|---|
| Unmasked | none | 0.0027 | row sums = 1.0 | `figures/unmasked_attention_heatmap.png` |
| Causal | lower-triangular, applied before softmax (both in training and at inference) | 0.0032 | row sums = 1.0; **upper-triangle attention mass = 0.0000000000** | `figures/causal_attention_heatmap.png` |

Both losses started near `ln(41) ≈ 3.71` (random-guess loss for a 41-token vocabulary) and
dropped to ~0.003, confirming the token/position embeddings and $W_Q, W_K, W_V$ projections were
genuinely learned, not left at random initialization.

**Unmasked heatmap:** attention is spread across full rows, including columns to the right of the
diagonal — every token can attend to tokens that come later in the sequence.

**Causal heatmap:** the entire strict upper triangle is exactly 0 (verified numerically, not just
visually, via `np.triu(attn, k=1).sum()`). Row 0 is a single spike of weight 1.0 at column 0;
each later row has one more available column than the last, producing the characteristic
triangular pattern of decoder self-attention described in *Attention Is All You Need*, Section
3.2.3.
