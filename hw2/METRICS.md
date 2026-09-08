# HW2 Metrics

SID4 = 1605, SEED = 1605. All numbers below are from the final SEED=1605 runs documented in
`RUN_LOG.txt` (0 cell errors on every notebook).

## Part 1 — Word2Vec transfer learning on IMDB

### Top-3 nearest neighbors, PRETRAINED (word2vec-google-news-300)

| word | rank | neighbor | cosine similarity |
|---|---|---|---|
| cast | 1 | casts | 0.7219 |
| cast | 2 | casting | 0.7188 |
| cast | 3 | Cast | 0.6638 |
| score | 1 | scoring | 0.7197 |
| score | 2 | scores | 0.6596 |
| score | 3 | scored | 0.6384 |
| plot | 1 | plots | 0.7625 |
| plot | 2 | Plot | 0.6524 |
| plot | 3 | plotting | 0.6328 |
| screen | 1 | screens | 0.7729 |
| screen | 2 | onscreen | 0.6115 |
| screen | 3 | LCD_screen | 0.5599 |
| review | 1 | reviewed | 0.6630 |
| review | 2 | reviewing | 0.6610 |
| review | 3 | reviews | 0.6380 |

### Top-3 nearest neighbors, FINE-TUNED (25k IMDB reviews, workers=1, 5 epochs, seed=1605)

| word | rank | neighbor | cosine similarity |
|---|---|---|---|
| cast | 1 | ensemble | 0.5751 |
| cast | 2 | actors | 0.5732 |
| cast | 3 | supporting | 0.5728 |
| score | 1 | nicolai | 0.6441 |
| score | 2 | ennio | 0.6208 |
| score | 3 | music | 0.6163 |
| plot | 1 | storyline | 0.6644 |
| plot | 2 | plots | 0.6002 |
| plot | 3 | plotline | 0.5955 |
| screen | 1 | screens | 0.5142 |
| screen | 2 | onscreen | 0.4771 |
| screen | 3 | pinter | 0.4120 |
| review | 1 | comment | 0.6064 |
| review | 2 | reviews | 0.5684 |
| review | 3 | reviewing | 0.5438 |

Pretrained neighbors are largely generic morphological variants; fine-tuned neighbors are
movie-domain specific (e.g. "score" pulls in film composers Nicola Piovani / Ennio Morricone;
"cast" pulls in "actors"/"supporting" instead of "casting").

### Self-similarity: original vector vs fine-tuned vector (same word, before vs after)

| word | self cosine similarity |
|---|---|
| score | 0.5296 |
| review | 0.5481 |
| screen | 0.6302 |
| cast | 0.6514 |
| plot | 0.6675 |

**Most shifted: "score"** (0.5296) — general-news "score" (tally/rate/musical-score-generic)
collapses toward the specific film-scoring/composer sense under IMDB fine-tuning.
**Least shifted: "plot"** (0.6675) — already leans toward its narrative/movie sense even in
general news text, so fine-tuning mostly reinforces rather than relocates it.

t-SNE 2D visualization of "score" (target word + its top-3 neighbors, before vs after,
distinct colors/markers) is rendered inline in `part1_word2vec.ipynb`.

Checkpoint: `hw2/checkpoints/imdb_finetuned_word2vec.model` (66.6 MB, gitignored — see
`RUN_LOG.txt` for the exact training command and how to regenerate it).

---

## Part 2 — RAG pipeline (10 movie Wikipedia pages)

Chunking (original config): `chunk_size=500, chunk_overlap=50`. Embeddings:
`sentence-transformers/all-MiniLM-L6-v2`. Vector store: FAISS. LLM: local `google/flan-t5-base`
(deterministic, `do_sample=False`, `num_beams=4`).

### Retrieval success (original config, k=3)

| # | question | top-3 contains answer? | rank of first relevant chunk |
|---|---|---|---|
| 1 | Who directed The Matrix and in what year was it released? | Yes | 1 |
| 2 | Which actor played the Joker in The Dark Knight, and what happened to him after filming? | Yes | 1 |
| 3 | What novel was The Godfather based on and who wrote it? | Yes | 1 |
| 4 | How much did Titanic gross worldwide and when was it released in the US? | Yes | 1 |
| 5 | Who directed Interstellar and which actor starred as the lead astronaut? | Yes | 1 |

**Retrieval Success Rate = 5/5 = 100%**

### Chunk-size sensitivity (2 of 5 questions re-run with an alternate configuration)

| question | original (500/50) answer | alt config | alt answer | effect |
|---|---|---|---|---|
| Q4 (Titanic gross) | "$2.264 billion" (US release date omitted) | 1000/100 | garbled ("...were used to make the film.") | bigger chunk kept the sentence intact but diluted relevance, LLM answer got worse |
| Q2 (Joker/Ledger) | "Ledger died from an accidental prescription drug overdose" | 200/20 | truncated ("Ledger died") | smaller chunk fragmented the fact, thinner answer |

### RAG failures found and categorized (>= 2, from the 5 original-config questions)

1. **Correct context retrieved, LLM answered incompletely** (Q1, Q3, Q5) — the rank-1 chunk
   contained both requested facts (e.g. director *and* year), but flan-t5-base's answer dropped
   one sub-fact of each compound question every time (e.g. gave "1999" for The Matrix but never
   named the Wachowskis). Root cause: small local LLM under-extracts on multi-fact questions,
   not a retrieval failure.
2. **Chunk boundary separated important information** (Q4) — the 500-char chunk boundary cut
   the sentence right after "$2.264 billion," severing the dollar figure from the clause
   explaining the US theatrical release date. Confirmed by rebuilding with chunk_size=1000
   (kept the sentence whole) — but that in turn diluted retrieval relevance and produced a
   worse, garbled answer, showing the tradeoff runs both directions.

---

## Part 3 — PyTorch training optimization techniques

Shared testbed: one 8-block residual MLP (`DeepMLP`), synthetic 4096-sample / 256-feature /
10-class classification data, `batch_size=64`, `steps=50`, `SEED=1605`. Timing = 1 untimed
warmup + best-of-3 repeats (Section 1: mean-of-5).

| Technique | Variant | Time (s) | Final loss | Memory |
|---|---|---|---|---|
| 1. Tensor creation | CPU create | 0.2151 | — | — |
| 1. Tensor creation | MPS create | 0.0061 | — | — |
| 1. Tensor creation | CPU matmul (4096²) | 0.1664 | — | — |
| 1. Tensor creation | MPS matmul (4096²) | 0.1698 | — | 0.98x vs CPU (near parity) |
| 2. Weight init | default | 0.4249 | 1.9414 | +0.7 MB RSS |
| 2. Weight init | xavier | 0.4229 | 1.7263 | −1.94 MB RSS |
| 2. Weight init | kaiming | 0.4241 | 2.1366 | +0.59 MB RSS |
| 2. Weight init | zeros | 0.4221 | **2.3017 (stuck at ln 10, never learns)** | +0.30 MB RSS |
| 3. Activation checkpointing | off | 5.3373 | 1.1995 | 50.0 MB (fwd activations) |
| 3. Activation checkpointing | on | 6.8586 (+28.5%) | 1.1995 (identical) | 15.0 MB (**−70%**) |
| 4. Gradient accumulation | direct B=64 | 0.4239 | 1.9414 | −0.31 MB RSS |
| 4. Gradient accumulation | accum 4×B=16 | 1.5421 (≈3.6x slower) | 1.9414 (max diff <1e-6 vs direct) | +1.09 MB RSS |
| 5. Mixed precision | fp32 (MPS) | 0.4211 | 1.9414 | −2.31 MB RSS |
| 5. Mixed precision | autocast fp16 (MPS) | 0.5428 | 1.9413 | +0.50 MB RSS |
| 5. Mixed precision | autocast bf16 (MPS) | 0.5367 | 1.9410 | −0.42 MB RSS |
| 5. Mixed precision | fp32 (CPU) | 0.4177 | 1.9414 | 0.00 MB RSS |
| 5. Mixed precision | autocast bf16 (CPU) | 19.2929 (**≈46x slower**) | 1.9417 | −10.89 MB RSS |

Raw data: `hw2/part3_results.csv`.

### Findings
- **Tensor creation**: MPS tensor allocation is ~35x faster than CPU; matmul throughput is
  near parity on this hardware/size (0.98x) — a materially different result from an earlier
  arbitrary-seed run that showed MPS ahead, corrected here rather than left stale.
- **Weight init**: all-zeros initialization is a genuine failure mode — with every neuron
  identical, gradients are identical too, so the network never breaks symmetry and loss sits at
  ln(10) (random-guess loss for 10 classes) for all 50 steps. Default/Xavier/Kaiming all learn
  normally.
- **Activation checkpointing**: textbook time/memory tradeoff — 70% less forward-activation
  memory, 28.5% more wall-clock time from recomputation, identical final loss (mathematically
  equivalent forward+backward, just recomputed).
- **Gradient accumulation**: loss trajectories match the direct full-batch run to under 1e-6 —
  confirms accumulation is mathematically equivalent to the larger batch — but costs ~3.6x more
  wall-clock time here due to the overhead of 4 separate forward/backward passes vs 1.
- **Mixed precision**: `torch.cuda.amp.GradScaler` was confirmed to silently self-disable off
  CUDA (`is_enabled()==False` + UserWarning) rather than error, so autocast was used without a
  scaler. fp16/bf16 autocast worked correctly on MPS (correct loss, no NaNs) but was *slower*
  than fp32 at this small model scale — casting overhead dominates when there isn't enough
  compute to amortize it. CPU bf16 was dramatically slower (~46x) since this CPU lacks fast
  native bf16 kernels. Documented as a real, hardware-specific limitation rather than an
  idealized "mixed precision is always faster" claim.
