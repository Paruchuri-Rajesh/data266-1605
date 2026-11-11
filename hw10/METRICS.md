# HW10 Metrics — RAG with Reranking, Chunking, and Conversational Memory

All numbers come from the final executed run of `HW10_RAG_Reranking_Chunking_Memory.ipynb` (0 cell errors; see `RUN_LOG.txt`). None were typed in by hand. Raw per-query results are in `data/results_*.csv`.

## Setup

| Item | Value |
|---|---|
| Data | `ccdv/arxiv-summarization`, config `document`, split `train`, first 100 rows (`article` field indexed) |
| Corpus size | 651,653 whitespace tokens → 506,531 after cleaning (22.3% removed: `@xmath`/`@xcite` placeholders, detached punctuation) |
| Chunking 1 | Fixed window, 200 words, 40 overlap → **3,194 chunks** (mean 197 words; 8.0% end at a sentence boundary) |
| Chunking 2 | Sentence-aware recursive, ≤200 words, 1-sentence overlap → **3,156 chunks** (mean 182 words; 97.9% end at a sentence boundary) |
| Embedder | `BAAI/bge-small-en-v1.5`, 384-d, normalized; 4.5% / 4.1% of chunks exceed 512 tokens and are truncated |
| Vector DB | ChromaDB 1.5.9 `PersistentClient` (`chroma_db/`), cosine HNSW, one collection per strategy |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top-20 Chroma candidates |
| Memory | Last 2 user turns prepended to the follow-up query |
| Evaluation | 10 single-turn queries + 5 two-turn conversations (turn 2 scored); P@5, R@5, P@10, R@10, MRR@10 |

## Single-turn: all four configurations (mean over 10 queries)

| Strategy | Rerank | P@5 | R@5 | P@10 | R@10 | MRR@10 |
|---|---|---|---|---|---|---|
| fixed | no | 0.88 | 0.580 | 0.64 | 0.758 | 1.00 |
| fixed | yes | 0.88 | 0.580 | 0.68 | 0.803 | 1.00 |
| sentence | no | 0.90 | 0.591 | 0.69 | 0.804 | 0.95 |
| sentence | yes | 0.88 | 0.595 | **0.70** | **0.837** | **1.00** |

## Headline comparisons

| Comparison | Setting | P@5 | R@5 | P@10 | R@10 | MRR@10 |
|---|---|---|---|---|---|---|
| Reranking | without | 0.89 | 0.585 | 0.665 | 0.781 | 0.975 |
| | with | 0.88 | 0.588 | 0.690 | 0.820 | 1.000 |
| Chunking | fixed window | 0.88 | 0.580 | 0.660 | 0.781 | 1.000 |
| | sentence-aware | 0.89 | 0.593 | 0.695 | 0.821 | 0.975 |
| Memory (turn-2 follow-ups) | without | 0.38 | 0.242 | 0.255 | 0.325 | 0.657 |
| | with | 0.48 | 0.396 | 0.435 | 0.689 | 0.750 |

## Memory: per configuration (turn 2 of 5 conversations)

| Memory | Strategy | Rerank | P@5 | R@5 | P@10 | R@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| off | fixed | no | 0.40 | 0.242 | 0.28 | 0.349 | 0.629 |
| off | fixed | yes | 0.44 | 0.282 | 0.26 | 0.327 | 0.800 |
| off | sentence | no | 0.36 | 0.231 | 0.22 | 0.289 | 0.600 |
| off | sentence | yes | 0.32 | 0.211 | 0.26 | 0.337 | 0.600 |
| on | fixed | no | 0.40 | 0.342 | 0.44 | 0.673 | 0.617 |
| on | fixed | yes | 0.48 | 0.399 | 0.44 | 0.679 | 0.750 |
| on | sentence | no | 0.48 | 0.378 | 0.40 | 0.639 | 0.767 |
| on | sentence | yes | **0.56** | **0.465** | **0.46** | **0.766** | **0.867** |

Per conversation (mean over the 4 configurations), memory helped C2, C3 and C5 and hurt C1 and C4:

| Conv | P@5 off → on | R@10 off → on | MRR off → on |
|---|---|---|---|
| C1 statefinder → "best-fit values … supernova data" | 0.55 → 0.30 | 0.39 → 0.46 | 1.00 → 0.62 |
| C2 white dwarf → "elements in its spectrum" | 0.35 → 0.60 | 0.22 → 0.52 | 1.00 → 0.88 |
| C3 solar cycle → "also reproduce grand minima?" | 0.05 → 0.45 | 0.10 → 0.75 | 0.29 → 1.00 |
| C4 MTJ synapse → "input frequency … LTP" | 0.95 → 0.70 | 0.92 → 0.79 | 1.00 → 0.88 |
| C5 bead-spring → "change the resolution on the fly?" | 0.00 → 0.35 | 0.00 → 0.92 | 0.00 → 0.38 |

Topic switch (polymer question, then the unrelated Q10): P@5 0.6 → 0.0 and MRR 1.0 → 0.0 with memory, in all 4 configurations.

## Relevance labels: how they changed

Relevance is a keyword rule per query (gold paper AND answer-bearing terms), written before retrieval. After the first run, I read all 119 pooled top-5 (query, chunk) pairs and made three changes:

1. **Phrase inclusions** (`MANUAL_INCLUDE`) for answer-bearing passages that paraphrase the key term (Q1 ×2, Q3, Q5, Q7, Q8).
2. **Bibliography chunks are never relevant.** Eight reference-list chunks of the original Q4 paper had matched on reference titles.
3. **The original Q4 was replaced.** It asked about SiC-nanopillar second-harmonic generation (doc 6), but doc 6's `article` holds only supplementary material, so the corpus had 0 or 1 answering chunks. The new Q4 asks about cosmology-independent GRB calibration (doc 72).

For transparency, here are the first-run single-turn results (original rules and original Q4; memory results were unaffected):

| Strategy | Rerank | P@5 | R@5 | P@10 | R@10 | MRR@10 |
|---|---|---|---|---|---|---|
| fixed | no | 0.74 | 0.580 | 0.58 | 0.822 | 1.00 |
| fixed | yes | 0.82 | 0.657 | 0.61 | 0.862 | 1.00 |
| sentence | no | 0.78 | 0.626 | 0.60 | 0.866 | 0.95 |
| sentence | yes | 0.80 | 0.641 | 0.63 | 0.905 | 0.95 |

Under the first-run labels, reranking appeared to improve P@5 by +0.05. Summed over queries that gain was +1.0 P@5: +0.4 came from Q4 (fixed), where reranking moved reference-list chunks up that should not have counted as relevant, and +0.2 each came from Q3 and Q8 (fixed). Those were rule misses: reranking brought up a passage the keywords had missed, and the corrected labels count it in both rankings. After the corrections, reranking does not help at P@5 (0.89 → 0.88) and helps a little at K = 10. **The label audit changed the reranking conclusion.** The chunking and memory conclusions did not change.
