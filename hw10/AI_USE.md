# AI Use Appendix — HW10

> **Before submitting:** this file describes what actually happened in this AI-assisted session.
> Read it, check it against your own understanding, and rewrite it in your own words. Be ready to
> explain any cell in the notebook yourself.

## What the assistant did

I used Claude Code (Claude Opus 5.5, Anthropic) for the whole assignment:

- `rag_core.py`: text cleaning, the sentence splitter, both chunkers, the `ConversationMemory` class, and the metric functions;
- `build_notebook.py`, which generates `HW10_RAG_Reranking_Chunking_Memory.ipynb`: loading the data, ChromaDB indexing,
  retrieval with and without the cross-encoder, the multi-turn demos, the evaluation, the figures and the findings;
- choosing the queries, writing the keyword relevance rules, and the hand audit of the labels;
- headless execution with `nbconvert`, and `METRICS.md`, `RUN_LOG.txt` and this file.

## Design decisions (and why)

- **Equal chunk budgets** (about 200 words) for both chunking strategies, so the comparison isolates boundary placement rather than size.
- **Retrieve 20, rerank, keep top-K.** The standard two-stage design: the cross-encoder is too slow to score all about 3,150 chunks.
- **LLM-free memory** (prepend the last 2 turns). It is simple, deterministic and needs no API key. Its failure modes (topic switch; diluting already-specific follow-ups) are measured, not hidden.
- **Relevance rules written before seeing results**, then audited by reading every pooled top-5 chunk.

## Things that went wrong, and how they were caught

1. **The label audit reversed a conclusion.** After the first run, reranking appeared to raise P@5 by +0.05. Reading the 119 pooled
   chunks showed that 8 "relevant" Q4 chunks were *reference lists* that matched on paper titles, and that several genuinely
   answer-bearing passages had been missed by the keywords. After correcting both (applied identically to both chunking strategies),
   reranking no longer helps at P@5 (0.89 → 0.88) and helps only modestly at K = 10. Both versions are recorded in `METRICS.md`.
2. **A query the corpus could not answer.** The same audit showed that doc 6's `article` field is only the paper's
   supplementary material. Q4 was replaced with a question on doc 72; its rule was written before running it.
3. **A wrong explanation.** The first draft of the findings said the reranker *demoted* equation-heavy chunks for Q1. Printing the
   actual rankings showed the opposite: it *promoted* them (c8, c10, c11) and demoted the explanatory chunks. The text was corrected.
4. **Runtime.** The first execution took 1 h 19 min, almost all of it embedding on MPS. Embeddings are now cached
   (`data/emb_*.npy`), so a rerun takes about 80 s.

## What to check yourself

- Read Section 11 of the notebook (the relevance audit) and decide whether you agree with the labels, especially the lenient Q2 and Q10 ones.
- The reranking and chunking differences are small (+0.03 to +0.04 over 10 queries) and not statistically meaningful. Make sure the
  write-up you submit does not overstate them.
