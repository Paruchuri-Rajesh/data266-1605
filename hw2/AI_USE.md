# AI Use Appendix — HW2

SID4 = 1605

## What I used an assistant for vs. wrote myself

I used Claude Code (Claude Sonnet 5, Anthropic) to implement all three notebooks end to end:
the Word2Vec fine-tuning pipeline (Part 1), the LangChain RAG pipeline (Part 2), and the PyTorch
optimization-technique experiments (Part 3), including the environment setup, library selection,
debugging of every runtime error hit along the way, and the final retrofit that added
`SID4`/`SEED`, explicit seeding, and reproducibility documentation required by the course's
Section 0 rules.

What I did myself, rather than delegate: I made the decisions the assistant explicitly asked me
for rather than letting it guess — which LLM the RAG pipeline should call (I chose a local
HuggingFace model over an API-based one, since I didn't want to hand out an API key), how the
deliverables should be organized (one notebook per part), and confirming the `word2vec-google-
news-300` download was acceptable. I also supplied my SID4/repo details and reviewed the final
notebook outputs, metrics tables, and this file for accuracy before submitting, rather than
accepting them unread. I did not hand-write any of the Python/PyTorch/gensim/LangChain code
myself.

I'm disclosing this fully rather than downplaying it, since a generic/vague answer here earns
zero credit per the syllabus and the actual balance of effort was assistant-heavy on
implementation, human-directed on scope and acceptance.

## One specific thing the assistant produced that was wrong

While fine-tuning the Word2Vec embeddings in Part 1, the code needed to seed the fine-tuning run
from the pretrained `word2vec-google-news-300` vectors using gensim's
`KeyedVectors.intersect_word2vec_format(..., binary=True)`. This function internally calls
`np.fromstring(data, sep='')` to parse the binary vector file. On this machine's NumPy version
(2.3.3), that call raises:

```
ValueError: The binary mode of fromstring is removed, use frombuffer instead
```

The first version of the fine-tuning cell the assistant wrote did not anticipate this — it
assumed gensim 4.4.0 was compatible with NumPy 2.x out of the box, which it isn't for this
specific code path (`np.fromstring` in binary mode was deprecated in NumPy 1.x and hard-removed
in NumPy 2.0, years after gensim 4.4.0 was released).

## How I discovered the failure

The cell raised the `ValueError` above when the notebook was executed top-to-bottom via
`jupyter nbconvert --execute` — it wasn't a silent/logical bug, it was a hard crash that stopped
the run, which is how I noticed it immediately rather than having to spot it in the output.

## What I changed and why the fix works

The fix (visible in `part1_word2vec.ipynb`, cell 3) is a small compatibility shim: it saves a
reference to the original `np.fromstring`, then monkey-patches `np.fromstring` so that calls
made in binary mode (`sep=''`) are transparently routed to `np.frombuffer` instead (which is
still supported and does the equivalent job — reads raw bytes into a typed array without
copying), while calls made in text mode (`sep != ''`, used elsewhere by unrelated code) still go
through the original function unchanged. This works because `np.frombuffer` and the removed
binary mode of `np.fromstring` have the same contract for this use case (raw bytes -> typed
array, given a dtype and count) — gensim's `intersect_word2vec_format` never notices the swap,
so no gensim code had to change, only the NumPy call it depends on.

## A secondary example: a wrong empirical claim, not just a crash

Separately, during the reproducibility retrofit (re-running everything under `SEED=1605` with
consistent warmup+best-of-3 timing), the assistant caught that an earlier draft of Part 3 had
claimed MPS matmul was ~1.2-2x faster than CPU matmul for a 4096x4096 multiply. Under the final,
more careful measurement methodology (warmup pass + best-of-3, fixed seed), the real result was
near parity (0.98x — CPU marginally faster). Rather than leaving the old, more flattering claim
in the notebook, it was corrected in the Section 1 findings, the summary table, and the overall
discussion in `part3_optimization.ipynb` and `METRICS.md` to match what the code actually
measured. I'm including this because it's a case where the "wrong output" wasn't a crash but a
plausible-sounding, unverified performance claim — the kind of error that's easy to miss if you
don't re-check assistant-generated benchmark numbers against a rerun.
