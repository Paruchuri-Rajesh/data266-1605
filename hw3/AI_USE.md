# AI Use Appendix — HW3

SID4 = 1605

## What I used an assistant for vs. wrote myself

I used Claude Code (Claude Sonnet 5, Anthropic) to implement both parts of this assignment end
to end in a single notebook, `attention_and_prompting.ipynb`: the LangChain prompt-engineering
section (Part A — Zero-Shot, Few-Shot, Chain-of-Thought, Zero-Shot CoT, Meta-Prompting, and
Tree-of-Thoughts, two examples each) and the from-scratch PyTorch self-attention section (Part B
— tokenization, trainable embeddings, single-head scaled dot-product attention, the autoregressive
training loop, and both the unmasked and causal-masked attention heatmaps). This includes
environment setup (cloning the existing `data266-1605` repo, creating an isolated `.venv/`,
registering it as a Jupyter kernel) and choosing the LLM backend.

What I directed rather than delegated: since this machine has no Anthropic/OpenAI/Google API key
configured, I had the assistant use the locally running Ollama server (`llama3.1:8b`) through
`langchain-ollama` instead of blocking on an API key, consistent with the local-model choice I
made in HW2's RAG pipeline for the same reason. I reviewed the executed notebook outputs (the
actual LLM responses and the attention heatmaps) before accepting them, rather than accepting
generated code unread.

## One specific thing the assistant produced that was wrong

The first version of `AttentionLM` (Part B) built the causal mask once in `__init__` sized to
the **full** sequence length (`seq_len=51`, matching the whole tokenized dataset), but the
training loop calls the model on `token_ids[:-1]` (**50** tokens) for the shifted next-token
objective. Applying the full 51x51 mask to a 50x50 attention-score matrix raised:

```
RuntimeError: The size of tensor a (50) must match the size of tensor b (51) at non-singleton dimension 1
```

## How I discovered the failure

This was a hard crash, not a silent error: `jupyter nbconvert --execute` stopped at the training
cell with the `RuntimeError` traceback above and a non-zero exit, which is how I noticed it
immediately when re-running the notebook top-to-bottom rather than having to spot a subtly wrong
number in the output.

## What I changed, and why the fix works

`AttentionLM.forward` (in `attention_and_prompting.ipynb`, Part B) now slices the precomputed mask down to
the actual input length at call time — `mask = self.mask[:L, :L] if self.mask is not None else
None`, where `L = ids.shape[-1]` — instead of assuming the input is always the full sequence.
This works because the causal mask for any prefix of length `L` is exactly the top-left `LxL`
block of the full `seq_len x seq_len` lower-triangular mask (masking still only ever depends on
relative position, not on the total sequence length), so the same buffer correctly serves both
the length-50 training calls and the length-51 full-sequence inference call used to produce the
heatmaps. After the fix, both the unmasked and causal-masked models trained without error and
their final cross-entropy losses dropped from `ln(41) approx 3.71` (vocabulary size, i.e.
random-guess loss) to 0.0027 and 0.0032 respectively, confirming the embeddings and Q/K/V
projections were genuinely learned rather than left at random initialization.

## A secondary example: a silent multi-hour hang, not a crash

Separately, the first full run of the Part A Tree-of-Thoughts cells (which sample 3 candidate
reasoning branches per question via `ChatOllama`) hung for **over 5 hours** with the Python
process sitting at near-0% CPU the whole time — no exception, no timeout, nothing in the log after
the last successful print. `ChatOllama` was constructed with no request timeout and no cap on
generated tokens, so when one generation apparently got stuck in a repetition loop server-side,
the client just waited on the open connection forever instead of failing.

## How I discovered the failure

There was no crash to catch it — I noticed only because the run had been "in progress" for far
longer than ~20 local LLM calls should ever take. Checking `ps` showed the Python process had
accumulated only ~8 seconds of actual CPU time across 5+ wall-clock hours (i.e. blocked on I/O,
not computing), and `ollama ps` showed the model was mid-way through being auto-unloaded for
inactivity — consistent with a request that would truly never return.

## What I changed, and why the fix works

Every `ChatOllama` instance now sets `num_predict=400` (hard cap on generated tokens, so a
repetition loop can't run unbounded) and `client_kwargs={"timeout": 120}` (hard client-side
request timeout, so a stuck call fails after 2 minutes instead of hanging forever). This works
because it bounds both failure modes directly at their source rather than working around symptoms:
even in the worst case (model loops and never emits a stop token), generation is forcibly cut off
at 400 tokens server-side, and even if the whole request never returns, the HTTP client itself
gives up after 120s and raises instead of blocking indefinitely. After adding both, the full
notebook (all ~20 LLM calls plus PyTorch training) ran end-to-end via `jupyter nbconvert --execute`
in about 37 minutes with 0 cell errors.

## Verification performed

- `hw3/attention_and_prompting.ipynb` was executed top-to-bottom via `jupyter nbconvert --execute`
  with 0 cell errors on the final run (see `RUN_LOG.txt`); all LLM outputs, loss numbers, and
  attention-weight statistics quoted in `METRICS.md` were read directly from that executed output,
  not invented — including the cases where the model got an answer wrong (Zero-Shot math, Few-Shot
  answering the wrong question, Zero-Shot-CoT self-contradicting, both Tree-of-Thoughts examples),
  which I kept and reported honestly rather than cherry-picking only correct runs.
- The causal-masking claim was verified numerically, not just visually: the executed notebook
  computes `np.triu(causal_attn_np, k=1).sum()` (total attention mass placed on strictly-future
  positions) and prints it as `0.0000000000`, confirming the mask blocks 100% of future-position
  attention rather than just looking approximately triangular in the heatmap.
