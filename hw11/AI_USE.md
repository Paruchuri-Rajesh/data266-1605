# AI Use Appendix — HW11

SID4 = 1605 | SEED = 1605

> **Before submitting:** this file describes what actually happened in this AI-assisted session.
> Read it, check it against your own understanding, and rewrite it in your own words. Be ready to
> explain any cell in the notebook yourself.

## What the assistant did

I used Claude Code (Claude Opus 5.5, Anthropic) for the whole assignment:

- `build_notebook.py`, which generates `HW11_CLIP_Similarity_ZeroShot_CIFAR10.ipynb`: loading CLIP and CIFAR-10, the
  embedding helpers, Part A (caption similarity, visualizations, additional captions) and Part B (zero-shot classifier,
  prompt comparison, per-class accuracy, confusion matrix, error examples);
- choosing the captions and prompt templates;
- headless execution with `nbconvert`, writing the findings from the executed outputs, and `README.md`, `METRICS.md`,
  `RUN_LOG.txt` and this file.

## Design decisions (and why)

- **ViT-B/32**: the standard, smallest CLIP model; it runs in about 2 minutes on a laptop GPU, and its published CIFAR-10
  zero-shot accuracy (~88–89%) gives a sanity check for the result (88.33%).
- **CIFAR-10 test split** for both parts, so Part A's 100-per-class subset is a subset of Part B's data, and the two can be cross-checked (Section 10: 100% agreement).
- **Raw cosine similarity** in Part A, not a softmax: a softmax over images for one caption would only rescale the ranking.
- **Three prompt strategies** in Part B, so the effect of prompt wording is measured rather than assumed.

## Things that went wrong, and how they were caught

1. **API change in `transformers` 5.x.** `get_image_features` returns an output object rather than a tensor. Because the text
   tower's hidden size is also 512, a shape check alone could not prove that `pooler_output` is the *projected* embedding; it
   was compared against `CLIPModel(...).image_embeds/text_embeds` before use.
2. **A wrong claim in the first draft of the findings.** It said cross-group errors were "mostly airplane ↔ bird and ship ↔ airplane",
   but ship and airplane are both vehicles. Counting from the confusion matrix gave 124 of 1,167 errors (10.6%) crossing
   groups, 70 of them airplane ↔ bird; the text was corrected.
3. **Wrong seed.** The first runs used SEED = 0. Before pushing, the earlier homeworks in the repo showed the standing rule
   SEED = 1605 (SID4), so the notebook was re-run with it and the Part A findings were rewritten from the new outputs
   (Part B does not depend on the seed).

## What to check yourself

- Look at the top/bottom-15 and the mistake figures yourself, and decide whether you agree with the explanations in Section 11
  (for example, that deer → horse comes from missing antlers at 32×32). These are interpretations, not measured facts.
- Part A's precision@100 depends on the random 100-per-class sample: 0.73 with SEED = 0, 0.77 with SEED = 1605.
