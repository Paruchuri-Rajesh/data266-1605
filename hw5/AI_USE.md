# AI Use Appendix — HW5

SID4 = 1605

## What I used an assistant for vs. wrote myself

I used Claude Code (Claude Opus 5.5, Anthropic) to implement the assignment end to end in a single
notebook, `lora_dialogsum.ipynb`: loading DialogSum and converting it to dialogue → summary pairs,
the zero-shot baseline, attaching LoRA adapters with PEFT, fine-tuning with the Hugging Face
`Seq2SeqTrainer`, re-running the same two test dialogues, the ROUGE evaluation, and the r=4 vs r=16
rank experiment. This includes environment setup (an isolated `.venv/` registered as the `hw5`
Jupyter kernel) and executing the notebook headless with `nbconvert`.

What I directed rather than delegated: I decided to stop the first `flan-t5-base` run when it had
been training for ~2 hours without finishing, and to keep the finished work honest rather than
fast — the notebook was re-run top to bottom after every fix below, and every number in
`METRICS.md` is read from the final executed notebook, not typed in by hand. I reviewed the actual
generated summaries (not just the ROUGE numbers) before accepting each run, which is how the
problems below were caught.

## One specific thing the assistant produced that was wrong

After the fine-tuned model was seen looping a clause under greedy decoding (*"Ms. Dawson asks
#Person1# to take a dictation for Ms. Dawson. Ms. Dawson asks #Person1# to take a dictation for Ms.
Dawson. …"*), the assistant "fixed" it by adding `no_repeat_ngram_size=3` to `model.generate` — the
textbook anti-repetition setting. That run completed with 0 cell errors and ROUGE still looked
reasonable (ROUGE-1 34.3 for r=16 vs 12.0 baseline), but the summaries were now corrupted:

```
#Person1# asks #Pperson2# to take a dictation for Ms. Dawson. Mss.
#Person1# is stuck in traffic again and #Pon2# thinks it's better to take the subway to work. #Pone2# suggests ...
```

## How I discovered the failure

No error and no bad-looking metric flagged it — it only showed up when reading the two
before/after summaries printed in Section 5. Tokenizing a speaker tag showed the cause:
`tok.tokenize("#Person2#")` → `['▁#', 'P', 'erson', '2', '#']`. DialogSum summaries refer to
speakers by these 5-token tags, so a trigram ban makes it *impossible* to write the same tag twice;
the decoder is forced to spell it differently (`#Pperson2#`, `#Pon2#`) the second time.

## What I changed, and why the fix works

Instead of guessing another setting, six decoding options were compared for the r=16 adapter on
**validation** dialogues 200–299 (held out from both training and the test set, so the choice isn't
tuned on test data), measuring ROUGE, a regex count of mangled tags, and a count of summaries that
repeat a sentence:

| Decoding | ROUGE-1 | mangled tags | looping |
|---|---|---|---|
| greedy | 40.70 | 0/100 | 2/100 |
| greedy + no_repeat_ngram 3 | 36.91 | **67/100** | 1/100 |
| greedy + no_repeat_ngram 6 | 40.99 | 12/100 | 1/100 |
| greedy + repetition_penalty 1.2 | 41.75 | 0/100 | 2/100 |
| beam 4 | 41.10 | 0/100 | 4/100 |
| **beam 4 + no_repeat_ngram 6** | **42.40** | **0/100** | **0/100** |

`summarize()` now uses `num_beams=4, no_repeat_ngram_size=6` for every model (baseline, r=4, r=16).
A 6-gram ban is longer than a speaker tag, so tags can repeat freely, but short enough to block the
~15-token clause loops; beam search keeps the more probable wording. In the final run, the four
fine-tuned summaries of the two fixed test dialogues contain 0 mangled tags and 0 repeated sentences
(checked by the same regexes in `METRICS.md`), and test ROUGE-1 rose to 37.6 (r=16) / 38.0 (r=4).

## A secondary example: a training loss that was silently 2× too high

The first flan-t5-small run used batch 4 × 2 gradient-accumulation steps. Its logged training loss
ended at **2.82** while the validation loss was **1.31** — a gap far too big for a model trained on
only 1,000 examples for 2 epochs to be *better* on unseen data. Rather than explain it away, I had
the saved r=16 adapter evaluated in eval mode on 200 of the very examples it was trained on: the
true loss was **1.31**, matching validation. So the model was fine and the *logged number* was
wrong: with accumulation, this transformers version (5.5.3) reported the loss summed over the 2
micro-batches instead of averaged. Since flan-t5-small fits in memory at batch 8, the fix was to
drop accumulation (`per_device_train_batch_size=8`, `gradient_accumulation_steps=1`, same effective
batch). The final run's logged training loss (1.41 at the last step) is now consistent with its
validation loss (1.32).

## Other issues hit along the way (briefly)

- **`flan-t5-base` does not fit in 16 GB alongside normal apps.** The first run (full train split,
  batch 8, 3 epochs) was still inside the first of two training runs after ~2 hours: 15.9 GB of swap
  in use and only 50 MB of the kernel resident. A second attempt at batch 4 on a 1,000-dialogue
  subset started at ~4.6 s/step but degraded to ~76 s/step as swap filled. Switched to
  `flan-t5-small` (77M params; the model the assignment suggests), which trains at ~3–4.5 s/step
  (12–18 min per run) without swapping to a halt.
- **The test split lists every dialogue three times** (`test_0_1`, `test_0_2`, `test_0_3`, one per
  human reference). The first version took test rows 0 and 1 as "two dialogues" — they were the
  same conversation — and computed ROUGE over the first 100 rows (only ~34 distinct dialogues). The
  notebook now groups the test split by unique dialogue (167), uses two *distinct* dialogues for
  the before/after comparison, and scores ROUGE against all references per dialogue.
- **`evaluate.load("rouge")` needs to download a script from the Hub**, and failed on this
  machine's flaky connection; replaced with the `rouge_score` package, which runs fully offline.

## Verification performed

- `lora_dialogsum.ipynb` was executed top to bottom via `nbconvert --execute` with 0 cell errors on
  the final run (see `RUN_LOG.txt`).
- The adapter-reload path was verified, not assumed: Section 5 loads `outputs/lora_adapter_r16`
  from disk onto a fresh copy of the base model and generates from that.
- Training is reproducible: the final run's first logged losses (2.4053 at step 10, 1.8900 at step
  20) match the previous run's exactly, since the seed, data subset and order are fixed.
