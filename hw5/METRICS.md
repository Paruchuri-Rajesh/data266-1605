# HW5 Metrics

SID4 = 1605, SEED = 1605. All numbers below are read directly from the final executed run of `lora_dialogsum.ipynb` (0 cell errors — see `RUN_LOG.txt`); this file is generated from `outputs/all_outputs.json` and the notebook outputs, not typed by hand.

## Setup

| Item | Value |
|---|---|
| Base model | `google/flan-t5-small` (76,961,152 params) |
| Dataset | `neil-code/dialogsum-test`: 1,999 train / 499 validation / 499 test rows (test = 167 unique dialogues: 166 with 3 human references, 1 with a single reference) |
| Training data | random 1,000-dialogue subset of train (seed 1605), 2 epochs, 250 optimizer steps |
| Optimizer | AdamW, lr 1e-3, linear decay, 5% warmup, batch 8, no gradient accumulation |
| LoRA | target modules `q`, `v` (all self- and cross-attention blocks), dropout 0.05, α/r = 2 |
| Decoding (all models) | beam search, 4 beams, `no_repeat_ngram_size=6`, max 128 new tokens |
| Hardware | Apple Silicon (MPS), 16 GB unified memory |

## LoRA parameters

| Config | Total params | Trainable params | % trainable | Adapter file |
|---|---|---|---|---|
| r=16, α=32 | 77,649,280 | 688,128 | 0.886% | 2.77 MB |
| r=4, α=8 | 77,133,184 | 172,032 | 0.223% | 0.70 MB |

Base model with no adapters: 76,961,152 parameters, all frozen during LoRA training.

## Fine-tuning (loss)

| Run | First logged train loss (step 10) | Final logged train loss (step 250) | Avg train loss | Val loss epoch 1 | Val loss epoch 2 | Train time |
|---|---|---|---|---|---|---|
| LoRA r=16 | 2.4053 | 1.4099 | 1.6016 | 1.3372 | 1.3167 | 18.3 min |
| LoRA r=4 | 2.5011 | 1.4522 | 1.6575 | 1.3529 | 1.3378 | 12.2 min |

Loss curves: `outputs/loss_curve_r16.png`, `outputs/loss_curve_r4_vs_r16.png`. Full per-step logs: `outputs/train_progress_r16.log`, `outputs/train_progress_r4.log`.

## ROUGE on the test set (167 unique dialogues, scored against all their human references, F1 × 100)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg summary length (words) |
|---|---|---|---|---|
| Baseline (zero-shot) | 17.45 | 5.33 | 14.89 | 10.7 |
| LoRA r=4 | 38.04 | 13.45 | 30.46 | 22.6 |
| LoRA r=16 | 37.63 | 12.78 | 29.89 | 23.0 |

Human reference summaries average 19.3 words.

## The two fixed test dialogues: before vs after

**Test dialogue 0** (unique test dialogue #0)

| Model | Summary |
|---|---|
| Reference | Ms. Dawson helps #Person1# to write a memo to inform every employee that they have to change the communication method and should not use Instant Messaging anymore. |
| Baseline | Ms. Dawson, Attached is a draft memo to all employees. |
| LoRA r=4 | #Person1# asks #Person2# to take a dictation for Ms. Dawson. Ms. Dawson asks #Person1# to go out as an intra-office memorandum to all employees by this afternoon. |
| LoRA r=16 | #Person1# asks #Person2# to take a dictation for Ms. Dawson. Ms. Dawson tells #Person1# that the use of instant messaging programs by employees during working hours is strictly prohibited. |

**Test dialogue 1** (unique test dialogue #1)

| Model | Summary |
|---|---|
| Reference | #Person2# arrives late because of traffic jam. #Person1# persuades #Person2# to use public transportations to keep healthy and to protect the environment. |
| Baseline | Talk to a friend. |
| LoRA r=4 | #Person1# is stuck in traffic again. #Person2# thinks it would be better if #Person1# started taking public transport system to work. #Person1# thinks the public transport system would be better for the environment. |
| LoRA r=16 | #Person1# tells #Person2# that #Person2#'s got stuck in traffic again. #Person2# thinks it's better if #Person1# started taking public transport system to work. #Person1# thinks biking to work would be better for the environment. |

Sanity checks on these fine-tuned summaries: mangled speaker tags = 0/4, summaries repeating a sentence = 0/4.

## Findings

### Before vs after fine-tuning

- **Before**, flan-t5-small does not really summarize: it emits one short, often off-topic line — *"Ms. Dawson, Attached is a draft memo to all employees."* reads like the start of the memo itself, and *"Talk to a friend."* is unrelated to a dialogue about commuting (baseline average 10.7 words, ROUGE-1 17.45 on all 167 test dialogues).
- **After** LoRA (r=16), outputs follow the DialogSum style: 2–3 third-person sentences, ~23 words (references average 19.3), naming speakers with `#Person1#`/`#Person2#` tags and covering the main events (the Instant-Messaging ban; being stuck in traffic → taking public transport / biking). ROUGE-1 rises 17.45 → 37.63 and ROUGE-2 5.33 → 12.78, having trained only 0.89% of the parameters.
- The remaining errors are about **who did what**: in dialogue 0 it says *Ms. Dawson* tells #Person1# about the ban (it is #Person1# dictating to her), and in dialogue 1 it reverses who suggests public transport. The fine-tuned model learned the format and main content, but not reliable speaker attribution.

### LoRA rank r=4 vs r=16

- **Capacity/loss:** r=16 trains 4× more parameters (688,128 vs 172,032) and fits slightly better: validation loss 1.317 vs 1.338 and final logged train loss 1.410 vs 1.452.
- **Output quality:** that small loss advantage **does not show up in ROUGE**: r=4 is marginally *higher* (ROUGE-1 38.04 vs 37.63, ROUGE-2 13.45 vs 12.78, ROUGE-L 30.46 vs 29.89). The gaps are under 0.7 points on 167 dialogues, most likely within run-to-run noise. Both produce well-formed summaries of similar length (22.6 vs 23.0 words). On the two fixed dialogues, r=16 captured slightly more of the key content: it names the Instant-Messaging ban in dialogue 0, where r=4 only restates the memo logistics, and it correctly says #Person2# got stuck in traffic in dialogue 1, where r=4 attributes it to #Person1#. Both still swap some speaker roles.
- **Conclusion:** for this small model and 1,000-dialogue subset, **r=4 is sufficient**; going to r=16 buys a little lower loss but no measurable ROUGE gain, at 4× the adapter size (0.70 MB vs 2.77 MB). (Wall-clock time, 12.2 vs 18.3 min, differed because of memory pressure on the machine during the r=16 run, not because of the rank — LoRA's rank adds negligible compute here.)
