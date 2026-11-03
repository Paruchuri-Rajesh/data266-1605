# Written answers (pulled into METRICS.md by make_report.py)

<!-- Each "## key" section replaces the matching TODO in METRICS.md. Edit here, then run `python make_report.py`. -->
<!-- Numbers quoted below come from results/kaggle_t4, results/colab_t4 and RUN_LOG.txt (Tesla T4s, 2026-11-03). -->

## diagnosis_1
Labelled neutral, but the text is a complaint about a weeks-long runaround, so this looks like a label error; "negative" is a reasonable prediction.

## diagnosis_2
Sarcasm: the literal words ("values your tweet", "complimenting") are positive while the meaning is a jab at United, and the gold label of neutral is itself debatable.

## diagnosis_3
The negative words ("Darn", "Terrible memory") are aimed at the writer's own memory, not the airline; the model scores sentiment words without knowing what they are aimed at.

## diagnosis_4
Wordplay on #burningman: "burn my dreams" plus "!!!" reads as distress, but it is a request about ticket sales, which the model cannot tell apart.

## diagnosis_5
A reply in someone else's thread: "at least they gave u a refund" implies a bad trip, and whether it counts as neutral depends on context the model never sees.

## q6
In the measured range, per-request latency never becomes worse than the HF batch-1 baseline. vLLM's p50 rises from 921 ms at batch 1 to 998 ms at batch 8 and 1,394 ms at batch 32, which is still 3.6× below HF's 4,999 ms, so the crossing point is above batch 32. Against vLLM's own batch 1, per-request latency gets worse at every step (p95 +9% at batch 8, +45% at batch 32) while total throughput rises 7.3× and 21× (138 → 1,008 → 2,937 tok/s). For an interactive application the per-request latency (especially p95, the tail users notice) is the number that matters, not aggregate tokens/s. On this T4, batch 8 is the sweet spot: 7× the throughput for about 9% more latency. (All requests in a batch are submitted together, so latency here is the batch's wall time; time-to-first-token was not measured.)

## partB
Part B was measured on two T4 sessions. The Kaggle T4 table covers gradient checkpointing, gradient accumulation and 8-bit AdamW. The Colab T4 table covers mixed precision and the combinations, each against a fp32 baseline rerun on that same GPU. Compare absolute numbers only within one table (the combined ranking uses percentages against each card's own baseline): the identical fp32 baseline ran at 43.3 samples/s on Kaggle (torch 2.8) and 52.4 on Colab (torch 2.11). The first Kaggle AMP runs used emulated bf16 and are kept only in `superseded/`.

Apply first: **mixed precision (fp16 + GradScaler)**. On the Colab T4 it raised throughput from 52.4 to 171.9 samples/s (3.3×, because the T4's tensor cores run fp16), cut peak memory from 3.99 to 3.24 GB, and doubled the max batch from 128 to 256. It is the only technique that saves memory and time at once. Test accuracy was 0.8067 vs 0.8293, while final val accuracy was higher (0.847 vs 0.840). Since the same fp32 baseline itself differed by 0.8 points between the two T4s, part of that gap is run-to-run noise. Next is **8-bit AdamW**, which saved 0.69 GB for 0.4% throughput. Combined with fp16 ("best combination") it ran at 171.2 samples/s in 2.58 GB. Gradient accumulation (8 × 16: 2.46 GB vs 11.67 GB for a direct batch of 128, 6% slower) is the tool for reaching a large effective batch. Gradient checkpointing gave the lowest memory (1.25 GB with fp16 + 8-bit, max batch 1,024), but it cost 26.2% throughput in fp32 and 22.4% on top of fp16 + 8-bit. On a 32 GB card I would not bother with gradient checkpointing (or accumulation): the HW4 settings need only 2.6–4 GB there, so the memory it frees is not needed and the throughput it costs is real.

## partD
The HW4.a serving run (08:13 in RUN_LOG.txt) and the Part D rerun (10:44) used the same Tesla T4, model (Qwen2.5-0.5B-Instruct), precision (float16) and attention backend, and agree within 2.5%. HF batch 1 gave 24.7 vs 25.0 tok/s. vLLM gave 135.1 vs 138.4 tok/s at batch 1, 994.9 vs 1,008.2 at batch 8 and 2,923.4 vs 2,937.3 at batch 32, with p50 within 2% (923/921, 1,017/998, 1,401/1,394 ms). Differences that small are run-to-run noise (GPU clock and thermal state on a shared Kaggle host), not a hardware or precision effect. Both runs write the same result files, so the tables above show the Part D run; the HW4.a numbers are in RUN_LOG.txt. One hardware caveat applies to both runs: the T4 has no FlashAttention 2, and vLLM's FlexAttention fallback crashed with an illegal memory access, so serving used `VLLM_ATTENTION_BACKEND=TRITON_ATTN`; an Ampere or newer card would be faster. For the KV cache, Qwen2.5-1.5B needs 28 KiB per token, and vLLM's 10.38 GB pool holds 362,144 tokens (45,268 per sequence at batch 8). The longest context at batch 8 is therefore capped by the model's 32,768-token limit rather than by memory: 8 × 32,768 tokens use 7.52 GB, and a real batch-8 run at that length succeeded.

## partE
I would use **mixed precision + 8-bit AdamW at batch 32 (the "best combination")**. On the Colab T4 it trains at 171.2 samples/s, so one full fine-tune on 70% of all 14,640 rows for 4 epochs costs **0.066 GPU-hours**, versus 0.217 for the fp32 baseline on the same GPU (and 0.263 on the Kaggle T4). It peaks at 2.58 GB, so memory is far from the limit. Its test accuracy, 0.8080, is within run-to-run variation of the baselines (0.8213 and 0.8293). Adding gradient checkpointing would raise the cost to 0.086 GPU-hours for memory that is not needed. Validation loss in this run was lowest at epoch 3 (0.450, rising to 0.461 at epoch 4), so stopping after 3 epochs would cut the cost to about 0.05 GPU-hours without losing accuracy. On a lab RTX 4090/5090, `--amp auto` would pick native bf16 instead of fp16 (no GradScaler needed), with the same choice of configuration.
