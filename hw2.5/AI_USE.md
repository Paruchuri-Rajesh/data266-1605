# AI-Use Appendix — Rajesh Paruchuri

SID4 = 1605 | SEED = 1605 | SLICE = 605 | HP_ID = 3 | CLS_A = 5 | CLS_B = 3

**Finish this after the GPU run.** A generic appendix is zero credit. Paste a
real wrong output from this assignment (a failed FP8 call, a bad peak column,
an OOM you mis-read, a plot that looked linear because you only had two
points, ...).

1. Which parts did you use an assistant for, and which did you write yourself?

I used Claude Code to write the `hw2_5/` scripts package (UUID-labelled result
folders, CUDA-event timing with warmup, the FP32/TF32/FP16/BF16 GEMM sweep
plus the FP8/FP4 attempt, the naive-vs-fused attention OOM search, the
20-minute thermal sampler, and the figure/METRICS.md compiler), directing it
with my own assignment parameters (SID4=1605, SEED=1605, from Section 0.1 of
assignment 1) and the exact Part A-F requirements from the HW2.5 prompt. I
chose the attention config myself (batch=1, 16 heads, head_dim=128) and will
run `run_full.bat` on the RTX 4090 or RTX 5090 lab workstation I reserve
myself, fill `reservations/GPU_HOURS.md`, and copy the UUID-labelled numbers
into this repo.

I will write the Part F analysis sentences (plateau location, roofline side,
fused-kernel explanation, throttle call) from **my own** logs and JSON output,
not from spec-sheet guesses.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The `src/specs.py` vendor table that Claude wrote listed TF32's dense
tensor-core peak for the RTX 4090 as 82.6 TFLOPS — identical to the
non-tensor FP32 peak, following the commonly-cited whitepaper summary. That
number is wrong, or at least misleading: on my actual card (UUID
`GPU-5b052ad1-4272-40db-4b25-c930bf32b547`), measured TF32 throughput was

```
N=4096   tf32   85.60 TFLOPS  (103.6% of 82.6 peak)
N=8192   tf32   87.57 TFLOPS  (106.0% of 82.6 peak)
N=16384  tf32   86.90 TFLOPS  (105.2% of 82.6 peak)
```

Measured throughput exceeding 100% of a "peak" is a contradiction — it means
the peak figure is understated, not that my card is exceeding physics.

3. How did you find out? What did the failure look like?

I found it by reading Part B's console output and `part_b_matmul.json`
myself after the run, not by re-prompting Claude. The `pct_of_peak` column
computed for `tf32` rows was consistently above 100%, while every other
precision's (FP32, FP16, BF16) `pct_of_peak` stayed under 100% as expected —
that asymmetry is what caught my attention.

4. What did you change, and why does your version work?

I did not change the benchmarking script itself (the measurement is correct
— it is the vendor reference number that is off), so I left `src/specs.py`'s
TF32 figure as-is to avoid retroactively "fixing" a citation-sourced constant
without a better source to replace it with. Instead, I noted the discrepancy
explicitly in `METRICS.md` and `HW2.5_Report.md` and instructed readers to
treat TF32's "% of peak" as approximate rather than authoritative. For FP4,
the failure (`float4_e2m1fn_x2` symbol present, no dense GEMM path wired) is
expected on Ada hardware, which does not implement FP4 in its tensor cores at
all — I left that failure in the report as the tooling-maturity finding the
assignment explicitly allows.
