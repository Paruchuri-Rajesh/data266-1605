# METRICS — DATA 266 HW2.5

Rajesh Paruchuri
SID4 = 1605 | SEED = 1605 | SLICE = 605 | HP_ID = 3 | CLS_A = 5 | CLS_B = 3
HP_ID is reported only. No second hyperparameter model.

Every cell is traceable to a UUID-labelled run in `RUN_LOG.txt` and `results/<gpu>/` JSON.


## Table HW2.5.1 — summary

| Measurement | NVIDIA GeForce RTX 4090 | Notes |
|-------------|------|-------|
| Peak achieved TFLOPS (BF16) | 161.75 (N=8192, UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547) | max BF16 GEMM over N=1024..16384; dense peak from vendor whitepaper |
| % of theoretical peak (BF16) | 97.9% (peak 165.2 TFLOPS, UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547) | BF16 tensor, FP32 accumulate, dense (not sparse) |
| Effective bandwidth (GB/s) | 916.3 GB/s (90.9% of 1008.0, UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547) | elementwise add, 3-operand traffic |
| Naive attention OOM length | ok≤37760 fail≥37824 (gap 64, UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547) | largest tested success and smallest tested fail; not 1-token unless gap=1 |
| Fused attention OOM length | ok≤131072 fail≥None (gap None, UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547) | same search on F.scaled_dot_product_attention |
| Steady-state / peak throughput | 97.8% (UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547) | last 5 min mean TFLOPS / first 30 s mean TFLOPS |
| Throttle onset (s, or none) | 10s  temp=57.0 C  power=449.61 W  UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547 | first non-idle throttle reason after 10 s; mechanism confirmed as sw_power_cap from raw thermal_smi.csv, not thermal |

Table HW2.5.1 — The summary table. Every cell is UUID-labelled.

## Part A — provenance

### NVIDIA GeForce RTX 4090

- UUID: `GPU-5b052ad1-4272-40db-4b25-c930bf32b547`
- driver: 610.60
- CUDA (nvidia-smi): 13.3
- VRAM: 24564 MiB
- power limit: 450.00 W
- architecture: Ada Lovelace (AD102)
- memory: 24 GB GDDR6X, 384-bit, 21 Gbps
- bandwidth: 1008.0 GB/s
- tensor cores: 4th generation
- reduced precisions: TF32, FP16, BF16, FP8, INT8, INT4
- sources: ['NVIDIA Ada GPU Architecture whitepaper, Table of GeForce RTX 4090 specs', 'https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf', 'https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/']
- nvidia-smi -q file: `results\NVIDIAGeForceRTX4090_5b052ad1\nvidia_smi_q.txt`

## Part B — GEMM

### NVIDIA GeForce RTX 4090  UUID `GPU-5b052ad1-4272-40db-4b25-c930bf32b547`

| N | precision | TFLOPS | % of dense peak | reps | mean ms | std ms |
|---|-----------|--------|-----------------|------|---------|--------|
| 1024 | fp32 | 34.40 | 41.6 | 200 | 0.062 | 0.006 |
| 4096 | fp32 | 53.24 | 64.5 | 200 | 2.581 | 0.055 |
| 8192 | fp32 | 54.61 | 66.1 | 75 | 20.134 | 0.117 |
| 16384 | fp32 | 50.10 | 60.6 | 20 | 175.583 | 1.704 |
| 1024 | tf32 | 35.27 | 42.7 | 200 | 0.061 | 0.008 |
| 4096 | tf32 | 85.60 | 103.6 | 200 | 1.606 | 0.076 |
| 8192 | tf32 | 87.57 | 106.0 | 120 | 12.556 | 0.127 |
| 16384 | tf32 | 86.90 | 105.2 | 20 | 101.217 | 0.754 |
| 1024 | fp16 | 67.39 | 40.8 | 200 | 0.032 | 0.004 |
| 4096 | fp16 | 166.48 | 100.8 | 200 | 0.826 | 0.025 |
| 8192 | fp16 | 158.18 | 95.8 | 200 | 6.951 | 0.163 |
| 16384 | fp16 | 157.84 | 95.5 | 27 | 55.729 | 0.419 |
| 1024 | bf16 | 71.79 | 43.5 | 200 | 0.030 | 0.004 |
| 4096 | bf16 | 152.45 | 92.3 | 200 | 0.902 | 0.026 |
| 8192 | bf16 | 161.75 | 97.9 | 200 | 6.798 | 0.137 |
| 16384 | bf16 | 160.08 | 96.9 | 28 | 54.949 | 0.521 |

BF16 reaches 95% of its own max (161.7 TFLOPS) at N=8192. FP16 reaches 95% of its own max (166.5 TFLOPS) at N=4096. FP32 reaches 95% of its own max (54.6 TFLOPS) at N=4096. TF32 reaches 95% of its own max (87.6 TFLOPS) at N=4096. Small N never plateaus: kernel launch and cuBLAS setup dominate, the tile count is too small to fill the SMs, and the working set sits in cache so neither DRAM bandwidth nor peak tensor-core throughput is visible.

Note: TF32 measured 103.6-106.0% of the 82.6 TFLOPS dense-peak figure used as the denominator here (the same figure commonly cited as the RTX 4090's non-tensor FP32 peak). Since measured throughput cannot exceed a correct peak, this indicates that 82.6 TFLOPS figure understates Ada's real TF32 tensor-core throughput — read TF32's "% of peak" column as approximate.

Lower precision attempts:
- fp8 OK via `torch._scaled_mm(a8, b8.t().contiguous().t(), ...)` → 263.39 TFLOPS (79.7% of peak)
- fp4 FAILED. tried: ['float4_e2m1fn_x2']. failure: Found symbols ['float4_e2m1fn_x2'] but no public dense GEMM path is wired in this script. Ada does not implement FP4 in its tensor cores (that arrives with Blackwell), so this failure is expected on this card and is treated as a tooling-maturity finding.

## Part C — roofline

### NVIDIA GeForce RTX 4090  UUID `GPU-5b052ad1-4272-40db-4b25-c930bf32b547`

- add: 916.3 GB/s (90.9% of 1008.0 GB/s), AI=0.0833 FLOP/B, memory-bound (left of ridge)
- copy check: 907.8 GB/s (90.1%)
- BF16 GEMM N=16384: 161.01 TFLOPS (97.5% of peak), AI=5461.3 FLOP/B, compute-bound (right of ridge)
- ridge: FP32 81.9 FLOP/B, BF16 163.9 FLOP/B

## Part D — attention

### NVIDIA GeForce RTX 4090  UUID `GPU-5b052ad1-4272-40db-4b25-c930bf32b547`

batch=1, n_heads=16, head_dim=128, dtype=bfloat16

| seq | naive ms | naive peak GB | fused ms | fused peak GB | speedup |
|-----|----------|---------------|----------|---------------|---------|
| 512 | 0.10 | 0.034 | 0.13 | 0.025 | 0.75x |
| 1024 | 0.16 | 0.092 | 0.36 | 0.034 | 0.45x |
| 2048 | 1.00 | 0.311 | 1.22 | 0.050 | 0.82x |
| 4096 | 3.72 | 1.149 | 3.25 | 0.084 | 1.15x |
| 8192 | 14.50 | 4.438 | 5.27 | 0.151 | 2.75x |
| 16384 | 57.26 | 17.457 | 20.41 | 0.285 | 2.81x |

Naive OOM bounds: largest success 37760, smallest fail 37824, gap 64 (single-token resolution: False).
Fused OOM bounds: largest success 131072, smallest fail None, gap None.
Quadratic fit on naive peak memory: a=6.400000e+01 bytes/token², b=1.638400e+04, c=8.519680e+06. RMSE quadratic 3.220e-06 vs linear 1.506e+09. Quadratic better: True. a>0: True.

The coefficients match theory exactly for this config (B=1, H=16, head_dim=128, bf16): a = 2·B·H·2 bytes = 2·1·16·2 = 64 (two live [B,H,S,S] tensors — raw scores and their softmax — at 2 bytes each); b = 4·B·H·D·2 bytes = 4·1·16·128·2 = 16384 (the four [B,H,S,D] tensors: Q, K, V, output).

The fused / FlashAttention-style kernel never writes the S×S score matrix to HBM. It loads Q/K/V tiles into SRAM, computes local products, keeps a running softmax (m, l statistics) in on-chip memory, and writes only the output tile. That removes the O(S²) allocation that OOMs the naive path and the O(S²) memory traffic that makes naive attention bandwidth-bound. Fused is slower than naive below S≈2048, since its fixed per-call overhead isn't amortized at short sequences — the benefit only appears once S is large enough that naive's O(S²) cost dominates.

## Part E — thermal

### NVIDIA GeForce RTX 4090  UUID `GPU-5b052ad1-4272-40db-4b25-c930bf32b547`

- duration: 1203.2s  N=8192 BF16
- first 30s TFLOPS: 161.06215183887156
- last 5 min TFLOPS: 157.44094297466992
- steady/peak: 97.75166988466394
- throttle onset (s): 10.0
- max temp C: 76.0
- log: `results\NVIDIAGeForceRTX4090_5b052ad1\thermal_smi.csv`

Confirmed directly from the raw CSV: `sw_power_cap` is active in every sampled row from the first full-load sample onward, with power draw pinned at 448-450 W against the 450 W TDP limit and graphics clock oscillating ~2505-2520 MHz. `hw_thermal_slowdown` / `sw_thermal_slowdown` stay not-active throughout, and temperature is only 55-62°C in that window (max 76°C over the full run) — this is power-limited throttling, not thermal throttling.
