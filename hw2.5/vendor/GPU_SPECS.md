# DATA 266 — HW2.5 vendor specs (Part A)

Dense theoretical peaks only (no 2:4 sparsity). PyTorch / cuBLAS GEMM in this
assignment does not use structured sparsity, so the sparse whitepaper column
would overstate "% of peak".

## GeForce RTX 4090

| Field | Value | Source |
|-------|-------|--------|
| Architecture | Ada Lovelace, AD102 | NVIDIA Ada GPU Architecture whitepaper |
| Memory | 24 GB GDDR6X, 384-bit, 21 Gbps | same, plus GeForce RTX 4090 product page |
| Bandwidth | 1008 GB/s | Ada whitepaper |
| Tensor cores | 4th generation (512) | Ada whitepaper |
| Reduced precisions on tensor cores | TF32, FP16, BF16, FP8, INT8, INT4 | Ada whitepaper |
| Peak FP32 (CUDA cores) | 82.6 TFLOPS | Ada whitepaper |
| Peak TF32 tensor (dense) | 82.6 TFLOPS | Ada whitepaper |
| Peak FP16 / BF16 tensor, FP32 accumulate (dense) | 165.2 TFLOPS | Ada whitepaper |
| Peak FP8 tensor, FP32 accumulate (dense) | 330.3 TFLOPS | Ada whitepaper |
| Peak FP8 tensor, FP16 accumulate (dense) | 660.6 TFLOPS | Ada whitepaper |
| TGP | 450 W | product page / Ada whitepaper |

Citations:
- https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
- https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/

## GeForce RTX 5090

| Field | Value | Source |
|-------|-------|--------|
| Architecture | Blackwell, GB202 | NVIDIA RTX Blackwell GPU Architecture whitepaper, Table 3 |
| Memory | 32 GB GDDR7, 512-bit, 28 Gbps | same |
| Bandwidth | 1792 GB/s (1.792 TB/s) | Blackwell whitepaper |
| Tensor cores | 5th generation (680) | Blackwell whitepaper |
| Reduced precisions on tensor cores | TF32, FP16, BF16, FP8, FP6, FP4, INT8 | Blackwell whitepaper |
| Peak FP32 (CUDA cores) | 104.8 TFLOPS | Table 3 |
| Peak TF32 tensor (dense) | 104.8 TFLOPS | Table 3 |
| Peak FP16 / BF16 tensor, FP32 accumulate (dense) | 209.5 TFLOPS | Table 3 |
| Peak FP8 tensor, FP32 accumulate (dense) | 419 TFLOPS | Table 3 |
| Peak FP8 tensor, FP16 accumulate (dense) | 838 TFLOPS | Table 3 |
| Peak FP4 tensor, FP32 accumulate (dense) | 1676 TFLOPS | Table 3 |
| TGP | 575 W | product page |

Citations:
- https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/

## How % of peak is computed in this assignment

| Benchmark precision | Peak column used |
|---------------------|------------------|
| FP32 (`allow_tf32=False`) | non-tensor FP32 |
| TF32 (`float32` + `allow_tf32=True`) | TF32 tensor dense |
| FP16 | FP16 tensor, FP32 accumulate, dense |
| BF16 | BF16 tensor, FP32 accumulate, dense |
| FP8 (if the stack can run it) | FP8 tensor, FP32 accumulate, dense |
| FP4 (if the stack can run it) | FP4 tensor, FP32 accumulate, dense |

Every measured number is labelled with the GPU UUID from `nvidia-smi -q`.
