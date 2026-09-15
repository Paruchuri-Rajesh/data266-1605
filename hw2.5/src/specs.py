"""
Vendor theoretical peaks used as the denominator for % of peak.

Dense (non-sparsity) numbers. Standard cuBLAS / PyTorch GEMM does not enable
structured 2:4 sparsity, so comparing against the sparse column would inflate
the percentage. First number in the NVIDIA whitepaper A/B pairs is dense.

Sources (cite these in METRICS.md / the report):
- NVIDIA Ada GPU Architecture whitepaper (RTX 4090 / AD102):
  https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
- NVIDIA RTX Blackwell GPU Architecture whitepaper (RTX 5090 / GB202):
  https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- GeForce RTX 4090 product page:
  https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
- GeForce RTX 5090 product page:
  https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/

PyTorch float16 / bfloat16 matmul on these cards uses Tensor Cores with
FP32 accumulate, so FP16/BF16 % of peak uses the FP32-accumulate column,
not the FP16-accumulate column.
"""

from __future__ import annotations

# All TFLOPS are dense (no 2:4 sparsity). Bandwidth in GB/s.
GPU_SPECS = {
    "rtx4090": {
        "match": ["4090"],
        "marketing_name": "GeForce RTX 4090",
        "architecture": "Ada Lovelace (AD102)",
        "memory_type": "24 GB GDDR6X, 384-bit, 21 Gbps",
        "memory_gb": 24,
        "bandwidth_gbs": 1008.0,
        "tensor_core_gen": "4th generation",
        "tensor_precisions": ["TF32", "FP16", "BF16", "FP8", "INT8", "INT4"],
        "tflops": {
            "fp32": 82.6,  # CUDA cores, non-tensor
            "tf32": 82.6,  # tensor, dense, FP32 accumulate
            "fp16": 165.2,  # tensor, dense, FP32 accumulate
            "bf16": 165.2,  # tensor, dense, FP32 accumulate
            "fp8": 330.3,  # tensor, dense, FP32 accumulate
            "fp8_fp16acc": 660.6,
            "fp4": None,  # not supported on Ada
        },
        "tdp_w": 450,
        "boost_mhz": 2520,
        "sources": [
            "NVIDIA Ada GPU Architecture whitepaper, Table of GeForce RTX 4090 specs",
            "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf",
            "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/",
        ],
    },
    "rtx5090": {
        "match": ["5090"],
        "marketing_name": "GeForce RTX 5090",
        "architecture": "Blackwell (GB202)",
        "memory_type": "32 GB GDDR7, 512-bit, 28 Gbps",
        "memory_gb": 32,
        "bandwidth_gbs": 1792.0,
        "tensor_core_gen": "5th generation",
        "tensor_precisions": ["TF32", "FP16", "BF16", "FP8", "FP6", "FP4", "INT8"],
        "tflops": {
            "fp32": 104.8,
            "tf32": 104.8,
            "fp16": 209.5,
            "bf16": 209.5,
            "fp8": 419.0,  # FP32 accumulate
            "fp8_fp16acc": 838.0,
            "fp4": 1676.0,  # FP32 accumulate; 3352 with sparsity / FP4 AI TOPS
        },
        "tdp_w": 575,
        "boost_mhz": 2407,
        "sources": [
            "NVIDIA RTX Blackwell GPU Architecture whitepaper, Table 3 (RTX 5090 vs 4090 vs 3090)",
            "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf",
            "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/",
        ],
    },
}


def identify_spec(gpu_name: str) -> dict | None:
    name = (gpu_name or "").lower()
    for spec in GPU_SPECS.values():
        if any(token in name for token in spec["match"]):
            return spec
    return None


def peak_tflops(gpu_name: str, precision: str) -> float | None:
    spec = identify_spec(gpu_name)
    if spec is None:
        return None
    return spec["tflops"].get(precision)


def ridge_point_flops_per_byte(peak_tflops_value: float, bandwidth_gbs: float) -> float:
    """Roofline ridge: peak FLOP/s divided by peak bytes/s -> FLOPs/byte."""
    return (peak_tflops_value * 1e12) / (bandwidth_gbs * 1e9)
