#!/usr/bin/env python3
"""Part C — memory-bound copy/add vs compute-bound GEMM, arithmetic intensity, roofline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.identity import append_run_log, gpu_info, results_dir, write_json  # noqa: E402
from src.params import SEED, banner, seed_everything  # noqa: E402
from src.specs import identify_spec, peak_tflops, ridge_point_flops_per_byte  # noqa: E402
from src.timing import cuda_times_ms, gemm_tflops  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print(banner())
    seed_everything(SEED)
    torch = __import__("torch")
    info = gpu_info(args.device)
    spec = identify_spec(info["name"])
    if spec is None:
        raise SystemExit(f"Unknown GPU '{info['name']}'. Cannot compute % of spec bandwidth.")

    device = args.device
    # ~2 GiB of fp32 per tensor on 4090/5090. Two inputs + output ~6 GiB.
    n_elem = 2**26 if args.quick else 2**28  # 256M or 1G floats? 2**28 = 1.0 GiB per tensor
    # 2**28 fp32 = 1 GiB. Three tensors = 3 GiB. Safe on 24 GB.
    x = torch.randn(n_elem, device=device, dtype=torch.float32)
    y = torch.randn(n_elem, device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    def add():
        torch.add(x, y, out=out)

    def copy():
        out.copy_(x)

    add_stats = cuda_times_ms(
        add,
        warmup=5 if args.quick else 20,
        min_reps=10 if args.quick else 50,
        min_seconds=0.3 if args.quick else 2.0,
    )
    copy_stats = cuda_times_ms(
        copy,
        warmup=5 if args.quick else 20,
        min_reps=10 if args.quick else 50,
        min_seconds=0.3 if args.quick else 2.0,
    )

    bytes_add = 3 * n_elem * 4  # two reads, one write
    bytes_copy = 2 * n_elem * 4  # one read, one write
    bw_add = bytes_add / (add_stats["mean_ms"] * 1e-3) / 1e9
    bw_copy = bytes_copy / (copy_stats["mean_ms"] * 1e-3) / 1e9
    spec_bw = spec["bandwidth_gbs"]

    # Compute-bound: large square BF16 GEMM.
    n = 4096 if args.quick else 16384
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)

    def gemm():
        return torch.mm(a, b)

    gemm_stats = cuda_times_ms(
        gemm,
        warmup=3 if args.quick else 10,
        min_reps=5 if args.quick else 20,
        min_seconds=0.3 if args.quick else 2.0,
    )
    tflops = gemm_tflops(n, gemm_stats["mean_ms"])
    peak_bf16 = peak_tflops(info["name"], "bf16")

    # Arithmetic intensity.
    # Elementwise add: 1 FLOP (add) per 12 bytes moved.
    ai_add = 1.0 / 12.0
    # Copy: 0 FLOPs per 8 bytes — purely memory.
    ai_copy = 0.0
    # GEMM: 2 N^3 FLOPs, 3 N^2 * 2 bytes (bf16) moved if no reuse modelled.
    bytes_gemm = 3 * n * n * 2
    flops_gemm = 2.0 * (n ** 3)
    ai_gemm = flops_gemm / bytes_gemm  # = 2N / 6 = N/3 for bf16

    ridge_bf16 = ridge_point_flops_per_byte(peak_bf16, spec_bw)
    ridge_fp32 = ridge_point_flops_per_byte(spec["tflops"]["fp32"], spec_bw)

    def side(ai, ridge):
        if ai <= 0:
            return "memory-bound (zero / near-zero arithmetic intensity)"
        if ai < ridge:
            return "memory-bound (left of ridge)"
        return "compute-bound (right of ridge)"

    payload = {
        "part": "C",
        "gpu": info,
        "uuid": info["uuid"],
        "vendor_spec": spec,
        "memory_bound": {
            "op": "elementwise add  out = x + y",
            "n_elem": n_elem,
            "bytes_moved": bytes_add,
            "effective_bandwidth_gbs": bw_add,
            "pct_of_spec_bandwidth": 100.0 * bw_add / spec_bw,
            "spec_bandwidth_gbs": spec_bw,
            "arithmetic_intensity_flops_per_byte": ai_add,
            "roofline_side": side(ai_add, ridge_fp32),
            "timing": {k: add_stats[k] for k in ("warmup", "reps", "mean_ms", "std_ms", "cv")},
            "copy_check": {
                "op": "out.copy_(x)",
                "bytes_moved": bytes_copy,
                "effective_bandwidth_gbs": bw_copy,
                "pct_of_spec_bandwidth": 100.0 * bw_copy / spec_bw,
                "arithmetic_intensity_flops_per_byte": ai_copy,
                "timing": {k: copy_stats[k] for k in ("warmup", "reps", "mean_ms", "std_ms", "cv")},
            },
        },
        "compute_bound": {
            "op": "square BF16 matmul",
            "n": n,
            "tflops": tflops,
            "peak_bf16_tflops": peak_bf16,
            "pct_of_peak": 100.0 * tflops / peak_bf16 if peak_bf16 else None,
            "bytes_modelled": bytes_gemm,
            "flops": flops_gemm,
            "arithmetic_intensity_flops_per_byte": ai_gemm,
            "roofline_side": side(ai_gemm, ridge_bf16),
            "timing": {k: gemm_stats[k] for k in ("warmup", "reps", "mean_ms", "std_ms", "cv")},
        },
        "ridge_point": {
            "fp32_flops_per_byte": ridge_fp32,
            "bf16_flops_per_byte": ridge_bf16,
            "note": (
                "Ridge = peak_FLOP/s / peak_bytes/s. Ops with AI below the ridge "
                "are bandwidth-bound; above it they are compute-bound."
            ),
        },
    }
    write_json(results_dir(info) / "part_c_roofline.json", payload)

    lines = [
        banner(),
        f"UUID: {info['uuid']}",
        f"spec bandwidth: {spec_bw} GB/s",
        f"add: {bw_add:.1f} GB/s  ({payload['memory_bound']['pct_of_spec_bandwidth']:.1f}% of spec)  "
        f"AI={ai_add:.4f} FLOP/B  {payload['memory_bound']['roofline_side']}",
        f"copy: {bw_copy:.1f} GB/s  ({payload['memory_bound']['copy_check']['pct_of_spec_bandwidth']:.1f}% of spec)",
        f"bf16 GEMM N={n}: {tflops:.2f} TFLOPS  "
        f"({payload['compute_bound']['pct_of_peak']:.1f}% of {peak_bf16})  "
        f"AI={ai_gemm:.1f} FLOP/B  {payload['compute_bound']['roofline_side']}",
        f"ridge FP32={ridge_fp32:.1f} FLOP/B  ridge BF16={ridge_bf16:.1f} FLOP/B",
    ]
    text = "\n".join(lines)
    print(text)
    append_run_log("PART C\n" + text, info)


if __name__ == "__main__":
    main()
