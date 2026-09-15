#!/usr/bin/env python3
"""Part B — dense GEMM throughput in FP32, TF32, FP16, BF16, plus lower precision."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.identity import append_run_log, gpu_info, results_dir, write_json  # noqa: E402
from src.params import SEED, banner, seed_everything  # noqa: E402
from src.specs import identify_spec, peak_tflops  # noqa: E402
from src.timing import cuda_times_ms, gemm_tflops  # noqa: E402


def configure_precision(torch, precision: str) -> None:
    if precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    elif precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def make_pair(torch, n: int, dtype, device: int):
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    return a, b


def try_fp8_scaled_mm(torch, n: int, device: int) -> dict:
    """Lower precision exposed on Ada (FP8) and Blackwell (FP8/FP4)."""
    attempts = []
    if not hasattr(torch, "float8_e4m3fn"):
        return {
            "ok": False,
            "precision": "fp8",
            "tried": "torch.float8_e4m3fn",
            "failure": "torch.float8_e4m3fn is not on this PyTorch build",
        }

    a_bf = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b_bf = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    a8 = a_bf.to(torch.float8_e4m3fn)
    b8 = b_bf.to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0, device=device)
    scale_b = torch.tensor(1.0, device=device)

    # Several PyTorch versions disagree on _scaled_mm argument names / layout.
    calls = []
    if hasattr(torch, "_scaled_mm"):
        calls.append(
            (
                "torch._scaled_mm(a8, b8, scale_a, scale_b, out_dtype=bf16)",
                lambda: torch._scaled_mm(
                    a8, b8, scale_a, scale_b, out_dtype=torch.bfloat16
                ),
            )
        )
        calls.append(
            (
                "torch._scaled_mm(a8, b8.t().contiguous().t(), ...)",
                lambda: torch._scaled_mm(
                    a8,
                    b8.t().contiguous().t(),
                    scale_a,
                    scale_b,
                    out_dtype=torch.bfloat16,
                ),
            )
        )
        try:
            calls.append(
                (
                    "torch._scaled_mm kwargs scale_a/scale_b",
                    lambda: torch._scaled_mm(
                        a8, b8, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
                    ),
                )
            )
        except TypeError:
            pass

    if not calls:
        return {
            "ok": False,
            "precision": "fp8",
            "tried": "torch._scaled_mm",
            "failure": "torch._scaled_mm is not on this PyTorch build",
        }

    last_err = None
    for label, fn in calls:
        try:
            out = fn()
            if out is None:
                raise RuntimeError("returned None")
            attempts.append({"call": label, "ok": True})
            stats = cuda_times_ms(lambda: fn(), warmup=5, min_reps=10, min_seconds=1.0)
            tflops = gemm_tflops(n, stats["mean_ms"])
            return {
                "ok": True,
                "precision": "fp8",
                "n": n,
                "call_that_worked": label,
                "attempts": attempts,
                "tflops": tflops,
                **{k: stats[k] for k in ("warmup", "reps", "mean_ms", "std_ms", "cv")},
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            attempts.append({"call": label, "ok": False, "error": last_err})

    return {
        "ok": False,
        "precision": "fp8",
        "n": n,
        "tried": [c[0] for c in calls],
        "attempts": attempts,
        "failure": last_err or "all scaled_mm signatures failed",
        "traceback": traceback.format_exc(),
    }


def try_fp4(torch, n: int, device: int) -> dict:
    names = [
        n for n in dir(torch) if "float4" in n.lower() or "fp4" in n.lower() or "e2m1" in n.lower()
    ]
    if not names:
        return {
            "ok": False,
            "precision": "fp4",
            "tried": "dir(torch) for float4 / fp4 / e2m1",
            "failure": (
                "No FP4 dtype on this PyTorch build. Blackwell 5th-gen tensor cores "
                "support FP4 in hardware (NVIDIA RTX Blackwell whitepaper), but the "
                "installed stack does not expose a callable FP4 GEMM API."
            ),
        }
    return {
        "ok": False,
        "precision": "fp4",
        "tried": names,
        "failure": (
            f"Found symbols {names} but no public dense GEMM path is wired in this script. "
            "Treated as tooling-maturity finding unless a later cell succeeds."
        ),
    }


def run_one(torch, n: int, precision: str, device: int, quick: bool) -> dict:
    configure_precision(torch, precision)
    dtype = {
        "fp32": torch.float32,
        "tf32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]
    a, b = make_pair(torch, n, dtype, device)

    def gemm():
        return torch.mm(a, b)

    warmup = 3 if quick else 10
    min_reps = 5 if quick else 20
    min_seconds = 0.2 if quick else 1.5
    max_reps = 30 if quick else 200
    stats = cuda_times_ms(
        gemm, warmup=warmup, min_reps=min_reps, min_seconds=min_seconds, max_reps=max_reps
    )
    tflops = gemm_tflops(n, stats["mean_ms"])
    del a, b
    torch.cuda.empty_cache()
    return {
        "ok": True,
        "n": n,
        "precision": precision,
        "dtype": str(dtype),
        "tflops": tflops,
        **{k: stats[k] for k in ("warmup", "reps", "mean_ms", "std_ms", "min_ms", "max_ms", "cv")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    quick = args.quick

    print(banner())
    seed_everything(SEED)
    torch = __import__("torch")
    info = gpu_info(args.device)
    spec = identify_spec(info["name"])
    sizes = [1024, 2048] if quick else [1024, 4096, 8192, 16384]
    precisions = ["fp32", "tf32", "fp16", "bf16"]

    rows = []
    lines = [banner(), f"UUID: {info['uuid']}", f"GPU: {info['name']}", f"sizes: {sizes}"]
    for precision in precisions:
        for n in sizes:
            print(f"GEMM {precision} N={n} ...", flush=True)
            row = run_one(torch, n, precision, args.device, quick)
            peak = peak_tflops(info["name"], precision)
            row["peak_tflops"] = peak
            row["pct_of_peak"] = (100.0 * row["tflops"] / peak) if peak else None
            row["uuid"] = info["uuid"]
            rows.append(row)
            lines.append(
                f"  {precision:5s} N={n:5d}  {row['tflops']:8.2f} TFLOPS  "
                f"({row['pct_of_peak']:.1f}% of {peak} peak)  "
                f"reps={row['reps']} mean={row['mean_ms']:.3f}ms std={row['std_ms']:.3f}ms cv={row['cv']:.3f}"
                if peak
                else f"  {precision:5s} N={n:5d}  {row['tflops']:8.2f} TFLOPS  (no peak table for this GPU name)"
            )

    lower = []
    n_low = 4096 if quick else 8192
    print(f"Trying FP8 at N={n_low} ...", flush=True)
    fp8 = try_fp8_scaled_mm(torch, n_low, args.device)
    fp8["uuid"] = info["uuid"]
    if fp8.get("ok"):
        peak = peak_tflops(info["name"], "fp8")
        fp8["peak_tflops"] = peak
        fp8["pct_of_peak"] = (100.0 * fp8["tflops"] / peak) if peak else None
        lines.append(
            f"  fp8   N={n_low}  {fp8['tflops']:.2f} TFLOPS  "
            f"via {fp8.get('call_that_worked')}  pct={fp8.get('pct_of_peak')}"
        )
    else:
        lines.append(f"  fp8 FAILED: tried {fp8.get('tried')} ; {fp8.get('failure')}")
    lower.append(fp8)

    print("Trying FP4 ...", flush=True)
    fp4 = try_fp4(torch, n_low, args.device)
    fp4["uuid"] = info["uuid"]
    lower.append(fp4)
    lines.append(f"  fp4: ok={fp4.get('ok')} {fp4.get('failure') or fp4.get('tried')}")

    payload = {
        "part": "B",
        "gpu": info,
        "vendor_spec": spec,
        "sizes": sizes,
        "precisions": precisions,
        "rows": rows,
        "lower_precision": lower,
        "notes": (
            "TF32 uses float32 tensors with allow_tf32=True. "
            "FP32 disables TF32. FP16/BF16 use tensor-core GEMM with FP32 accumulate "
            "as the % of peak denominator. Warmup then timed CUDA events."
        ),
    }
    out = results_dir(info)
    write_json(out / "part_b_matmul.json", payload)
    text = "\n".join(lines)
    print(text)
    append_run_log("PART B\n" + text, info)


if __name__ == "__main__":
    main()
