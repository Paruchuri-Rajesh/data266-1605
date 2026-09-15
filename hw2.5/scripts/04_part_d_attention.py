#!/usr/bin/env python3
"""Part D — naive vs fused scaled-dot-product attention: latency, peak memory, OOM."""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.identity import append_run_log, gpu_info, results_dir, write_json  # noqa: E402
from src.params import SEED, banner, seed_everything  # noqa: E402
from src.timing import cuda_times_ms  # noqa: E402

# One batch size and one head dimension, stated here and in METRICS.md.
# head_dim=128 with 16 heads (2048 model width, a common LLM head size) at
# bf16 makes the S×S score tensor large enough to OOM on 24-32 GB inside the
# assigned sequence sweep (16384), which is the point of the lab.
ATTN_BATCH = 1
ATTN_HEADS = 16
ATTN_HEAD_DIM = 128
ATTN_DTYPE_NAME = "bfloat16"

COARSE_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]


def dtype_of(torch):
    return torch.bfloat16


def naive_sdpa(q, k, v):
    """Materialize the full sequence-by-sequence attention matrix."""
    scale = ATTN_HEAD_DIM ** -0.5
    scores = q @ k.transpose(-2, -1)  # [B, H, S, S]
    scores = scores * scale
    weights = torch_softmax(scores)
    return weights @ v


def torch_softmax(scores):
    import torch

    return torch.softmax(scores, dim=-1)


def fused_sdpa(q, k, v):
    import torch
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)


def alloc_qkv(torch, seq: int, device: int):
    shape = (ATTN_BATCH, ATTN_HEADS, seq, ATTN_HEAD_DIM)
    q = torch.randn(*shape, device=device, dtype=dtype_of(torch))
    k = torch.randn(*shape, device=device, dtype=dtype_of(torch))
    v = torch.randn(*shape, device=device, dtype=dtype_of(torch))
    return q, k, v


def free(*tensors):
    import torch

    for t in tensors:
        del t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def measure_forward(torch, seq: int, kind: str, device: int, quick: bool) -> dict:
    free()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(device)
    try:
        q, k, v = alloc_qkv(torch, seq, device)
        fn = naive_sdpa if kind == "naive" else fused_sdpa
        out = fn(q, k, v)
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated(device))
        allocated = int(torch.cuda.memory_allocated(device))
        # latency with tensors already on device
        stats = cuda_times_ms(
            lambda: fn(q, k, v),
            warmup=2 if quick else 5,
            min_reps=3 if quick else 10,
            min_seconds=0.15 if quick else 0.8,
            max_reps=15 if quick else 80,
        )
        result = {
            "ok": True,
            "kind": kind,
            "seq": seq,
            "peak_memory_bytes": peak,
            "allocated_after_fwd_bytes": allocated,
            "allocated_before_bytes": mem_before,
            "mean_latency_ms": stats["mean_ms"],
            "std_latency_ms": stats["std_ms"],
            "reps": stats["reps"],
            "warmup": stats["warmup"],
            "cv": stats["cv"],
            "out_shape": list(out.shape),
        }
        free(q, k, v, out)
        return result
    except RuntimeError as e:
        err = str(e)
        oom = "out of memory" in err.lower()
        free()
        return {
            "ok": False,
            "kind": kind,
            "seq": seq,
            "oom": oom,
            "error": err.split("\n")[0][:400],
            "traceback": traceback.format_exc()[-1500:],
        }


def coarse_sweep(torch, kind: str, device: int, lengths, quick: bool) -> list:
    rows = []
    for seq in lengths:
        print(f"  {kind} seq={seq} ...", flush=True)
        row = measure_forward(torch, seq, kind, device, quick)
        rows.append(row)
        if row.get("oom") or (not row.get("ok") and row.get("oom") is True):
            # further lengths will also OOM
            print(f"    OOM at seq={seq}, stopping coarse sweep for {kind}", flush=True)
            for later in lengths[lengths.index(seq) + 1 :]:
                rows.append(
                    {
                        "ok": False,
                        "kind": kind,
                        "seq": later,
                        "oom": True,
                        "error": "skipped after earlier OOM",
                    }
                )
            break
        if not row.get("ok"):
            print(f"    FAIL seq={seq}: {row.get('error')}", flush=True)
    return rows


def largest_ok_smallest_fail(rows: list) -> dict:
    oks = [r["seq"] for r in rows if r.get("ok")]
    fails = [r["seq"] for r in rows if r.get("oom")]
    return {
        "largest_success": max(oks) if oks else None,
        "smallest_fail": min(fails) if fails else None,
    }


def refine_oom(torch, kind: str, device: int, lo_ok: int | None, hi_fail: int | None, quick: bool) -> dict:
    """
    Binary search between largest success and smallest fail.
    Assignment: report both bounds; do not claim a single-token boundary
    unless we actually searched at step 1.
    """
    if lo_ok is None or hi_fail is None:
        # maybe never failed — grow until OOM or cap
        if lo_ok is None:
            return {
                "refined": False,
                "reason": "no successful length in the coarse sweep",
                "largest_success": None,
                "smallest_fail": hi_fail,
                "step": None,
            }
        cap = 32768 if quick else 131072
        s = lo_ok
        probed = []
        while s * 2 <= cap:
            s = s * 2
            print(f"  {kind} grow seq={s} ...", flush=True)
            row = measure_forward(torch, s, kind, device, quick=True)
            probed.append(row)
            if row.get("oom"):
                hi_fail = s
                break
            if row.get("ok"):
                lo_ok = s
            else:
                break
        if hi_fail is None:
            return {
                "refined": False,
                "reason": f"no OOM up to seq={lo_ok} (cap {cap})",
                "largest_success": lo_ok,
                "smallest_fail": None,
                "probes": probed,
                "step": None,
            }

    step_floor = 256 if quick else 64
    probes = []
    lo, hi = lo_ok, hi_fail
    while hi - lo > step_floor:
        mid = (lo + hi) // 2
        # keep even lengths
        mid = max(lo + 1, mid)
        print(f"  {kind} refine seq={mid} (lo_ok={lo} hi_fail={hi}) ...", flush=True)
        row = measure_forward(torch, mid, kind, device, quick=True)
        probes.append(row)
        if row.get("ok"):
            lo = mid
        else:
            hi = mid
    return {
        "refined": True,
        "largest_success": lo,
        "smallest_fail": hi,
        "gap_tokens": hi - lo,
        "step_floor": step_floor,
        "single_token_resolution": (hi - lo) == 1,
        "probes": [
            {k: p.get(k) for k in ("seq", "ok", "oom", "peak_memory_bytes", "error")}
            for p in probes
        ],
        "note": (
            "Bounds are the largest tested length that succeeded and the smallest "
            "tested length that OOM'd. Not a single-token boundary unless gap is 1."
        ),
    }


def backend_note() -> dict:
    import torch

    info = {
        "torch": torch.__version__,
        "flash_sdp_enabled": None,
        "mem_efficient_sdp_enabled": None,
        "math_sdp_enabled": None,
        "flash_attn_package": False,
    }
    try:
        info["flash_sdp_enabled"] = torch.backends.cuda.flash_sdp_enabled()
        info["mem_efficient_sdp_enabled"] = torch.backends.cuda.mem_efficient_sdp_enabled()
        info["math_sdp_enabled"] = torch.backends.cuda.math_sdp_enabled()
    except Exception as e:
        info["backend_query_error"] = str(e)
    try:
        import flash_attn  # noqa: F401

        info["flash_attn_package"] = True
    except Exception:
        info["flash_attn_package"] = False
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print(banner())
    seed_everything(SEED)
    torch = __import__("torch")
    info = gpu_info(args.device)
    lengths = [512, 1024, 2048] if args.quick else COARSE_LENGTHS

    print("Naive attention (materializes SxS) ...", flush=True)
    naive_rows = coarse_sweep(torch, "naive", args.device, lengths, args.quick)
    naive_bounds = largest_ok_smallest_fail(naive_rows)
    print("Refine naive OOM ...", flush=True)
    naive_refine = refine_oom(
        torch,
        "naive",
        args.device,
        naive_bounds["largest_success"],
        naive_bounds["smallest_fail"],
        args.quick,
    )

    print("Fused / memory-efficient attention ...", flush=True)
    fused_ok = True
    fused_error = None
    try:
        q, k, v = alloc_qkv(torch, 512, args.device)
        _ = fused_sdpa(q, k, v)
        free(q, k, v)
    except Exception as e:
        fused_ok = False
        fused_error = f"{type(e).__name__}: {e}"

    if fused_ok:
        fused_rows = coarse_sweep(torch, "fused", args.device, lengths, args.quick)
        fused_bounds = largest_ok_smallest_fail(fused_rows)
        print("Refine fused OOM ...", flush=True)
        fused_refine = refine_oom(
            torch,
            "fused",
            args.device,
            fused_bounds["largest_success"],
            fused_bounds["smallest_fail"],
            args.quick,
        )
    else:
        fused_rows = []
        fused_refine = {
            "refined": False,
            "reason": fused_error,
            "largest_success": None,
            "smallest_fail": None,
        }

    speedups = []
    fused_by_seq = {r["seq"]: r for r in fused_rows if r.get("ok")}
    for r in naive_rows:
        if not r.get("ok"):
            continue
        f = fused_by_seq.get(r["seq"])
        if not f:
            continue
        speedups.append(
            {
                "seq": r["seq"],
                "naive_ms": r["mean_latency_ms"],
                "fused_ms": f["mean_latency_ms"],
                "speedup": r["mean_latency_ms"] / f["mean_latency_ms"] if f["mean_latency_ms"] else None,
                "naive_peak_bytes": r["peak_memory_bytes"],
                "fused_peak_bytes": f["peak_memory_bytes"],
            }
        )

    payload = {
        "part": "D",
        "gpu": info,
        "uuid": info["uuid"],
        "config": {
            "batch_size": ATTN_BATCH,
            "n_heads": ATTN_HEADS,
            "head_dim": ATTN_HEAD_DIM,
            "dtype": ATTN_DTYPE_NAME,
            "why": (
                "batch=1 and head_dim=128 as the two chosen knobs (a common LLM head "
                "size, e.g. 16 heads x 128 = 2048 model width). n_heads=16 keeps the "
                "naive S×S tensor large enough to OOM on 24-32 GB inside the assigned "
                "sequence list."
            ),
        },
        "backends": backend_note(),
        "fused_available": fused_ok,
        "fused_error": fused_error,
        "naive": {
            "coarse": naive_rows,
            "oom_bounds": naive_refine,
        },
        "fused": {
            "coarse": fused_rows,
            "oom_bounds": fused_refine,
        },
        "speedups": speedups,
        "fused_kernel_explanation": (
            "The fused / FlashAttention-style kernel never writes the S×S score matrix "
            "to HBM. It loads Q/K/V tiles into SRAM, computes local products, keeps a "
            "running softmax (m, l statistics) in on-chip memory, and writes only the "
            "output tile. That removes the O(S²) allocation that OOMs the naive path "
            "and the O(S²) memory traffic that makes naive attention bandwidth-bound."
        ),
    }
    write_json(results_dir(info) / "part_d_attention.json", payload)

    lines = [
        banner(),
        f"UUID: {info['uuid']}",
        f"config: B={ATTN_BATCH} H={ATTN_HEADS} D={ATTN_HEAD_DIM} dtype={ATTN_DTYPE_NAME}",
        f"naive OOM: largest ok={naive_refine.get('largest_success')}  "
        f"smallest fail={naive_refine.get('smallest_fail')}  "
        f"gap={naive_refine.get('gap_tokens')}",
        f"fused available: {fused_ok}  OOM largest ok={fused_refine.get('largest_success')}  "
        f"smallest fail={fused_refine.get('smallest_fail')}",
    ]
    for s in speedups:
        lines.append(
            f"  seq={s['seq']:5d}  naive={s['naive_ms']:.2f}ms  fused={s['fused_ms']:.2f}ms  "
            f"speedup={s['speedup']:.2f}x  mem naive={s['naive_peak_bytes']/1e9:.2f}GB  "
            f"fused={s['fused_peak_bytes']/1e9:.2f}GB"
        )
    for r in naive_rows:
        if r.get("ok"):
            lines.append(
                f"  naive seq={r['seq']:5d}  {r['mean_latency_ms']:.2f}ms  "
                f"peak={r['peak_memory_bytes']/1e9:.3f} GB  reps={r['reps']}"
            )
        else:
            lines.append(f"  naive seq={r['seq']:5d}  FAIL oom={r.get('oom')} {r.get('error')}")
    text = "\n".join(lines)
    print(text)
    append_run_log("PART D\n" + text, info)


if __name__ == "__main__":
    main()
