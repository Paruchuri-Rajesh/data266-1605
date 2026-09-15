"""CUDA event timing with warmup and enough repetitions for small variance."""

from __future__ import annotations

import statistics
from typing import Callable

from .identity import require_cuda


def cuda_times_ms(
    fn: Callable[[], None],
    warmup: int = 10,
    min_reps: int = 20,
    min_seconds: float = 1.0,
    max_reps: int = 200,
) -> dict:
    torch = require_cuda()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    import time

    wall0 = time.perf_counter()
    while True:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
        elapsed = time.perf_counter() - wall0
        if len(times) >= min_reps and elapsed >= min_seconds:
            break
        if len(times) >= max_reps:
            break

    mean = statistics.fmean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return {
        "warmup": warmup,
        "reps": len(times),
        "times_ms": times,
        "mean_ms": mean,
        "std_ms": std,
        "min_ms": min(times),
        "max_ms": max(times),
        "cv": (std / mean) if mean else 0.0,
        "min_seconds_target": min_seconds,
    }


def gemm_tflops(n: int, mean_ms: float) -> float:
    """Square GEMM FLOPs = 2 N^3. mean_ms is milliseconds."""
    if mean_ms <= 0:
        return 0.0
    return (2.0 * (n ** 3)) / (mean_ms * 1e-3) / 1e12
