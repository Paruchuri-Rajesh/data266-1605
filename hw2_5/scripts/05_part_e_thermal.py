#!/usr/bin/env python3
"""Part E — 20-minute sustained GEMM with nvidia-smi sampling every 5 seconds."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.identity import (  # noqa: E402
    append_run_log,
    find_nvidia_smi,
    gpu_info,
    results_dir,
    write_json,
)
from src.params import SEED, banner, seed_everything  # noqa: E402

QUERY = (
    "timestamp,uuid,name,clocks.current.graphics,clocks.current.memory,"
    "clocks.max.graphics,temperature.gpu,power.draw,power.limit,"
    "utilization.gpu,utilization.memory,"
    "clocks_throttle_reasons.gpu_idle,"
    "clocks_throttle_reasons.sw_power_cap,"
    "clocks_throttle_reasons.hw_slowdown,"
    "clocks_throttle_reasons.hw_thermal_slowdown,"
    "clocks_throttle_reasons.sw_thermal_slowdown,"
    "clocks_throttle_reasons.active"
)


def start_sampler(csv_path: Path, device: int) -> subprocess.Popen:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    logf = csv_path.open("w")
    proc = subprocess.Popen(
        [
            find_nvidia_smi(),
            f"--id={device}",
            f"--query-gpu={QUERY}",
            "--format=csv",
            "-l",
            "5",
        ],
        stdout=logf,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._logf = logf  # type: ignore[attr-defined]
    return proc


def stop_sampler(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        # SIGINT is unreliable on Windows lab PCs; terminate the nvidia-smi loop.
        try:
            proc.terminate()
            proc.wait(timeout=8)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    logf = getattr(proc, "_logf", None)
    if logf:
        try:
            logf.close()
        except Exception:
            pass


def run_load(torch, device: int, seconds: int, n: int) -> list:
    """Sustained BF16 GEMM. Record per-window TFLOPS for first 30s vs last 5 min."""
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    # warmup so the first 30s is not still compiling / paging
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()

    samples = []
    t_end = time.time() + seconds
    t0 = time.time()
    iters = 0
    window_t0 = t0
    window_iters = 0
    while time.time() < t_end:
        torch.mm(a, b)
        iters += 1
        window_iters += 1
        now = time.time()
        if now - window_t0 >= 1.0:
            torch.cuda.synchronize()
            dt = time.time() - window_t0
            tflops = (window_iters * 2.0 * (n ** 3)) / dt / 1e12
            elapsed = time.time() - t0
            samples.append(
                {
                    "elapsed_s": elapsed,
                    "window_s": dt,
                    "iters": window_iters,
                    "tflops": tflops,
                }
            )
            window_t0 = time.time()
            window_iters = 0
    torch.cuda.synchronize()
    total_s = time.time() - t0
    return samples, iters, total_s, n


def summarize_throughput(samples: list, total_s: float) -> dict:
    first = [s["tflops"] for s in samples if s["elapsed_s"] <= 30.0]
    last = [s["tflops"] for s in samples if s["elapsed_s"] >= max(0.0, total_s - 300.0)]
    if not last:
        last = [s["tflops"] for s in samples[-5:]] if samples else []
    peak_first = sum(first) / len(first) if first else None
    steady = sum(last) / len(last) if last else None
    ratio = None
    if peak_first and steady is not None and peak_first > 0:
        ratio = 100.0 * steady / peak_first
    return {
        "first_30s_mean_tflops": peak_first,
        "last_5min_mean_tflops": steady,
        "steady_over_peak_pct": ratio,
        "n_windows_first_30s": len(first),
        "n_windows_last_5min": len(last),
    }


def parse_throttle(csv_path: Path) -> dict:
    if not csv_path.exists() or csv_path.stat().st_size < 10:
        return {"throttle_onset_s": None, "reason": "no sampler log"}
    text = csv_path.read_text().strip().splitlines()
    if len(text) < 2:
        return {"throttle_onset_s": None, "reason": "sampler log empty"}
    reader = csv.DictReader(text)
    rows = list(reader)
    if not rows:
        return {"throttle_onset_s": None, "reason": "no rows"}

    def col(row, *keys):
        for k in row:
            kl = k.strip().lower()
            for want in keys:
                if want in kl:
                    return row[k].strip()
        return ""

    t0 = None
    onset = None
    onset_reason = None
    onset_temp = None
    onset_power = None
    max_temp = None
    clocks = []
    temps = []
    for i, row in enumerate(rows):
        ts = col(row, "timestamp")
        # nvidia-smi timestamp like 2026/09/14 15:04:01.123
        elapsed = i * 5.0
        gfx = col(row, "clocks.current.graphics", "graphics")
        temp = col(row, "temperature.gpu")
        power = col(row, "power.draw")
        sw_power = col(row, "sw_power_cap")
        hw_slow = col(row, "hw_slowdown")
        hw_therm = col(row, "hw_thermal_slowdown")
        sw_therm = col(row, "sw_thermal_slowdown")
        active = col(row, "throttle_reasons.active", "active")
        clocks.append(gfx)
        try:
            tval = float(temp)
            temps.append(tval)
            max_temp = tval if max_temp is None else max(max_temp, tval)
        except ValueError:
            tval = None
        flags = {
            "sw_power_cap": sw_power,
            "hw_slowdown": hw_slow,
            "hw_thermal_slowdown": hw_therm,
            "sw_thermal_slowdown": sw_therm,
            "active": active,
        }
        throttling = any(
            str(v).lower() in {"active", "yes", "true", "1"}
            for k, v in flags.items()
            if k != "active"
        )
        # "Active" field can be a bitmask string; treat Not Active / 0x0000000000000000 as none
        active_l = active.lower()
        if active and active_l not in {"not active", "none", "0x0000000000000000", "0", ""}:
            if "idle" not in active_l:
                throttling = True
        if throttling and onset is None and elapsed >= 10:
            onset = elapsed
            onset_reason = flags
            onset_temp = tval
            onset_power = power
        _ = ts, t0
    return {
        "n_samples": len(rows),
        "throttle_onset_s": onset,
        "throttle_flags_at_onset": onset_reason,
        "temp_c_at_onset": onset_temp,
        "power_at_onset": onset_power,
        "max_temp_c": max_temp,
        "note": (
            "Onset is the first sample after 10 s where a non-idle throttle reason "
            "is active. GPU-idle at the start of the run is ignored."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seconds", type=int, default=None)
    args = parser.parse_args()

    seconds = args.seconds
    if seconds is None:
        seconds = 60 if args.quick else 20 * 60
    n = 4096 if args.quick else 8192

    print(banner())
    print(f"Sustained load: {seconds}s  GEMM N={n} BF16  sample every 5s", flush=True)
    seed_everything(SEED)
    torch = __import__("torch")
    info = gpu_info(args.device)
    out = results_dir(info)
    csv_path = out / "thermal_smi.csv"

    proc = start_sampler(csv_path, args.device)
    time.sleep(2.0)  # let nvidia-smi write the header + first row
    try:
        samples, iters, total_s, n = run_load(torch, args.device, seconds, n)
    finally:
        stop_sampler(proc)

    thru = summarize_throughput(samples, total_s)
    throttle = parse_throttle(csv_path)
    payload = {
        "part": "E",
        "gpu": info,
        "uuid": info["uuid"],
        "duration_s_requested": seconds,
        "duration_s_actual": total_s,
        "gemm_n": n,
        "dtype": "bfloat16",
        "sample_period_s": 5,
        "thermal_csv": str(csv_path.relative_to(ROOT)),
        "throughput_windows": samples,
        "throughput_summary": thru,
        "throttle": throttle,
        "total_gemm_iters": iters,
    }
    write_json(out / "part_e_thermal.json", payload)
    write_json(out / "thermal_throughput.json", {"windows": samples, **thru})

    lines = [
        banner(),
        f"UUID: {info['uuid']}",
        f"ran {total_s:.1f}s  N={n} BF16  iters={iters}",
        f"first 30s mean TFLOPS: {thru['first_30s_mean_tflops']}",
        f"last 5min mean TFLOPS: {thru['last_5min_mean_tflops']}",
        f"steady/peak %: {thru['steady_over_peak_pct']}",
        f"throttle onset s: {throttle.get('throttle_onset_s')}",
        f"max temp C: {throttle.get('max_temp_c')}",
        f"smi log: {csv_path}",
    ]
    text = "\n".join(lines)
    print(text)
    append_run_log("PART E\n" + text, info)


if __name__ == "__main__":
    main()
