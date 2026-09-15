#!/usr/bin/env python3
"""Build figures + METRICS.md from results/<gpu_slug>/ JSON after the GPU run(s)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.identity import ROOT as HW_ROOT  # noqa: E402
from src.params import CLS_A, CLS_B, HP_ID, SEED, SID4, SLICE, STUDENT  # noqa: E402


def load_runs(results_root: Path) -> list[dict]:
    runs = []
    if not results_root.exists():
        return runs
    for d in sorted(p for p in results_root.iterdir() if p.is_dir()):
        bundle = {"dir": d, "slug": d.name}
        for name in [
            "part_a_provenance.json",
            "part_b_matmul.json",
            "part_c_roofline.json",
            "part_d_attention.json",
            "part_e_thermal.json",
        ]:
            p = d / name
            if p.exists():
                bundle[name] = json.loads(p.read_text())
        if any(k.startswith("part_") for k in bundle):
            runs.append(bundle)
    return runs


def uuid_of(bundle: dict) -> str:
    for key in (
        "part_a_provenance.json",
        "part_b_matmul.json",
        "part_c_roofline.json",
        "part_d_attention.json",
        "part_e_thermal.json",
    ):
        blob = bundle.get(key) or {}
        gpu = blob.get("gpu") or {}
        if gpu.get("uuid"):
            return gpu["uuid"]
        if blob.get("uuid"):
            return blob["uuid"]
    return "unknown"


def gpu_name(bundle: dict) -> str:
    for key in bundle:
        if isinstance(bundle[key], dict):
            gpu = bundle[key].get("gpu") or {}
            if gpu.get("name"):
                return gpu["name"]
    return bundle.get("slug", "GPU")


def plot_matmul(bundle: dict, dest: Path) -> None:
    data = bundle.get("part_b_matmul.json")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    by_p: dict[str, list] = {}
    for row in data["rows"]:
        by_p.setdefault(row["precision"], []).append(row)
    for prec, rows in by_p.items():
        rows = sorted(rows, key=lambda r: r["n"])
        ax.plot([r["n"] for r in rows], [r["tflops"] for r in rows], marker="o", label=prec.upper())
    ax.set_xlabel("Matrix size N")
    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title(f"Part B — GEMM throughput\n{gpu_name(bundle)}\nUUID {uuid_of(bundle)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def plateau_note(data: dict) -> str:
    if not data:
        return "no Part B data"
    lines = []
    by_p: dict[str, list] = {}
    for row in data["rows"]:
        by_p.setdefault(row["precision"], []).append(row)
    for prec, rows in sorted(by_p.items()):
        rows = sorted(rows, key=lambda r: r["n"])
        t = [r["tflops"] for r in rows]
        if not t:
            continue
        peak = max(t)
        plateau_n = None
        for r in rows:
            if r["tflops"] >= 0.95 * peak:
                plateau_n = r["n"]
                break
        lines.append(
            f"{prec.upper()} reaches 95% of its own max ({peak:.1f} TFLOPS) at N={plateau_n}."
        )
    lines.append(
        "Small N never plateaus: kernel launch and cuBLAS setup dominate, the tile "
        "count is too small to fill the SMs, and the working set sits in cache so "
        "neither DRAM bandwidth nor peak tensor-core throughput is visible."
    )
    return " ".join(lines)


def plot_attention_memory(bundle: dict, dest: Path) -> dict | None:
    data = bundle.get("part_d_attention.json")
    if not data:
        return None
    rows = [r for r in data["naive"]["coarse"] if r.get("ok")]
    if len(rows) < 3:
        return None
    s = np.array([r["seq"] for r in rows], dtype=float)
    m = np.array([r["peak_memory_bytes"] for r in rows], dtype=float)
    coeff2 = np.polyfit(s, m, 2)
    coeff1 = np.polyfit(s, m, 1)
    pred2 = np.polyval(coeff2, s)
    pred1 = np.polyval(coeff1, s)
    rmse2 = float(np.sqrt(np.mean((pred2 - m) ** 2)))
    rmse1 = float(np.sqrt(np.mean((pred1 - m) ** 2)))
    a, b, c = [float(x) for x in coeff2]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s, m / 1e9, "o", label="measured peak memory")
    s_line = np.linspace(s.min(), s.max(), 200)
    ax.plot(s_line, np.polyval(coeff2, s_line) / 1e9, "-", label="quadratic fit")
    ax.plot(s_line, np.polyval(coeff1, s_line) / 1e9, "--", label="linear fit")
    fused_rows = [r for r in data.get("fused", {}).get("coarse", []) if r.get("ok")]
    if fused_rows:
        ax.plot(
            [r["seq"] for r in fused_rows],
            [r["peak_memory_bytes"] / 1e9 for r in fused_rows],
            "s",
            label="fused peak memory",
        )
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Peak allocated memory (GB)")
    ax.set_title(f"Part D — attention memory\n{gpu_name(bundle)}\nUUID {uuid_of(bundle)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)

    # Confirm quadratic: degree-2 RMSE must beat degree-1, and a > 0.
    return {
        "a_bytes_per_s2": a,
        "b_bytes_per_s": b,
        "c_bytes": c,
        "rmse_quadratic": rmse2,
        "rmse_linear": rmse1,
        "quadratic_better": rmse2 < rmse1,
        "a_positive": a > 0,
        "n_points": len(rows),
        "formula": "peak_bytes ≈ a S^2 + b S + c",
    }


def plot_thermal(bundle: dict, dest: Path) -> None:
    data = bundle.get("part_e_thermal.json")
    if not data:
        return
    csv_rel = data.get("thermal_csv")
    csv_path = HW_ROOT / csv_rel if csv_rel else bundle["dir"] / "thermal_smi.csv"
    if not csv_path.exists():
        csv_path = bundle["dir"] / "thermal_smi.csv"
    if not csv_path.exists():
        return
    lines = csv_path.read_text().strip().splitlines()
    if len(lines) < 2:
        return
    reader = csv.DictReader(lines)
    rows = list(reader)
    if not rows:
        return

    def pick(row, *needles):
        for k, v in row.items():
            kl = k.strip().lower()
            if any(n in kl for n in needles):
                return v.strip().replace(" MHz", "").replace(" W", "").replace(" %", "")
        return ""

    gfx, temp, t = [], [], []
    for i, row in enumerate(rows):
        t.append(i * 5.0)
        try:
            gfx.append(float(pick(row, "clocks.current.graphics", "graphics")))
        except ValueError:
            gfx.append(np.nan)
        try:
            temp.append(float(pick(row, "temperature.gpu")))
        except ValueError:
            temp.append(np.nan)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.plot(t, gfx, color="C0", label="graphics clock (MHz)")
    ax2.plot(t, temp, color="C3", label="temperature (C)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Graphics clock (MHz)", color="C0")
    ax2.set_ylabel("Temperature (C)", color="C3")
    ax1.set_title(f"Part E — clock and temperature\n{gpu_name(bundle)}\nUUID {uuid_of(bundle)}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def cell(bundle: dict, kind: str) -> str:
    uid = uuid_of(bundle)
    if kind == "peak_bf16":
        data = bundle.get("part_b_matmul.json")
        if not data:
            return "—"
        rows = [r for r in data["rows"] if r["precision"] == "bf16"]
        if not rows:
            return "—"
        best = max(rows, key=lambda r: r["tflops"])
        return f"{best['tflops']:.2f} (N={best['n']}, UUID {uid})"
    if kind == "pct_bf16":
        data = bundle.get("part_b_matmul.json")
        if not data:
            return "—"
        rows = [r for r in data["rows"] if r["precision"] == "bf16" and r.get("pct_of_peak")]
        if not rows:
            return "—"
        best = max(rows, key=lambda r: r["tflops"])
        return f"{best['pct_of_peak']:.1f}% (peak {best['peak_tflops']} TFLOPS, UUID {uid})"
    if kind == "bw":
        data = bundle.get("part_c_roofline.json")
        if not data:
            return "—"
        mb = data["memory_bound"]
        return (
            f"{mb['effective_bandwidth_gbs']:.1f} GB/s "
            f"({mb['pct_of_spec_bandwidth']:.1f}% of {mb['spec_bandwidth_gbs']}, UUID {uid})"
        )
    if kind == "naive_oom":
        data = bundle.get("part_d_attention.json")
        if not data:
            return "—"
        b = data["naive"]["oom_bounds"]
        return (
            f"ok≤{b.get('largest_success')} fail≥{b.get('smallest_fail')} "
            f"(gap {b.get('gap_tokens')}, UUID {uid})"
        )
    if kind == "fused_oom":
        data = bundle.get("part_d_attention.json")
        if not data:
            return "—"
        b = data["fused"]["oom_bounds"]
        return (
            f"ok≤{b.get('largest_success')} fail≥{b.get('smallest_fail')} "
            f"(gap {b.get('gap_tokens')}, UUID {uid})"
        )
    if kind == "steady":
        data = bundle.get("part_e_thermal.json")
        if not data:
            return "—"
        s = data["throughput_summary"]
        pct = s.get("steady_over_peak_pct")
        return f"{pct:.1f}% (UUID {uid})" if pct is not None else f"— (UUID {uid})"
    if kind == "throttle":
        data = bundle.get("part_e_thermal.json")
        if not data:
            return "—"
        th = data["throttle"]
        onset = th.get("throttle_onset_s")
        if onset is None:
            return f"none (max {th.get('max_temp_c')} C, UUID {uid})"
        return (
            f"{onset:.0f}s  temp={th.get('temp_c_at_onset')} C  "
            f"power={th.get('power_at_onset')}  UUID {uid}"
        )
    return "—"


def write_metrics(runs: list[dict], fits: dict, dest: Path) -> None:
    header = (
        f"# METRICS — DATA 266 HW2.5\n\n"
        f"{STUDENT}  \n"
        f"SID4 = {SID4:04d} | SEED = {SEED} | SLICE = {SLICE} | "
        f"HP_ID = {HP_ID} | CLS_A = {CLS_A} | CLS_B = {CLS_B}  \n"
        f"HP_ID is reported only. No second hyperparameter model.\n\n"
        f"Every cell is traceable to a UUID-labelled run in `RUN_LOG.txt` and "
        f"`results/<gpu>/` JSON.\n\n"
    )
    if not runs:
        dest.write_text(
            header
            + "No `results/` yet. Run `run_full.bat` (or `run_full.sh`) on the lab GPU, "
            "then rerun `scripts/06_compile_report.py`.\n"
        )
        return

    cols = [gpu_name(b) for b in runs]
    keys = [
        ("Peak achieved TFLOPS (BF16)", "peak_bf16"),
        ("% of theoretical peak (BF16)", "pct_bf16"),
        ("Effective bandwidth (GB/s)", "bw"),
        ("Naive attention OOM length", "naive_oom"),
        ("Fused attention OOM length", "fused_oom"),
        ("Steady-state / peak throughput", "steady"),
        ("Throttle onset (s, or none)", "throttle"),
    ]

    lines = [header, "## Table HW2.5.1 — summary\n"]
    sep_header = "| Measurement | " + " | ".join(cols) + " | Notes |"
    sep = "|-------------|" + "|".join(["------"] * len(cols)) + "|-------|"
    lines.append(sep_header)
    lines.append(sep)
    notes = {
        "peak_bf16": "max BF16 GEMM over N=1024..16384; dense peak from vendor whitepaper",
        "pct_bf16": "BF16 tensor, FP32 accumulate, dense (not sparse)",
        "bw": "elementwise add, 3-operand traffic",
        "naive_oom": "largest tested success and smallest tested fail; not 1-token unless gap=1",
        "fused_oom": "same search on F.scaled_dot_product_attention",
        "steady": "last 5 min mean TFLOPS / first 30 s mean TFLOPS",
        "throttle": "first non-idle throttle reason after 10 s",
    }
    for label, kind in keys:
        vals = " | ".join(cell(b, kind) for b in runs)
        lines.append(f"| {label} | {vals} | {notes[kind]} |")
    lines.append("")
    lines.append("Table HW2.5.1 — The summary table. Every cell is UUID-labelled.\n")

    lines.append("## Part A — provenance\n")
    for b in runs:
        a = b.get("part_a_provenance.json") or {}
        req = a.get("required_fields") or {}
        gpu = a.get("gpu") or {}
        spec = a.get("vendor_spec") or {}
        lines.append(f"### {gpu_name(b)}\n")
        lines.append(f"- UUID: `{req.get('uuid') or uuid_of(b)}`")
        lines.append(f"- driver: {req.get('driver_version')}")
        lines.append(f"- CUDA (nvidia-smi): {req.get('cuda_version')}")
        lines.append(f"- VRAM: {req.get('vram_capacity')}")
        lines.append(f"- power limit: {req.get('power_limit')}")
        if spec:
            lines.append(f"- architecture: {spec.get('architecture')}")
            lines.append(f"- memory: {spec.get('memory_type')}")
            lines.append(f"- bandwidth: {spec.get('bandwidth_gbs')} GB/s")
            lines.append(f"- tensor cores: {spec.get('tensor_core_gen')}")
            lines.append(f"- reduced precisions: {', '.join(spec.get('tensor_precisions') or [])}")
            lines.append(f"- sources: {spec.get('sources')}")
        lines.append(f"- nvidia-smi -q file: `{a.get('nvidia_smi_q_file')}`")
        lines.append("")

    lines.append("## Part B — GEMM\n")
    for b in runs:
        data = b.get("part_b_matmul.json")
        lines.append(f"### {gpu_name(b)}  UUID `{uuid_of(b)}`\n")
        if not data:
            lines.append("missing\n")
            continue
        lines.append("| N | precision | TFLOPS | % of dense peak | reps | mean ms | std ms |")
        lines.append("|---|-----------|--------|-----------------|------|---------|--------|")
        for row in data["rows"]:
            pct = f"{row['pct_of_peak']:.1f}" if row.get("pct_of_peak") is not None else "—"
            lines.append(
                f"| {row['n']} | {row['precision']} | {row['tflops']:.2f} | {pct} | "
                f"{row['reps']} | {row['mean_ms']:.3f} | {row['std_ms']:.3f} |"
            )
        lines.append("")
        lines.append(plateau_note(data))
        lines.append("")
        lines.append("Lower precision attempts:")
        for lp in data.get("lower_precision") or []:
            if lp.get("ok"):
                lines.append(
                    f"- {lp.get('precision')} OK via `{lp.get('call_that_worked')}` "
                    f"→ {lp.get('tflops'):.2f} TFLOPS ({lp.get('pct_of_peak')}% of peak)"
                )
            else:
                lines.append(
                    f"- {lp.get('precision')} FAILED. tried: {lp.get('tried')}. "
                    f"failure: {lp.get('failure')}"
                )
        lines.append("")

    lines.append("## Part C — roofline\n")
    for b in runs:
        data = b.get("part_c_roofline.json")
        lines.append(f"### {gpu_name(b)}  UUID `{uuid_of(b)}`\n")
        if not data:
            lines.append("missing\n")
            continue
        mb = data["memory_bound"]
        cb = data["compute_bound"]
        ridge = data["ridge_point"]
        lines.append(
            f"- add: {mb['effective_bandwidth_gbs']:.1f} GB/s "
            f"({mb['pct_of_spec_bandwidth']:.1f}% of {mb['spec_bandwidth_gbs']} GB/s), "
            f"AI={mb['arithmetic_intensity_flops_per_byte']:.4f} FLOP/B, "
            f"{mb['roofline_side']}"
        )
        lines.append(
            f"- copy check: {mb['copy_check']['effective_bandwidth_gbs']:.1f} GB/s "
            f"({mb['copy_check']['pct_of_spec_bandwidth']:.1f}%)"
        )
        lines.append(
            f"- BF16 GEMM N={cb['n']}: {cb['tflops']:.2f} TFLOPS "
            f"({cb['pct_of_peak']:.1f}% of peak), "
            f"AI={cb['arithmetic_intensity_flops_per_byte']:.1f} FLOP/B, "
            f"{cb['roofline_side']}"
        )
        lines.append(
            f"- ridge: FP32 {ridge['fp32_flops_per_byte']:.1f} FLOP/B, "
            f"BF16 {ridge['bf16_flops_per_byte']:.1f} FLOP/B"
        )
        lines.append("")

    lines.append("## Part D — attention\n")
    for b in runs:
        data = b.get("part_d_attention.json")
        lines.append(f"### {gpu_name(b)}  UUID `{uuid_of(b)}`\n")
        if not data:
            lines.append("missing\n")
            continue
        cfg = data["config"]
        lines.append(
            f"batch={cfg['batch_size']}, n_heads={cfg['n_heads']}, "
            f"head_dim={cfg['head_dim']}, dtype={cfg['dtype']}"
        )
        lines.append("")
        lines.append("| seq | naive ms | naive peak GB | fused ms | fused peak GB | speedup |")
        lines.append("|-----|----------|---------------|----------|---------------|---------|")
        fused = {r["seq"]: r for r in data.get("fused", {}).get("coarse", [])}
        for r in data["naive"]["coarse"]:
            seq = r["seq"]
            if r.get("ok"):
                n_ms = f"{r['mean_latency_ms']:.2f}"
                n_gb = f"{r['peak_memory_bytes']/1e9:.3f}"
            else:
                n_ms, n_gb = "OOM" if r.get("oom") else "FAIL", "—"
            f = fused.get(seq)
            if f and f.get("ok"):
                f_ms = f"{f['mean_latency_ms']:.2f}"
                f_gb = f"{f['peak_memory_bytes']/1e9:.3f}"
                sp = (
                    f"{r['mean_latency_ms']/f['mean_latency_ms']:.2f}x"
                    if r.get("ok") and f["mean_latency_ms"]
                    else "—"
                )
            else:
                f_ms = "OOM" if (f or {}).get("oom") else ("—" if not f else "FAIL")
                f_gb, sp = "—", "—"
            lines.append(f"| {seq} | {n_ms} | {n_gb} | {f_ms} | {f_gb} | {sp} |")
        nb = data["naive"]["oom_bounds"]
        fb = data["fused"]["oom_bounds"]
        lines.append("")
        lines.append(
            f"Naive OOM bounds: largest success {nb.get('largest_success')}, "
            f"smallest fail {nb.get('smallest_fail')}, gap {nb.get('gap_tokens')} "
            f"(single-token resolution: {nb.get('single_token_resolution')})."
        )
        lines.append(
            f"Fused OOM bounds: largest success {fb.get('largest_success')}, "
            f"smallest fail {fb.get('smallest_fail')}, gap {fb.get('gap_tokens')}."
        )
        fit = fits.get(uuid_of(b))
        if fit:
            lines.append(
                f"Quadratic fit on naive peak memory: a={fit['a_bytes_per_s2']:.6e} bytes/token², "
                f"b={fit['b_bytes_per_s']:.6e}, c={fit['c_bytes']:.6e}. "
                f"RMSE quadratic {fit['rmse_quadratic']:.3e} vs linear {fit['rmse_linear']:.3e}. "
                f"Quadratic better: {fit['quadratic_better']}. a>0: {fit['a_positive']}."
            )
        lines.append("")
        lines.append(data.get("fused_kernel_explanation", ""))
        lines.append("")

    lines.append("## Part E — thermal\n")
    for b in runs:
        data = b.get("part_e_thermal.json")
        lines.append(f"### {gpu_name(b)}  UUID `{uuid_of(b)}`\n")
        if not data:
            lines.append("missing\n")
            continue
        s = data["throughput_summary"]
        th = data["throttle"]
        lines.append(f"- duration: {data.get('duration_s_actual'):.1f}s  N={data.get('gemm_n')} BF16")
        lines.append(f"- first 30s TFLOPS: {s.get('first_30s_mean_tflops')}")
        lines.append(f"- last 5 min TFLOPS: {s.get('last_5min_mean_tflops')}")
        lines.append(f"- steady/peak: {s.get('steady_over_peak_pct')}")
        lines.append(f"- throttle onset (s): {th.get('throttle_onset_s')}")
        lines.append(f"- max temp C: {th.get('max_temp_c')}")
        lines.append(f"- log: `{data.get('thermal_csv')}`")
        lines.append("")

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=HW_ROOT / "results")
    args = parser.parse_args()
    runs = load_runs(args.results_dir)
    fig_root = HW_ROOT / "figures"
    fig_root.mkdir(exist_ok=True)
    fits = {}
    for b in runs:
        slug = b["slug"]
        local = b["dir"] / "figures"
        local.mkdir(exist_ok=True)
        p1 = fig_root / f"part_b_tflops_{slug}.png"
        p2 = fig_root / f"part_d_attn_mem_{slug}.png"
        p3 = fig_root / f"part_e_thermal_{slug}.png"
        plot_matmul(b, p1)
        plot_matmul(b, local / "part_b_tflops.png")
        fit = plot_attention_memory(b, p2)
        plot_attention_memory(b, local / "part_d_attn_mem.png")
        if fit:
            fits[uuid_of(b)] = fit
            (b["dir"] / "part_d_quadratic_fit.json").write_text(json.dumps(fit, indent=2) + "\n")
        plot_thermal(b, p3)
        plot_thermal(b, local / "part_e_thermal.png")
        print(f"figures for {slug}: {p1.name}, {p2.name}, {p3.name}")
    if runs:
        write_metrics(runs, fits, HW_ROOT / "METRICS.md")
        print(f"wrote {HW_ROOT / 'METRICS.md'} from {len(runs)} GPU run(s)")
    else:
        print("No results yet — that is expected before the lab machine. METRICS.md left as placeholder.")


if __name__ == "__main__":
    main()
