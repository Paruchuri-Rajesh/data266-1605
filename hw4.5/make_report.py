"""Build figures/ and METRICS.md from everything under results/<card>/.

    python make_report.py

Nothing in METRICS.md is typed by hand except the written answers, which come from ANSWERS.md (one `## key` section
each; a missing key stays TODO): every number comes from the JSON written by train_bert.py, sweep.py, ddp_bert.py and
serve_bench.py.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots in fixed order (one per card / series), recessive ink for text and grid
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2, "lines.markersize": 7, "figure.dpi": 130, "savefig.bbox": "tight"})
TECH_NAMES = {"baseline": "Baseline (HW4 settings)", "amp": "Mixed precision", "grad_ckpt": "Gradient checkpointing",
              "grad_accum": "Gradient accumulation", "adamw8bit": "8-bit optimiser", "best": "Best combination",
              "best_ckpt": "Best combination + grad ckpt"}


def load_answers(path="ANSWERS.md"):
    """`## key` headings -> answer text, so written answers survive regenerating METRICS.md."""
    out, key = {}, None
    for line in (open(path).read().splitlines() if os.path.exists(path) else []):
        if line.startswith("## "):
            key = line[3:].strip()
            out[key] = []
        elif key and not line.startswith("<!--"):
            out[key].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items() if "\n".join(v).strip()}


ANSWERS = load_answers()


def answer(key, todo):
    return ANSWERS.get(key, todo)


def load(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None


def f(x, nd=1):
    return "—" if x is None else (f"{x:,.{nd}f}" if isinstance(x, float) else str(x))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def cards():
    return sorted(d for d in os.listdir("results") if os.path.isdir(f"results/{d}")) if os.path.isdir("results") else []


# ----------------------------------------------------------------------------- data + hardware (HW4.5 clarifications)

def data_section(md, dataset_name):
    """Dataset name, exact row range, and proof that every HW4.5 training run used the HW4.a rows and split."""
    lock = load("data/hw4a_data_lock.json")
    md += ["## Data (same as HW4.a) and GPU used", ""]
    if not lock:
        md += ["No HW4.a data lock found (`data/hw4a_data_lock.json`); run `./run_card.sh <card> hw4a` first.", ""]
        return
    sizes = lock["split_sizes"]
    hw4a = [load(p) for p in sorted(glob.glob("results/*/hw4a.json"))]
    fname = os.path.basename(hw4a[0]["data"]["file"]) if hw4a and hw4a[0] else "?"
    lines = hw4a[0]["data"].get("csv_line_range", "") if hw4a and hw4a[0] else ""
    md += [f"- Dataset: **{dataset_name}**, file `{fname}` "
           f"(sha256 `{lock['file_sha256'][:16]}…`), text column `{lock['text_col']}`, label column `{lock['label_col']}`.",
           f"- Exact row range: **{lock['row_range']}**" + (f" = CSV lines {lines}" if lines else "")
           + ", i.e. up to 5,000 consecutive rows starting at SLICE; "
           f"{lock['rows_used']} rows used, labels {lock['labels']}.",
           f"- Split: train/val/test = {sizes[0]}/{sizes[1]}/{sizes[2]} (70/15/15) with **SEED = {lock['seed']}**; "
           f"split fingerprint `{lock['split_sha256'][:16]}…`.", ""]
    runs, bad = 0, []
    for p in sorted(glob.glob("results/**/*.json", recursive=True)):
        r = load(p)
        fp = (r or {}).get("data", {}).get("fingerprint") if isinstance(r, dict) else None
        if fp is None:
            continue
        runs += 1
        if any(fp.get(k) != v for k, v in lock.items()):
            bad.append(p)
    md += [f"Same-data check: **{runs - len(bad)} of {runs}** training result files (HW4.a, Parts A, B, C) used exactly these "
           f"rows and this split." + (f" Mismatched: {', '.join(bad)}." if bad else " No rows were re-picked and nothing was reshuffled."), ""]
    for p in sorted(glob.glob("env/uuid_*.txt")):
        lines = [l for l in open(p).read().splitlines() if l.startswith("GPU ")]
        md += [f"GPU (`nvidia-smi -L`, {os.path.basename(p)}):", "", "```", *(lines or ["(no nvidia-smi -L output captured)"]), "```", ""]
    if not glob.glob("env/uuid_*.txt"):
        md += ["TODO: no `env/uuid_*.txt` found; run `./run_card.sh <card> env` to record `nvidia-smi -L`.", ""]


# ----------------------------------------------------------------------------- HW4.a

def hw4a_section(card, md, dataset_name):
    r = load(f"results/{card}/hw4a.json")
    if not r or r.get("status") != "ok":
        return None
    m, d, t, es = r["metrics"], r["data"], r["test"], r.get("early_stopping", {})
    md += [f"### HW4.a fine-tune on `{card}` ({r['gpu']['name']})", "",
           f"- Dataset: **{dataset_name}** (`{d['file']}`), rows **{d['row_range']}** (SLICE={r['config']['slice_start']}, SEED={r['config']['seed']}), "
           f"{d['rows_used']} rows used ({d['rows_dropped_na']} dropped for missing values); "
           f"split train/val/test = {d['split_sizes']['train']}/{d['split_sizes']['val']}/{d['split_sizes']['test']} (70/15/15, SEED).",
           f"- Settings: {r['config']['model_name']}, batch {r['config']['batch_size']}, lr {r['config']['lr']}, "
           f"epochs {r['config']['epochs']}, max_len {r['config']['max_len']}, fp32, AdamW.", ""]
    md.append(table(["Epoch", "Train loss", "Val loss", "Val acc", "Epoch time (s)"],
                    [[h["epoch"], f(h["train_loss"], 4), f(h["val_loss"], 4), f(h["val_acc"], 4), f(h["epoch_time_s"])]
                     for h in m["history"]]))
    md += ["", f"Validation loss stopped improving after **epoch {es.get('best_epoch_by_val_loss')}** "
               f"(best val accuracy at epoch {es.get('best_epoch_by_val_acc')}). "
               + (f"Training {es['epochs_past_best']} more epoch(s) cost {es['extra_time_s']:.1f} s of GPU time, "
                  f"raised val loss by {es['val_loss_increase']:+.4f} and changed val accuracy by {es['val_acc_change']:+.4f}."
                  if es.get("epochs_past_best") else "No epochs were trained past that point."), ""]
    names = t["labels"]
    md += [f"**Test accuracy {t['test_acc']:.4f}.** Confusion matrix (rows = true, columns = predicted):", ""]
    md.append(table(["true \\ pred", *names], [[n, *row] for n, row in zip(names, t["confusion_matrix"])]))
    md += [""]
    md.append(table(["Class", "Precision", "Recall", "F1", "Support"],
                    [[n, f(v["precision"], 3), f(v["recall"], 3), f(v["f1"], 3), v["support"]] for n, v in t["per_class"].items()]))
    md += ["", f"Worst class (lowest F1): **{t['worst_class']}**. Five misclassified test examples (most confident mistakes first):", ""]
    if t["worst_class_misclassified"]:
        md.append(table(["#", "Tweet", "Predicted (conf)", "Diagnosis"],
                        [[i + 1, e["text"].replace("|", "\\|").replace("\n", " "), f"{e['pred']} ({e['pred_conf']})", answer(f"diagnosis_{i + 1}", e["diagnosis"])]
                         for i, e in enumerate(t["worst_class_misclassified"])]))
    else:
        md.append("(no misclassified test examples in this class)")
    md += ["", f"**HW4 baseline for HW4.5:** peak GPU memory **{m['peak_alloc_gb']:.2f} GB** allocated "
               f"({m['peak_reserved_gb']:.2f} GB reserved), wall-clock training time **{m['train_time_s']:.1f} s** "
               f"({m['wall_s_incl_eval']:.1f} s including per-epoch validation).", ""]
    if r.get("custom_inference"):
        md += ["Reloaded-model custom inference:", ""]
        md.append(table(["Text", "Prediction", "Confidence"], [[c["text"], c["pred"], c["conf"]] for c in r["custom_inference"]]))
        md.append("")
    # training curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ep = [h["epoch"] for h in m["history"]]
    a1.plot(ep, [h["train_loss"] for h in m["history"]], "o-", color=SERIES[0], label="train loss")
    a1.plot(ep, [h["val_loss"] for h in m["history"]], "o-", color=SERIES[1], label="val loss")
    a1.set(xlabel="epoch", ylabel="loss", title="Loss", xticks=ep)
    a1.legend(frameon=False)
    a2.plot(ep, [h["val_acc"] for h in m["history"]], "o-", color=SERIES[0])
    a2.set(xlabel="epoch", ylabel="accuracy", title="Validation accuracy", xticks=ep)
    fig.savefig(f"figures/{card}_hw4a_curves.png")
    plt.close(fig)
    md += [f"![training curves](figures/{card}_hw4a_curves.png)", ""]
    return r


# ----------------------------------------------------------------------------- serving

def serving_rows(card):
    rows = []
    for name, label in [("hf_b1", "HF pipeline, batch 1"), ("vllm_b1", "vLLM, batch 1"),
                        ("vllm_b8", "vLLM, batch 8"), ("vllm_b32", "vLLM, batch 32")]:
        r = load(f"results/{card}/serving/{name}.json")
        rows.append((label, r))
    return rows


def vllm_demo_section(card, md):
    d = load(f"results/{card}/serving/vllm_demo.json")
    if not d:
        return
    md += [f"### HW4.a.2 — vLLM inference on `{card}` with `{d['model']}` (a different Hugging Face model, own prompt)", ""]
    for r in d["results"]:
        md += [f"Prompt: *{r['prompt']}*", "", "Output:", "", *[f"> {l}" for l in r["output"].strip().splitlines()], ""]


def serving_section(card, md):
    rows = serving_rows(card)
    if not any(r for _, r in rows):
        return
    model = next(r["model"] for _, r in rows if r)
    md += [f"#### Serving on `{card}` — model `{model}`, {next(r['max_tokens'] for _, r in rows if r)} new tokens/request", ""]
    md.append(table(["Configuration", "Tokens/sec", "p50 latency (ms)", "p95 latency (ms)", "Requests/sec"],
                    [[lab, f(r and r["tokens_per_s"]), f(r and r["p50_ms"]), f(r and r["p95_ms"]), f(r and r["requests_per_s"], 2)]
                     for lab, r in rows]))
    hf = rows[0][1]
    v = [r for _, r in rows[1:] if r]
    if hf and v:
        crossing = None
        for prev, cur in zip(v, v[1:]):
            if cur["p50_ms"] > hf["p50_ms"] and cur["tokens_per_s"] > prev["tokens_per_s"]:
                crossing = cur["batch"]
                break
        md += ["", "Measured facts for question 6: " + (
            f"per-request p50 latency first exceeds the HF batch-1 baseline ({hf['p50_ms']:.0f} ms) at **vLLM batch {crossing}**, "
            f"while total throughput is still rising." if crossing else
            f"vLLM per-request p50 latency stays at or below the HF batch-1 baseline ({hf['p50_ms']:.0f} ms) at every batch "
            f"measured; the latency crossing is above batch {v[-1]['batch']}.")]
    md.append("")


def kv_section(card, md):
    for p in sorted(glob.glob(f"results/{card}/serving/kv_*.json")):
        k = load(p)
        md += [f"#### KV cache on `{card}` — `{k['model']}` ({k['dtype']}), batch {k['batch']}, "
               f"gpu_memory_utilization {k['gpu_memory_utilization']}", "",
               f"KV bytes/token = 2 (K,V) × {k['layers']} layers × {k['kv_heads']} KV heads × {k['head_dim']} head_dim × "
               f"{k['dtype_bytes']} B = **{k['bytes_per_token'] / 1024:.1f} KiB**. vLLM reserved a KV pool of "
               f"**{k['kv_capacity_tokens']:,} tokens ({k['kv_capacity_gb']:.2f} GB)**, so the longest context at batch "
               f"{k['batch']} is **{k['longest_context_at_batch']:,} tokens** (limited by {k['limited_by']}; "
               f"verified by a real batch-{k['batch']} run: {'yes' if k['verified'] else 'NO'}).", ""]
        md.append(table(["Context", f"KV cache at batch {k['batch']} (GB)", "Fits"],
                        [[f"{c['context']:,}", f(c["kv_gb_at_batch"], 2), "yes" if c["fits"] else "no"] for c in k["curve"]]))
        md.append("")


# ----------------------------------------------------------------------------- Part A

def part_a(cs, md):
    data = {}
    for c in cs:
        pts = []
        for p in glob.glob(f"results/{c}/partA/bs*.json"):
            r = load(p)
            if r and r.get("status") == "ok":
                pts.append(r)
        if pts:
            data[c] = sorted(pts, key=lambda r: r["config"]["batch_size"])
    if not data:
        return {}
    md += ["## Part A — Batch scaling to the memory wall", ""]
    out = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (c, pts) in enumerate(data.items()):
        bs = [r["config"]["batch_size"] for r in pts]
        thr = [r["metrics"]["throughput_sps"] for r in pts]
        knee = next((bs[j] for j in range(len(bs) - 1) if thr[j + 1] < thr[j] * 1.05), bs[-1])
        oom = load(f"results/{c}/partA/bs{bs[-1] * 2}.json")
        out[c] = {"max_batch": bs[-1], "knee": knee, "vram_gb": pts[0]["gpu"]["total_gb"], "gpu": pts[0]["gpu"]["name"]}
        md += [f"### `{c}` ({pts[0]['gpu']['name']}, {pts[0]['gpu']['total_gb']} GB)", ""]
        md.append(table(["Batch", "Throughput (samp/s)", "Peak alloc (GB)", "Peak reserved (GB)", "Time / epoch (s)"],
                        [[r["config"]["batch_size"], f(r["metrics"]["throughput_sps"]), f(r["metrics"]["peak_alloc_gb"], 2),
                          f(r["metrics"]["peak_reserved_gb"], 2),
                          f(r["metrics"]["time_per_epoch_s"]) + (" (est.)" if r["metrics"].get("epoch_time_estimated") else "")]
                         for r in pts]))
        md += ["", f"Largest batch that runs: **{bs[-1]}**" + (f" (batch {bs[-1] * 2}: {oom['status'].upper()})" if oom else "")
               + f". Throughput stops improving (next doubling < 5% gain) beyond batch **{knee}**.", ""]
        ax.plot(bs, thr, "o-", color=SERIES[i], label=f"{c} ({pts[0]['gpu']['total_gb']:.0f} GB)")
        ax.annotate(f"max {bs[-1]}", (bs[-1], thr[-1]), textcoords="offset points", xytext=(6, -4), color=INK2, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set(xlabel="batch size (log2)", ylabel="training throughput (samples/s)", title="BERT fine-tune throughput vs batch size")
    ax.legend(frameon=False)
    fig.savefig("figures/partA_throughput_vs_batch.png")
    plt.close(fig)
    md += ["![throughput vs batch](figures/partA_throughput_vs_batch.png)", ""]
    if len(out) == 1:
        md += ["Only one GPU was used (one GPU is enough per the HW4.5 clarifications), so the max-batch ratio vs VRAM "
               "ratio comparison between two cards does not apply.", ""]
    if len(out) >= 2:
        (c1, a), (c2, b) = sorted(out.items(), key=lambda kv: -kv[1]["vram_gb"])[:2]
        md += [f"Max-batch ratio {c1}/{c2} = {a['max_batch']}/{b['max_batch']} = **{a['max_batch'] / b['max_batch']:.2f}**; "
               f"VRAM ratio = {a['vram_gb']}/{b['vram_gb']} = **{a['vram_gb'] / b['vram_gb']:.2f}**. "
               f"Throughput knee: {a['knee']} on {c1}, {b['knee']} on {c2} "
               f"({'same' if a['knee'] == b['knee'] else 'different'}).", "",
               "TODO (2–3 sentences): why the ratios differ — batch sizes are restricted to powers of two, so the max batch can "
               "only change by 2×; a fixed chunk of memory (weights, gradients, AdamW states, CUDA context, allocator "
               "fragmentation) does not scale with batch, so the memory left for activations is not proportional to VRAM.", ""]
    return out


# ----------------------------------------------------------------------------- Part B

def part_b(card, md, base_a):
    root = f"results/{card}/partB"
    summ = load(f"{root}/summary.json")
    if not summ:
        return None
    full = {k: load(f"{root}/{k}/full.json") for k in TECH_NAMES if k != "grad_accum"}
    full = {k: v for k, v in full.items() if v and v.get("status") == "ok"}
    acc = {k: load(f"{root}/grad_accum/{k}.json") for k in ("accum", "direct")}
    gpu = next(iter(full.values()))["gpu"] if full else {"name": card, "total_gb": "?"}
    md += [f"### Table HW4.5.1 — `{card}` ({gpu['name']}, {gpu['total_gb']} GB)", "",
           "Throughput, peak memory and accuracy are from a full HW4-settings run "
           + (f"(batch {full['baseline']['config']['batch_size']}, {full['baseline']['config']['epochs']} epochs); " if "baseline" in full else "; ")
           + "max batch is the largest power of two that trains.", ""]
    rows = []
    for k, name in TECH_NAMES.items():
        if k == "grad_accum":
            ga, a = summ.get("grad_accum"), acc.get("accum")
            if ga and a and a.get("status") == "ok":
                rows.append([f"{name} ({ga['physical_batch']}×{ga['accum_steps']})",
                             f"{ga['max_effective_batch']} effective ({ga['max_physical_batch']} physical)",
                             f(a["metrics"]["throughput_sps"]), f(a["metrics"]["peak_alloc_gb"], 2), f(a["test"]["test_acc"], 4)])
            continue
        r = full.get(k)
        if r:
            dt = r["metrics"].get("amp_dtype")
            rows.append([f"{name} ({dt})" if dt and dt != "fp32" else name, summ.get(k, {}).get("max_batch"), f(r["metrics"]["throughput_sps"]),
                         f(r["metrics"]["peak_alloc_gb"], 2), f(r["test"]["test_acc"], 4)])
    md.append(table(["Technique", "Max batch", "Throughput (samp/s)", "Peak memory (GB)", "Eval accuracy"], rows))
    md.append("")
    b = full.get("baseline")
    if not b:
        return None
    bt, bm = b["metrics"]["throughput_sps"], b["metrics"]["peak_alloc_gb"]
    if "grad_ckpt" in full:
        t = full["grad_ckpt"]["metrics"]["throughput_sps"]
        md += [f"Gradient checkpointing: throughput {t:.1f} vs {bt:.1f} samp/s = **{(1 - t / bt):.1%} slower** "
               f"(the extra forward recomputation during backward).", ""]
    if acc.get("accum") and acc.get("direct"):
        a, d = acc["accum"], acc["direct"]
        md.append(table(["Run", "Physical batch", "Effective batch", "Optimiser steps", "Train time (s)", "Peak (GB)", "Test acc"],
                        [[lab, r["config"]["batch_size"], r["metrics"]["effective_batch"], r["metrics"]["opt_steps"],
                          f(r["metrics"]["train_time_s"]), f(r["metrics"]["peak_alloc_gb"], 2), f(r["test"]["test_acc"], 4)]
                         for lab, r in (("accumulation", a), ("direct large batch", d)) if r.get("status") == "ok"]
                        + ([] if d.get("status") == "ok" else [["direct large batch", d["config"]["batch_size"], "", "", "OOM", "", ""]])))
        md.append("")
    # ranking: memory saved per unit throughput lost, each against its own reference
    cand = []
    for k in ("amp", "grad_ckpt", "adamw8bit"):
        if k in full:
            cand.append((TECH_NAMES[k], bm - full[k]["metrics"]["peak_alloc_gb"], bt - full[k]["metrics"]["throughput_sps"], bt))
    if (acc.get("accum") or {}).get("status") == "ok" and (acc.get("direct") or {}).get("status") == "ok":
        a, d = acc["accum"]["metrics"], acc["direct"]["metrics"]
        cand.append(("Gradient accumulation (vs direct effective batch)", d["peak_alloc_gb"] - a["peak_alloc_gb"],
                     d["throughput_sps"] - a["throughput_sps"], d["throughput_sps"]))
    md += ["Ranking by memory saved per unit of throughput lost (GB saved ÷ % throughput lost; ∞ = saves memory with no "
           "throughput loss):", ""]
    md.append(ranking_table(cand))
    md.append("")
    return {"full": full, "acc": acc, "summary": summ, "cand": cand}


def rank_score(c):
    saved, lost, ref = c[1], c[2], c[3]
    pct_lost = 100 * lost / ref
    return float("inf") if pct_lost <= 0 and saved > 0 else (saved / pct_lost if pct_lost > 0 else float("-inf"))


def ranking_table(cand):
    cand = sorted(cand, key=rank_score, reverse=True)
    return table(["Rank", "Technique", "Memory saved (GB)", "Throughput lost (%)", "GB per % lost"],
                 [[i + 1, n, f(s, 2), f(100 * l / ref, 1) if l > 0 else f"none ({f(-100 * l / ref, 1)}% faster)",
                   "∞" if rank_score(c) == float("inf") else f(rank_score(c), 3)]
                  for i, c in enumerate(cand) for n, s, l, ref in [c]])


# ----------------------------------------------------------------------------- Part C

def part_c(cs, md):
    found = False
    for c in cs:
        one, two = load(f"results/{c}/partC/ddp_ws1.json"), load(f"results/{c}/partC/ddp_ws2.json")
        if not two:
            continue
        found = True
        t = two["topology"]
        md += ["## Part C — Two cards at once (DDP)", "",
               f"Pairing `{c}`: {', '.join(w['gpu'] + ' @ ' + w['host'] for w in t['workers'])} — "
               f"**{'homogeneous' if t['homogeneous'] else 'heterogeneous'}**, **{'single-node' if t['single_node'] else 'multi-node'}**"
               f"; NCCL {t['nccl_version']}; GPU peer access: {t['p2p_access']}. "
               f"Raw all-reduce of the {two['allreduce']['bytes'] / 1e6:.0f} MB fp32 gradient: {two['allreduce']['time_ms']:.1f} ms, "
               f"bus bandwidth **{two['allreduce']['busbw_GBps']:.2f} GB/s**.", "",
               "TODO: state the interconnect path (check `env/topo_*.txt` and the `NCCL INFO ... via P2P/SHM/NET` lines in RUN_LOG.txt).", ""]
        single = {r["per_gpu_batch"]: r for r in (one or {}).get("rows", [])}
        rows = []
        for r in two["rows"]:
            s_same = single.get(r["per_gpu_batch"])
            s_glob = single.get(r["global_batch"])
            eff = r["throughput_sps"] / (2 * s_same["throughput_sps"]) if s_same else None
            spd = r["throughput_sps"] / s_glob["throughput_sps"] if s_glob else None
            rows.append([r["per_gpu_batch"], r["global_batch"], f(s_same and s_same["throughput_sps"]), f(r["throughput_sps"]),
                         f(eff and 100 * eff) + ("%" if eff else ""), f(spd, 2), f(r["step_ms"]), f(r["comm_overhead_ms"]),
                         f(100 * r["comm_fraction"]) + "%", " / ".join(f(x) for x in r["straggler_wait_ms_per_rank"]),
                         f(r["effective_comm_GBps"], 2)])
        md.append(table(["Batch/GPU", "Global batch", "1-GPU samp/s", "2-GPU samp/s", "Scaling eff.", "Speed-up vs 1 GPU @ same global",
                         "Step (ms)", "Comm overhead (ms)", "Comm %", "Straggler wait per rank (ms)", "Effective comm GB/s"], rows))
        worth = next((r["per_gpu_batch"] for r in two["rows"] if single.get(r["global_batch"])
                      and r["throughput_sps"] > single[r["global_batch"]]["throughput_sps"]), None)
        md += ["", f"Scaling efficiency = 2-GPU throughput ÷ (2 × 1-GPU throughput at the same per-GPU batch). "
                   f"Comm overhead = DDP step − the same step under `no_sync()` (communication not hidden behind backward). "
                   + (f"Multi-GPU first beats one GPU running the same global batch at **{worth} per GPU**." if worth else
                      "Multi-GPU never beat one GPU at the same global batch in the measured range."), "",
               "TODO: explain the scaling behaviour (heterogeneous → waiting on the slower worker; homogeneous → measured comm "
               "overhead), and whether communication or computation dominates at your batch size.", ""]
        fig, ax = plt.subplots(figsize=(7, 4))
        if single:
            ks = sorted(single)
            ax.plot(ks, [single[k]["throughput_sps"] for k in ks], "o-", color=SERIES[0], label="1 GPU")
        ax.plot([r["global_batch"] for r in two["rows"]], [r["throughput_sps"] for r in two["rows"]], "o-", color=SERIES[1], label="2 GPUs (DDP)")
        ax.set_xscale("log", base=2)
        ax.set(xlabel="global batch size (log2)", ylabel="throughput (samples/s)", title=f"DDP scaling on {c}")
        ax.legend(frameon=False)
        fig.savefig(f"figures/{c}_partC_ddp.png")
        plt.close(fig)
        md += [f"![ddp](figures/{c}_partC_ddp.png)", ""]
    return found


# ----------------------------------------------------------------------------- Part E

def part_e(cs, md, b_results, full_epochs):
    md += ["## Part E — Analysis and summary (HW4.5 summary — not an HW5 baseline)", ""]
    for c in cs:
        br = b_results.get(c)
        if not br:
            continue
        full = br["full"]
        d = next(iter(full.values()))["data"]
        epochs = full_epochs or next(iter(full.values()))["config"]["epochs"]  # default: the HW4 run's epochs
        n_train_full = int(0.70 * d["file_rows"])
        md += [f"### `{c}`", "", f"One full fine-tune = 70% of all {d['file_rows']:,} rows = {n_train_full:,} training samples × "
               f"{epochs} epochs = {n_train_full * epochs:,} samples. GPU-hours = samples ÷ throughput ÷ 3600:", ""]
        md.append(table(["Configuration", "Throughput (samp/s)", "Calculation", "GPU-hours"],
                        [[TECH_NAMES[k], f(r["metrics"]["throughput_sps"]),
                          f"{n_train_full * epochs:,} / {r['metrics']['throughput_sps']:.1f} / 3600",
                          f(n_train_full * epochs / r["metrics"]["throughput_sps"] / 3600, 3)] for k, r in full.items()]))
        mem = min(full.items(), key=lambda kv: kv[1]["metrics"]["peak_alloc_gb"])
        thr = max(full.items(), key=lambda kv: kv[1]["metrics"]["throughput_sps"])
        wall = min(full.items(), key=lambda kv: kv[1]["metrics"]["train_time_s"])
        md += ["", "**HW4.5 summary (best measured configurations):**", ""]
        md.append(table(["Criterion", "Configuration", "Peak memory (GB)", "Throughput (samp/s)", "Wall-clock train time (s)", "Test acc"],
                        [[crit, TECH_NAMES[k], f(r["metrics"]["peak_alloc_gb"], 2), f(r["metrics"]["throughput_sps"]),
                          f(r["metrics"]["train_time_s"]), f(r["test"]["test_acc"], 4)]
                         for crit, (k, r) in (("Lowest peak memory", mem), ("Highest throughput", thr), ("Shortest wall-clock", wall))]))
        md.append("")
    md += [answer("partE", "TODO: the single configuration you would use for a full fine-tune on the lab hardware, justified from the tables above."), ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-epochs", type=int, default=0, help="epochs of one full fine-tune (default: the HW4 run's)")
    ap.add_argument("--dataset-name", default="Twitter US Airline Sentiment", help="name of the HW4.a Twitter dataset")
    a = ap.parse_args()
    os.makedirs("figures", exist_ok=True)
    cs = cards()
    md = ["# METRICS — HW4.a and HW4.5 (GPU Assignment II)", "",
          "Generated by `make_report.py` from `results/`. GPU UUIDs: see `env/`. Written answers come from `ANSWERS.md`.", ""]
    data_section(md, a.dataset_name)
    md += ["## HW4.a — BERT fine-tune and vLLM inference", ""]
    for c in cs:
        hw4a_section(c, md, a.dataset_name)
    for c in cs:
        if os.path.exists(f"results/{c}/hw4a.json"):
            vllm_demo_section(c, md)
    md += ["### Table HW4.1 — Serving comparison", ""]
    for c in cs:
        if os.path.exists(f"results/{c}/hw4a.json"):  # the card HW4.a ran on; Part D below shows every card
            serving_section(c, md)
    md += [answer("q6", "TODO (Q6, 3–4 sentences): at which batch size per-request latency becomes worse than the baseline while total "
           "throughput still improves, and which number matters for an interactive application."), ""]
    part_a(cs, md)
    md += ["## Part B — Memory techniques, measured one at a time", ""]
    b_results = {}
    for c in cs:
        r = part_b(c, md, None)
        if r:
            b_results[c] = r
    if len(b_results) > 1:  # Part B.5 asks for one ranking of the four techniques; each is scored against its own card's reference
        allc = [(f"{n} (`{c}`)", *rest) for c, r in b_results.items() for n, *rest in r["cand"]]
        md += ["### Ranking of the four techniques (all cards)", "",
               "Each technique's memory saved and throughput lost are measured against the baseline on the card it ran on, so the "
               "percentages are comparable even though the cards' absolute throughputs differ.", "",
               ranking_table(allc), ""]
    md += [answer("partB", "TODO: which technique you would apply first, and which you would not bother with on a 32 GB card."), ""]
    if not part_c(cs, md):
        md += ["## Part C — Two cards at once", "", "**Part C skipped (optional)**", ""]
    md += ["## Part D — Serving, revisited on real hardware", ""]
    for c in cs:
        serving_section(c, md)
        kv_section(c, md)
    md += [answer("partD", "TODO: compare with your HW4 serving numbers; if they differ, explain (hardware, precision, or measurement flaw)."), ""]
    part_e(cs, md, b_results, a.full_epochs)
    open("METRICS.md", "w").write("\n".join(md))
    print(f"wrote METRICS.md and figures/ for cards: {cs}")


if __name__ == "__main__":
    main()
