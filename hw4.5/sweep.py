"""HW4.5 Part A (batch scaling to the memory wall) and Part B (memory techniques, one at a time).

Each configuration runs in its own process so an OOM cannot poison the next measurement.

    python sweep.py partA --card rtx4090 --slice-start $SLICE --seed $SEED
    python sweep.py partB --card rtx4090 --slice-start $SLICE --seed $SEED
"""
import argparse
import json
import os
import subprocess
import sys

MAX_POW = 14  # stop at batch 16384 even if it still fits

# Part B: one knob each, then the best combinations.
TECHNIQUES = {
    "baseline": [],
    "amp": ["--amp", "auto"],
    "grad_ckpt": ["--grad-ckpt", "true"],
    "adamw8bit": ["--optim", "adamw8bit"],
    "best": ["--amp", "auto", "--optim", "adamw8bit"],
    "best_ckpt": ["--amp", "auto", "--optim", "adamw8bit", "--grad-ckpt", "true"],
}


def run(args, common, extra, out):
    cmd = [sys.executable, "train_bert.py", *common, *extra, "--out", out]
    print("$ " + " ".join(cmd), flush=True)
    if os.path.exists(out):
        os.remove(out)  # never read a stale result from an earlier attempt
    rc = subprocess.call(cmd)
    status = json.load(open(out))["status"] if os.path.exists(out) else "error"
    return rc, status


def max_batch_search(args, common, extra, outdir, start=1, epoch_or_probe=False):
    """Double batch size until OOM. Returns the largest batch that ran."""
    best, bs = None, start
    while bs <= 2 ** MAX_POW:
        out = f"{outdir}/bs{bs}.json"
        mode = ["--epoch-or-probe"] if epoch_or_probe else ["--probe-steps", "10", "--eval", "false"]
        rc, status = run(args, common, [*extra, "--batch-size", str(bs), "--tag", f"{os.path.basename(outdir)}_bs{bs}", *mode], out)
        if status != "ok":
            print(f"--> stopped at batch {bs}: {status}", flush=True)
            break
        best, bs = bs, bs * 2
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["partA", "partB"])
    ap.add_argument("--card", required=True, help="label for this GPU, e.g. rtx4090")
    ap.add_argument("--slice-start", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", default="data/Tweets.csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="airline_sentiment")
    ap.add_argument("--hw4-batch", type=int, default=32, help="batch size of the HW4.a run")
    ap.add_argument("--accum-physical", type=int, default=8)
    ap.add_argument("--accum-steps", type=int, default=16)
    ap.add_argument("--only", nargs="*", help="Part B: subset of techniques to run")
    a = ap.parse_args()
    common = ["--slice-start", str(a.slice_start), "--seed", str(a.seed), "--data", a.data,
              "--text-col", a.text_col, "--label-col", a.label_col]
    root = f"results/{a.card}/{a.part}"
    os.makedirs(root, exist_ok=True)

    if a.part == "partA":
        best = max_batch_search(a, common, [], root, start=1, epoch_or_probe=True)
        json.dump({"max_batch": best}, open(f"{root}/summary.json", "w"), indent=2)
        print(f"Part A max batch on {a.card}: {best}")
        return

    summary = {}
    for name, extra in TECHNIQUES.items():
        if a.only and name not in a.only:
            continue
        d = f"{root}/{name}"
        os.makedirs(f"{d}/search", exist_ok=True)
        mb = max_batch_search(a, common, extra, f"{d}/search", start=a.hw4_batch)
        # full HW4 run (same batch, epochs, lr) for throughput, peak memory and eval accuracy
        rc, status = run(a, common, [*extra, "--batch-size", str(a.hw4_batch), "--tag", f"B_{name}"], f"{d}/full.json")
        summary[name] = {"max_batch": mb, "full_status": status}

    if not a.only or "grad_accum" in a.only:
        d = f"{root}/grad_accum"
        os.makedirs(d, exist_ok=True)
        eff = a.accum_physical * a.accum_steps
        run(a, common, ["--batch-size", str(a.accum_physical), "--accum-steps", str(a.accum_steps),
                        "--tag", f"B_accum_{a.accum_physical}x{a.accum_steps}"], f"{d}/accum.json")
        # reference: the same effective batch taken in one physical step, same number of optimiser steps
        run(a, common, ["--batch-size", str(eff), "--tag", f"B_direct_{eff}"], f"{d}/direct.json")
        mb = max_batch_search(a, common, ["--accum-steps", str(a.accum_steps)], f"{d}/search", start=a.accum_physical)
        summary["grad_accum"] = {"physical_batch": a.accum_physical, "accum_steps": a.accum_steps,
                                 "effective_batch": eff, "max_physical_batch": mb,
                                 "max_effective_batch": mb * a.accum_steps if mb else None}
    prev = f"{root}/summary.json"
    if os.path.exists(prev):
        summary = {**json.load(open(prev)), **summary}
    json.dump(summary, open(prev, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
