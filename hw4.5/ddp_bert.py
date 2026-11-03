"""HW4.5 Part C: distributed data parallel BERT fine-tune throughput, communication overhead and scaling.

Run the SAME script with 1 process (single-GPU baseline) and 2 processes (the assigned pair):
    torchrun --standalone --nproc_per_node 1 ddp_bert.py --card kaggle_t4x2 --slice-start $SLICE --seed $SEED
    NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node 2 ddp_bert.py --card kaggle_t4x2 --slice-start $SLICE --seed $SEED
Multi-node pair (one process per node):
    torchrun --nnodes 2 --nproc_per_node 1 --node_rank {0|1} --rdzv_backend c10d --rdzv_endpoint HOST:29500 ddp_bert.py ...

For each per-GPU batch size it measures
  t_sync    mean step time with normal DDP gradient all-reduce
  t_nosync  mean step time inside model.no_sync() (identical work, no all-reduce)
  comm overhead per step = t_sync - t_nosync   (communication NOT hidden behind backward)
and separately times a raw all_reduce of a gradient-sized buffer to get the effective bus bandwidth.
"""
import argparse
import contextlib
import os
import socket
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from bert_common import TrainConfig, build_datasets, build_model, build_optimizer, resolve_amp, set_seed, write_json


def timed_steps(model, opt, scaler, amp_dtype, ds, bs, n, warmup, device, gen, no_sync):
    times = []
    for i in range(warmup + n):
        idx = torch.randint(len(ds), (bs,), generator=gen)
        ids, mask, y = (t[idx].to(device, non_blocking=True) for t in ds.tensors)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ctx = model.no_sync() if no_sync else contextlib.nullcontext()
        with ctx:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(ids, attention_mask=mask, labels=y).loss
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def allreduce_bandwidth(numel, device, world, iters=10):
    buf = torch.ones(numel, dtype=torch.float32, device=device)
    for _ in range(3):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / iters
    nbytes = numel * 4
    bus = nbytes / t * (2 * (world - 1) / world) if world > 1 else 0.0  # NCCL "busbw" convention
    return {"bytes": nbytes, "time_ms": t * 1e3, "algbw_GBps": nbytes / t / 1e9, "busbw_GBps": bus / 1e9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--slice-start", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", default="data/Tweets.csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="airline_sentiment")
    ap.add_argument("--amp", default="none")
    # keep these below the Part A max batch: an OOM on one rank can leave the other blocked in all-reduce
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    cfg = TrainConfig(data=a.data, text_col=a.text_col, label_col=a.label_col,
                      slice_start=a.slice_start, seed=a.seed, amp=a.amp)
    ds, names, info, _, _ = build_datasets(cfg)
    set_seed(cfg.seed)
    model = DDP(build_model(cfg, len(names), device), device_ids=[local])
    opt = build_optimizer(cfg, model)
    amp_dtype = resolve_amp(cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    gen = torch.Generator().manual_seed(cfg.seed + rank)

    me = {"rank": rank, "host": socket.gethostname(), "local_rank": local,
          "gpu": torch.cuda.get_device_name(local),
          "total_gb": round(torch.cuda.get_device_properties(local).total_memory / 1e9, 1)}
    peers = [None] * world
    dist.all_gather_object(peers, me)
    hosts = sorted({p["host"] for p in peers})
    gpus = sorted({p["gpu"] for p in peers})
    topo = {
        "world_size": world, "workers": peers,
        "homogeneous": len(gpus) == 1, "nodes": len(hosts),
        "single_node": len(hosts) == 1,
        "p2p_access": (torch.cuda.can_device_access_peer(0, 1)
                       if len(hosts) == 1 and torch.cuda.device_count() > 1 else None),
        "nccl_version": ".".join(map(str, torch.cuda.nccl.version())),
    }
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bw = allreduce_bandwidth(n_params, device, world)

    rows = []
    for bs in a.batches:
        try:
            t_sync = timed_steps(model, opt, scaler, amp_dtype, ds["train"], bs, a.steps, a.warmup, device, gen, False)
            t_nosync = timed_steps(model, opt, scaler, amp_dtype, ds["train"], bs, a.steps, a.warmup, device, gen, True)
            ok = torch.tensor(1, device=device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            t_sync = t_nosync = float("nan")
            ok = torch.tensor(0, device=device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        per_rank = [None] * world
        dist.all_gather_object(per_rank, {"rank": rank, "t_sync": t_sync, "t_nosync": t_nosync})
        if not ok.item():
            if rank == 0:
                print(f"per-GPU batch {bs}: OOM on at least one rank, stopping sweep", flush=True)
            break
        t_s = max(r["t_sync"] for r in per_rank)              # step is gated by the slowest rank
        compute = [r["t_nosync"] for r in per_rank]
        row = {
            "per_gpu_batch": bs, "global_batch": bs * world,
            "step_ms": t_s * 1e3,
            "compute_ms_per_rank": [c * 1e3 for c in compute],
            "comm_overhead_ms": (t_s - max(compute)) * 1e3,
            "straggler_wait_ms_per_rank": [(max(compute) - c) * 1e3 for c in compute],
            "throughput_sps": bs * world / t_s,
        }
        row["comm_fraction"] = row["comm_overhead_ms"] / row["step_ms"]
        row["effective_comm_GBps"] = (bw["bytes"] * 2 * (world - 1) / world / (row["comm_overhead_ms"] / 1e3) / 1e9
                                      if world > 1 and row["comm_overhead_ms"] > 0 else None)
        rows.append(row)
        if rank == 0:
            print(f"bs/gpu {bs:5d}  step {row['step_ms']:8.1f} ms  comm {row['comm_overhead_ms']:7.1f} ms "
                  f"({row['comm_fraction']:.0%})  thr {row['throughput_sps']:8.1f} samp/s", flush=True)

    if rank == 0:
        out = f"results/{a.card}/partC/ddp_ws{world}.json"
        write_json(out, {"topology": topo, "grad_params": n_params, "allreduce": bw,
                         "amp": a.amp, "data": info, "rows": rows})
        print(f"wrote {out}\n{topo}\nall-reduce of {bw['bytes']/1e6:.0f} MB: {bw['time_ms']:.1f} ms, "
              f"busbw {bw['busbw_GBps']:.2f} GB/s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
