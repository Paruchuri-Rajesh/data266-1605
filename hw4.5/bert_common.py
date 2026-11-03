"""Shared data / model / training code for the HW4.a BERT fine-tune and every HW4.5 experiment.

The HW4.a settings live in TrainConfig defaults. HW4.5 Part A/B change exactly one knob at a time
(batch size, amp, grad_ckpt, accum_steps, optim) so every effect can be attributed.
"""
import json
import math
import os
import random
import time
import hashlib
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

MAX_ROWS = 5000


@dataclass
class TrainConfig:
    # data
    data: str = "data/Tweets.csv"
    text_col: str = "text"
    label_col: str = "airline_sentiment"
    slice_start: int = 0          # SLICE
    seed: int = 42                # SEED
    max_len: int = 128
    # model / HW4.a hyper-parameters (the "unchanged" baseline)
    model_name: str = "bert-base-uncased"
    batch_size: int = 32
    epochs: int = 4
    lr: float = 2e-5
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    # HW4.5 memory techniques (all off = baseline)
    amp: str = "none"             # none | bf16 | fp16 | auto
    grad_ckpt: bool = False
    accum_steps: int = 1
    optim: str = "adamw"          # adamw | adamw8bit
    # run control
    max_opt_steps: int = 0        # stop after this many optimiser steps (0 = full epochs)
    probe_steps: int = 0          # >0: short throughput/memory probe on sampled-with-replacement batches
    warmup_steps: int = 3         # micro-steps excluded from steady-state throughput
    eval: bool = True
    save_dir: str = ""
    tag: str = "run"
    out: str = ""
    # same-data guarantee: the HW4.a run writes the lock, every HW4.5 run must match it
    data_lock: str = "data/hw4a_data_lock.json"
    write_lock: bool = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_info():
    if not torch.cuda.is_available():
        return {"name": "cpu", "total_gb": 0.0}
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "total_gb": round(p.total_memory / 1e9, 2),
            "capability": f"{p.major}.{p.minor}", "torch": torch.__version__,
            "cuda": torch.version.cuda}


# --------------------------------------------------------------------------- data

def load_slice(cfg):
    """Up to MAX_ROWS consecutive rows starting at SLICE; returns texts, label ids, label names, info."""
    df = pd.read_csv(cfg.data, encoding_errors="replace")
    if cfg.slice_start >= len(df):
        raise ValueError(f"SLICE={cfg.slice_start} is past the end of {cfg.data} ({len(df)} rows)")
    end = min(cfg.slice_start + MAX_ROWS, len(df))
    sub = df.iloc[cfg.slice_start:end][[cfg.text_col, cfg.label_col]]
    n_raw = len(sub)
    sub = sub.dropna()
    names = sorted(sub[cfg.label_col].astype(str).unique())
    to_id = {n: i for i, n in enumerate(names)}
    info = {
        "file": cfg.data, "file_rows": len(df),
        "row_range": f"{cfg.slice_start}..{end - 1} (0-based data rows, inclusive)",
        "csv_line_range": f"{cfg.slice_start + 2}..{end + 1} (1-based file lines, header = line 1)",
        "slice_start": cfg.slice_start, "slice_end_exclusive": end,
        "rows_in_range": n_raw, "rows_dropped_na": n_raw - len(sub), "rows_used": len(sub),
        "labels": names,
    }
    texts = sub[cfg.text_col].astype(str).tolist()
    labels = np.array([to_id[str(v)] for v in sub[cfg.label_col]])
    return texts, labels, names, info


def split_70_15_15(labels, seed):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(labels))
    try:
        tr, rest = train_test_split(idx, test_size=0.30, random_state=seed, stratify=labels)
        va, te = train_test_split(rest, test_size=0.50, random_state=seed, stratify=labels[rest])
    except ValueError:  # a class too small to stratify
        tr, rest = train_test_split(idx, test_size=0.30, random_state=seed)
        va, te = train_test_split(rest, test_size=0.50, random_state=seed)
    return tr, va, te


def tokenize(tokenizer, texts, labels, max_len):
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_fingerprint(cfg, info, labels, tr, va, te):
    """Everything that must not change between HW4.a and HW4.5: file, rows, labels and the exact split."""
    split = hashlib.sha256()
    for ix in (tr, va, te):
        split.update(np.asarray(ix, dtype=np.int64).tobytes())
    split.update(np.asarray(labels, dtype=np.int64).tobytes())
    return {"file_sha256": sha256_file(cfg.data), "text_col": cfg.text_col, "label_col": cfg.label_col,
            "row_range": info["row_range"], "rows_used": info["rows_used"], "labels": info["labels"],
            "seed": cfg.seed, "split_sizes": [len(tr), len(va), len(te)], "split_sha256": split.hexdigest()}


def check_data_lock(cfg, fp):
    """HW4.5 must reuse the HW4.a data: same file, same rows, same 70/15/15 split with the same SEED."""
    if cfg.write_lock and not os.path.exists(cfg.data_lock):
        write_json(cfg.data_lock, fp)
        print(f"[data] wrote {cfg.data_lock} (HW4.a data lock)", flush=True)
        return
    if not os.path.exists(cfg.data_lock):
        raise SystemExit(f"[data] {cfg.data_lock} not found: run the HW4.a fine-tune first (./run_card.sh <card> hw4a) "
                         f"so HW4.5 is measured on the same rows and split")
    lock = json.load(open(cfg.data_lock))
    diff = [k for k in lock if lock[k] != fp.get(k)]
    if diff:
        raise SystemExit(f"[data] this run's data differs from the HW4.a lock in {diff}. HW4.5 must use the same file, "
                         f"SLICE and SEED as HW4.a (see {cfg.data_lock}).")
    print(f"[data] matches HW4.a lock: rows {fp['row_range']}, split {fp['split_sha256'][:12]}", flush=True)


def build_datasets(cfg):
    texts, labels, names, info = load_slice(cfg)
    tr, va, te = split_70_15_15(labels, cfg.seed)
    info["fingerprint"] = data_fingerprint(cfg, info, labels, tr, va, te)
    check_data_lock(cfg, info["fingerprint"])
    tok = BertTokenizer.from_pretrained(cfg.model_name)
    pick = lambda ix: ([texts[i] for i in ix], labels[ix])
    ds = {k: tokenize(tok, *pick(ix), cfg.max_len) for k, ix in (("train", tr), ("val", va), ("test", te))}
    info["split_sizes"] = {k: len(v) for k, v in ds.items()}
    test_texts = [texts[i] for i in te]
    return ds, names, info, tok, test_texts


# --------------------------------------------------------------------------- model / optimiser

def resolve_amp(amp):
    if amp == "none" or not torch.cuda.is_available():
        return None
    if amp == "auto":  # native bf16 needs Ampere+ (cc 8.0); is_bf16_supported() is also True for slow emulation on a T4
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]


def build_model(cfg, num_labels, device):
    model = BertForSequenceClassification.from_pretrained(cfg.model_name, num_labels=num_labels)
    if cfg.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(device)


def build_optimizer(cfg, model):
    if cfg.optim == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=cfg.eps, weight_decay=cfg.weight_decay)
    if cfg.optim == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=cfg.lr, eps=cfg.eps, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optim)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, ds, batch_size, device, amp_dtype):
    model.eval()
    loader = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=max(batch_size, 64))
    total_loss, preds, probs, gold = 0.0, [], [], []
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(ids, attention_mask=mask, labels=y)
        total_loss += out.loss.item() * len(y)
        p = out.logits.float().softmax(-1)
        probs.append(p.cpu())
        preds.append(p.argmax(-1).cpu())
        gold.append(y.cpu())
    preds, gold, probs = torch.cat(preds).numpy(), torch.cat(gold).numpy(), torch.cat(probs).numpy()
    return {"loss": total_loss / len(gold), "acc": float((preds == gold).mean()),
            "preds": preds, "gold": gold, "probs": probs}


# --------------------------------------------------------------------------- training

def train(cfg, ds, num_labels, device, log=print):
    """Train and return (model, metrics). Raises torch.OutOfMemoryError on OOM."""
    set_seed(cfg.seed)
    model = build_model(cfg, num_labels, device)
    opt = build_optimizer(cfg, model)
    amp_dtype = resolve_amp(cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    train_ds = ds["train"]
    g = torch.Generator().manual_seed(cfg.seed)
    if cfg.probe_steps:  # fixed-length probe; sampling with replacement lets batch exceed the train split
        sampler = RandomSampler(train_ds, replacement=True, num_samples=cfg.batch_size * cfg.probe_steps, generator=g)
        epochs = 1
    else:
        sampler = RandomSampler(train_ds, generator=g)
        epochs = cfg.epochs
    loader = DataLoader(train_ds, sampler=sampler, batch_size=cfg.batch_size, pin_memory=True)
    opt_steps_per_epoch = math.ceil(len(loader) / cfg.accum_steps)
    total_opt_steps = cfg.max_opt_steps or opt_steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(opt, 0, total_opt_steps)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    history, opt_step, micro = [], 0, 0
    steady_t0 = steady_samples = None
    samples = 0
    sync()
    t_start = time.perf_counter()
    done = False
    for epoch in range(1, epochs + 1):
        model.train()
        sync()
        t_ep = time.perf_counter()
        ep_loss, ep_n = 0.0, 0
        for i, (ids, mask, y) in enumerate(loader):
            ids, mask, y = ids.to(device, non_blocking=True), mask.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(ids, attention_mask=mask, labels=y).loss
            scaler.scale(loss / cfg.accum_steps).backward()
            ep_loss += loss.item() * len(y)
            ep_n += len(y)
            samples += len(y)
            micro += 1
            if micro % cfg.accum_steps == 0 or i + 1 == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                opt_step += 1
            if micro == cfg.warmup_steps:
                sync()
                steady_t0, steady_samples = time.perf_counter(), samples
            if cfg.max_opt_steps and opt_step >= cfg.max_opt_steps:
                done = True
                break
        sync()
        ep_time = time.perf_counter() - t_ep
        rec = {"epoch": epoch, "train_loss": ep_loss / max(ep_n, 1), "epoch_time_s": ep_time, "opt_steps": opt_step}
        if cfg.eval and not cfg.probe_steps:
            v = evaluate(model, ds["val"], cfg.batch_size, device, amp_dtype)
            rec.update(val_loss=v["loss"], val_acc=v["acc"])
        history.append(rec)
        log(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}))
        if done:
            break
    sync()
    wall = time.perf_counter() - t_start
    train_only = sum(h["epoch_time_s"] for h in history)  # epoch_time_s excludes the val pass
    if cfg.probe_steps and steady_t0:
        thr = (samples - steady_samples) / (time.perf_counter() - steady_t0)  # steady state, warm-up excluded
    else:
        thr = samples / train_only
    m = {
        "samples": samples, "opt_steps": opt_step, "wall_s_incl_eval": wall, "train_time_s": train_only,
        "throughput_sps": thr,
        "time_per_epoch_s": float(np.mean([h["epoch_time_s"] for h in history])),
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0,
        "effective_batch": cfg.batch_size * cfg.accum_steps,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "fp32",
        "history": history,
    }
    return model, m


def early_stop_summary(history):
    """Epoch where val loss stopped improving and what training past it cost."""
    if not history or "val_loss" not in history[0]:
        return {}
    best = min(history, key=lambda h: h["val_loss"])
    after = [h for h in history if h["epoch"] > best["epoch"]]
    last = history[-1]
    return {
        "best_epoch_by_val_loss": best["epoch"],
        "best_epoch_by_val_acc": max(history, key=lambda h: h["val_acc"])["epoch"],
        "epochs_past_best": len(after),
        "extra_time_s": sum(h["epoch_time_s"] for h in after),
        "val_loss_increase": last["val_loss"] - best["val_loss"],
        "val_acc_change": last["val_acc"] - best["val_acc"],
    }


def test_report(ev, names, test_texts, n_examples=5):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    labels = list(range(len(names)))
    cm = confusion_matrix(ev["gold"], ev["preds"], labels=labels)
    p, r, f, s = precision_recall_fscore_support(ev["gold"], ev["preds"], labels=labels, zero_division=0)
    per_class = {names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                 for i in labels}
    worst = int(np.argmin(f))
    wrong = [i for i in range(len(ev["gold"])) if ev["gold"][i] == worst and ev["preds"][i] != worst]
    wrong.sort(key=lambda i: -ev["probs"][i][ev["preds"][i]])  # most confident mistakes first
    examples = [{"text": test_texts[i], "true": names[worst], "pred": names[int(ev["preds"][i])],
                 "pred_conf": round(float(ev["probs"][i][ev["preds"][i]]), 3),
                 "diagnosis": "TODO: one-line diagnosis"} for i in wrong[:n_examples]]
    return {"test_acc": ev["acc"], "test_loss": ev["loss"], "labels": names,
            "confusion_matrix": cm.tolist(), "per_class": per_class,
            "worst_class": names[worst], "worst_class_misclassified": examples}


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)


def config_dict(cfg):
    return asdict(cfg)
