"""HW4.a BERT sentiment fine-tune, also the single-run worker for HW4.5 Parts A and B.

HW4.a full run (train, evaluate, save, reload, custom inference):
    python train_bert.py --slice-start $SLICE --seed $SEED --save-dir models/bert_hw4a \
        --tag hw4a --out results/$CARD/hw4a.json
"""
import argparse
import dataclasses
import json
import sys
import time

import torch

from bert_common import (TrainConfig, build_datasets, train, evaluate, resolve_amp, early_stop_summary,
                         test_report, write_json, config_dict, gpu_info)

CUSTOM_SENTENCES = [
    "The flight was delayed three hours and nobody told us anything.",
    "Crew was friendly and we landed early, great job!",
    "Is there wifi on the 7am flight to Denver?",
]


def parse_args():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(TrainConfig):
        name = "--" + f.name.replace("_", "-")
        if f.type is bool or isinstance(f.default, bool):
            ap.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            ap.add_argument(name, type=type(f.default), default=f.default)
    ap.add_argument("--epoch-or-probe", action="store_true",
                    help="Part A: run one real epoch, or a 10-step probe if batch > train split")
    a = ap.parse_args()
    extra = {"epoch_or_probe": a.epoch_or_probe}
    del a.epoch_or_probe
    return TrainConfig(**vars(a)), extra


@torch.no_grad()
def custom_inference(model_dir, names, sentences, device, max_len):
    from transformers import BertTokenizer, BertForSequenceClassification
    tok = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    enc = tok(sentences, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    probs = model(**enc).logits.softmax(-1).cpu()
    return [{"text": s, "pred": names[int(p.argmax())], "conf": round(float(p.max()), 3)}
            for s, p in zip(sentences, probs)]


def main():
    cfg, extra = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds, names, info, tok, test_texts = build_datasets(cfg)
    if extra["epoch_or_probe"]:
        cfg.epochs, cfg.eval = 1, False
        if cfg.batch_size > len(ds["train"]):
            cfg.probe_steps = 10
    print(f"[{cfg.tag}] data {info['row_range']} splits={info['split_sizes']} labels={names}", flush=True)
    result = {"tag": cfg.tag, "config": config_dict(cfg), "data": info, "gpu": gpu_info(),
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        model, m = train(cfg, ds, len(names), device, log=lambda s: print(f"[{cfg.tag}] {s}", flush=True))
    except torch.OutOfMemoryError as e:
        result.update(status="oom", error=str(e).splitlines()[0])
        print(f"[{cfg.tag}] OOM at batch {cfg.batch_size}", flush=True)
        if cfg.out:
            write_json(cfg.out, result)
        sys.exit(3)
    result.update(status="ok", metrics=m)
    if cfg.probe_steps:
        result["metrics"]["epoch_time_estimated"] = True
        result["metrics"]["time_per_epoch_s"] = len(ds["train"]) / m["throughput_sps"]
    if cfg.eval and not cfg.probe_steps:
        result["early_stopping"] = early_stop_summary(m["history"])
        ev = evaluate(model, ds["test"], cfg.batch_size, device, resolve_amp(cfg.amp))
        result["test"] = test_report(ev, names, test_texts)
        print(f"[{cfg.tag}] test acc {ev['acc']:.4f}  worst class {result['test']['worst_class']}", flush=True)
    if cfg.save_dir:
        model.save_pretrained(cfg.save_dir)
        tok.save_pretrained(cfg.save_dir)
        del model
        result["custom_inference"] = custom_inference(cfg.save_dir, names, CUSTOM_SENTENCES, device, cfg.max_len)
        print(json.dumps(result["custom_inference"], indent=1), flush=True)
    if cfg.out:
        write_json(cfg.out, result)
    print(f"[{cfg.tag}] thr {m['throughput_sps']:.1f} samp/s  peak {m['peak_alloc_gb']:.2f} GB  "
          f"epoch {result['metrics']['time_per_epoch_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
