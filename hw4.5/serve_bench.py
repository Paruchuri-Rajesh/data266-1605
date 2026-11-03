"""HW4.a vLLM inference + serving benchmark (Table HW4.1), re-used for HW4.5 Part D.

    python serve_bench.py demo  --card rtx4090                       # 4.a.2: vLLM with your own prompt
    python serve_bench.py hf    --card rtx4090                       # HF pipeline, batch 1 baseline
    python serve_bench.py vllm  --card rtx4090 --batches 1 8 32      # vLLM at batch 1, 8, 32
    python serve_bench.py kv    --card rtx4090 --model Qwen/Qwen2.5-7B-Instruct   # Part D.2

Every request asks for exactly --max-tokens new tokens (greedy, EOS ignored) so tokens/sec is comparable.
Latency: a batch of B requests is submitted together and all B finish at the same step, so each request's
latency is the wall time of its batch. p50/p95 are taken over all requests; tokens/sec = all generated
tokens / total wall time.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "Explain in two sentences why GPUs are faster than CPUs for training neural networks.",
    "Write a short haiku about a delayed flight.",
    "Summarise the plot of a heist movie in three sentences.",
    "What are three tips for writing a clear bug report?",
    "Describe the water cycle to a ten-year-old.",
    "Give a one-paragraph pitch for a campus food-delivery app.",
    "List four differences between TCP and UDP.",
    "Write a polite email declining a meeting invitation.",
]


def smi():
    """GPU facts from nvidia-smi, so this process never opens a CUDA context that vLLM would have to share."""
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,uuid",
                                   "--format=csv,noheader,nounits", "-i", "0"], text=True)
    name, mem, cap, uuid = [x.strip() for x in out.strip().split(",")]
    return {"gpu": name, "gpu_total_gb": round(float(mem) / 1024, 1), "compute_cap": cap, "uuid": uuid}


def pick_dtype(name):
    if name != "auto":
        return name
    return "bfloat16" if float(smi()["compute_cap"]) >= 8.0 else "float16"


def stats(lat_s, tokens, wall_s):
    lat_ms = np.array(lat_s) * 1e3
    return {"requests": len(lat_s), "tokens": int(tokens), "wall_s": wall_s,
            "tokens_per_s": tokens / wall_s, "requests_per_s": len(lat_s) / wall_s,
            "p50_ms": float(np.percentile(lat_ms, 50)), "p95_ms": float(np.percentile(lat_ms, 95))}


def save(card, name, obj):
    path = f"results/{card}/serving/{name}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)
    print(f"wrote {path}\n{json.dumps(obj, indent=1)[:1500]}", flush=True)


def gpu_meta():
    import torch
    return {**smi(), "torch": torch.__version__}


def bench_hf(a):
    import torch
    from transformers import pipeline
    dtype = pick_dtype(a.dtype)
    try:  # transformers >= 4.56 takes dtype=, older releases only torch_dtype=
        pipe = pipeline("text-generation", model=a.model, dtype=getattr(torch, dtype), device=0)
    except TypeError:
        pipe = pipeline("text-generation", model=a.model, torch_dtype=getattr(torch, dtype), device=0)
    kw = dict(max_new_tokens=a.max_tokens, min_new_tokens=a.max_tokens, do_sample=False, return_full_text=False)
    for p in PROMPTS[:2]:
        pipe(p, **kw)  # warm-up
    lat = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(a.requests):
        s = time.perf_counter()
        pipe(PROMPTS[i % len(PROMPTS)], **kw)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - s)
    wall = time.perf_counter() - t0
    r = {"engine": "hf_pipeline", "batch": 1, "model": a.model, "dtype": dtype, "max_tokens": a.max_tokens,
         **gpu_meta(), **stats(lat, a.requests * a.max_tokens, wall)}
    save(a.card, "hf_b1", r)


def make_llm(a, **kw):
    from vllm import LLM
    return LLM(model=a.model, dtype=pick_dtype(a.dtype), gpu_memory_utilization=a.gpu_util,
               seed=0, **kw)


def bench_vllm(a):
    from vllm import SamplingParams
    llm = make_llm(a, max_model_len=a.max_model_len)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, ignore_eos=True)
    llm.generate(PROMPTS, sp, use_tqdm=False)  # warm-up / CUDA graph capture
    for b in a.batches:
        lat, tokens = [], 0
        n_batches = max(1, a.requests // b) if b > 1 else a.requests
        t0 = time.perf_counter()
        for k in range(n_batches):
            prompts = [PROMPTS[(k * b + j) % len(PROMPTS)] for j in range(b)]
            s = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            e = time.perf_counter() - s
            lat += [e] * b
            tokens += sum(len(o.outputs[0].token_ids) for o in outs)
        wall = time.perf_counter() - t0
        r = {"engine": "vllm", "batch": b, "model": a.model, "dtype": pick_dtype(a.dtype),
             "max_tokens": a.max_tokens, "gpu_memory_utilization": a.gpu_util, **gpu_meta(),
             **stats(lat, tokens, wall)}
        save(a.card, f"vllm_b{b}", r)


def demo(a):
    from vllm import SamplingParams
    llm = make_llm(a, max_model_len=a.max_model_len)
    prompts = [a.prompt] if a.prompt else PROMPTS[:3]
    outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200))
    res = [{"prompt": o.prompt, "output": o.outputs[0].text} for o in outs]
    save(a.card, "vllm_demo", {"model": a.model, "results": res})


# ----------------------------------------------------------------------------- Part D.2: KV cache

def kv_bytes_per_token(model, dtype):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model)
    c = getattr(c, "text_config", c)
    kv_heads = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    nbytes = 2 if dtype in ("bfloat16", "float16") else 4
    return {"layers": c.num_hidden_layers, "kv_heads": kv_heads, "head_dim": head_dim, "dtype_bytes": nbytes,
            "bytes_per_token": 2 * c.num_hidden_layers * kv_heads * head_dim * nbytes,
            "max_position_embeddings": c.max_position_embeddings}


def kv_child(a):
    """Runs in a subprocess so vLLM's 'GPU KV cache size' log line can be captured from its output."""
    from vllm import SamplingParams
    llm = make_llm(a, max_model_len=a.max_model_len, max_num_seqs=a.kv_batch)
    if a.verify_len:
        ids = list(range(1000, 1000 + a.verify_len - a.max_tokens))
        ids = [ids] * a.kv_batch
        s = time.perf_counter()
        outs = llm.generate([{"prompt_token_ids": x} for x in ids],
                            SamplingParams(temperature=0.0, max_tokens=a.max_tokens, ignore_eos=True), use_tqdm=False)
        print(f"VERIFY_OK batch={len(outs)} len={a.verify_len} secs={time.perf_counter() - s:.2f}", flush=True)


def run_child(a, max_model_len, verify_len=0):
    cmd = [sys.executable, __file__, "kv-child", "--card", a.card, "--model", a.model, "--dtype", a.dtype,
           "--gpu-util", str(a.gpu_util), "--max-model-len", str(max_model_len), "--kv-batch", str(a.kv_batch),
           "--max-tokens", str(a.max_tokens), "--verify-len", str(verify_len)]
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    sys.stdout.write(out[-4000:])
    return p.returncode, out


def kv(a):
    dtype = pick_dtype(a.dtype)
    a.dtype = dtype
    geo = kv_bytes_per_token(a.model, dtype)
    probe_len = min(4096, geo["max_position_embeddings"])
    rc, out = run_child(a, probe_len)
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", out)
    if m:
        cap = int(m.group(1).replace(",", ""))
    else:  # older vLLM (V0) logs block count instead
        m = re.search(r"# GPU blocks:\s*(\d+)", out)
        cap = int(m.group(1)) * 16 if m else None
    if cap is None:
        sys.exit("could not find the KV cache size in vLLM's log; see output above")
    longest = min(cap // a.kv_batch, geo["max_position_embeddings"])
    curve, L = [], 256
    while L <= geo["max_position_embeddings"]:
        curve.append({"context": L, "kv_gb_at_batch": geo["bytes_per_token"] * L * a.kv_batch / 1e9,
                      "fits": L * a.kv_batch <= cap})
        L *= 2
    rc2, out2 = run_child(a, longest, verify_len=longest)
    save(a.card, f"kv_{a.model.split('/')[-1]}", {
        "model": a.model, "dtype": dtype, **smi(), "gpu_memory_utilization": a.gpu_util, "batch": a.kv_batch, **geo,
        "kv_capacity_tokens": cap, "kv_capacity_gb": cap * geo["bytes_per_token"] / 1e9,
        "longest_context_at_batch": longest,
        "limited_by": "model max_position_embeddings" if longest == geo["max_position_embeddings"] else "KV cache memory",
        "verified": "VERIFY_OK" in out2, "curve": curve})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["demo", "hf", "vllm", "kv", "kv-child"])
    ap.add_argument("--card", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--requests", type=int, default=64, help="requests per configuration")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--kv-batch", type=int, default=8)
    ap.add_argument("--verify-len", type=int, default=0)
    a = ap.parse_args()
    {"demo": demo, "hf": bench_hf, "vllm": bench_vllm, "kv": kv, "kv-child": kv_child}[a.mode](a)


if __name__ == "__main__":
    main()
