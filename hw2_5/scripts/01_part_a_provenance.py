#!/usr/bin/env python3
"""Part A — capture nvidia-smi -q and write UUID-labelled provenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.identity import (  # noqa: E402
    append_run_log,
    dump_nvidia_smi_q,
    gpu_info,
    results_dir,
    write_json,
)
from src.params import banner  # noqa: E402
from src.specs import identify_spec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    print(banner())
    info = gpu_info(args.device)
    out = results_dir(info)
    smi_path = dump_nvidia_smi_q(out / "nvidia_smi_q.txt", device=args.device)
    spec = identify_spec(info["name"])
    payload = {
        "part": "A",
        "gpu": info,
        "vendor_spec": spec,
        "nvidia_smi_q_file": str(smi_path.relative_to(ROOT)),
        "required_fields": {
            "uuid": info["uuid"],
            "driver_version": info["driver_version"],
            "cuda_version": info["cuda_version_smi"],
            "vram_capacity": info["vram_total_smi"] or f"{info['vram_total_bytes']} bytes",
            "power_limit": info["power_limit"],
        },
    }
    write_json(out / "part_a_provenance.json", payload)

    lines = [
        banner(),
        f"hostname: {info['hostname']}",
        f"GPU name: {info['name']}",
        f"GPU UUID: {info['uuid']}",
        f"driver: {info['driver_version']}",
        f"CUDA (nvidia-smi): {info['cuda_version_smi']}",
        f"CUDA (torch): {info['cuda_version_torch']}",
        f"torch: {info['torch_version']}",
        f"VRAM: {info['vram_total_smi']}  ({info['vram_total_bytes']} bytes)",
        f"power limit: {info['power_limit']}",
        f"capability: {info['capability']}",
        f"nvidia-smi -q written to: {smi_path}",
    ]
    if spec:
        lines += [
            f"architecture: {spec['architecture']}",
            f"memory type: {spec['memory_type']}",
            f"bandwidth: {spec['bandwidth_gbs']} GB/s",
            f"tensor core gen: {spec['tensor_core_gen']}",
            f"tensor precisions: {', '.join(spec['tensor_precisions'])}",
        ]
    else:
        lines.append(
            "WARNING: GPU name did not match RTX 4090 or RTX 5090. "
            "Record architecture from nvidia-smi and vendor docs by hand."
        )
    text = "\n".join(lines)
    print(text)
    append_run_log("PART A\n" + text, info)


if __name__ == "__main__":
    main()
