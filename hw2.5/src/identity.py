"""GPU identity, result directories, and JSON helpers. Every number is UUID-labelled."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_NVIDIA_SMI = None


def find_nvidia_smi() -> str:
    """Windows lab boxes often have nvidia-smi but not on PATH."""
    global _NVIDIA_SMI
    if _NVIDIA_SMI:
        return _NVIDIA_SMI
    found = shutil.which("nvidia-smi")
    candidates = [
        found,
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        "/usr/bin/nvidia-smi",
    ]
    for c in candidates:
        if c and Path(c).exists():
            _NVIDIA_SMI = c
            return _NVIDIA_SMI
    raise SystemExit(
        "nvidia-smi not found. On the Windows lab PC it is usually "
        r"C:\Windows\System32\nvidia-smi.exe — add it to PATH or use that PC's NVIDIA driver."
    )


def require_cuda():
    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "PyTorch is not installed. On the GPU lab machine install the CUDA "
            "build of torch, then: pip install numpy matplotlib pandas"
        ) from e
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU visible. HW2.5 must run on an RTX 4090 or RTX 5090 "
            "lab workstation, not on a laptop CPU/MPS."
        )
    return torch


def nvidia_smi(*args: str, timeout: int = 60) -> str:
    cmd = [find_nvidia_smi(), *args]
    try:
        proc = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError as e:
        raise SystemExit("nvidia-smi not found. This is not a NVIDIA GPU machine.") from e
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"nvidia-smi failed:\n{e.stderr or e.stdout}") from e
    return proc.stdout


def parse_smi_query(text: str) -> dict:
    def grab(pattern: str, default: str = "") -> str:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return m.group(1).strip() if m else default

    uuid = grab(r"GPU UUID\s*:\s*(GPU-[0-9a-fA-F-]+)")
    if not uuid:
        uuid = grab(r"^\s*GPU UUID\s*:\s*(\S+)")
    return {
        "name": grab(r"Product Name\s*:\s*(.+)"),
        "uuid": uuid,
        "driver_version": grab(r"Driver Version\s*:\s*([0-9.]+)"),
        "cuda_version": grab(r"CUDA Version\s*:\s*([0-9.]+)"),
        "vram_total": grab(r"FB Memory Usage[\s\S]*?Total\s*:\s*(.+)"),
        "power_limit": grab(r"(?:Power Limit|Enforced Power Limit)\s*:\s*(.+)"),
        "persistence_mode": grab(r"Persistence Mode\s*:\s*(\S+)"),
        "compute_mode": grab(r"Compute Mode\s*:\s*(.+)"),
        "gpu_clock_max": grab(r"Max Clocks[\s\S]*?Graphics\s*:\s*(.+)"),
        "mem_clock_max": grab(r"Max Clocks[\s\S]*?Memory\s*:\s*(.+)"),
    }


def gpu_info(device: int = 0) -> dict:
    torch = require_cuda()
    props = torch.cuda.get_device_properties(device)
    smi_q = nvidia_smi("-q", "-i", str(device))
    parsed = parse_smi_query(smi_q)
    uuid = parsed.get("uuid") or f"cuda-device-{device}"
    name = parsed.get("name") or torch.cuda.get_device_name(device)
    return {
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_index": device,
        "name": name,
        "uuid": uuid,
        "driver_version": parsed.get("driver_version", ""),
        "cuda_version_smi": parsed.get("cuda_version", ""),
        "cuda_version_torch": getattr(torch.version, "cuda", None),
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "vram_total_smi": parsed.get("vram_total", ""),
        "vram_total_bytes": int(props.total_memory),
        "power_limit": parsed.get("power_limit", ""),
        "capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "nvidia_smi_parsed": parsed,
    }


def gpu_slug(info: dict | None = None) -> str:
    info = info or gpu_info()
    name = re.sub(r"[^A-Za-z0-9]+", "", info["name"]) or "GPU"
    uuid_short = (info["uuid"] or "unknown").replace("GPU-", "")[:8]
    return f"{name}_{uuid_short}"


def results_dir(info: dict | None = None, create: bool = True) -> Path:
    override = os.environ.get("HW2_5_RESULTS_DIR")
    if override:
        d = Path(override)
    else:
        d = ROOT / "results" / gpu_slug(info)
    if create:
        d.mkdir(parents=True, exist_ok=True)
        (d / "figures").mkdir(exist_ok=True)
    return d


def figures_dir(info: dict | None = None) -> Path:
    d = results_dir(info) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def append_run_log(text: str, info: dict | None = None) -> None:
    d = results_dir(info)
    log = d / "RUN_LOG.txt"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with log.open("a") as f:
        f.write(f"\n----- {stamp} -----\n")
        f.write(text.rstrip() + "\n")
    # also keep a combined file at assignment root (appended, labelled)
    root_log = ROOT / "RUN_LOG.txt"
    with root_log.open("a") as f:
        slug = gpu_slug(info) if info else "unknown"
        f.write(f"\n===== {stamp}  uuid={info['uuid'] if info else '?'}  {slug} =====\n")
        f.write(text.rstrip() + "\n")


def dump_nvidia_smi_q(dest: Path, device: int = 0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = nvidia_smi("-q", "-i", str(device), timeout=90)
    dest.write_text(text)
    return dest
