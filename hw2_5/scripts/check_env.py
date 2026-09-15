#!/usr/bin/env python3
"""Fail fast if this is not a CUDA GPU box."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import banner  # noqa: E402


def main() -> None:
    print(banner())
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch missing. Use the lab CUDA environment.")
    print("torch", torch.__version__)
    print("cuda available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU. Stop. Copy hw2_5 onto the 4090 or 5090 workstation.")
    print("GPU", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    print("ok")


if __name__ == "__main__":
    main()
