#!/usr/bin/env bash
# Standalone nvidia-smi -q capture (Part A). Prefer scripts/run_all.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$ROOT/scripts/01_part_a_provenance.py" "$@"
