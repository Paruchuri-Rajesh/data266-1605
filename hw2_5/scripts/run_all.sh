#!/usr/bin/env bash
# Optional. Windows lab PCs should use run_full.bat / run_quick.bat instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec "${PYTHON:-python3}" "$ROOT/scripts/run_all.py" "$@"
