#!/usr/bin/env python3
"""Cross-platform driver. On the Windows lab PC:  python scripts\\run_all.py"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(py: str, script: Path, extra: list[str]) -> None:
    cmd = [py, str(script), *extra]
    print("\n>>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HW2.5 GPU lab driver. Run this on the RTX 4090 / 5090 workstation you reserved."
    )
    parser.add_argument("--quick", action="store_true", help="smaller sizes, ~60s thermal")
    parser.add_argument("--skip-thermal", action="store_true")
    parser.add_argument("--parts", default="A,B,C,D,E,F", help="e.g. A,B,C")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    py = sys.executable
    print("hw2_5 root:", ROOT)
    print("python:", py)
    print("parts:", args.parts, "quick:", args.quick, "skip_thermal:", args.skip_thermal)

    check = subprocess.run([py, str(ROOT / "scripts" / "check_env.py")])
    if check.returncode != 0:
        raise SystemExit(check.returncode)

    extra = ["--device", str(args.device)]
    if args.quick:
        extra.append("--quick")
    parts = {p.strip().upper() for p in args.parts.split(",") if p.strip()}

    mapping = {
        "A": ROOT / "scripts" / "01_part_a_provenance.py",
        "B": ROOT / "scripts" / "02_part_b_matmul.py",
        "C": ROOT / "scripts" / "03_part_c_roofline.py",
        "D": ROOT / "scripts" / "04_part_d_attention.py",
        "E": ROOT / "scripts" / "05_part_e_thermal.py",
    }
    for key in ["A", "B", "C", "D"]:
        if key in parts:
            run(py, mapping[key], extra)
    if "E" in parts:
        if args.skip_thermal:
            print("===== PART E skipped (--skip-thermal) =====", flush=True)
        else:
            run(py, mapping["E"], extra)
    if "F" in parts:
        run(py, ROOT / "scripts" / "06_compile_report.py", [])

    print(
        "\nDone. Copy the results\\<GPU>_<uuid> folder off this PC "
        "(USB / zip / git). Fill reservations\\GPU_HOURS.md before you log out."
    )


if __name__ == "__main__":
    main()
