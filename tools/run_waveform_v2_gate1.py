#!/usr/bin/env python3
"""Generate and validate phase-1 waveform statistical evidence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "run_isac_sim.py"
CHECK = ROOT / "tools" / "check_waveform_v2_gate1.py"


def checked_run(cmd: list[str]) -> None:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results_waveform_v2_gate1"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    for mode in ("waveform-check", "waveform-detection", "waveform-detection-grid"):
        cmd = [
            sys.executable,
            str(RUN),
            "--mode",
            mode,
            "--preset",
            "target-local-waveform-v2-phase1",
            "--seed",
            str(args.seed),
            "--out",
            str(args.out),
        ]
        if not args.plots:
            cmd.append("--no-plots")
        checked_run(cmd)

    grid_csv = (
        args.out
        / "waveform-detection-grid"
        / "waveform-detection-grid.csv"
    )
    checked_run([sys.executable, str(CHECK), str(grid_csv)])
    print(f"Waveform V2 Gate 1 evidence saved under {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
