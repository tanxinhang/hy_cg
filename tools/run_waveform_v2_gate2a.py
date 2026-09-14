#!/usr/bin/env python3
"""Run the MC=100 ideal/representative/stress waveform screening gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "tools" / "rerun_waveform_v2_phase1.py"
CHECKER = ROOT / "tools" / "check_waveform_v2_gate2a.py"

# Dimensionless diagnostic settings, fixed before seeing Gate-2a results.  They
# are sensitivity points, not fitted or claimed physical nominal values.
SCENARIOS = {
    "ideal": (0.0, 0.0, 0.0, 0.0, 0.0),
    "representative": (0.25, 0.10, 0.10, 0.05, -0.05),
    "stress": (0.50, 0.25, 0.20, 0.10, -0.10),
}


def checked_run(cmd: list[str]) -> None:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results_waveform_v2_gate2a"
    )
    parser.add_argument("--no-plots", action="store_true", default=True)
    args = parser.parse_args()
    if args.mc < 100:
        parser.error("Gate 2a requires --mc >= 100")
    if args.workers < 1:
        parser.error("--workers must be at least one")

    for name, values in SCENARIOS.items():
        clutter, multipath, unresolved, sync_delay, sync_doppler = values
        cmd = [
            sys.executable,
            str(RUNNER),
            "--mc",
            str(args.mc),
            "--seed",
            str(args.seed),
            "--workers",
            str(args.workers),
            "--out",
            str(args.out / name),
            "--skip-tests",
            "--no-plots",
            "--record-runtime",
            "--clutter-inr",
            str(clutter),
            "--multipath-inr",
            str(multipath),
            "--unresolved-target-inr",
            str(unresolved),
            "--sync-delay-bins",
            str(sync_delay),
            "--sync-doppler-bins",
            str(sync_doppler),
        ]
        checked_run(cmd)

    checked_run([sys.executable, str(CHECKER), str(args.out)])
    print(f"Waveform V2 Gate 2a evidence saved under {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
