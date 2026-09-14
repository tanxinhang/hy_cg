#!/usr/bin/env python3
"""Run the frozen three-scenario MC=1000 waveform confirmation gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCENARIO_RUNNER = ROOT / "tools" / "run_waveform_v2_gate2a.py"
GATE2B_CHECKER = ROOT / "tools" / "check_waveform_v2_gate2b.py"
GATE3_CHECKER = ROOT / "tools" / "check_waveform_v2_gate3.py"


def checked_run(cmd: list[str]) -> None:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results_waveform_v2_gate3"
    )
    args = parser.parse_args()
    if args.mc < 1000:
        parser.error("Gate 3 requires --mc >= 1000")

    checked_run([
        sys.executable,
        str(SCENARIO_RUNNER),
        "--mc",
        str(args.mc),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--out",
        str(args.out),
    ])
    checked_run([
        sys.executable,
        str(GATE2B_CHECKER),
        str(args.out),
        "--min-trials",
        "1000",
        "--pd-noninferiority-margin",
        "0.005",
    ])
    checked_run([sys.executable, str(GATE3_CHECKER), str(args.out)])
    print(f"Waveform V2 Gate 3 evidence saved under {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
