#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run every experiment used in the paper and collect results into results/.

The paper (SimulationResults.tex, Table I) specifies M=15, Q=10, UAV speed
30-60 m/s and target speed 50-90 m/s, with 1000 Monte-Carlo trials.  The
package defaults already match most of Table I; only the kinematic ranges
differ from the shipped defaults, so we override them here and run every
mode the paper cites:

    fig2  main          -> results/main/
    fig3  lambda-sweep  -> results/lambda-sweep/
    fig5  comm-sweep    -> results/comm-sweep/
    Table II fair-ablation + ablation -> results/fair-ablation/, results/ablation/
    DD ablation + C2F   -> results/dd-ablation/, results/c2f/
    fig6  robustness    -> results/robustness/  (3 axes)

Usage:
    python tools/rerun_paper.py --mc 1000
    python tools/rerun_paper.py --mc 20 --out results_preview
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "run_isac_sim.py"

# Paper Table I kinematic values (the only fields that differ from the
# package defaults).
PAPER_SET = [
    "geometry.uav_speed_min=30",
    "geometry.target_speed_min=50",
    "geometry.target_speed_max=90",
]

# (mode, extra argv).  Every mode uses the same paper parameter set + MC.
# Robustness is run per axis; each axis needs its own output subdirectory so
# the three runs don't overwrite each other's CSV.
MODES: List[Tuple[str, List[str]]] = [
    ("main", []),
    ("c2f", []),
    ("dd-ablation", []),
    ("fair-ablation", []),
    ("ablation", []),
    ("comm-sweep", []),
    ("lambda-sweep", []),
    ("robustness", ["--axis", "comm_model"]),
    ("robustness", ["--axis", "error_sigma"]),
    ("robustness", ["--axis", "residual_direct"]),
]


def run_one(mode: str, extra: List[str], mc: int, out: Path, seed: int) -> float:
    cmd = [
        sys.executable,
        str(RUN),
        "--mode", mode,
        "--mc", str(mc),
        "--seed", str(seed),
        "--out", str(out),
        "--quiet",
    ]
    for s in PAPER_SET:
        cmd += ["--set", s]
    cmd += extra
    t0 = time.time()
    print(f"\n===== {mode} {' '.join(extra)} =====", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[FAILED] {mode} {' '.join(extra)}", flush=True)
    elapsed = time.time() - t0
    print(f"[done] {mode} {' '.join(extra)} in {elapsed:.0f}s", flush=True)
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for mode, extra in MODES:
        # Robustness runs write to <out>/robustness_<axis>/ so the three axes
        # coexist instead of overwriting robustness.csv.
        mode_out = out
        if "--axis" in extra:
            axis = extra[extra.index("--axis") + 1]
            mode_out = out / f"robustness_{axis}"
        run_one(mode, extra, args.mc, mode_out, args.seed)

    print(f"\nAll modes finished in {(time.time() - t_start) / 60:.1f} min; "
          f"output in {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
