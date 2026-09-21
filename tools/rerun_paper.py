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

PAPER_METHODS = [
    "proposed_c2f",
    "global_topk_deflection",
    "cost_aware_greedy",
    "exact_marginal_greedy",
    "sense_sinr",
    "all_neighbor",
]

# (mode, extra argv, output subdirectory).  Every mode uses the same paper
# parameter set + MC.  Robustness is run per axis, each into its own
# subdirectory, so the axes don't overwrite each other's CSV.
MODES: List[Tuple[str, List[str], str]] = [
    ("main", ["--methods", *PAPER_METHODS], "main"),
    # Control run under the conservative full-concurrent interference model,
    # so the gain from the active-set model is visible in the summary.
    ("main", ["--set", "comm.interference_model=full_concurrent",
              "--methods", *PAPER_METHODS], "main_full_concurrent"),
    ("c2f", [], "c2f"),
    ("dd-ablation", [], "dd-ablation"),
    ("fair-ablation", [], "fair-ablation"),
    ("ablation", [], "ablation"),
    ("comm-sweep", [], "comm-sweep"),
    ("lambda-sweep", [], "lambda-sweep"),
    ("robustness", ["--axis", "comm_model"], "robustness_comm_model"),
    ("robustness", ["--axis", "error_sigma"], "robustness_error_sigma"),
    ("robustness", ["--axis", "residual_direct"], "robustness_residual_direct"),
    # (``--axis direct_cancellation`` was removed together with the
    #  ``interference.direct_cancellation_db`` field it swept.)
    # Model-refinement experiments (§V).
    ("interference-consistency", [], "interference-consistency"),
    ("belief-mismatch", [], "belief-mismatch"),
    ("fbl-sweep", [], "fbl-sweep"),
    ("correlation-ablation", [], "correlation-ablation"),
    ("submodularity", [], "submodularity"),
    ("same-objective-gap", [], "same-objective-gap"),
]


def run_one(
    mode: str, extra: List[str], mc: int, out: Path, seed: int, workers: int
) -> float:
    cmd = [
        sys.executable,
        str(RUN),
        "--mode", mode,
        "--mc", str(mc),
        "--seed", str(seed),
        "--workers", str(workers),
        "--out", str(out),
        "--preset", "paper-canonical",
        "--quiet",
    ]
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
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel Monte-Carlo worker processes")
    parser.add_argument("--only", nargs="+", default=None, metavar="SUBDIR",
                        help="run only these output subdirectories (e.g. --only "
                             "lambda-sweep robustness_comm_model); the rest are "
                             "skipped so already-completed runs are not repeated")
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    selected = MODES
    if args.only:
        wanted = set(args.only)
        selected = [m for m in MODES if m[2] in wanted]
        missing = wanted - {m[2] for m in MODES}
        if missing:
            raise SystemExit(f"unknown subdir(s): {sorted(missing)}")
        print(f"running only: {[m[2] for m in selected]}")

    t_start = time.time()
    for mode, extra, subdir in selected:
        mode_out = out / subdir
        run_one(mode, extra, args.mc, mode_out, args.seed, args.workers)

    print(f"\nAll modes finished in {(time.time() - t_start) / 60:.1f} min; "
          f"output in {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
