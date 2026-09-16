#!/usr/bin/env python3
"""Run the frozen target-local V1 contract tests and main evaluation.

This runner never writes to ``results/`` or ``results_release/`` by default.
It fixes the V1 preset, method roster, paired reference, and seed so that a
candidate result cannot accidentally be presented under the paper-canonical
protocol.

The archived label erasure at source 69f3300 denoted Gaussian replacement.
Pin the current semantic name here; current true erasure is a different model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "run_isac_sim.py"
METHODS = [
    "proposed_c2f_adaptive_pd",
    "exact_marginal_greedy",
    "sense_sinr",
]


def checked_run(cmd: list[str]) -> None:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results_target_local_v1"
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.mc < 1:
        parser.error("--mc must be at least one")
    if args.workers < 1:
        parser.error("--workers must be at least one")

    if not args.skip_tests:
        checked_run(
            [
                sys.executable,
                "-m",
                "unittest",
                "tests.test_target_local_v1",
                "tests.test_canonical_consistency",
            ]
        )

    cmd = [
        sys.executable,
        str(RUN),
        "--mode",
        "main",
        "--preset",
        "target-local-v1",
        "--set",
        "detect.comm_error_model=gaussian_replacement",
        "--mc",
        str(args.mc),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--methods",
        *METHODS,
        "--paired-reference",
        "proposed_c2f_adaptive_pd",
        "--out",
        str(args.out),
        "--quiet",
    ]
    if args.no_plots:
        cmd.append("--no-plots")
    checked_run(cmd)
    print(f"Target-local V1 evaluation saved under {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
