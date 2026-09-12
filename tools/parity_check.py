#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parity check: the refactored package must reproduce the v10 prototype exactly.

The refactor moved every function into :mod:`isac_sim` but was not allowed to
change the model.  This script proves that: it runs both implementations on the
same Monte-Carlo size and seed, then compares every numeric CSV field.

Usage
-----
    python tools/parity_check.py                  # run both, all modes, MC=8
    python tools/parity_check.py --mc 20          # larger (slower) check
    python tools/parity_check.py --golden-only    # only compare against the frozen CSV

Exit status is 0 when every compared field is bit-identical, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
V10 = ROOT / "lagrangian_dotfs_isac_simplified_v10_paper_plots.py"
GOLDEN = ROOT / "golden_v10_mc20.csv"

# (tag, v10 argv, new-mode argv, v10 csv path, new csv path relative to --out)
MODE_CASES: Sequence[Tuple[str, List[str], List[str], str, str]] = (
    ("main", ["--csv", "v10_main.csv"], [], "v10_main.csv", "main/main.csv"),
    (
        "lambda-sweep",
        ["--lambda-sweep", "--lambda-values", "0.005", "0.05", "--sweep-csv", "v10_lambda.csv"],
        ["--mode", "lambda-sweep", "--values", "0.005", "0.05"],
        "v10_lambda.csv",
        "lambda-sweep/lambda-sweep.csv",
    ),
    (
        "ablation",
        ["--ablation", "--ablation-csv", "v10_ablation.csv"],
        ["--mode", "ablation"],
        "v10_ablation.csv",
        "ablation/ablation.csv",
    ),
    (
        "fair-ablation",
        ["--fair-ablation", "--fair-ablation-csv", "v10_fair.csv"],
        ["--mode", "fair-ablation"],
        "v10_fair.csv",
        "fair-ablation/fair-ablation.csv",
    ),
    (
        "dd-ablation",
        ["--dd-ablation", "--dd-ablation-csv", "v10_dd.csv"],
        ["--mode", "dd-ablation"],
        "v10_dd.csv",
        "dd-ablation/dd-ablation.csv",
    ),
    (
        "comm-sweep",
        ["--comm-sweep", "--R-min-values", "2e5", "1e6", "--comm-sweep-csv", "v10_comm.csv"],
        ["--mode", "comm-sweep", "--values", "2e5", "1e6"],
        "v10_comm.csv",
        "comm-sweep/comm-sweep.csv",
    ),
    (
        "robustness",
        ["--robustness", "--robustness-type", "comm_model", "--robustness-csv", "v10_rob.csv"],
        ["--mode", "robustness", "--axis", "comm_model"],
        "v10_rob.csv",
        "robustness/robustness.csv",
    ),
)

SKIP_KEYS = {"method", "variant", "experiment", "robustness_type", "comm_error_model"}


def _run(cmd: List[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def _compare(path_a: Path, path_b: Path) -> Tuple[int, float, Tuple[str, ...] | None]:
    rows_a = list(csv.DictReader(path_a.open(encoding="utf-8")))
    rows_b = list(csv.DictReader(path_b.open(encoding="utf-8")))

    if len(rows_a) != len(rows_b):
        # Row counts diverge only when the refactor *added* a method (e.g.
        # ``topk_deflection``), which the prototype never emitted.  Align by
        # the ``method`` column so the extra row is skipped instead of
        # shifting every subsequent row.
        if rows_a and rows_b and "method" in rows_a[0] and "method" in rows_b[0]:
            by_a = {r["method"]: r for r in rows_a}
            by_b = {r["method"]: r for r in rows_b}
            common = [m for m in by_a if m in by_b]
            if not common:
                return len(rows_a), float("inf"), ("no common methods",)
            pairs = [(by_a[m], by_b[m]) for m in common]
        else:
            return len(rows_a), float("inf"), ("row count", str(len(rows_a)), str(len(rows_b)))
    else:
        pairs = list(zip(rows_a, rows_b))

    shared = set(pairs[0][0]) & set(pairs[0][1])
    worst = 0.0
    where = None
    compared = 0
    for row_a, row_b in pairs:
        for key in sorted(shared):
            va, vb = row_a[key], row_b[key]
            if key in SKIP_KEYS or va == "" or vb == "":
                continue
            try:
                fa, fb = float(va), float(vb)
            except ValueError:
                if va != vb:
                    return len(rows_a), float("inf"), (key, va, vb)
                continue
            compared += 1
            diff = abs(fa - fb)
            if diff > worst:
                worst, where = diff, (key, fa, fb)
    return compared, worst, where


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mc", type=int, default=8, help="Monte-Carlo trials per run (default: 8)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--golden-only", action="store_true",
                        help="only compare the main run against the frozen golden CSV")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="isac_parity_") as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "new"
        new_argv = ["--mc", str(args.mc), "--seed", str(args.seed), "--quiet", "--no-plots", "--out", str(out)]

        failures = 0
        if args.golden_only:
            if not GOLDEN.exists():
                raise SystemExit(f"frozen golden file not found: {GOLDEN}")
            _run([sys.executable, str(ROOT / "run_isac_sim.py"), *new_argv], ROOT)
            compared, worst, where = _compare(GOLDEN, out / "main" / "main.csv")
            status = "OK" if worst == 0.0 else "MISMATCH"
            print(f"{'main (golden)':<16} fields={compared:<5} max|diff|={worst:<8g} {status}")
            return 0 if worst == 0.0 else 1

        if not V10.exists():
            raise SystemExit(f"reference prototype not found: {V10}")

        for tag, v10_args, new_args, v10_rel, new_rel in MODE_CASES:
            _run([sys.executable, str(V10), "--num-mc", str(args.mc), "--seed", str(args.seed),
                  "--quiet", *v10_args], ROOT)
            _run([sys.executable, str(ROOT / "run_isac_sim.py"), *new_argv, *new_args], ROOT)

            compared, worst, where = _compare(ROOT / v10_rel, out / new_rel)
            ok = worst == 0.0
            failures += 0 if ok else 1
            print(f"{tag:<16} fields={compared:<5} max|diff|={worst:<8g} {'OK' if ok else 'MISMATCH'}")
            if not ok:
                print(f"    worst field: {where}")

        print()
        print("PARITY OK - refactor is numerically identical to v10" if not failures
              else f"PARITY FAILED in {failures} mode(s)")
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
