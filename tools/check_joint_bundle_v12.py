#!/usr/bin/env python3
"""Check V1.2 feasibility/calibration and exact-pricing oracle agreement."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def check(condition: bool, message: str, failures: list[str]) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {message}")
    if not condition: failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path("results_joint_bundle_v12"))
    parser.add_argument("--target-pfa", type=float, default=0.05)
    parser.add_argument("--max-pfa-error", type=float, default=0.01)
    args = parser.parse_args()
    failures: list[str] = []

    main_path = args.root / "main" / "main.csv"
    oracle_path = args.root / "oracle" / "oracle_gap.csv"
    check(main_path.is_file(), "main comparison exists", failures)
    check(oracle_path.is_file(), "small-system oracle audit exists", failures)
    if main_path.is_file():
        with main_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        methods = {row["method"] for row in rows}
        required = {"joint_bundle_cg", "fixed_fusion_bundle", "local_only_bundle"}
        check(required <= methods, "factorial bundle baselines are present", failures)
        for row in rows:
            pfa = float(row["P_FA_active"])
            low = float(row["P_FA_cluster_ci95_low"])
            high = float(row["P_FA_cluster_ci95_high"])
            check(
                abs(pfa - args.target_pfa) <= args.max_pfa_error
                and low <= args.target_pfa <= high,
                f"{row['method']} realised P_FA is calibrated",
                failures,
            )
    if oracle_path.is_file():
        with oracle_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        check(bool(rows), "oracle audit contains trials", failures)
        check(
            bool(rows) and all(float(row["exact_lexicographic_match"]) == 1.0 for row in rows),
            "exact-pricing CG matches the joint oracle on every trial",
            failures,
        )

    if failures:
        print(f"V1.2 correctness gate FAILED ({len(failures)} checks).")
        return 1
    print("V1.2 correctness gate PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
