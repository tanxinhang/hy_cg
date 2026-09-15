#!/usr/bin/env python3
"""Apply the preregistered V1.2 performance-promotion criteria."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COMPARATORS = (
    "fixed_fusion_bundle",
    "proposed_c2f_adaptive_pd",
    "sense_sinr_budgeted",
    "local_only_bundle",
)


def require(condition: bool, message: str, failures: list[str]) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {message}")
    if not condition: failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path("results_joint_bundle_v12"))
    parser.add_argument("--minimum-mc", type=int, default=200)
    parser.add_argument("--target-pfa", type=float, default=0.05)
    parser.add_argument("--max-pfa-error", type=float, default=0.01)
    args = parser.parse_args()
    failures: list[str] = []

    with (args.root / "main" / "main.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    require("joint_bundle_cg" in rows, "joint method is present", failures)
    require(all(method in rows for method in COMPARATORS),
            "all promotion comparators are present", failures)
    if "joint_bundle_cg" not in rows:
        return 1

    candidate = rows["joint_bundle_cg"]
    mc = int(float(candidate.get("mc", 0)))
    require(mc >= args.minimum_mc,
            f"independent Monte Carlo count is at least {args.minimum_mc}", failures)
    for method, row in rows.items():
        pfa = float(row["P_FA_active"])
        low = float(row["P_FA_cluster_ci95_low"])
        high = float(row["P_FA_cluster_ci95_high"])
        require(
            abs(pfa - args.target_pfa) <= args.max_pfa_error
            and low <= args.target_pfa <= high,
            f"{method} active-target P_FA is calibrated",
            failures,
        )
    for comparator in COMPARATORS:
        if comparator not in rows:
            continue
        row = rows[comparator]
        lower = float(row["paired_reference_delta_ci95_low"])
        require(
            lower > 0.0,
            f"joint bundle has a positive paired P_D lower bound versus {comparator}",
            failures,
        )

    if failures:
        print(f"V1.2 promotion gate: NO-GO ({len(failures)} failed criteria).")
        return 1
    print("V1.2 promotion gate: GO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
