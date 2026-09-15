#!/usr/bin/env python3
"""Gate realised active-target P_FA for every method in a result table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_csv", type=Path)
    parser.add_argument("--target", type=float, default=0.05)
    parser.add_argument("--max-absolute-error", type=float, default=0.01)
    parser.add_argument("--methods", nargs="*")
    args = parser.parse_args()

    with args.main_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    requested = set(args.methods or ())
    if requested:
        rows = [row for row in rows if row.get("method") in requested]
    if not rows:
        print("[FAIL] no matching method rows")
        return 1

    failures: list[str] = []
    for row in rows:
        method = row["method"]
        pfa_text = row.get("P_FA_active") or row.get("P_FA")
        if pfa_text is None:
            print(f"[FAIL] {method}: missing P_FA/P_FA_active column")
            failures.append(method)
            continue
        pfa = float(pfa_text)
        error = abs(pfa - args.target)
        low_text = row.get("P_FA_cluster_ci95_low", "")
        high_text = row.get("P_FA_cluster_ci95_high", "")
        interval_ok = bool(low_text and high_text)
        if interval_ok:
            low, high = float(low_text), float(high_text)
            interval_ok = low <= args.target <= high
        error_ok = error <= args.max_absolute_error
        passed = interval_ok and error_ok
        print(
            f"[{'PASS' if passed else 'FAIL'}] {method}: "
            f"P_FA={pfa:.6f}, error={error:.6f}, "
            f"cluster_CI_covers_target={interval_ok}"
        )
        if not passed:
            failures.append(method)

    if failures:
        print(f"V1.2 realised-P_FA gate FAILED: {', '.join(failures)}")
        return 1
    print("V1.2 realised-P_FA gate PASSED for every checked method.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
