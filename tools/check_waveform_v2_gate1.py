#!/usr/bin/env python3
"""Validate the phase-1 waveform detection grid with simultaneous tolerances."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import NormalDist


ROOT = Path(__file__).resolve().parent.parent


def require(condition: bool, message: str, failures: list[str]) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=(
            ROOT
            / "results_waveform_v2_gate1"
            / "waveform-detection-grid"
            / "waveform-detection-grid.csv"
        ),
    )
    parser.add_argument("--target-pfa", type=float, default=0.05)
    parser.add_argument("--family-alpha", type=float, default=0.05)
    parser.add_argument("--min-cases", type=int, default=24)
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(f"missing grid CSV: {args.csv_path}")
    if not 0.0 < args.target_pfa < 1.0:
        parser.error("--target-pfa must lie in (0, 1)")
    if not 0.0 < args.family_alpha < 1.0:
        parser.error("--family-alpha must lie in (0, 1)")

    with args.csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    failures: list[str] = []
    require(len(rows) >= args.min_cases,
            f"grid contains at least {args.min_cases} cases", failures)
    if not rows:
        return 1

    # Two empirical checks per case.  Bonferroni control makes the gate a
    # simultaneous family-wise test instead of 2N unrelated pointwise tests.
    test_count = max(2 * len(rows), 1)
    z_sim = NormalDist().inv_cdf(
        1.0 - args.family_alpha / (2.0 * test_count)
    )
    pfa_failures: list[int] = []
    pd_failures: list[int] = []
    max_pfa_error = 0.0
    max_pd_error = 0.0

    numeric_fields = (
        "raw_gamma",
        "gamma_effective",
        "captured_energy",
        "local_window_energy",
        "calibrated_exact_pd",
        "calibrated_empirical_pd",
        "calibrated_exact_pfa",
        "calibrated_empirical_pfa",
        "n_trials",
    )
    finite = True
    for index, row in enumerate(rows):
        values = {field: float(row[field]) for field in numeric_fields}
        finite = finite and all(math.isfinite(value) for value in values.values())
        n_trials = max(int(values["n_trials"]), 1)

        empirical_pfa = values["calibrated_empirical_pfa"]
        pfa_error = abs(empirical_pfa - args.target_pfa)
        pfa_se = math.sqrt(
            args.target_pfa * (1.0 - args.target_pfa) / n_trials
        )
        pfa_tolerance = z_sim * pfa_se + 1.0 / n_trials
        max_pfa_error = max(max_pfa_error, pfa_error)
        if pfa_error > pfa_tolerance:
            pfa_failures.append(index)

        exact_pd = values["calibrated_exact_pd"]
        empirical_pd = values["calibrated_empirical_pd"]
        pd_error = abs(empirical_pd - exact_pd)
        pd_se = math.sqrt(max(exact_pd * (1.0 - exact_pd), 0.0) / n_trials)
        pd_tolerance = z_sim * pd_se + 1.0 / n_trials
        max_pd_error = max(max_pd_error, pd_error)
        if pd_error > pd_tolerance:
            pd_failures.append(index)

    require(finite, "all gate metrics are finite", failures)
    require(not pfa_failures,
            f"all empirical P_FA values pass simultaneous tolerance; failures={pfa_failures}",
            failures)
    require(not pd_failures,
            f"all empirical P_D values match the exact mixture within simultaneous tolerance; failures={pd_failures}",
            failures)
    require(
        all(abs(float(row["calibrated_exact_pfa"]) - args.target_pfa) <= 1e-8
            for row in rows),
        "every exact calibrated null tail equals the configured P_FA",
        failures,
    )

    require(min(float(row["report_success"]) for row in rows) <= 0.90,
            "grid includes a report-failure stress regime", failures)
    require(max(float(row["interference_gamma"]) for row in rows) > 0.0,
            "grid includes unresolved-target interference", failures)
    require(max(float(row["clutter_gamma"]) for row in rows) > 0.0,
            "grid includes clutter", failures)
    require(max(float(row["multipath_ratio"]) for row in rows) > 0.0,
            "grid includes multipath", failures)
    require(any(abs(float(row["sync_error_k"])) > 0.0
                or abs(float(row["sync_error_l"])) > 0.0 for row in rows),
            "grid includes synchronization mismatch", failures)
    require(len({float(row["raw_gamma"]) for row in rows}) >= 3,
            "grid includes at least three desired-SINR levels", failures)

    print(f"\nSimultaneous z threshold: {z_sim:.4f}")
    print(f"Maximum empirical P_FA error: {max_pfa_error:.6f}")
    print(f"Maximum empirical-exact P_D error: {max_pd_error:.6f}")
    if failures:
        print(f"\nWaveform V2 Gate 1 FAILED ({len(failures)} checks).")
        return 1
    print("\nWaveform V2 Gate 1 PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
