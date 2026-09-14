#!/usr/bin/env python3
"""Apply the preregistered confirmatory waveform-calibration criteria."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
REFERENCE = "proposed_c2f_adaptive_pd"
CANDIDATE = "proposed_c2f_adaptive_pd_calibrated"
SCENARIOS = ("ideal", "representative", "stress")


def require(condition: bool, message: str, failures: list[str]) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        failures.append(message)


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng([2026, seed])
    means = np.empty(10_000, dtype=float)
    for start in range(0, means.size, 200):
        stop = min(start + 200, means.size)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = np.mean(values[indices], axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_root",
        nargs="?",
        type=Path,
        default=ROOT / "results_waveform_v2_gate3",
    )
    parser.add_argument("--target-pfa", type=float, default=0.05)
    parser.add_argument("--max-pfa-error", type=float, default=0.002)
    parser.add_argument("--pd-noninferiority-margin", type=float, default=0.005)
    parser.add_argument("--max-calibrated-detection-ms", type=float, default=250.0)
    args = parser.parse_args()

    failures: list[str] = []
    display: list[tuple[object, ...]] = []
    closer_count = 0
    for scenario_index, scenario in enumerate(SCENARIOS):
        result_dir = args.result_root / scenario / "main"
        main_path = result_dir / "main.csv"
        trial_path = result_dir / "trials.csv"
        require(main_path.is_file() and trial_path.is_file(),
                f"{scenario}: confirmatory outputs exist", failures)
        if not (main_path.is_file() and trial_path.is_file()):
            continue

        with main_path.open(newline="", encoding="utf-8-sig") as handle:
            summary = {row["method"]: row for row in csv.DictReader(handle)}
        with trial_path.open(newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle))
        ref_rows = sorted(
            (row for row in records if row["method"] == REFERENCE),
            key=lambda row: int(row["trial"]),
        )
        cand_rows = sorted(
            (row for row in records if row["method"] == CANDIDATE),
            key=lambda row: int(row["trial"]),
        )
        require(len(ref_rows) >= 1000 and len(cand_rows) == len(ref_rows),
                f"{scenario}: 1000 complete paired trials are recorded", failures)
        if len(ref_rows) < 1000 or len(cand_rows) != len(ref_rows):
            continue

        ref_pd = np.asarray([float(row["detected_fraction"]) for row in ref_rows])
        cand_pd = np.asarray([float(row["detected_fraction"]) for row in cand_rows])
        cand_pfa_trials = np.asarray([
            float(row["false_alarm_fraction"]) for row in cand_rows
        ])
        pd_delta_ci = bootstrap_ci(cand_pd - ref_pd, 9300 + scenario_index)
        pfa_ci = bootstrap_ci(cand_pfa_trials, 9400 + scenario_index)

        ref_pfa = float(summary[REFERENCE]["P_FA"])
        cand_pfa = float(summary[CANDIDATE]["P_FA"])
        ref_error = abs(ref_pfa - args.target_pfa)
        cand_error = abs(cand_pfa - args.target_pfa)
        if cand_error < ref_error:
            closer_count += 1

        require(pfa_ci[0] <= args.target_pfa <= pfa_ci[1],
                f"{scenario}: calibrated P_FA cluster interval covers target", failures)
        require(cand_error <= args.max_pfa_error,
                f"{scenario}: calibrated P_FA meets absolute accuracy", failures)
        require(cand_error <= ref_error + 0.0005,
                f"{scenario}: calibrated P_FA is not materially less accurate than V1", failures)
        require(pd_delta_ci[0] >= -args.pd_noninferiority_margin,
                f"{scenario}: calibrated P_D passes the confirmatory noninferiority bound", failures)

        runtime_ms = float(summary[CANDIDATE]["detection_runtime_mean_ms"])
        require(runtime_ms <= args.max_calibrated_detection_ms,
                f"{scenario}: calibrated detector meets the absolute runtime budget", failures)
        display.append((
            scenario,
            float(summary[REFERENCE]["P_D"]),
            float(summary[CANDIDATE]["P_D"]),
            ref_pfa,
            cand_pfa,
            pfa_ci,
            pd_delta_ci,
            runtime_ms,
        ))

    require(closer_count >= 2,
            "calibrated P_FA is closer to target in at least two of three scenarios",
            failures)

    if display:
        print("\nscenario         V1_PD  cal_PD  V1_PFA  cal_PFA  delta_PD_CI95       det_ms")
        for row in display:
            print(
                f"{row[0]:<16s} {row[1]:.4f}  {row[2]:.4f}  {row[3]:.4f}   "
                f"{row[4]:.4f}   [{row[6][0]:+.4f},{row[6][1]:+.4f}]  {row[7]:7.3f}"
            )

    if failures:
        print(f"\nWaveform V2 Gate 3 FAILED ({len(failures)} checks).")
        return 1
    print("\nWaveform V2 Gate 3 PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
