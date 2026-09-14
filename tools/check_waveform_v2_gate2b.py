#!/usr/bin/env python3
"""Check MC=200 phase-1 noninferiority, cluster intervals, and runtime."""

from __future__ import annotations

import argparse
import csv
import math
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
    for start in range(0, means.size, 250):
        stop = min(start + 250, means.size)
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
        default=ROOT / "results_waveform_v2_gate2b",
    )
    parser.add_argument("--target-pfa", type=float, default=0.05)
    parser.add_argument("--min-trials", type=int, default=200)
    parser.add_argument("--pd-noninferiority-margin", type=float, default=0.015)
    parser.add_argument("--max-calibrated-detection-ms", type=float, default=250.0)
    args = parser.parse_args()

    failures: list[str] = []
    display_rows: list[tuple[object, ...]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        result_dir = args.result_root / scenario / "main"
        main_path = result_dir / "main.csv"
        trial_path = result_dir / "trials.csv"
        require(main_path.is_file() and trial_path.is_file(),
                f"{scenario}: summary and trial records exist", failures)
        if not (main_path.is_file() and trial_path.is_file()):
            continue

        with main_path.open(newline="", encoding="utf-8-sig") as handle:
            summary = {row["method"]: row for row in csv.DictReader(handle)}
        with trial_path.open(newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle))
        grouped = {
            method: sorted(
                (row for row in records if row["method"] == method),
                key=lambda row: int(row["trial"]),
            )
            for method in (REFERENCE, CANDIDATE)
        }
        ref_rows, cand_rows = grouped[REFERENCE], grouped[CANDIDATE]
        require(len(ref_rows) >= args.min_trials and len(cand_rows) == len(ref_rows),
                f"{scenario}: at least {args.min_trials} complete paired trials are recorded", failures)
        if not ref_rows or len(ref_rows) != len(cand_rows):
            continue

        identity_fields = (
            "selected_links", "selected_observations", "remote_reports",
            "local_observations", "overhead_bits", "overhead_delay_ms",
            "D_mean", "D_median",
        )
        require(
            all(
                int(ref["trial"]) == int(cand["trial"])
                and all(ref[field] == cand[field] for field in identity_fields)
                for ref, cand in zip(ref_rows, cand_rows)
            ),
            f"{scenario}: phase-1 selection and resources equal V1 trial by trial",
            failures,
        )

        ref_pd = np.asarray([float(row["detected_fraction"]) for row in ref_rows])
        cand_pd = np.asarray([float(row["detected_fraction"]) for row in cand_rows])
        ref_pfa = np.asarray([float(row["false_alarm_fraction"]) for row in ref_rows])
        cand_pfa = np.asarray([float(row["false_alarm_fraction"]) for row in cand_rows])
        pd_delta = cand_pd - ref_pd
        pd_ci = bootstrap_ci(pd_delta, 8200 + scenario_index)
        cand_pd_ci = bootstrap_ci(cand_pd, 8300 + scenario_index)
        cand_pfa_ci = bootstrap_ci(cand_pfa, 8400 + scenario_index)

        require(pd_ci[0] >= -args.pd_noninferiority_margin,
                f"{scenario}: paired cluster P_D lower bound passes noninferiority",
                failures)
        require(cand_pfa_ci[0] <= args.target_pfa <= cand_pfa_ci[1],
                f"{scenario}: target P_FA lies inside the trial-cluster interval",
                failures)

        runtime = np.asarray([
            float(row.get("detection_runtime_ms", "nan")) for row in cand_rows
        ])
        runtime_ok = np.all(np.isfinite(runtime)) and np.all(runtime > 0.0)
        require(runtime_ok, f"{scenario}: calibrated detection runtime is recorded", failures)
        runtime_mean = float(np.mean(runtime)) if runtime_ok else float("nan")
        require(runtime_ok and runtime_mean <= args.max_calibrated_detection_ms,
                f"{scenario}: calibrated detection runtime stays within the provisional budget",
                failures)

        ref_runtime = np.asarray([
            float(row.get("detection_runtime_ms", "nan")) for row in ref_rows
        ])
        ratio = runtime_mean / max(float(np.mean(ref_runtime)), 1e-12)
        display_rows.append((
            scenario,
            float(np.mean(ref_pd)),
            float(np.mean(cand_pd)),
            cand_pd_ci,
            float(np.mean(cand_pfa)),
            cand_pfa_ci,
            pd_ci,
            runtime_mean,
            ratio,
        ))

        # Cross-check that summary export contains the newly required fields.
        require(
            all(field in summary[CANDIDATE] for field in (
                "P_D_cluster_ci95_low", "P_D_cluster_ci95_high",
                "P_FA_cluster_ci95_low", "P_FA_cluster_ci95_high",
                "detection_runtime_mean_ms", "detection_runtime_p90_ms",
            )),
            f"{scenario}: summary exports cluster and runtime fields",
            failures,
        )

    if display_rows:
        print("\nscenario         V1_PD  cal_PD  cal_PFA  delta_PD_CI95       det_ms  runtime_x")
        for row in display_rows:
            print(
                f"{row[0]:<16s} {row[1]:.4f}  {row[2]:.4f}  {row[4]:.4f}   "
                f"[{row[6][0]:+.4f},{row[6][1]:+.4f}]  {row[7]:7.3f}  {row[8]:8.2f}"
            )

    if failures:
        print(f"\nWaveform V2 Gate 2b FAILED ({len(failures)} checks).")
        return 1
    print("\nWaveform V2 Gate 2b PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
