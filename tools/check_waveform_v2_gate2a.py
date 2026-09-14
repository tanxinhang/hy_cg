#!/usr/bin/env python3
"""Check phase-1 MC screening without promoting it to confirmatory evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
REFERENCE = "proposed_c2f_adaptive_pd"
CANDIDATE = "proposed_c2f_adaptive_pd_calibrated"
SCENARIOS = {
    "ideal": (0.0, 0.0, 0.0, 0.0, 0.0),
    "representative": (0.25, 0.10, 0.10, 0.05, -0.05),
    "stress": (0.50, 0.25, 0.20, 0.10, -0.10),
}


def require(condition: bool, message: str, failures: list[str]) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        failures.append(message)


def paired_ci95(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values)) if values.size else 0.0
    half = (
        1.96 * float(np.std(values, ddof=1)) / math.sqrt(values.size)
        if values.size > 1 else 0.0
    )
    return mean, mean - half, mean + half


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_root",
        nargs="?",
        type=Path,
        default=ROOT / "results_waveform_v2_gate2a",
    )
    parser.add_argument("--target-pfa", type=float, default=0.05)
    parser.add_argument("--pfa-screen-tolerance", type=float, default=0.01)
    parser.add_argument("--pd-noninferiority-margin", type=float, default=0.03)
    args = parser.parse_args()

    failures: list[str] = []
    summaries: list[tuple[str, float, float, float, float, float]] = []
    for scenario, expected in SCENARIOS.items():
        result_dir = args.result_root / scenario / "main"
        config_path = result_dir / "config.json"
        main_path = result_dir / "main.csv"
        trial_path = result_dir / "trials.csv"
        require(config_path.is_file() and main_path.is_file() and trial_path.is_file(),
                f"{scenario}: config, summary, and trial records exist", failures)
        if not (config_path.is_file() and main_path.is_file() and trial_path.is_file()):
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        wi = config["waveform_impairments"]
        actual = (
            float(wi["clutter_inr"]),
            float(wi["multipath_inr"]),
            float(wi["unresolved_target_inr"]),
            float(wi["sync_delay_bins"]),
            float(wi["sync_doppler_bins"]),
        )
        require(bool(wi["enable"]), f"{scenario}: waveform adapter is enabled", failures)
        require(actual == expected, f"{scenario}: preregistered impairment tuple is recorded", failures)
        require(int(config["run"]["num_mc"]) >= 100,
                f"{scenario}: at least 100 trials are recorded", failures)

        with main_path.open(newline="", encoding="utf-8-sig") as handle:
            summary = {row["method"]: row for row in csv.DictReader(handle)}
        require({REFERENCE, CANDIDATE}.issubset(summary),
                f"{scenario}: V1 and calibrated candidate are present", failures)

        with trial_path.open(newline="", encoding="utf-8-sig") as handle:
            trial_rows = list(csv.DictReader(handle))
        by_key = {(int(row["trial"]), row["method"]): row for row in trial_rows}
        trials = sorted({trial for trial, method in by_key if method == REFERENCE})
        require(len(trials) >= 100, f"{scenario}: at least 100 paired trial rows exist", failures)

        identity_fields = (
            "selected_links",
            "selected_observations",
            "remote_reports",
            "local_observations",
            "overhead_bits",
            "overhead_delay_ms",
            "D_mean",
            "D_median",
        )
        identity_ok = True
        pd_diffs: list[float] = []
        for trial in trials:
            ref = by_key.get((trial, REFERENCE))
            cand = by_key.get((trial, CANDIDATE))
            if ref is None or cand is None:
                identity_ok = False
                continue
            identity_ok = identity_ok and all(
                ref[field] == cand[field] for field in identity_fields
            )
            pd_diffs.append(float(cand["detected_fraction"]) - float(ref["detected_fraction"]))
        require(identity_ok,
                f"{scenario}: selected sets and resource metrics match V1 trial by trial",
                failures)

        if {REFERENCE, CANDIDATE}.issubset(summary):
            ref_pd = float(summary[REFERENCE]["P_D"])
            cand_pd = float(summary[CANDIDATE]["P_D"])
            ref_pfa = float(summary[REFERENCE]["P_FA"])
            cand_pfa = float(summary[CANDIDATE]["P_FA"])
            mean_delta, delta_low, delta_high = paired_ci95(np.asarray(pd_diffs))
            require(abs(cand_pfa - args.target_pfa) <= args.pfa_screen_tolerance,
                    f"{scenario}: calibrated P_FA is within screening tolerance",
                    failures)
            require(
                abs(cand_pfa - args.target_pfa)
                <= abs(ref_pfa - args.target_pfa) + 0.005,
                f"{scenario}: calibration is not materially farther from target than V1",
                failures,
            )
            require(delta_low >= -args.pd_noninferiority_margin,
                    f"{scenario}: paired P_D lower bound exceeds the screening noninferiority margin",
                    failures)
            summaries.append((scenario, ref_pd, cand_pd, ref_pfa, cand_pfa, mean_delta))

    if summaries:
        print("\nscenario         V1_PD   cal_PD  V1_PFA  cal_PFA  cal-minus-V1_PD")
        for row in summaries:
            print(f"{row[0]:<16s} {row[1]:.4f}  {row[2]:.4f}  {row[3]:.4f}   {row[4]:.4f}   {row[5]:+.4f}")

    if failures:
        print(f"\nWaveform V2 Gate 2a FAILED ({len(failures)} checks).")
        return 1
    print("\nWaveform V2 Gate 2a PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
