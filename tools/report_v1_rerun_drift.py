#!/usr/bin/env python3
"""Classify a V1 rerun against an earlier snapshot: selection path vs detection.

Motivation
----------
The V1 release drifted once already, and the drift was initially mis-attributed
to a threshold swap when the real cause was the detection stage moving from a
shared sequential RNG stream to per-link keyed streams.  The signature of that
change is precise and reusable as a test:

    selection-path columns must be bit-identical
    detection-count columns are allowed to move

If a selection-path column moves, something changed the physical/optimisation
model rather than the sampling of the detector's soft statistic -- and that is a
far more serious event than a small ``P_D`` shift.  Run this before believing any
"the rerun is just sampling noise" claim.

Usage::

    python tools/report_v1_rerun_drift.py \
        --old archive/results_target_local_v1_pre_rngfix_2026-09-16 \
        --new results_target_local_v1

Exit code is 1 when a selection-path column differs, else 0.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent.parent

# Columns that are fully determined by geometry, feasibility, the selector and
# the reporting MAC.  None of them sample the detector's soft statistic, so a
# rerun that only changed the detector RNG stream must reproduce them exactly.
SELECTION_PATH = {
    "B_mean_bits", "B_std_bits", "T_mean_ms", "T_std_ms",
    "selected_links_mean", "selected_links_std",
    "selected_observations_mean", "selected_observations_std",
    "active_target_ratio_mean", "feasible_target_ratio_mean",
    "feasible_links_mean", "comm_feasible_edge_ratio_mean",
    "selected_rate_mean_mbps", "selected_rate_min_mbps_mean",
    "selected_rate_p10_mbps_mean", "selected_chi_mean",
    "selected_chi_min_mean", "selected_chi_p10_mean",
    "selected_gamma_comm_mean_db", "selected_rate_satisfaction_ratio_mean",
    "selected_chi_ge_min_ratio_mean",
    "D_mean", "D_median", "D_p10", "D_p90", "D_per_kbit", "D_per_ms",
    "fine_eval_full_mean", "fine_eval_c2f_mean", "belief_capture_rate_mean",
    "worst_target_D_mean",
}

# Columns derived from the sampled detection decision.  These are expected to
# move when the detector's sampling path changes.
DETECTION = {
    "P_D", "P_D_ci95_low", "P_D_ci95_high", "P_D_ci95_half_width",
    "P_D_per_kbit", "P_D_per_ms",
    "P_FA", "P_FA_ci95_low", "P_FA_ci95_high", "P_FA_ci95_half_width",
    "P_FA_overall", "P_FA_overall_ci95_low", "P_FA_overall_ci95_high",
    "P_FA_overall_ci95_half_width",
    "all_targets_satisfied_prob", "worst_target_satisfied_prob",
    "actual_mean_target_P_D", "actual_worst_target_P_D",
    "actual_best_target_P_D",
    "paired_proposed_delta_P_D", "paired_proposed_delta_ci95_low",
    "paired_proposed_delta_ci95_high",
    "paired_reference_delta_P_D", "paired_reference_delta_ci95_low",
    "paired_reference_delta_ci95_high",
}

# Condition/identity columns: never compared numerically.
KEY_COLUMNS = ("method", "experiment", "condition", "fusion_rule", "error_level",
               "R_min_mbps", "area_xy_m", "lambda_c", "n_block", "M", "Q",
               "belief_sigma_pos_m", "belief_sigma_vel_mps", "paired_reference_method")


def classify(column: str) -> str:
    if column in SELECTION_PATH:
        return "selection"
    if column in DETECTION:
        return "detection"
    return "other"


def key_of(row: Dict[str, str]) -> Tuple[str, ...]:
    return tuple(row.get(c, "") for c in KEY_COLUMNS if c in row)


def compare_csv(old: Path, new: Path) -> Dict[str, object]:
    with old.open(encoding="utf-8", newline="") as handle:
        old_rows = {key_of(r): r for r in csv.DictReader(handle)}
    with new.open(encoding="utf-8", newline="") as handle:
        new_rows = {key_of(r): r for r in csv.DictReader(handle)}

    common = sorted(set(old_rows) & set(new_rows))
    missing = sorted(set(old_rows) - set(new_rows))
    added = sorted(set(new_rows) - set(old_rows))

    buckets: Dict[str, List[str]] = {"selection": [], "detection": [], "other": []}
    worst: Dict[str, float] = {}

    if common:
        columns = [c for c in old_rows[common[0]]
                   if c not in KEY_COLUMNS and c in new_rows[common[0]]]
        for column in columns:
            kind = classify(column)
            max_abs = 0.0
            for key in common:
                try:
                    a = float(old_rows[key][column])
                    b = float(new_rows[key][column])
                except (TypeError, ValueError):
                    if old_rows[key][column] != new_rows[key][column]:
                        max_abs = float("inf")
                    continue
                max_abs = max(max_abs, abs(a - b))
            worst[column] = max_abs
            if max_abs > 0.0:
                buckets[kind].append(column)

    return {
        "common": len(common), "missing": missing, "added": added,
        "buckets": buckets, "worst": worst,
    }


def iter_pairs(old_root: Path, new_root: Path) -> Iterable[Tuple[str, Path, Path]]:
    for old in sorted(old_root.rglob("*.csv")):
        rel = old.relative_to(old_root)
        new = new_root / rel
        # ``main/trials.csv`` has no archived counterpart.
        if new.exists():
            yield str(rel).replace("\\", "/"), old, new


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    failures: List[str] = []
    checked = 0
    for label, old, new in iter_pairs(args.old, args.new):
        checked += 1
        result = compare_csv(old, new)
        sel = result["buckets"]["selection"]
        det = result["buckets"]["detection"]
        oth = result["buckets"]["other"]
        print(f"\n=== {label} ===")
        print(f"  rows matched: {result['common']}  "
              f"(missing {len(result['missing'])}, added {len(result['added'])})")
        for kind, names, head in (("selection", sel, 8), ("detection", det, 12),
                                  ("other", oth, 8)):
            if names:
                shown = ", ".join(names[:head])
                more = "" if len(names) <= head else f" (+{len(names)-head} more)"
                print(f"  [{kind}] {len(names)} differ: {shown}{more}")
            else:
                print(f"  [{kind}] all identical")
        if args.verbose and sel:
            for column in sel:
                print(f"    selection drift {column}: {result['worst'][column]:.6g}")
        if sel:
            failures.append(f"{label}: selection-path columns moved ({', '.join(sel)})")
        if result["missing"]:
            failures.append(f"{label}: {len(result['missing'])} rows missing in rerun")

    print(f"\n{checked} CSV file(s) compared.")
    if failures:
        print("\nFAIL — the rerun is not detector-sampling-only:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("PASS — selection path is bit-identical; only detection counts moved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
