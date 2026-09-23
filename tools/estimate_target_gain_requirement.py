#!/usr/bin/env python3
"""Cheap, oracle-assisted effective-gain diagnostic for one GLRT score.

The calculation holds the *single-observation* calibrated H0 gate and the
detector rank fixed, and scales only the target noncentrality.  It is useful
for ruling out small target-direction gains; it is not a multi-CPI simulator
or a deployable detector.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2


def _rows(path: Path, arm: str) -> dict[str, list[dict]]:
    out = {"calibration": [], "test": []}
    with (path / "records.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] == arm:
                out[row["split"]].append(row)
    if not out["calibration"] or not out["test"]:
        raise ValueError("both calibration and test records are required")
    return out


def _required_gain(lam: np.ndarray, threshold: float, dof: int,
                   target_pd: float) -> float:
    def mean_pd(gain: float) -> float:
        return float(np.mean(ncx2.sf(2.0 * threshold, dof, 2.0 * gain * lam)))
    if mean_pd(1.0) >= target_pd:
        return 1.0
    high = 2.0
    while mean_pd(high) < target_pd and high < 1e6:
        high *= 2.0
    if mean_pd(high) < target_pd:
        return float("inf")
    return float(brentq(lambda gain: mean_pd(gain) - target_pd, 1.0, high))


def run(path: Path, arm: str = "perfect_channel",
        targets: tuple[float, ...] = (0.5, 0.8)) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("target_dictionary") != "truth" or manifest.get("target_glrt_mode") != "centre":
        raise ValueError("diagnostic requires the truth-template centre GLRT")
    rows = _rows(path, arm)
    with (path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle) if row["arm"] == arm]
    if len(matches) != 1:
        raise ValueError("expected one summary row for the selected arm")
    threshold = float(matches[0]["threshold"])
    dofs = {int(row["dof_real"]) for split in rows.values() for row in split}
    if len(dofs) != 1 or next(iter(dofs)) <= 0:
        raise ValueError("the detector rank must be fixed and positive")
    dof = dofs.pop()
    cal_lam = np.asarray([float(row["ncp_unit"]) for row in rows["calibration"]])
    test_lam = np.asarray([float(row["ncp_unit"]) for row in rows["test"]])
    if np.any(cal_lam < 0) or np.any(test_lam < 0):
        raise ValueError("noncentrality must be nonnegative")

    def model_pd(lam: np.ndarray, gain: float = 1.0) -> float:
        return float(np.mean(ncx2.sf(2.0 * threshold, dof, 2.0 * gain * lam)))

    requirements = []
    for target_pd in targets:
        if not 0.0 < target_pd < 1.0:
            raise ValueError("target PD must lie in (0, 1)")
        gain = _required_gain(cal_lam, threshold, dof, target_pd)
        requirements.append({
            "target_mean_pd": target_pd,
            "calibration_effective_gain_linear": gain,
            "calibration_effective_gain_db": float(10.0 * np.log10(gain)),
            "heldout_model_pd_at_calibrated_gain": model_pd(test_lam, gain),
        })

    ideal_gate = float(chi2.isf(float(manifest["p_fa"]), dof) / 2.0)
    empirical_test_pd = float(np.mean([
        float(row["stat_h1"]) > threshold for row in rows["test"]
    ]))
    return {
        "protocol": "truth_centre_effective_gain_diagnostic_v1",
        "source": str(path.resolve()),
        "n_cal": len(cal_lam), "n_test": len(test_lam), "dof_real": dof,
        "ideal_chi2_gate": ideal_gate,
        "calibrated_h0_gate": threshold,
        "calibration_ncp_quantiles": np.quantile(
            cal_lam, [0.1, 0.5, 0.9]
        ).tolist(),
        "calibration_model_pd_at_baseline": model_pd(cal_lam),
        "heldout_model_pd_at_baseline": model_pd(test_lam),
        "heldout_empirical_pd_at_baseline": empirical_test_pd,
        "requirements": requirements,
        "interpretation": (
            "Scales target noncentrality with detector rank and calibrated H0 "
            "gate fixed. Does not model new CPI degrees of freedom, changing "
            "interference, changed geometry, or a new H0 threshold. Oracle "
            "truth is used only to diagnose resource scale."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
