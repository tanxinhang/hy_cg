#!/usr/bin/env python3
"""Calibrate a frozen multi-receiver statistic on independent H0 scenes.

This is a fixed-subset gate check, not a learned association experiment.
Each input benchmark must cover the same scenes with one realization per
scene. No test score enters the threshold or the fusion definition.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def _read(path: Path, arm: str, target: int, boost: float) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["direct_error_scope"] != "scene":
        raise ValueError(f"{path}: direct error must be fixed within scene")
    if manifest["direct_mismatch_covariance_model"] != "sigma_point":
        raise ValueError(f"{path}: expected sigma_point covariance")
    if manifest["target_glrt_mode"] != "neighbourhood_max":
        raise ValueError(f"{path}: expected neighbourhood GLRT")
    if int(manifest["realisations_per_scene_target_receiver"]) != 1:
        raise ValueError(f"{path}: one realization per independent scene required")
    rows = {}
    with (path / "records.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["arm"] != arm or int(row["target"]) != target or
                    float(row["direct_gain_boost_db"]) != boost):
                continue
            key = (row["split"], int(row["scene_id"]))
            if key in rows:
                raise ValueError(f"{path}: duplicate score for {key}")
            rows[key] = row
    if not rows:
        raise ValueError(f"{path}: no matching records")
    return {"manifest": manifest, "rows": rows,
            "receiver": int(next(iter(rows.values()))["receiver"])}


def _auc(h0: np.ndarray, h1: np.ndarray) -> float:
    ranks = rankdata(np.r_[h0, h1], method="average")
    n0, n1 = len(h0), len(h1)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def run(paths: list[Path], *, arm: str = "tp_uic_full", target: int = 1,
        boost: float = 30.0, alpha: float = 0.05) -> dict:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    sources = [_read(path, arm, target, boost) for path in paths]
    receivers = [s["receiver"] for s in sources]
    if len(set(receivers)) != len(receivers):
        raise ValueError("receiver inputs must be distinct")
    baseline = sources[0]["manifest"]
    common_fields = (
        "master_seed", "calibration_scenes", "test_scenes", "target_rcs_m2",
        "area_xy_m", "m_rx", "total_power_w_per_uav", "sensing_fraction",
        "belief_sigma_pos_m", "belief_sigma_vel_mps", "direct_mismatch_in_cres",
        "direct_mismatch_covariance_scale", "oracle_direct_residual_in_cres",
        "target_statistic_normalization", "direct_gain_boost_db_grid",
        "direct_estimation_sigma_delay_bins",
        "direct_estimation_sigma_doppler_bins", "interference_tangent_order",
        "interference_uncertainty_weighted", "target_glrt_radius_bins",
        "target_glrt_grid_points", "max_protected_targets",
    )
    for source in sources[1:]:
        for field in common_fields:
            if source["manifest"].get(field) != baseline.get(field):
                raise ValueError(f"input manifest mismatch: {field}")
    if any(s["manifest"]["oracle_direct_residual_in_cres"] for s in sources):
        raise ValueError("oracle truth leakage is excluded from this pilot")
    keys = set(sources[0]["rows"])
    if any(set(source["rows"]) != keys for source in sources[1:]):
        raise ValueError("receiver inputs must have identical split/scene keys")
    by_split = {"calibration": [], "test": []}
    for split, scene in sorted(keys):
        if split not in by_split:
            raise ValueError(f"unexpected split {split!r}")
        h0 = h1 = 0.0
        for source in sources:
            row = source["rows"][(split, scene)]
            rank = float(row["dof_real"]) / 2.0
            if rank <= 0:
                raise ValueError("zero-rank local detector")
            h0 += float(row["stat_h0"]) / rank
            h1 += float(row["stat_h1"]) / rank
        by_split[split].append((scene, h0, h1))
    cal = by_split["calibration"]
    test = by_split["test"]
    if not cal or not test:
        raise ValueError("both calibration and test scenes are required")
    if (len(cal) != int(baseline["calibration_scenes"]) or
            len(test) != int(baseline["test_scenes"])):
        raise ValueError("incomplete scene coverage")
    n = len(cal)
    order = math.ceil((n + 1) * (1.0 - alpha))
    threshold = (float("inf") if order > n else
                 float(np.sort([v[1] for v in cal])[order - 1]))
    h0 = np.asarray([v[1] for v in test])
    h1 = np.asarray([v[2] for v in test])
    local_h0 = np.asarray([
        [float(source["rows"][("test", scene)]["stat_h0"])
         for source in sources] for scene, _, _ in test
    ])
    corr = np.corrcoef(local_h0, rowvar=False)
    return {
        "protocol": "fixed_multi_receiver_postfusion_conformal_v1",
        "inputs": [str(p.resolve()) for p in paths],
        "receivers": receivers, "target": target, "boost_db": boost,
        "fusion": "sum of local neighbourhood GLRT statistics divided by local complex rank",
        "selection": "fixed before calibration",
        "alpha": alpha, "n_cal_independent_scenes": n,
        "n_test_independent_scenes": len(test),
        "conformal_order": order, "threshold": threshold,
        "conformal_marginal_bound": ((n + 1 - order) / (n + 1)
                                     if order <= n else 0.0),
        "test_false_alarms": int(np.sum(h0 > threshold)),
        "test_detections": int(np.sum(h1 > threshold)),
        "test_pfa": float(np.mean(h0 > threshold)),
        "test_pd": float(np.mean(h1 > threshold)),
        "test_auc": _auc(h0, h1),
        "test_h0_receiver_correlation": corr.tolist(),
        "test_scores": [{"scene_id": scene, "h0": s0, "h1": s1}
                        for scene, s0, s1 in test],
        "scope": "one target, fixed receiver subset, marginal across independent scenes",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", default="tp_uic_full")
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--boost-db", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    result = run(args.input, arm=args.arm, target=args.target,
                 boost=args.boost_db, alpha=args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "test_scores"}, indent=2))


if __name__ == "__main__":
    main()
