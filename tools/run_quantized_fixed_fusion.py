#!/usr/bin/env python3
"""Fixed-pair, fixed-bit reporting pilot on real TP-UIC local scores.

Only scalar payload quantization is modelled. This is not a transport-link or
multi-target global-PFA experiment. The quantizer is fixed before calibration.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from tools.run_fixed_fusion_conformal import _auc, _read
except ModuleNotFoundError:  # direct invocation: python tools/run_quantized_fixed_fusion.py
    from run_fixed_fusion_conformal import _auc, _read


def quantize(score: float, bits: int, clip: float) -> float:
    if bits < 1 or not math.isfinite(clip) or clip <= 0:
        raise ValueError("bits must be positive and clip must be finite positive")
    if not math.isfinite(score) or score < 0:
        raise ValueError("local GLRT score must be finite and nonnegative")
    levels = (1 << bits) - 1
    return round(min(score, clip) * levels / clip) * clip / levels


def run(paths: list[Path], *, bits: int = 3, clip: float = 8.0,
        alpha: float = 0.05, target: int = 1, boost: float = 30.0) -> dict:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    sources = [_read(path, "tp_uic_full", target, boost) for path in paths]
    if len(sources) != 2 or len({s["receiver"] for s in sources}) != 2:
        raise ValueError("exactly two distinct receivers are required")
    baseline = sources[0]["manifest"]
    for source in sources:
        manifest = source["manifest"]
        for key in ("master_seed", "calibration_scenes", "test_scenes",
                    "target_rcs_m2", "area_xy_m", "m_rx", "total_power_w_per_uav",
                    "sensing_fraction", "belief_sigma_pos_m", "belief_sigma_vel_mps",
                    "target_glrt_radius_bins", "target_glrt_grid_points",
                    "direct_gain_boost_db_grid", "direct_estimation_sigma_delay_bins",
                    "direct_estimation_sigma_doppler_bins", "interference_tangent_order",
                    "interference_uncertainty_weighted", "direct_mismatch_in_cres",
                    "direct_mismatch_covariance_scale", "max_protected_targets"):
            if manifest.get(key) != baseline.get(key):
                raise ValueError(f"receiver manifests differ: {key}")
        if manifest.get("oracle_direct_residual_in_cres"):
            raise ValueError("oracle covariance is excluded")
    keys = set(sources[0]["rows"])
    if keys != set(sources[1]["rows"]):
        raise ValueError("receiver scene keys are not aligned")
    by_split: dict[str, list[tuple[int, float, float]]] = {
        "calibration": [], "test": []}
    clipped = {split: {"h0": 0, "h1": 0} for split in by_split}
    for split, scene in sorted(keys):
        if split not in by_split:
            raise ValueError(f"unexpected split: {split}")
        pair = []
        for source in sources:
            row = source["rows"][(split, scene)]
            rank = float(row["dof_real"]) / 2
            if rank <= 0:
                raise ValueError("zero-rank local detector")
            local_h0 = float(row["stat_h0"]) / rank
            local_h1 = float(row["stat_h1"]) / rank
            clipped[split]["h0"] += int(local_h0 > clip)
            clipped[split]["h1"] += int(local_h1 > clip)
            pair.append((quantize(local_h0, bits, clip),
                         quantize(local_h1, bits, clip)))
        by_split[split].append((scene, sum(v[0] for v in pair),
                                sum(v[1] for v in pair)))
    cal, test = by_split["calibration"], by_split["test"]
    if len(cal) != int(baseline["calibration_scenes"]) or len(test) != int(baseline["test_scenes"]):
        raise ValueError("incomplete independent scene coverage")
    n = len(cal)
    order = math.ceil((n + 1) * (1 - alpha))
    threshold = float("inf") if order > n else float(np.sort([v[1] for v in cal])[order - 1])
    h0 = np.asarray([v[1] for v in test])
    h1 = np.asarray([v[2] for v in test])
    return {
        "protocol": "quantized_fixed_pair_postfusion_conformal_v1",
        "receivers": [s["receiver"] for s in sources], "target": target,
        "boost_db": boost, "bits_per_uav_payload": bits,
        "total_payload_bits_per_target": 2 * bits,
        "quantizer": {"type": "fixed_uniform_nearest", "range": [0, clip]},
        "clipped_local_reports": clipped,
        "communication_scope": "error-free scalar payload only; no header, coding, outage, latency or transport model",
        "selection": "fixed before calibration", "alpha": alpha,
        "n_cal_independent_scenes": n, "n_test_independent_scenes": len(test),
        "conformal_order": order, "threshold": threshold,
        "marginal_pfa_bound": 0.0 if order > n else (n + 1 - order) / (n + 1),
        "test_false_alarms": int(np.sum(h0 > threshold)),
        "test_detections": int(np.sum(h1 > threshold)),
        "test_auc": _auc(h0, h1),
        "scope": "one fixed target; not multi-target/global family-wise PFA",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--clip", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--boost-db", type=float, default=30.0)
    args = parser.parse_args()
    result = run(args.input, bits=args.bits, clip=args.clip,
                 alpha=args.alpha, target=args.target, boost=args.boost_db)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
