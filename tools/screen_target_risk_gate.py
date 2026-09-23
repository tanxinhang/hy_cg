#!/usr/bin/env python3
"""Necessary-condition screen for target-conditioned, risk-gated TP-UIC."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.receiver.joint_observation import generate_joint_observation_pair
from tools import run_tpuic_receiver_benchmark as bench


ARMS = ("plain_ls", "targeted_tpuic_full", "tp_uic_full")
RISK_GATE = 0.8


def _cfg(args):
    defaults = bench.build_parser().parse_args(["--out", str(args.out)])
    defaults.master_seed = args.master_seed
    defaults.uavs = args.uavs
    defaults.targets_count = args.targets
    defaults.target_rcs = args.target_rcs
    defaults.area_xy = args.area_xy
    defaults.direct_dd_sigma = args.direct_dd_sigma
    defaults.interference_tangent_order = 1
    defaults.interference_uncertainty_weighted = True
    defaults.direct_mismatch_covariance_model = "sigma_point"
    defaults.target_glrt_mode = "neighbourhood_max"
    defaults.target_glrt_radius_bins = 0.25
    defaults.target_glrt_grid_points = 3
    defaults.max_protected_targets = min(1, args.targets)
    return bench._make_cfg(defaults)


def _score(cfg, obs, arm, results):
    result = results[arm]
    model = gl.residual_model(cfg, obs, arm, results)
    score = gl.target_neighbourhood_glrt(
        cfg, obs, result, model, target=obs.weak_index,
        p_fa=float(cfg.detect.Pfa_target), dictionary="belief",
    )
    info = gl.detection_information(
        cfg, obs, result, model, target=obs.weak_index
    )
    return result, score, info


def _rank_correlation(x, y):
    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        out = np.empty(len(values), float)
        out[order] = np.arange(len(values), dtype=float)
        return out
    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(rx) > 1 else float("nan")


def _quantile(rows, key, q):
    return float(np.quantile([row[key] for row in rows], q))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--master-seed", type=int, default=20261002)
    parser.add_argument("--uavs", type=int, default=6)
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--target-rcs", type=float, default=0.05)
    parser.add_argument("--area-xy", type=float, default=800.0)
    parser.add_argument("--boost-db", type=float, default=30.0)
    parser.add_argument("--direct-dd-sigma", type=float, default=0.10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(args)
    rows = []
    for scene in range(args.scenes):
        truth, belief, base0 = bench._scene(
            cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz"
        )
        base = bench._boost_direct(base0, args.boost_db)
        for receiver in range(args.uavs):
            pair = generate_joint_observation_pair(
                cfg, truth, belief, base, scene_id=scene, receiver=receiver
            )
            for target in range(args.targets):
                obs0, obs1 = pair.for_target(target)
                results0 = cx.cancellation_arms(cfg, obs0, weak_target=target)
                results1 = cx.cancellation_arms(cfg, obs1, weak_target=target)
                scored0 = {arm: _score(cfg, obs0, arm, results0) for arm in ARMS}
                scored1 = {arm: _score(cfg, obs1, arm, results1) for arm in ARMS}
                plain = scored1["plain_ls"][0]
                row = {
                    "scene": scene, "receiver": receiver, "target": target,
                    "plain_predicted_risk": plain.eta_survive_risk_q,
                    "plain_truth_survival": plain.eta_survive_q,
                }
                for arm in ARMS:
                    result1, score1, info1 = scored1[arm]
                    _, score0, _ = scored0[arm]
                    row[f"{arm}_ncp"] = info1.value
                    row[f"{arm}_stat_h0"] = score0.statistic
                    row[f"{arm}_stat_h1"] = score1.statistic
                    row[f"{arm}_truth_survival"] = result1.eta_survive_q
                    row[f"{arm}_kappa_db"] = result1.kappa_db
                row["delta_ncp_targeted_plain"] = (
                    row["targeted_tpuic_full_ncp"] - row["plain_ls_ncp"]
                )
                rows.append(row)

    risky = [row for row in rows if row["plain_predicted_risk"] < RISK_GATE]
    safe = [row for row in rows if row["plain_predicted_risk"] >= RISK_GATE]
    correlation = _rank_correlation(
        [row["plain_predicted_risk"] for row in rows],
        [row["plain_truth_survival"] for row in rows],
    )
    risky_fraction = len(risky) / len(rows)
    risky_median = _quantile(risky, "delta_ncp_targeted_plain", 0.5) if risky else None
    risky_q25 = _quantile(risky, "delta_ncp_targeted_plain", 0.25) if risky else None
    passed = bool(
        correlation >= 0.5 and risky_fraction >= 0.2
        and risky_median is not None and risky_median > 0.0 and risky_q25 >= 0.0
    )
    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": vars(args) | {"risk_gate": RISK_GATE, "arms": ARMS},
        "preregistered_gate": {
            "risk_truth_rank_correlation_at_least": 0.5,
            "risky_fraction_at_least": 0.2,
            "risky_median_delta_ncp_positive": True,
            "risky_q25_delta_ncp_nonnegative": True,
        },
        "result": {
            "count": len(rows), "risky_count": len(risky), "safe_count": len(safe),
            "risky_fraction": risky_fraction,
            "risk_truth_rank_correlation": correlation,
            "risky_median_delta_ncp": risky_median,
            "risky_q25_delta_ncp": risky_q25,
        },
        "decision": "proceed" if passed else "stop_risk_gate",
        "scope": "mechanism screen only; oracle survival is diagnostic, never a gate input",
    }
    payload["protocol"]["out"] = str(payload["protocol"]["out"])
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
