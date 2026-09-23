#!/usr/bin/env python3
"""Three-way gate for train/H0-learned sigma-point covariance scaling."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from isac_sim.core.config import apply_overrides
from isac_sim.detection.evaluation_split import EvaluationPartition
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver.cancellation_glrt.covariance_scale import (
    select_covariance_scale,
)
from isac_sim.sensing.model import radar_hardware_gain
from tools import run_tpuic_receiver_benchmark as bench


SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)
AUC_MARGIN = 0.02
INFLATION_RANGE = (0.8, 1.25)


def _configs(args):
    defaults = bench.build_parser().parse_args(["--out", str(args.out)])
    defaults.master_seed = args.master_seed
    defaults.receivers = str(args.receiver)
    defaults.targets = str(args.target)
    defaults.direct_gain_boost_db = str(args.boost_db)
    defaults.direct_dd_sigma = args.direct_dd_sigma
    defaults.direct_error_scope = "scene"
    defaults.interference_tangent_order = 1
    defaults.interference_uncertainty_weighted = True
    defaults.direct_mismatch_covariance_model = "sigma_point"
    defaults.target_glrt_mode = "neighbourhood_max"
    defaults.target_glrt_radius_bins = 0.25
    defaults.target_glrt_grid_points = 3
    base = bench._make_cfg(defaults)
    return {
        scale: apply_overrides(base, {
            "cancellation.direct_mismatch_covariance_scale": scale,
        }) for scale in SCALES
    }


def _observations(cfg, truth, belief, base, args, scene):
    m = int(cfg.scale.M)
    power = np.full(m, float(cfg.radio.P_default) * float(cfg.radio.rho))
    obs_rng = np.random.default_rng([
        int(cfg.run.seed), scene, args.receiver, args.target, 0, 404,
    ])
    error_rng = np.random.default_rng([
        int(cfg.run.seed), scene, args.receiver, args.target, 405,
    ])
    return cx.build_observation_pair(
        cfg, truth, belief.as_geometry(truth), base, args.receiver,
        rng=obs_rng, direct_error_rng=error_rng, sense_power=power,
        radiated_power=power, processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)), exclude_target=args.target,
        active_mask=np.ones(m, dtype=bool), weak_index=args.target,
        share_noise=False,
    )


def _auc(h0, h1):
    return float(bench._auc(np.asarray(h0, float), np.asarray(h1, float)))


def _split_name(scene, args):
    if scene < args.train_scenes:
        return "train"
    if scene < args.train_scenes + args.calibration_scenes:
        return "calibration"
    return "test"


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=20)
    parser.add_argument("--calibration-scenes", type=int, default=20)
    parser.add_argument("--test-scenes", type=int, default=40)
    parser.add_argument("--master-seed", type=int, default=20261001)
    parser.add_argument("--receiver", type=int, default=0)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--boost-db", type=float, default=30.0)
    parser.add_argument("--direct-dd-sigma", type=float, default=0.10)
    args = parser.parse_args()
    if min(args.train_scenes, args.calibration_scenes, args.test_scenes) <= 0:
        raise ValueError("all three split sizes must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    configs = _configs(args)
    base_cfg = configs[1.0]
    records = []
    total = args.train_scenes + args.calibration_scenes + args.test_scenes
    for scene in range(total):
        truth, belief, base0 = bench._scene(
            base_cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz"
        )
        base = bench._boost_direct(base0, args.boost_db)
        obs1, obs0 = _observations(base_cfg, truth, belief, base, args, scene)
        results1 = cx.cancellation_arms(base_cfg, obs1, weak_target=args.target)
        results0 = cx.cancellation_arms(base_cfg, obs0, weak_target=args.target)
        h0, h1, inflation = {}, {}, {}
        for scale, cfg in configs.items():
            _, _, score1 = bench._run_arm(cfg, obs1, "tp_uic_full", results1)
            _, _, score0 = bench._run_arm(cfg, obs0, "tp_uic_full", results0)
            h0[scale] = float(score0.statistic)
            h1[scale] = float(score1.statistic)
            inflation[scale] = float(
                score0.raw_statistic / max(score0.dof_real / 2.0, 1.0)
            )
        records.append({
            "split": _split_name(scene, args), "scene_id": scene,
            "h0_by_scale": h0, "h1_by_scale": h1,
            "h0_inflation_by_scale": inflation,
        })

    partition = EvaluationPartition.from_records(records, require_train=True)
    selected = select_covariance_scale(partition.train, SCALES)
    evaluated = {}
    for label, scale in (("fixed", 1.0), ("learned", selected.scale)):
        cal0 = [row["h0_by_scale"][scale] for row in partition.calibration.rows]
        threshold = bench._calibrated_threshold(
            np.asarray(cal0), 0.05, "split_conformal"
        )
        test0 = [row["h0_by_scale"][scale] for row in partition.test.rows]
        test1 = [row["h1_by_scale"][scale] for row in partition.test.rows]
        infl = [row["h0_inflation_by_scale"][scale] for row in partition.test.rows]
        evaluated[label] = {
            "scale": scale, "threshold": threshold,
            "auc": _auc(test0, test1),
            "pfa": float(np.mean(np.asarray(test0) > threshold)),
            "pd": float(np.mean(np.asarray(test1) > threshold)),
            "median_h0_inflation": float(np.median(infl)),
        }
    fixed, learned = evaluated["fixed"], evaluated["learned"]
    passed = bool(
        learned["auc"] - fixed["auc"] >= AUC_MARGIN
        and learned["pd"] >= fixed["pd"]
        and INFLATION_RANGE[0] <= learned["median_h0_inflation"] <= INFLATION_RANGE[1]
    )
    flat_rows = []
    for row in records:
        for scale in SCALES:
            flat_rows.append({
                "split": row["split"], "scene_id": row["scene_id"], "scale": scale,
                "stat_h0": row["h0_by_scale"][scale],
                "stat_h1": row["h1_by_scale"][scale],
                "h0_inflation": row["h0_inflation_by_scale"][scale],
            })
    _write_csv(args.out / "records.csv", flat_rows)
    payload = {
        "protocol": vars(args) | {"scales": SCALES, "p_fa": 0.05},
        "selection": vars(selected), "evaluation": evaluated,
        "gate": {
            "auc_improvement_at_least": AUC_MARGIN, "pd_no_worse": True,
            "test_median_h0_inflation_range": INFLATION_RANGE,
        },
        "decision": "proceed_to_structured_weights" if passed else "stop_scalar_learning",
    }
    payload["protocol"]["out"] = str(payload["protocol"]["out"])
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
