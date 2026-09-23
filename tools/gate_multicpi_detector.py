#!/usr/bin/env python3
"""Paired gate for noncoherent accumulation of independent TP-UIC CPIs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from isac_sim.detection.evaluation_split import EvaluationPartition
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.sensing.model import radar_hardware_gain
from tools import run_tpuic_receiver_benchmark as bench


TEST_AUC_MARGIN = 0.02


def _cfg(args):
    defaults = bench.build_parser().parse_args(["--out", str(args.out)])
    defaults.master_seed = args.master_seed
    defaults.area_xy = args.area_xy
    defaults.direct_dd_sigma = args.direct_dd_sigma
    defaults.interference_tangent_order = 1
    defaults.interference_uncertainty_weighted = True
    defaults.direct_mismatch_covariance_model = "sigma_point_replacement"
    defaults.target_glrt_radius_bins = 0.25
    defaults.target_glrt_grid_points = 3
    defaults.max_protected_targets = 1
    return bench._make_cfg(defaults)


def _pair(cfg, truth, belief, base, args, scene, cpi):
    m = int(cfg.scale.M)
    power = np.full(m, float(cfg.radio.P_default) * float(cfg.radio.rho))
    rng = np.random.default_rng([
        int(cfg.run.seed), scene, args.receiver, args.target, cpi, 404,
    ])
    direct_rng = np.random.default_rng([
        int(cfg.run.seed), scene, args.receiver, args.target, 405,
    ])
    return cx.build_observation_pair(
        cfg, truth, belief.as_geometry(truth), base, args.receiver,
        rng=rng, direct_error_rng=direct_rng, sense_power=power,
        radiated_power=power, processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)), exclude_target=args.target,
        active_mask=np.ones(m, dtype=bool), weak_index=args.target,
        share_noise=False,
    )


def _score(cfg, obs, target):
    results = cx.cancellation_arms(cfg, obs, weak_target=target)
    result = results["tp_uic_full"]
    model = gl.residual_model(cfg, obs, "tp_uic_full", results)
    return gl.target_neighbourhood_glrt(
        cfg, obs, result, model, target=target, aggregation="max"
    ).statistic


def _role(scene, args):
    if scene < args.train_scenes:
        return "train"
    if scene < args.train_scenes + args.calibration_scenes:
        return "calibration"
    return "test"


def _auc(rows, prefix):
    return float(bench._auc(
        [row[f"{prefix}_h0"] for row in rows],
        [row[f"{prefix}_h1"] for row in rows],
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=20)
    parser.add_argument("--calibration-scenes", type=int, default=20)
    parser.add_argument("--test-scenes", type=int, default=40)
    parser.add_argument("--master-seed", type=int, default=20261004)
    parser.add_argument("--receiver", type=int, default=0)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--area-xy", type=float, default=400.0)
    parser.add_argument("--boost-db", type=float, default=10.0)
    parser.add_argument("--direct-dd-sigma", type=float, default=0.10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(args)
    total = args.train_scenes + args.calibration_scenes + args.test_scenes
    records = []
    for scene in range(total):
        truth, belief, base0 = bench._scene(
            cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz"
        )
        base = bench._boost_direct(base0, args.boost_db)
        scores0, scores1 = [], []
        for cpi in range(2):
            obs1, obs0 = _pair(cfg, truth, belief, base, args, scene, cpi)
            scores0.append(_score(cfg, obs0, args.target))
            scores1.append(_score(cfg, obs1, args.target))
        records.append({
            "split": _role(scene, args), "scene_id": scene,
            "one_cpi_h0": scores0[0], "one_cpi_h1": scores1[0],
            "two_cpi_h0": float(sum(scores0)),
            "two_cpi_h1": float(sum(scores1)),
        })

    partition = EvaluationPartition.from_records(records, require_train=True)
    train_auc = {name: _auc(partition.train.rows, name)
                 for name in ("one_cpi", "two_cpi")}
    evaluation = {}
    for name in ("one_cpi", "two_cpi"):
        cal0 = np.asarray([row[f"{name}_h0"] for row in partition.calibration.rows])
        threshold = bench._calibrated_threshold(cal0, 0.05, "split_conformal")
        test0 = np.asarray([row[f"{name}_h0"] for row in partition.test.rows])
        test1 = np.asarray([row[f"{name}_h1"] for row in partition.test.rows])
        evaluation[name] = {
            "threshold": threshold, "auc": _auc(partition.test.rows, name),
            "pfa": float(np.mean(test0 > threshold)),
            "pd": float(np.mean(test1 > threshold)),
        }
    one, two = evaluation["one_cpi"], evaluation["two_cpi"]
    passed = bool(
        two["auc"] >= one["auc"] + TEST_AUC_MARGIN and two["pd"] >= one["pd"]
    )
    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "protocol": vars(args) | {"p_fa": 0.05, "cpi_count": 2,
                                    "combiner": "noncoherent_sum"},
        "gate": {"test_auc_margin": TEST_AUC_MARGIN, "test_pd_no_worse": True},
        "train_auc": train_auc, "evaluation": evaluation,
        "decision": "promote_two_cpi" if passed else "retain_one_cpi",
        "cost": {"observation_blocks": 2, "receiver_compute_approx_x": 2.0},
    }
    payload["protocol"]["out"] = str(payload["protocol"]["out"])
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
