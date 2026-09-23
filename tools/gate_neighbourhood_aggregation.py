#!/usr/bin/env python3
"""Train/calibration/test gate for nonlinear DD-hypothesis aggregation."""
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


MODES = ("max", "logmeanexp")
TRAIN_MARGIN = 0.01
TEST_MARGIN = 0.02


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


def _pair(cfg, truth, belief, base, args, scene):
    m = int(cfg.scale.M)
    power = np.full(m, float(cfg.radio.P_default) * float(cfg.radio.rho))
    rng = np.random.default_rng([
        int(cfg.run.seed), scene, args.receiver, args.target, 0, 404,
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


def _role(scene, args):
    if scene < args.train_scenes:
        return "train"
    if scene < args.train_scenes + args.calibration_scenes:
        return "calibration"
    return "test"


def _auc(rows, mode):
    return float(bench._auc(
        [row[f"{mode}_h0"] for row in rows],
        [row[f"{mode}_h1"] for row in rows],
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=20)
    parser.add_argument("--calibration-scenes", type=int, default=20)
    parser.add_argument("--test-scenes", type=int, default=40)
    parser.add_argument("--master-seed", type=int, default=20261003)
    parser.add_argument("--receiver", type=int, default=0)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--boost-db", type=float, default=30.0)
    parser.add_argument("--area-xy", type=float, default=800.0)
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
        obs1, obs0 = _pair(cfg, truth, belief, base, args, scene)
        results1 = cx.cancellation_arms(cfg, obs1, weak_target=args.target)
        results0 = cx.cancellation_arms(cfg, obs0, weak_target=args.target)
        model1 = gl.residual_model(cfg, obs1, "tp_uic_full", results1)
        model0 = gl.residual_model(cfg, obs0, "tp_uic_full", results0)
        row = {"split": _role(scene, args), "scene_id": scene}
        for mode in MODES:
            out1 = gl.target_neighbourhood_glrt(
                cfg, obs1, results1["tp_uic_full"], model1,
                target=args.target, aggregation=mode,
            )
            out0 = gl.target_neighbourhood_glrt(
                cfg, obs0, results0["tp_uic_full"], model0,
                target=args.target, aggregation=mode,
            )
            row[f"{mode}_h0"] = out0.statistic
            row[f"{mode}_h1"] = out1.statistic
        records.append(row)

    partition = EvaluationPartition.from_records(records, require_train=True)
    train_auc = {mode: _auc(partition.train.rows, mode) for mode in MODES}
    selected = (
        "logmeanexp"
        if train_auc["logmeanexp"] >= train_auc["max"] + TRAIN_MARGIN else "max"
    )
    evaluation = {}
    for mode in ("max", selected):
        cal0 = np.asarray([row[f"{mode}_h0"] for row in partition.calibration.rows])
        threshold = bench._calibrated_threshold(cal0, 0.05, "split_conformal")
        test0 = np.asarray([row[f"{mode}_h0"] for row in partition.test.rows])
        test1 = np.asarray([row[f"{mode}_h1"] for row in partition.test.rows])
        evaluation[mode] = {
            "threshold": threshold, "auc": _auc(partition.test.rows, mode),
            "pfa": float(np.mean(test0 > threshold)),
            "pd": float(np.mean(test1 > threshold)),
        }
    baseline, chosen = evaluation["max"], evaluation[selected]
    passed = bool(
        selected != "max" and chosen["auc"] >= baseline["auc"] + TEST_MARGIN
        and chosen["pd"] >= baseline["pd"]
    )
    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "protocol": vars(args) | {"modes": MODES, "p_fa": 0.05},
        "gate": {"train_auc_margin": TRAIN_MARGIN, "test_auc_margin": TEST_MARGIN,
                 "test_pd_no_worse": True},
        "train_auc": train_auc, "selected": selected,
        "evaluation": evaluation,
        "decision": "promote_logmeanexp" if passed else "retain_max",
    }
    payload["protocol"]["out"] = str(payload["protocol"]["out"])
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
