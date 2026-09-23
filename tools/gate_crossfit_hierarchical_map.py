#!/usr/bin/env python3
"""Cross-fitted two-look gate for hierarchical direct-DD MAP refinement."""
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


def _cfg(args):
    defaults = bench.build_parser().parse_args(["--out", str(args.out)])
    defaults.master_seed = args.master_seed
    defaults.area_xy = 400.0
    defaults.direct_dd_sigma = 0.10
    defaults.interference_tangent_order = 1
    defaults.interference_uncertainty_weighted = True
    defaults.direct_mismatch_covariance_model = "sigma_point_replacement"
    defaults.target_glrt_radius_bins = 0.25
    defaults.target_glrt_grid_points = 3
    defaults.max_protected_targets = 1
    return bench._make_cfg(defaults)


def _role(scene, args):
    if scene < args.train_scenes:
        return "train"
    if scene < args.train_scenes + args.calibration_scenes:
        return "calibration"
    return "test"


def _pair(cfg, truth, belief, base, args, scene, look, boost_index):
    m = int(cfg.scale.M)
    power = np.full(m, float(cfg.radio.P_default) * float(cfg.radio.rho))
    rng = np.random.default_rng([
        int(cfg.run.seed), scene, args.receiver, args.target, look, boost_index, 601,
    ])
    direct_rng = np.random.default_rng([
        int(cfg.run.seed), scene, args.receiver, args.target, boost_index, 602,
    ])
    return cx.build_observation_pair(
        cfg, truth, belief.as_geometry(truth), base, args.receiver,
        rng=rng, direct_error_rng=direct_rng, sense_power=power,
        radiated_power=power,
        processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)), exclude_target=args.target,
        active_mask=np.ones(m, dtype=bool), weak_index=args.target,
        share_noise=False,
    )


def _score(cfg, obs, target, arm):
    results = cx.cancellation_arms(cfg, obs, weak_target=target)
    model = gl.residual_model(cfg, obs, arm, results)
    return gl.target_neighbourhood_glrt(
        cfg, obs, results[arm], model, target=target, aggregation="max"
    ).statistic


def _crossfit(cfg, observations, target):
    total = 0.0
    for held_index in range(2):
        reference = observations[1 - held_index]
        sources = cx.refine_direct_dd_joint(cfg, [reference])
        held = cx.apply_direct_dd(cfg, observations[held_index], sources)
        total += _score(cfg, held, target, "tp_uic_full")
    return float(total)


def _auc(rows, prefix):
    return float(bench._auc(
        [r[f"{prefix}_h0"] for r in rows], [r[f"{prefix}_h1"] for r in rows]
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=2)
    parser.add_argument("--calibration-scenes", type=int, default=40)
    parser.add_argument("--test-scenes", type=int, default=40)
    parser.add_argument("--master-seed", type=int, default=20261007)
    parser.add_argument("--receiver", type=int, default=0)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--boost-db", default="10,50")
    args = parser.parse_args()
    boosts = tuple(float(v) for v in args.boost_db.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(args)
    total = args.train_scenes + args.calibration_scenes + args.test_scenes
    records = []
    for boost_index, boost in enumerate(boosts):
        for scene in range(total):
            truth, belief, base0 = bench._scene(
                cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz"
            )
            base = bench._boost_direct(base0, boost)
            pairs = [_pair(cfg, truth, belief, base, args, scene, look, boost_index)
                     for look in range(2)]
            h1 = [pair[0] for pair in pairs]
            h0 = [pair[1] for pair in pairs]
            row = {"split": _role(scene, args), "scene_id": scene,
                   "boost_db": boost}
            for arm, prefix in (("tp_uic_full", "two_cpi"),
                                ("perfect_channel", "perfect_two_cpi")):
                row[f"{prefix}_h0"] = sum(_score(cfg, obs, args.target, arm) for obs in h0)
                row[f"{prefix}_h1"] = sum(_score(cfg, obs, args.target, arm) for obs in h1)
            row["crossfit_map_h0"] = _crossfit(cfg, h0, args.target)
            row["crossfit_map_h1"] = _crossfit(cfg, h1, args.target)
            records.append(row)

    evaluation = []
    for boost in boosts:
        partition = EvaluationPartition.from_records(
            [r for r in records if r["boost_db"] == boost], require_train=True
        )
        by_name = {}
        for name in ("two_cpi", "crossfit_map", "perfect_two_cpi"):
            cal0 = np.asarray([r[f"{name}_h0"] for r in partition.calibration.rows])
            threshold = bench._calibrated_threshold(cal0, 0.05, "split_conformal")
            test0 = np.asarray([r[f"{name}_h0"] for r in partition.test.rows])
            test1 = np.asarray([r[f"{name}_h1"] for r in partition.test.rows])
            result = {"boost_db": boost, "receiver": name,
                      "threshold": float(threshold),
                      "test_auc": _auc(partition.test.rows, name),
                      "test_pfa": float(np.mean(test0 > threshold)),
                      "test_pd": float(np.mean(test1 > threshold)),
                      "n_calibration": len(partition.calibration.rows),
                      "n_test": len(partition.test.rows)}
            evaluation.append(result)
            by_name[name] = result
        oracle = by_name["perfect_two_cpi"]
        for name in ("two_cpi", "crossfit_map"):
            by_name[name]["oracle_auc_gap"] = (
                oracle["test_auc"] - by_name[name]["test_auc"]
            )
            by_name[name]["oracle_pd_gap"] = (
                oracle["test_pd"] - by_name[name]["test_pd"]
            )
    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {"protocol": {"master_seed": args.master_seed, "boost_db": boosts,
                             "looks": 2, "fit": "opposite-look cross-fit",
                             "test_scenes": args.test_scenes},
               "evaluation": evaluation}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
