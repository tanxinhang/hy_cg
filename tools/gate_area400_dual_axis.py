#!/usr/bin/env python3
"""Factorial 400 m gate: cancellation loss versus evidence-limited detection."""
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


ARMS = ("tp_uic_full", "refined_tp_uic", "perfect_channel")
DEFAULT_CPI_COUNTS = (1, 2, 4)


def _cfg(args):
    defaults = bench.build_parser().parse_args(["--out", str(args.out)])
    defaults.master_seed = args.master_seed
    defaults.area_xy = 400.0
    defaults.direct_dd_sigma = args.direct_dd_sigma
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


def _observation_pair(cfg, truth, belief, base, args, scene, cpi, boost_index):
    m = int(cfg.scale.M)
    power = np.full(m, float(cfg.radio.P_default) * float(cfg.radio.rho))
    rng = np.random.default_rng([
        int(cfg.run.seed), scene, args.receiver, args.target, cpi, boost_index, 501,
    ])
    # Hold the receiver's DD belief error fixed across looks in one scene.
    direct_rng = np.random.default_rng([
        int(cfg.run.seed), scene, args.receiver, args.target, boost_index, 502,
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


def _scores(cfg, obs, target):
    results = cx.cancellation_arms(cfg, obs, weak_target=target)
    out = {}
    for arm in ("tp_uic_full", "perfect_channel"):
        model = gl.residual_model(cfg, obs, arm, results)
        out[arm] = gl.target_neighbourhood_glrt(
            cfg, obs, results[arm], model, target=target, aggregation="max"
        ).statistic
    refined = cx.refine_direct_dd(cfg, obs)
    refined_results = cx.cancellation_arms(cfg, refined, weak_target=target)
    refined_model = gl.residual_model(
        cfg, refined, "tp_uic_full", refined_results
    )
    out["refined_tp_uic"] = gl.target_neighbourhood_glrt(
        cfg, refined, refined_results["tp_uic_full"], refined_model,
        target=target, aggregation="max",
    ).statistic
    return out


def _auc(rows, arm, count):
    prefix = f"{arm}_{count}cpi"
    return float(bench._auc(
        [row[f"{prefix}_h0"] for row in rows],
        [row[f"{prefix}_h1"] for row in rows],
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=8)
    parser.add_argument("--calibration-scenes", type=int, default=20)
    parser.add_argument("--test-scenes", type=int, default=40)
    parser.add_argument("--master-seed", type=int, default=20261005)
    parser.add_argument("--receiver", type=int, default=0)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--boost-db", default="10,30,50")
    parser.add_argument("--cpi-counts", default="1,2,4")
    parser.add_argument("--direct-dd-sigma", type=float, default=0.10)
    args = parser.parse_args()
    boosts = tuple(float(v) for v in args.boost_db.split(","))
    cpi_counts = tuple(sorted({int(v) for v in args.cpi_counts.split(",")}))
    if not cpi_counts or cpi_counts[0] < 1:
        raise ValueError("cpi-counts must contain positive integers")
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
            cumulative0 = {arm: 0.0 for arm in ARMS}
            cumulative1 = {arm: 0.0 for arm in ARMS}
            row = {"split": _role(scene, args), "scene_id": scene,
                   "boost_db": boost}
            for cpi in range(1, max(cpi_counts) + 1):
                obs1, obs0 = _observation_pair(
                    cfg, truth, belief, base, args, scene, cpi - 1, boost_index
                )
                score0 = _scores(cfg, obs0, args.target)
                score1 = _scores(cfg, obs1, args.target)
                for arm in ARMS:
                    cumulative0[arm] += score0[arm]
                    cumulative1[arm] += score1[arm]
                    if cpi in cpi_counts:
                        row[f"{arm}_{cpi}cpi_h0"] = cumulative0[arm]
                        row[f"{arm}_{cpi}cpi_h1"] = cumulative1[arm]
            records.append(row)

    evaluations = []
    for boost in boosts:
        subset = [r for r in records if r["boost_db"] == boost]
        partition = EvaluationPartition.from_records(subset, require_train=True)
        by_key = {}
        for arm in ARMS:
            for count in cpi_counts:
                prefix = f"{arm}_{count}cpi"
                cal0 = np.asarray([
                    row[f"{prefix}_h0"] for row in partition.calibration.rows
                ])
                threshold = bench._calibrated_threshold(cal0, 0.05, "split_conformal")
                test0 = np.asarray([row[f"{prefix}_h0"] for row in partition.test.rows])
                test1 = np.asarray([row[f"{prefix}_h1"] for row in partition.test.rows])
                result = {
                    "boost_db": boost, "arm": arm, "cpi_count": count,
                    "train_auc": _auc(partition.train.rows, arm, count),
                    "threshold": float(threshold),
                    "auc": _auc(partition.test.rows, arm, count),
                    "pfa": float(np.mean(test0 > threshold)),
                    "pd": float(np.mean(test1 > threshold)),
                }
                evaluations.append(result)
                by_key[(arm, count)] = result
        for count in cpi_counts:
            oracle = by_key[("perfect_channel", count)]
            for arm in ("tp_uic_full", "refined_tp_uic"):
                result = by_key[(arm, count)]
                result["oracle_auc_gap"] = oracle["auc"] - result["auc"]
                result["oracle_pd_gap"] = oracle["pd"] - result["pd"]

    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "protocol": {
            "master_seed": args.master_seed, "area_xy_m": 400.0,
            "boost_db": boosts, "cpi_counts": cpi_counts,
            "arms": ARMS, "train_scenes": args.train_scenes,
            "calibration_scenes": args.calibration_scenes,
            "test_scenes": args.test_scenes, "p_fa": 0.05,
            "combiner": "noncoherent_sum",
        },
        "evaluation": evaluations,
    }
    (args.out / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
