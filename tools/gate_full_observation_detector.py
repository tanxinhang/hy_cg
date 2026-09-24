#!/usr/bin/env python3
"""Formal single-block GN-MAP TP-UIC detector gate."""
from __future__ import annotations

import argparse
import csv
import json
from copy import copy
from pathlib import Path

import numpy as np

from isac_sim.detection.evaluation_split import EvaluationPartition
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from tools import gate_crossfit_hierarchical_map as single
from tools import run_tpuic_receiver_benchmark as bench


def _score(cfg, obs, target, arm):
    only = arm if arm != "perfect_channel" else None
    results = cx.cancellation_arms(cfg, obs, weak_target=target, only=only)
    model = gl.residual_model(cfg, obs, arm, results)
    return gl.target_neighbourhood_glrt(
        cfg, obs, results[arm], model, target=target, aggregation="max").statistic


def _role(scene, args):
    if scene < args.train_scenes:
        return "train"
    if scene < args.train_scenes + args.calibration_scenes:
        return "calibration"
    return "test"


def _metrics(rows, name, threshold):
    h0 = np.asarray([row[f"{name}_h0"] for row in rows])
    h1 = np.asarray([row[f"{name}_h1"] for row in rows])
    return {"auc": bench._auc(h0, h1), "pfa": float(np.mean(h0 > threshold)),
            "pd": float(np.mean(h1 > threshold))}


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = single._cfg(args)
    total = args.train_scenes + args.calibration_scenes + args.test_scenes
    records = []
    for scene in range(total):
        truth, belief, base0 = bench._scene(
            cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz")
        base = bench._boost_direct(base0, args.boost_db)
        pair_args = copy(args)
        pairs = [single._pair(
            cfg, truth, belief, base, pair_args, scene, look) for look in range(2)]
        row = {"split": _role(scene, args), "scene_id": scene,
               "boost_db": args.boost_db}
        for hypothesis, index in (("h1", 0), ("h0", 1)):
            reference, held = pairs[1][index], pairs[0][index]
            sources = cx.refine_direct_dd_joint_gn(
                cfg, [reference], max_nfev=args.gn_max_nfev)
            refined = cx.apply_direct_dd(cfg, held, sources)
            row[f"nominal_{hypothesis}"] = _score(
                cfg, refined, args.target, "tp_uic_full")
            row[f"perfect_{hypothesis}"] = _score(
                cfg, held, args.target, "perfect_channel")
        records.append(row)
    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader()
        writer.writerows(records)
    partition = EvaluationPartition.from_records(records, require_train=True)
    evaluation = {}
    for name in ("nominal", "perfect"):
        cal0 = np.asarray([row[f"{name}_h0"] for row in partition.calibration.rows])
        threshold = bench._calibrated_threshold(cal0, args.p_fa, "split_conformal")
        evaluation[name] = {"threshold": threshold,
                            **_metrics(partition.test.rows, name, threshold)}
    payload = {"protocol": vars(args) | {"out": str(args.out)},
               "evaluation": evaluation,
               "oracle_gap_auc": evaluation["perfect"]["auc"]
               - evaluation["nominal"]["auc"],
               "eligible": args.calibration_scenes >= 40 and args.test_scenes >= 40}
    (args.out / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=1)
    parser.add_argument("--calibration-scenes", type=int, default=40)
    parser.add_argument("--test-scenes", type=int, default=40)
    parser.add_argument("--master-seed", type=int, default=20261014)
    parser.add_argument("--receiver", type=int, default=1)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--boost-db", type=float, default=50.0)
    parser.add_argument("--gn-max-nfev", type=int, default=4)
    parser.add_argument("--p-fa", type=float, default=0.05)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
