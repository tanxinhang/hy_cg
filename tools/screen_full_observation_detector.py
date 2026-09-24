#!/usr/bin/env python3
"""Full-observation 1/2-block detector headroom screen."""
from __future__ import annotations

import argparse
import csv
import json
from copy import copy
from pathlib import Path

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


def _nominal(cfg, reference, held, target, max_nfev):
    sources = cx.refine_direct_dd_joint_gn(
        cfg, [reference], max_nfev=max_nfev)
    return _score(cfg, cx.apply_direct_dd(cfg, held, sources), target, "tp_uic_full")


def _role(scene, args):
    if scene < args.train_scenes:
        return "train"
    if scene < args.train_scenes + args.calibration_scenes:
        return "calibration"
    return "test"


def run(args):
    cfg = single._cfg(args)
    total = args.train_scenes + args.calibration_scenes + args.test_scenes
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in range(total):
        truth, belief, base0 = bench._scene(
            cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz")
        for boost in args.boost_db:
            base = bench._boost_direct(base0, boost)
            pair_args = copy(args)
            pairs = [single._pair(
                cfg, truth, belief, base, pair_args, scene, look
            ) for look in range(4)]
            scores = {name: {hyp: [] for hyp in ("h1", "h0")}
                      for name in ("perfect", "nominal")}
            for block, (held_index, reference_index) in enumerate(((0, 1), (2, 3))):
                for hypothesis, index in (("h1", 0), ("h0", 1)):
                    held = pairs[held_index][index]
                    reference = pairs[reference_index][index]
                    scores["perfect"][hypothesis].append(
                        _score(cfg, held, args.target, "perfect_channel"))
                    scores["nominal"][hypothesis].append(_nominal(
                        cfg, reference, held, args.target, args.gn_max_nfev))
            row = {"split": _role(scene, args), "scene_id": scene,
                   "boost_db": boost}
            for name in scores:
                for hypothesis in scores[name]:
                    values = scores[name][hypothesis]
                    row[f"{name}_one_{hypothesis}"] = values[0]
                    row[f"{name}_two_{hypothesis}"] = sum(values)
            rows.append(row)
    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows(rows)
    test = [row for row in rows if row["split"] == "test"]
    summary = {}
    for boost in args.boost_db:
        group = [row for row in test if row["boost_db"] == boost]
        summary[str(boost)] = {}
        for name in ("perfect", "nominal"):
            for blocks in ("one", "two"):
                key = f"{name}_{blocks}"
                summary[str(boost)][key] = bench._auc(
                    [row[f"{key}_h0"] for row in group],
                    [row[f"{key}_h1"] for row in group])
        summary[str(boost)]["two_block_oracle_gap"] = (
            summary[str(boost)]["perfect_two"]
            - summary[str(boost)]["nominal_two"])
    payload = {"protocol": vars(args) | {"out": str(args.out)}, "test_auc": summary,
               "decision_scope": "screen_only_auc"}
    (args.out / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=1)
    parser.add_argument("--calibration-scenes", type=int, default=2)
    parser.add_argument("--test-scenes", type=int, default=8)
    parser.add_argument("--master-seed", type=int, default=20261013)
    parser.add_argument("--receiver", type=int, default=1)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--boost-db", type=lambda v: tuple(map(float, v.split(","))),
                        default=(10.0, 30.0, 50.0))
    parser.add_argument("--gn-max-nfev", type=int, default=4)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
