#!/usr/bin/env python3
"""Small shared-scene screen for distributed full-observation detector headroom."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from copy import copy
from pathlib import Path

import numpy as np

from isac_sim.receiver import cancellation as cx
from tools import gate_crossfit_hierarchical_map as single
from tools import gate_full_observation_detector as gate
from tools import run_tpuic_receiver_benchmark as bench


def _role(scene, args):
    if scene < args.train_scenes:
        return "train"
    if scene < args.train_scenes + args.calibration_scenes:
        return "calibration"
    return "test"


def _receiver_scores(cfg, truth, belief, base, args, scene, receiver):
    local = copy(args); local.receiver = receiver
    pairs = [single._pair(cfg, truth, belief, base, local, scene, look)
             for look in range(2)]
    scores = []
    for index in (1, 0):  # H0 then H1
        reference, held = pairs[1][index], pairs[0][index]
        sources = cx.refine_direct_dd_joint_gn(
            cfg, [reference], max_nfev=args.gn_max_nfev)
        refined = cx.apply_direct_dd(cfg, held, sources)
        scores.append(gate._score(cfg, refined, args.target, "tp_uic_full"))
    return scores


def _auc(test, subset, mode):
    values = np.asarray([r["scores"] for r in test], dtype=float)[:, :, subset]
    aggregate = np.sum(values, axis=2) if mode == "sum" else np.max(values, axis=2)
    return float(bench._auc(aggregate[:, 0], aggregate[:, 1]))


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = single._cfg(args)
    receivers = tuple(range(int(cfg.scale.M)))
    total = args.train_scenes + args.calibration_scenes + args.test_scenes
    rows = []
    for scene in range(total):
        truth, belief, base0 = bench._scene(
            cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz")
        base = bench._boost_direct(base0, args.boost_db)
        scores = np.asarray([_receiver_scores(
            cfg, truth, belief, base, args, scene, receiver)
            for receiver in receivers]).T
        rows.append({"split": _role(scene, args), "scene_id": scene,
                     "scores": scores.tolist()})
    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("split", "scene_id", "scores_json"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"split": row["split"], "scene_id": row["scene_id"],
                             "scores_json": json.dumps(row["scores"])})
    test = [row for row in rows if row["split"] == "test"]
    fixed = {"receiver_1": (1,), "pair_0_1": (0, 1),
             "triple_0_1_2": (0, 1, 2), "all_6": receivers}
    evaluation = {name: {mode: _auc(test, subset, mode) for mode in ("sum", "max")}
                  for name, subset in fixed.items()}
    oracle = {}
    for size in range(1, len(receivers) + 1):
        candidates = [(subset, _auc(test, subset, "sum"))
                      for subset in itertools.combinations(receivers, size)]
        subset, auc = max(candidates, key=lambda item: item[1])
        oracle[str(size)] = {"subset": list(subset), "test_auc": auc}
    payload = {
        "protocol": vars(args) | {"out": str(args.out)},
        "shared_scene_contract": True, "evaluation": evaluation,
        "test_selected_oracle_sum": oracle,
        "limitations": ["AUC-only screen", "oracle subsets use test and are ceilings only",
                        "no communication loss or report budget"],
    }
    (args.out / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=1)
    parser.add_argument("--calibration-scenes", type=int, default=2)
    parser.add_argument("--test-scenes", type=int, default=8)
    parser.add_argument("--master-seed", type=int, default=20261016)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--boost-db", type=float, default=50.0)
    parser.add_argument("--gn-max-nfev", type=int, default=4)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
