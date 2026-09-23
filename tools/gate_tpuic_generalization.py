#!/usr/bin/env python3
"""Receiver-target generalization gate for the frozen GN-MAP TP-UIC."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from copy import copy
from pathlib import Path

import numpy as np

from isac_sim.detection.evaluation_split import EvaluationPartition
from tools import gate_crossfit_hierarchical_map as single
from tools import run_tpuic_receiver_benchmark as bench


NAMES = ("two_cpi", "crossfit_map", "perfect_two_cpi")


def _indices(value: str, size: int) -> tuple[int, ...]:
    result = tuple(range(size)) if value == "all" else tuple(
        int(item) for item in value.split(",")
    )
    if not result or min(result) < 0 or max(result) >= size:
        raise ValueError(f"selection {value!r} is outside [0, {size})")
    return result


def _metrics(rows, name, threshold):
    h0 = np.asarray([row[f"{name}_h0"] for row in rows], dtype=float)
    h1 = np.asarray([row[f"{name}_h1"] for row in rows], dtype=float)
    return {"auc": bench._auc(h0, h1), "pfa": float(np.mean(h0 > threshold)),
            "pd": float(np.mean(h1 > threshold))}


def _evaluate(records, args):
    cells = defaultdict(list)
    for row in records:
        cells[(row["boost_db"], row["receiver"], row["target"])].append(row)
    evaluated = []
    for (boost, receiver, target), rows in sorted(cells.items()):
        part = EvaluationPartition.from_records(rows, require_train=True)
        result = {"boost_db": boost, "receiver": receiver, "target": target}
        for name in NAMES:
            cal0 = np.asarray([r[f"{name}_h0"] for r in part.calibration.rows])
            threshold = bench._calibrated_threshold(cal0, args.p_fa, "split_conformal")
            result[name] = {"threshold": threshold,
                            **_metrics(part.test.rows, name, threshold)}
        evaluated.append(result)
    aggregate = []
    for boost in sorted({row["boost_db"] for row in evaluated}):
        group = [row for row in evaluated if row["boost_db"] == boost]
        base = np.asarray([[row["two_cpi"][key] for row in group]
                           for key in ("auc", "pd", "pfa")])
        method = np.asarray([[row["crossfit_map"][key] for row in group]
                             for key in ("auc", "pd", "pfa")])
        oracle = np.asarray([[row["perfect_two_cpi"][key] for row in group]
                             for key in ("auc", "pd", "pfa")])
        aggregate.append({"boost_db": boost, "cells": len(group),
            "macro_two_cpi": dict(zip(("auc", "pd", "pfa"), base.mean(axis=1))),
            "macro_crossfit_map": dict(zip(("auc", "pd", "pfa"), method.mean(axis=1))),
            "macro_perfect": dict(zip(("auc", "pd", "pfa"), oracle.mean(axis=1))),
            "map_minus_base_auc": float(np.mean(method[0] - base[0])),
            "map_minus_base_pd": float(np.mean(method[1] - base[1])),
            "nonnegative_auc_cell_fraction": float(np.mean(method[0] >= base[0])),
            "nonnegative_pd_cell_fraction": float(np.mean(method[1] >= base[1]))})
    formal = args.calibration_scenes >= 40 and args.test_scenes >= 40
    return evaluated, aggregate, {"eligible": formal,
        "status": "pending_formal_criteria" if formal else "screen_only"}


def run(args):
    cfg = single._cfg(args)
    receivers = _indices(args.receivers, int(cfg.scale.M))
    targets = _indices(args.targets, int(cfg.scale.Q))
    boosts = tuple(float(value) for value in args.boost_db.split(","))
    total = args.train_scenes + args.calibration_scenes + args.test_scenes
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for scene in range(total):
        truth, belief, base0 = bench._scene(
            cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz")
        for boost_index, boost in enumerate(boosts):
            base = bench._boost_direct(base0, boost)
            for receiver in receivers:
                for target in targets:
                    pair_args = copy(args)
                    pair_args.receiver, pair_args.target = receiver, target
                    pairs = [single._pair(cfg, truth, belief, base, pair_args, scene, look,
                                          boost_index) for look in range(2)]
                    row = {"split": single._role(scene, args), "scene_id": scene,
                           "boost_db": boost, "receiver": receiver, "target": target}
                    for arm, name in (("tp_uic_full", "two_cpi"),
                                      ("perfect_channel", "perfect_two_cpi")):
                        row[f"{name}_h1"] = sum(single._score(cfg, p[0], target, arm) for p in pairs)
                        row[f"{name}_h0"] = sum(single._score(cfg, p[1], target, arm) for p in pairs)
                    row["crossfit_map_h1"], _ = single._crossfit(
                        cfg, [p[0] for p in pairs], target, "gn", args.gn_max_nfev)
                    row["crossfit_map_h0"], _ = single._crossfit(
                        cfg, [p[1] for p in pairs], target, "gn", args.gn_max_nfev)
                    records.append(row)
    evaluated, aggregate, decision = _evaluate(records, args)
    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    payload = {"protocol": vars(args) | {"out": str(args.out)}, "cells": evaluated,
               "aggregate": aggregate, "decision": decision}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-scenes", type=int, default=1)
    parser.add_argument("--calibration-scenes", type=int, default=4)
    parser.add_argument("--test-scenes", type=int, default=4)
    parser.add_argument("--master-seed", type=int, default=20261010)
    parser.add_argument("--receivers", default="0,1,2,3")
    parser.add_argument("--targets", default="0,1,2")
    parser.add_argument("--boost-db", default="10,30,50")
    parser.add_argument("--gn-max-nfev", type=int, default=4)
    parser.add_argument("--p-fa", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
