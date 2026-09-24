#!/usr/bin/env python3
"""Formal shared-scene gate for train-selected distributed receiver subsets."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from tools import gate_crossfit_hierarchical_map as single
from tools import run_tpuic_receiver_benchmark as bench
from tools.screen_distributed_detector_headroom import _receiver_scores


def _role(scene, args):
    if scene < args.train_scenes:
        return "train"
    if scene < args.train_scenes + args.calibration_scenes:
        return "calibration"
    return "test"


def _aggregate(rows, subset):
    values = np.asarray([row["scores"] for row in rows], dtype=float)[:, :, subset]
    return np.sum(values, axis=2)


def _auc(rows, subset):
    values = _aggregate(rows, subset)
    return float(bench._auc(values[:, 0], values[:, 1]))


def _evaluate(cal, test, subset, p_fa):
    cal_values = _aggregate(cal, subset)[:, 0]
    test_values = _aggregate(test, subset)
    threshold = bench._calibrated_threshold(
        cal_values, p_fa, "split_conformal")
    return {"subset": list(subset), "threshold": threshold,
            "auc": float(bench._auc(test_values[:, 0], test_values[:, 1])),
            "pfa": float(np.mean(test_values[:, 0] > threshold)),
            "pd": float(np.mean(test_values[:, 1] > threshold)),
            "h0_cal_test_ks": float(ks_2samp(
                cal_values, test_values[:, 0]).statistic)}


def _paired_auc_delta_ci(test, left, right, seed=20261018, draws=10000):
    a = _aggregate(test, left); b = _aggregate(test, right)
    rng = np.random.default_rng(seed)
    deltas = []
    for index in rng.integers(0, len(test), size=(draws, len(test))):
        deltas.append(bench._auc(a[index, 0], a[index, 1])
                      - bench._auc(b[index, 0], b[index, 1]))
    return np.quantile(deltas, [.025, .5, .975]).tolist()


def _scene_scores(cfg, args, scene, path):
    if path.exists():
        return np.load(path)["scores"]
    truth, belief, base0 = bench._scene(
        cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz")
    base = bench._boost_direct(base0, args.boost_db)
    scores = np.asarray([_receiver_scores(
        cfg, truth, belief, base, args, scene, receiver)
        for receiver in range(int(cfg.scale.M))]).T
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, scores=scores)
    return scores


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = single._cfg(args)
    receivers = tuple(range(int(cfg.scale.M)))
    total = args.train_scenes + args.calibration_scenes + args.test_scenes
    rows = []
    for scene in range(total):
        path = args.out / "scores" / f"scene_{scene:05d}.npz"
        rows.append({"scene_id": scene, "split": _role(scene, args),
                     "scores": _scene_scores(cfg, args, scene, path).tolist()})
    train = [row for row in rows if row["split"] == "train"]
    cal = [row for row in rows if row["split"] == "calibration"]
    test = [row for row in rows if row["split"] == "test"]
    selected = {}
    train_tables = {}
    for size in (1, 2, 3):
        candidates = [(subset, _auc(train, subset))
                      for subset in itertools.combinations(receivers, size)]
        train_tables[str(size)] = [
            {"subset": list(subset), "auc": auc} for subset, auc in candidates]
        subset, auc = max(candidates, key=lambda item: (item[1], item[0]))
        selected[f"train_selected_k{size}"] = (subset, auc)
    selected["receiver_1"] = ((1,), _auc(train, (1,)))
    selected["all_6"] = (receivers, _auc(train, receivers))
    evaluation = {}
    for name, (subset, train_auc) in selected.items():
        evaluation[name] = {"train_auc": train_auc,
                            **_evaluate(cal, test, subset, args.p_fa)}
    baseline = selected["train_selected_k1"][0]
    paired = {name: _paired_auc_delta_ci(test, subset, baseline)
              for name, (subset, _) in selected.items()
              if name != "train_selected_k1" and name != "receiver_1"}
    payload = {
        "protocol": vars(args) | {"out": str(args.out)},
        "shared_scene_contract": True, "selection": "train AUC, sum statistic",
        "evaluation": evaluation, "train_candidate_tables": train_tables,
        "paired_auc_delta_vs_train_selected_k1_ci95": paired,
        "limitations": ["single target and +50 dB only",
                        "ideal central score fusion without communication loss"],
    }
    (args.out / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=20)
    parser.add_argument("--calibration-scenes", type=int, default=40)
    parser.add_argument("--test-scenes", type=int, default=40)
    parser.add_argument("--master-seed", type=int, default=20261017)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--boost-db", type=float, default=50.0)
    parser.add_argument("--gn-max-nfev", type=int, default=4)
    parser.add_argument("--p-fa", type=float, default=.05)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
