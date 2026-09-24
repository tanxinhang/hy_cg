#!/usr/bin/env python3
"""Receiver-consistent C2F shortlist and train-only fine subset selection."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from tools import run_tpuic_receiver_benchmark as bench


def _load_scores(root, start, stop):
    return np.asarray([np.load(root / "scores" / f"scene_{i:05d}.npz")["scores"]
                       for i in range(start, stop)], dtype=float)


def _coarse_receiver_score(root, scenes, target, boost_db):
    values = []
    boost = 10.0 ** (boost_db / 10.0)
    for scene in range(scenes):
        data = np.load(root / "scenes" / f"scene_{scene:05d}.npz")
        gain = data["base_target_gain"][:, :, target]
        eta = data["base_eta_fine"][:, :, target]
        valid = data["base_valid_dd"][:, :, target]
        direct = data["base_direct_gain"] * boost
        signal = np.sum(gain * eta * valid, axis=0)
        interference = np.sum(direct, axis=0)
        values.append(np.log10((signal + 1e-30) / (interference + 1e-30)))
    return np.median(np.asarray(values), axis=0)


def _auc(values, subset):
    ids = np.asarray(subset, dtype=int)
    score = np.sum(values[:, :, ids], axis=2)
    return float(bench._auc(score[:, 0], score[:, 1]))


def _select(train, shortlist, size):
    candidates = [(subset, _auc(train, subset))
                  for subset in itertools.combinations(shortlist, size)]
    return max(candidates, key=lambda item: (item[1], item[0])), candidates


def _evaluate(cal, test, subset, p_fa):
    ids = np.asarray(subset, dtype=int)
    c0 = np.sum(cal[:, 0, ids], axis=1)
    t0 = np.sum(test[:, 0, ids], axis=1)
    t1 = np.sum(test[:, 1, ids], axis=1)
    threshold = bench._calibrated_threshold(c0, p_fa, "split_conformal")
    return {"subset": list(subset), "threshold": threshold,
            "auc": float(bench._auc(t0, t1)),
            "pfa": float(np.mean(t0 > threshold)),
            "pd": float(np.mean(t1 > threshold))}


def _delta_ci(test, subset, baseline, seed=20261020, draws=10000):
    def scores(ids):
        ids = np.asarray(ids, dtype=int)
        return np.sum(test[:, :, ids], axis=2)
    left, right = scores(subset), scores(baseline)
    rng = np.random.default_rng(seed)
    delta = []
    for index in rng.integers(0, len(test), size=(draws, len(test))):
        delta.append(bench._auc(left[index, 0], left[index, 1])
                     - bench._auc(right[index, 0], right[index, 1]))
    return np.quantile(delta, [.025, .5, .975]).tolist()


def run(args):
    train = _load_scores(args.input, 0, args.train_scenes)
    cal = _load_scores(args.input, args.train_scenes,
                       args.train_scenes + args.calibration_scenes)
    test = _load_scores(args.input, args.train_scenes + args.calibration_scenes,
                        args.train_scenes + args.calibration_scenes + args.test_scenes)
    coarse = _coarse_receiver_score(
        args.input, args.train_scenes, args.target, args.boost_db)
    shortlist = tuple(np.argsort(coarse)[::-1][:args.shortlist_size].tolist())
    evaluation, fine_tables, subsets = {}, {}, {}
    for size in (1, 2, 3):
        (subset, train_auc), candidates = _select(train, shortlist, size)
        name = f"c2f_k{size}"; subsets[name] = subset
        evaluation[name] = {"train_auc": train_auc,
                            **_evaluate(cal, test, subset, args.p_fa)}
        fine_tables[name] = [{"subset": list(s), "train_auc": auc}
                             for s, auc in candidates]
    all_ids = tuple(range(train.shape[2])); subsets["all_6"] = all_ids
    evaluation["all_6"] = _evaluate(cal, test, all_ids, args.p_fa)
    baseline = subsets["c2f_k1"]
    paired = {name: _delta_ci(test, subset, baseline)
              for name, subset in subsets.items() if name != "c2f_k1"}
    payload = {"protocol": vars(args) | {"input": str(args.input)},
               "coarse_proxy": "median log10(sum target_gain*eta_fine*valid / boosted direct_gain)",
               "coarse_scores": coarse.tolist(), "shortlist": list(shortlist),
               "evaluation": evaluation, "fine_train_tables": fine_tables,
               "paired_auc_delta_vs_c2f_k1_ci95": paired,
               "status": "exploratory reuse of frozen test; no promotion"}
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=20)
    parser.add_argument("--calibration-scenes", type=int, default=40)
    parser.add_argument("--test-scenes", type=int, default=40)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--boost-db", type=float, default=50.0)
    parser.add_argument("--shortlist-size", type=int, default=4)
    parser.add_argument("--p-fa", type=float, default=.05)
    args = parser.parse_args(); args.out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
