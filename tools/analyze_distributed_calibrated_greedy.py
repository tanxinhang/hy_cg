#!/usr/bin/env python3
"""Train-only robust scaling and correlation-aware subset screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools import run_tpuic_receiver_benchmark as bench


def _load(root, start, stop):
    return np.asarray([np.load(root / "scores" / f"scene_{i:05d}.npz")["scores"]
                       for i in range(start, stop)], dtype=float)


def _fit_scale(train):
    h0 = train[:, 0]
    centre = np.median(h0, axis=0)
    q25, q75 = np.quantile(h0, [.25, .75], axis=0)
    scale = np.maximum((q75 - q25) / 1.3489795, 1e-6)
    return centre, scale


def _transform(values, centre, scale):
    return (values - centre[None, None, :]) / scale[None, None, :]


def _deflection(train, subset):
    h0, h1 = train[:, 0], train[:, 1]
    delta = np.mean(h1 - h0, axis=0)
    cov = np.cov(h0, rowvar=False, ddof=1)
    cov = .3 * cov + .7 * np.diag(np.diag(cov))
    ids = np.asarray(subset, dtype=int)
    signal = max(float(np.sum(delta[ids])), 0.0)
    variance = max(float(np.sum(cov[np.ix_(ids, ids)])), 1e-9)
    return signal * signal / variance


def _greedy(train, size):
    selected = []
    while len(selected) < size:
        candidates = [(j, _deflection(train, selected + [j]))
                      for j in range(train.shape[2]) if j not in selected]
        selected.append(max(candidates, key=lambda item: (item[1], -item[0]))[0])
    return tuple(selected)


def _evaluate(cal, test, subset, p_fa):
    ids = np.asarray(subset, dtype=int)
    cal0 = np.sum(cal[:, 0, ids], axis=1)
    t0 = np.sum(test[:, 0, ids], axis=1)
    t1 = np.sum(test[:, 1, ids], axis=1)
    threshold = bench._calibrated_threshold(cal0, p_fa, "split_conformal")
    return {"subset": list(subset), "threshold": threshold,
            "auc": float(bench._auc(t0, t1)),
            "pfa": float(np.mean(t0 > threshold)),
            "pd": float(np.mean(t1 > threshold))}


def _delta_ci(test, subset, baseline, seed=20261019, draws=10000):
    def scores(ids):
        ids = np.asarray(ids, dtype=int)
        return np.sum(test[:, :, ids], axis=2)
    left, right = scores(subset), scores(baseline)
    rng = np.random.default_rng(seed)
    values = []
    for index in rng.integers(0, len(test), size=(draws, len(test))):
        values.append(bench._auc(left[index, 0], left[index, 1])
                      - bench._auc(right[index, 0], right[index, 1]))
    return np.quantile(values, [.025, .5, .975]).tolist()


def run(args):
    train = _load(args.input, 0, args.train_scenes)
    cal = _load(args.input, args.train_scenes,
                args.train_scenes + args.calibration_scenes)
    test = _load(args.input, args.train_scenes + args.calibration_scenes,
                 args.train_scenes + args.calibration_scenes + args.test_scenes)
    centre, scale = _fit_scale(train)
    train, cal, test = (_transform(v, centre, scale) for v in (train, cal, test))
    local_auc = [bench._auc(train[:, 0, j], train[:, 1, j])
                 for j in range(train.shape[2])]
    ranking = tuple(np.argsort(local_auc)[::-1].tolist())
    subsets = {"info_k1": ranking[:1], "info_k2": ranking[:2],
               "info_k3": ranking[:3], "greedy_k2": _greedy(train, 2),
               "greedy_k3": _greedy(train, 3),
               "all_6": tuple(range(train.shape[2]))}
    evaluation = {name: _evaluate(cal, test, subset, args.p_fa)
                  for name, subset in subsets.items()}
    baseline = subsets["info_k1"]
    paired = {name: _delta_ci(test, subset, baseline)
              for name, subset in subsets.items() if name != "info_k1"}
    payload = {"protocol": vars(args) | {"input": str(args.input)},
               "normalization": {"centre": centre.tolist(), "scale": scale.tolist()},
               "train_local_auc": local_auc, "evaluation": evaluation,
               "paired_auc_delta_vs_info_k1_ci95": paired,
               "status": "exploratory reuse of frozen test; no method promotion"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=20)
    parser.add_argument("--calibration-scenes", type=int, default=40)
    parser.add_argument("--test-scenes", type=int, default=40)
    parser.add_argument("--p-fa", type=float, default=.05)
    print(json.dumps(run(parser.parse_args()), indent=2, default=str))


if __name__ == "__main__":
    main()
