#!/usr/bin/env python3
"""Robust, non-enumerative, common-subset UAV association pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_nonenum_association_pilot import _auc, _load


def _h0_covariance(h0):
    """Noise/residual dependence is estimated under H0 only (CFAR-consistent)."""
    return np.atleast_2d(np.cov(h0, rowvar=False, ddof=1))


def _shrunk_cov(h0, h1, shrink=0.7, diagonal=False):
    cov = _h0_covariance(h0)
    diag = np.diag(np.diag(cov))
    if diagonal:
        out = diag
    else:
        out = (1.0 - shrink) * cov + shrink * diag
    scale = max(float(np.trace(out)) / out.shape[0], 1e-12)
    return out + 1e-3 * scale * np.eye(out.shape[0])


def _information(data, subset, shrink=0.7, diagonal=False, rows=None):
    idx = list(subset)
    h0, h1 = data["train_h0"][:, idx], data["train_h1"][:, idx]
    if rows is not None:
        h0, h1 = h0[rows], h1[rows]
    delta = h1.mean(0) - h0.mean(0)
    cov = _shrunk_cov(h0, h1, shrink=shrink, diagonal=diagonal)
    return float(delta @ np.linalg.solve(cov, delta))


def _robust_utility(datasets, subset, bootstrap_rows, shrink=0.7, q=0.2):
    # Lower quantile over resampled calibration sets, then worst target.
    per_target = []
    for target in sorted(datasets):
        vals = [_information(datasets[target], subset, shrink, rows=rows)
                for rows in bootstrap_rows[target]]
        per_target.append(float(np.quantile(vals, q)))
    return min(per_target)


def _plain_utility(datasets, subset, diagonal):
    return min(_information(data, subset, diagonal=diagonal)
               for data in datasets.values())


def _greedy(receivers, k, utility):
    selected = tuple()
    evaluated = set()
    for _ in range(k):
        candidates = [tuple(sorted((*selected, r))) for r in receivers if r not in selected]
        scores = []
        for candidate in candidates:
            evaluated.add(candidate)
            scores.append((utility(candidate), candidate))
        selected = max(scores, key=lambda item: (item[0], tuple(-v for v in item[1])))[1]
    return selected, evaluated


def _conditional_gain(data, selected, candidate, shrink=0.7):
    """Schur-complement detection information added by one receiver."""
    idx = list(selected) + [candidate]
    h0, h1 = data["train_h0"][:, idx], data["train_h1"][:, idx]
    mu = h1.mean(0) - h0.mean(0)
    cov = _shrunk_cov(h0, h1, shrink=shrink)
    if not selected:
        return float(mu[-1] ** 2 / max(cov[-1, -1], 1e-12))
    r_ss, r_is = cov[:-1, :-1], cov[-1, :-1]
    inv_mu = np.linalg.solve(r_ss, mu[:-1])
    inv_r = np.linalg.solve(r_ss, cov[:-1, -1])
    innovation = float(mu[-1] - r_is @ inv_mu)
    conditional_var = float(cov[-1, -1] - r_is @ inv_r)
    return innovation * innovation / max(conditional_var, 1e-12)


def _cig_greedy(receivers, k, datasets, shrink=0.7):
    selected = tuple()
    evaluated = 0
    trace = []
    for _ in range(k):
        scored = []
        for receiver in receivers:
            if receiver in selected:
                continue
            gains = [_conditional_gain(data, selected, receiver, shrink)
                     for data in datasets.values()]
            scored.append((min(gains), receiver, gains))
            evaluated += 1
        score, receiver, gains = max(scored, key=lambda item: (item[0], -item[1]))
        selected = tuple(sorted((*selected, receiver)))
        trace.append({"receiver": int(receiver), "worst_gain": float(score),
                      "per_target_gain": [float(v) for v in gains]})
    return selected, evaluated, trace


def _one_swap(start, receivers, utility, evaluated):
    best = (utility(start), start)
    for drop in start:
        for add in receivers:
            if add in start:
                continue
            candidate = tuple(sorted((set(start) - {drop}) | {add}))
            evaluated.add(candidate)
            score = utility(candidate)
            if score > best[0] + 1e-12:
                best = (score, candidate)
    return best[1]


def _evaluate(datasets, subset, shrink=0.7):
    per_target = {}
    for target, data in datasets.items():
        idx = list(subset)
        tr0, tr1 = data["train_h0"][:, idx], data["train_h1"][:, idx]
        delta = tr1.mean(0) - tr0.mean(0)
        cov = _shrunk_cov(tr0, tr1, shrink=shrink)
        weights = np.linalg.solve(cov, delta)
        te0, te1 = data["test_h0"][:, idx], data["test_h1"][:, idx]
        threshold = float(np.quantile(tr0 @ weights, 0.95, method="higher"))
        s0, s1 = te0 @ weights, te1 @ weights
        per_target[int(target)] = {
            "test_auc": _auc(s0, s1),
            "test_pfa": float(np.mean(s0 > threshold)),
            "test_pd": float(np.mean(s1 > threshold)),
        }
    return {
        "subset": [int(v) for v in subset],
        "worst_test_auc": min(v["test_auc"] for v in per_target.values()),
        "mean_test_auc": float(np.mean([v["test_auc"] for v in per_target.values()])),
        "worst_test_pd": min(v["test_pd"] for v in per_target.values()),
        "per_target": per_target,
    }


def run(records, arm="tp_uic_full", boost=30.0, k=3, seed=20260922,
        bootstrap=100, shrink=0.7):
    receivers, targets, datasets = _load(records, arm, boost)
    rng = np.random.default_rng(seed)
    bootstrap_rows = {}
    for target, data in datasets.items():
        n = data["train_h0"].shape[0]
        bootstrap_rows[target] = [rng.integers(0, n, n) for _ in range(bootstrap)]

    # Common SINR baseline: maximize the worst per-target normalized rank.
    ranks = []
    for target in targets:
        order = np.argsort(np.argsort(datasets[target]["sinr"]))
        ranks.append(order / max(len(receivers) - 1, 1))
    sinr_subset = tuple(sorted(np.argsort(np.min(ranks, axis=0))[-k:].tolist()))

    independent_subset, independent_seen = _greedy(
        receivers, k, lambda s: _plain_utility(datasets, s, diagonal=True)
    )
    robust_utility = lambda s: _robust_utility(
        datasets, s, bootstrap_rows, shrink=shrink
    )
    robust_subset, robust_seen = _greedy(receivers, k, robust_utility)
    robust_subset = _one_swap(robust_subset, receivers, robust_utility, robust_seen)
    cig_subset, cig_evaluated, cig_trace = _cig_greedy(
        receivers, k, datasets, shrink=shrink
    )

    return {
        "protocol": "robust_common_subset_association_pilot_v1",
        "input": str(Path(records).resolve()),
        "settings": {"arm": arm, "boost_db": boost, "k": k,
                     "bootstrap": bootstrap, "shrink": shrink},
        "methods": {
            "common_sinr_topk": _evaluate(datasets, sinr_subset, shrink),
            "common_independent_greedy": _evaluate(datasets, independent_subset, shrink),
            "robust_corr_greedy_swap": _evaluate(datasets, robust_subset, shrink),
            "h0_schur_cig_greedy": _evaluate(datasets, cig_subset, shrink),
        },
        "search_budget": {
            "independent_states": len(independent_seen),
            "robust_states": len(robust_seen),
            "robust_complete_k_subsets": sum(len(s) == k for s in robust_seen),
            "cig_candidate_evaluations": cig_evaluated,
        },
        "cig_trace": cig_trace,
        "limitations": [
            "only six calibration and six held-out samples per target",
            "fixed shrinkage is used because nested tuning is unsupported at this sample size",
            "post-IC SINR is a truth-assisted diagnostic baseline",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arm", default="tp_uic_full")
    p.add_argument("--boost-db", type=float, default=30.0)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--bootstrap", type=int, default=100)
    p.add_argument("--shrink", type=float, default=0.7)
    args = p.parse_args()
    result = run(args.records, args.arm, args.boost_db, args.k, args.seed,
                 args.bootstrap, args.shrink)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
