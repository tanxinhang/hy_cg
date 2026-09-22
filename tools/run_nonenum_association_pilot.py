#!/usr/bin/env python3
"""Non-enumerative UAV association pilot from paired receiver statistics.

Candidate generation is deliberately budgeted: SINR and independent-information
starts, stratified random subsets, narrow beam search, and one-swap local search.
No full subset enumeration is used in the experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def _auc(h0, h1):
    x0, x1 = np.asarray(h0, float), np.asarray(h1, float)
    ranks = rankdata(np.r_[x0, x1], method="average")
    return float((ranks[x0.size:].sum() - x1.size * (x1.size + 1) / 2) /
                 (x0.size * x1.size))


def _fit(h0, h1, correlated=True, ridge=0.1):
    delta = h1.mean(0) - h0.mean(0)
    pooled = np.cov(h0, rowvar=False, ddof=1) + np.cov(h1, rowvar=False, ddof=1)
    pooled = np.atleast_2d(pooled) / 2.0
    if not correlated:
        pooled = np.diag(np.diag(pooled))
    scale = max(float(np.trace(pooled)) / pooled.shape[0], 1e-12)
    cov = (1.0 - ridge) * pooled + ridge * scale * np.eye(pooled.shape[0])
    weights = np.linalg.solve(cov, delta)
    score = float(delta @ weights)
    return weights, cov, score


def _evaluate_subset(data, subset):
    idx = list(subset)
    tr0, tr1 = data["train_h0"][:, idx], data["train_h1"][:, idx]
    te0, te1 = data["test_h0"][:, idx], data["test_h1"][:, idx]
    wc, cov, score = _fit(tr0, tr1, correlated=True)
    wi, _, independent_score = _fit(tr0, tr1, correlated=False)
    cal = tr0 @ wc
    threshold = float(np.quantile(cal, 0.95, method="higher"))
    s0, s1 = te0 @ wc, te1 @ wc
    return {
        "subset": [int(v) for v in subset],
        "train_joint_information": score,
        "train_independent_information": independent_score,
        "test_auc_corr": _auc(s0, s1),
        "test_auc_independent": _auc(te0 @ wi, te1 @ wi),
        "test_pfa_corr": float(np.mean(s0 > threshold)),
        "test_pd_corr": float(np.mean(s1 > threshold)),
        "mean_abs_offdiag_correlation": float(
            np.mean(np.abs((cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov))))
                           [~np.eye(len(idx), dtype=bool)]))
        ) if len(idx) > 1 else 0.0,
    }


def _beam(data, receivers, k, width, seen):
    beam = [tuple()]
    for _size in range(1, k + 1):
        candidates = set()
        for partial in beam:
            for receiver in receivers:
                if receiver not in partial:
                    candidates.add(tuple(sorted((*partial, receiver))))
        scored = []
        for subset in candidates:
            seen.add(subset)
            scored.append((_evaluate_subset(data, subset)["train_joint_information"], subset))
        beam = [subset for _, subset in sorted(scored, reverse=True)[:width]]
    return beam[0]


def _swap(data, start, receivers, shortlist, seen, rounds=1):
    current = tuple(sorted(start))
    current_score = _evaluate_subset(data, current)["train_joint_information"]
    seen.add(current)
    for _ in range(rounds):
        best = (current_score, current)
        outside = [r for r in shortlist if r not in current]
        for drop in current:
            for add in outside:
                candidate = tuple(sorted((set(current) - {drop}) | {add}))
                seen.add(candidate)
                score = _evaluate_subset(data, candidate)["train_joint_information"]
                if score > best[0] + 1e-12:
                    best = (score, candidate)
        if best[1] == current:
            break
        current_score, current = best
    return current


def _budget_summary(seen, k, receiver_count):
    total = int(math.comb(receiver_count, k))
    evaluated = sum(len(s) == k for s in seen)
    return {
        "evaluated_search_states_total": len(seen),
        "evaluated_unique_k_subsets": evaluated,
        "total_possible_k_subsets": total,
        "k_subset_fraction_evaluated": evaluated / total,
    }


def _load(path, arm, boost):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] == arm and float(row["direct_gain_boost_db"]) == boost:
                rows.append(row)
    receivers = sorted({int(r["receiver"]) for r in rows})
    targets = sorted({int(r["target"]) for r in rows})
    out = {}
    for target in targets:
        per_split = {}
        sinr = np.zeros(len(receivers))
        for split in ("calibration", "test"):
            selected = [r for r in rows if int(r["target"]) == target and r["split"] == split]
            samples = defaultdict(dict)
            for r in selected:
                key = (int(r["scene_id"]), int(r["realisation"]))
                samples[key][int(r["receiver"])] = (
                    float(r["stat_h0"]), float(r["stat_h1"]), float(r["post_ic_sinr"])
                )
            complete = [samples[key] for key in sorted(samples)
                        if all(receiver in samples[key] for receiver in receivers)]
            per_split[split] = {
                "h0": np.asarray([[s[r][0] for r in receivers] for s in complete]),
                "h1": np.asarray([[s[r][1] for r in receivers] for s in complete]),
            }
            if split == "calibration":
                sinr = np.median([[s[r][2] for r in receivers] for s in complete], axis=0)
        out[target] = {
            "train_h0": per_split["calibration"]["h0"],
            "train_h1": per_split["calibration"]["h1"],
            "test_h0": per_split["test"]["h0"],
            "test_h1": per_split["test"]["h1"],
            "sinr": sinr,
        }
    return receivers, targets, out


def run(path, arm, boost, k, seed, beam_width, random_count):
    receivers, targets, datasets = _load(path, arm, boost)
    total_k_subsets = int(math.comb(len(receivers), k))
    if random_count > total_k_subsets:
        raise ValueError(
            f"random_count={random_count} exceeds the {total_k_subsets} distinct K-subsets"
        )
    rng = np.random.default_rng(seed)
    methods = defaultdict(dict)
    budgets = {}
    for target in targets:
        data = datasets[target]
        delta = data["train_h1"].mean(0) - data["train_h0"].mean(0)
        var = (data["train_h0"].var(0, ddof=1) + data["train_h1"].var(0, ddof=1)) / 2
        independent = delta * delta / np.maximum(var, 1e-12)
        sinr_start = tuple(sorted(np.argsort(data["sinr"])[-k:].tolist()))
        info_start = tuple(sorted(np.argsort(independent)[-k:].tolist()))
        seen = {sinr_start, info_start}

        shortlist = list(np.argsort(independent)[::-1][:min(len(receivers), k + 1)])
        beam = _beam(data, receivers, k, beam_width, seen)
        swap_sinr = _swap(data, sinr_start, receivers, shortlist, seen)
        swap_info = _swap(data, info_start, receivers, shortlist, seen)

        random_subsets = set()
        while len(random_subsets) < random_count:
            random_subsets.add(tuple(sorted(rng.choice(receivers, k, replace=False).tolist())))
        for subset in random_subsets:
            seen.add(subset)
        random_best = max(
            random_subsets,
            key=lambda s: _evaluate_subset(data, s)["train_joint_information"],
        )

        choices = {
            "sinr_topk": sinr_start,
            "independent_topk": info_start,
            "corr_beam": beam,
            "corr_swap_sinr": swap_sinr,
            "corr_swap_independent": swap_info,
            "random_best_train": random_best,
        }
        for name, subset in choices.items():
            methods[name][target] = _evaluate_subset(data, subset)
        budgets[target] = _budget_summary(seen, k, len(receivers))

    summary = {}
    for name, by_target in methods.items():
        summary[name] = {
            "worst_test_auc_corr": min(v["test_auc_corr"] for v in by_target.values()),
            "mean_test_auc_corr": float(np.mean([v["test_auc_corr"] for v in by_target.values()])),
            "worst_test_pd_corr": min(v["test_pd_corr"] for v in by_target.values()),
            "per_target": by_target,
        }
    return {
        "protocol": "nonenumerative_residual_aware_association_pilot_v1",
        "input": str(Path(path).resolve()),
        "arm": arm,
        "boost_db": boost,
        "k": k,
        "receivers": receivers,
        "targets": targets,
        "candidate_budget": budgets,
        "summary": summary,
        "limitations": [
            "six calibration and six held-out samples per target",
            "post_ic_sinr is a truth-assisted diagnostic baseline",
            "screening result only; no significance claim",
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
    p.add_argument("--beam-width", type=int, default=1)
    p.add_argument("--random-count", type=int, default=2)
    args = p.parse_args()
    result = run(args.records, args.arm, args.boost_db, args.k, args.seed,
                 args.beam_width, args.random_count)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
