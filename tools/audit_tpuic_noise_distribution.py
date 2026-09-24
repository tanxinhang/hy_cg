#!/usr/bin/env python3
"""Audit conditional TP-UIC noise-energy distributions without changing it."""
from __future__ import annotations

import argparse
import csv
import json
from copy import copy
from pathlib import Path

import numpy as np

from isac_sim.receiver import cancellation as cx
from tools import gate_crossfit_hierarchical_map as single
from tools import run_tpuic_receiver_benchmark as bench
from tools.tpuic_residual_decomposition import noise_energy_spectrum


def _distribution(weights, unit_count, sigma2, draws, rng):
    sample = rng.gamma(unit_count, 1.0, draws) if unit_count else np.zeros(draws)
    for weight in weights:
        sample += float(weight) * rng.exponential(size=draws)
    sample *= float(sigma2)
    mean = float(sigma2) * (unit_count + float(np.sum(weights)))
    variance = float(sigma2) ** 2 * (
        unit_count + float(np.sum(np.asarray(weights) ** 2)))
    third_sum = unit_count + float(np.sum(np.asarray(weights) ** 3))
    centered = np.maximum(sample - mean, 0.0)
    return {
        "mean": mean, "std": float(np.sqrt(variance)),
        "cv": float(np.sqrt(variance) / mean),
        "effective_shape": float(mean * mean / variance),
        "skewness": float(2.0 * third_sum / (variance / sigma2 ** 2) ** 1.5),
        "probability_above_mean": float(np.mean(sample > mean)),
        "mean_positive_excess": float(np.mean(centered)),
        "positive_excess_over_mean": float(np.mean(centered) / mean),
        "q90_over_mean": float(np.quantile(sample, 0.90) / mean),
        "q95_over_mean": float(np.quantile(sample, 0.95) / mean),
        "q99_over_mean": float(np.quantile(sample, 0.99) / mean),
    }


def run(args):
    cfg_args = argparse.Namespace(out=args.source, master_seed=args.master_seed)
    cfg = single._cfg(cfg_args)
    rows = []
    for scene in args.scenes:
        truth, belief, base0 = bench._scene(
            cfg, scene, args.source / "scenes" / f"scene_{scene:05d}.npz")
        for boost in args.boost_db:
            pair_args = copy(args)
            base = bench._boost_direct(base0, boost)
            pairs = [single._pair(
                cfg, truth, belief, base, pair_args, scene, look
            ) for look in range(2)]
            for held_index in range(2):
                reference = pairs[1 - held_index][1]
                sources = cx.refine_direct_dd_joint_gn(
                    cfg, [reference], max_nfev=args.gn_max_nfev)
                held = cx.apply_direct_dd(cfg, pairs[held_index][1], sources)
                results = cx.cancellation_arms(cfg, held, weak_target=args.target)
                weights, unit_count = noise_energy_spectrum(
                    cfg, held, results, "tp_uic_full")
                stats = _distribution(
                    weights, unit_count, held.sigma2, args.draws,
                    np.random.default_rng([args.seed, scene, int(boost), held_index]))
                rows.append({"scene": scene, "boost_db": boost,
                             "held_index": held_index, "dimension": held.y.size,
                             "exceptional_rank": len(weights), **stats})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows(rows)
    summary = {str(boost): {key: float(np.mean([row[key] for row in rows
        if row["boost_db"] == boost])) for key in rows[0] if key not in {
            "scene", "boost_db", "held_index"}} for boost in args.boost_db}
    payload = {"protocol": vars(args) | {"source": str(args.source),
        "out": str(args.out)}, "summary_by_boost": summary}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", type=lambda v: tuple(map(int, v.split(","))),
                        default=(2, 3, 4, 5))
    parser.add_argument("--boost-db", type=lambda v: tuple(map(float, v.split(","))),
                        default=(10.0, 30.0, 50.0))
    parser.add_argument("--receiver", type=int, default=1)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--master-seed", type=int, default=20261010)
    parser.add_argument("--gn-max-nfev", type=int, default=4)
    parser.add_argument("--draws", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261011)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
