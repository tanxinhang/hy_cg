#!/usr/bin/env python3
"""Gate whether receiver-known local bases cover TP-UIC's oracle residual.

The oracle direction is used only as a diagnostic label.  Every candidate
basis is assembled from receiver-estimated direct and belief-target
dictionaries, then evaluated in the sigma-point detector covariance geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.receiver.cancellation_glrt.local_residual_basis import (
    projection_coverage,
    receiver_known_local_bases,
    whitened_projection_coverage,
)
from tools import diagnose_tpuic_detection_gap as probe
from tools import run_tpuic_receiver_benchmark as bench


MEDIAN_GATE = 0.80
Q25_GATE = 0.50


def _config(args):
    defaults = bench.build_parser().parse_args(["--out", str(args.out)])
    defaults.area_xy = args.area_xy
    defaults.uavs = args.uavs
    defaults.targets_count = args.targets_count
    defaults.target_rcs = args.target_rcs
    defaults.master_seed = args.master_seed
    defaults.max_protected_targets = args.targets_count
    defaults.direct_dd_sigma = args.direct_dd_sigma
    defaults.m_rx = args.m_rx
    defaults.interference_tangent_order = 1
    defaults.interference_uncertainty_weighted = True
    defaults.direct_mismatch_covariance_model = "sigma_point"
    return bench._make_cfg(defaults)


def _summary(rows):
    output = {}
    for name in dict.fromkeys(row["basis"] for row in rows):
        chosen = [row for row in rows if row["basis"] == name]
        values = np.asarray([row["whitened_coverage"] for row in chosen], dtype=float)
        median = float(np.median(values))
        q25 = float(np.quantile(values, 0.25))
        output[name] = {
            "count": int(values.size),
            "median": median,
            "q25": q25,
            "minimum": float(np.min(values)),
            "median_rank": float(np.median([row["rank"] for row in chosen])),
            "raw_median": float(np.median([
                row["raw_coverage"] for row in chosen
            ])),
            "passes": bool(median >= MEDIAN_GATE and q25 >= Q25_GATE),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--realisations", type=int, default=2)
    parser.add_argument("--receiver", type=int, default=0)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--boost-db", type=float, default=30.0)
    parser.add_argument("--master-seed", type=int, default=20260922)
    parser.add_argument("--uavs", type=int, default=6)
    parser.add_argument("--targets-count", type=int, default=3)
    parser.add_argument("--target-rcs", type=float, default=0.05)
    parser.add_argument("--area-xy", type=float, default=800.0)
    parser.add_argument("--m-rx", type=int, default=4)
    parser.add_argument("--direct-dd-sigma", type=float, default=0.10)
    args = parser.parse_args()
    if args.scenes <= 0 or args.realisations <= 0:
        raise ValueError("scenes and realisations must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = _config(args)
    rows = []
    for scene in range(args.scenes):
        truth, belief, base0 = bench._scene(
            cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz"
        )
        base = bench._boost_direct(base0, args.boost_db)
        for realisation in range(args.realisations):
            obs = probe._observation(
                cfg, truth, belief, base, args.receiver, args.target,
                scene, realisation,
            )
            results = cx.cancellation_arms(cfg, obs, weak_target=args.target)
            model = gl.residual_model(cfg, obs, "tp_uic_full", results)
            oracle_direction = model.signal_transfer(obs.x_direct[:, None])[:, 0]
            for name, basis in receiver_known_local_bases(obs, args.target).items():
                # Both oracle label and candidate basis must live after the same
                # receiver map.  Comparing pre-cancellation templates against a
                # post-cancellation residual would be a category error.
                residual_basis = model.signal_transfer(basis)
                coverage, rank = whitened_projection_coverage(
                    model.cov, oracle_direction, residual_basis
                )
                raw_coverage, _ = projection_coverage(
                    oracle_direction, residual_basis
                )
                rows.append({
                    "scene": scene, "realisation": realisation,
                    "receiver": args.receiver, "target": args.target,
                    "basis": name, "rank": rank,
                    "whitened_coverage": coverage,
                    "raw_coverage": raw_coverage,
                })

    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows)
    payload = {
        "protocol": vars(args) | {
            "covariance": "sigma_point", "tangent_order": 1,
            "uncertainty_weighted": True,
        },
        "gate": {"median_at_least": MEDIAN_GATE, "q25_at_least": Q25_GATE},
        "summary": summary,
        "decision": "proceed" if any(item["passes"] for item in summary.values()) else "stop",
        "warning": "x_direct is used only as an oracle diagnostic label, never in a candidate basis",
    }
    payload["protocol"]["out"] = str(payload["protocol"]["out"])
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
