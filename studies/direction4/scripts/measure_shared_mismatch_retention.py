#!/usr/bin/env python3
"""Measure information retention under a shared target-position mismatch.

This is a cheap mechanism experiment.  It does not run target-state search or
threshold calibration.  For each receiver, it projects the truth target
dictionary onto the subspace spanned by the mismatched belief dictionary:

    eta_j = ||Q_belief^H A_truth||_F^2 / ||A_truth||_F^2.

For a receiver subset, the fused retention is the truth-energy-weighted mean
of eta_j.  Averaging over every subset of size M separates a systematic common
bias from receiver-selection luck.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isac_sim.cooperation.formation import minimum_uav_separation
from isac_sim.receiver.cancellation.dictionaries import target_dictionary
from isac_sim.scenario.belief.state import BeliefState
from isac_sim.sensing.model import build_base_gains, generate_geometry
from tools import gate_crossfit_target_state_map as gate


def _target_sources(sources, target: int):
    return tuple(src for src in (sources or ()) if int(src.target) == int(target))


def _retention(truth_dictionary, belief_dictionary):
    truth_energy = float(np.linalg.norm(truth_dictionary, "fro") ** 2)
    if truth_energy <= 0.0 or belief_dictionary.size == 0:
        return 0.0, truth_energy
    q_belief, _ = np.linalg.qr(belief_dictionary, mode="reduced")
    projected = q_belief.conj().T @ truth_dictionary
    retained = float(np.linalg.norm(projected, "fro") ** 2 / truth_energy)
    return float(np.clip(retained, 0.0, 1.0)), truth_energy


def _cfg_and_args(args):
    defaults = argparse.Namespace(
        out=args.out.parent, master_seed=args.master_seed, area_xy=400.0,
        uavs=6, targets_count=3,
        min_uav_separation_m=args.min_uav_separation_m, m_rx=4,
        target_rcs=args.target_rcs, prior_sigma_m=args.prior_sigma_m,
        glrt_radius=0.25, glrt_points=3, target=args.target,
    )
    return gate._cfg(defaults), defaults


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--radii-m", type=float, nargs="+",
                        default=(0.0, 50.0, 150.0, 250.0, 400.0))
    parser.add_argument("--target", type=int, default=2)
    parser.add_argument("--prior-sigma-m", type=float, default=150.0)
    parser.add_argument("--target-rcs", type=float, default=0.5)
    parser.add_argument("--min-uav-separation-m", type=float, default=20.0)
    parser.add_argument("--master-seed", type=int, default=20261007)
    parser.add_argument("--out", type=Path, default=Path(
        "studies/direction4/data/shared_mismatch_retention/summary.json"))
    args = parser.parse_args()
    cfg, gate_args = _cfg_and_args(args)
    receivers = tuple(range(int(cfg.scale.M)))
    rows = []

    for scene in range(args.scenes):
        truth = generate_geometry(
            cfg, np.random.default_rng([cfg.run.seed, scene, 101]))
        separation = minimum_uav_separation(truth.p_uav)
        if separation + 1e-9 < args.min_uav_separation_m:
            raise RuntimeError(f"scene {scene} violates UAV separation: {separation}")
        belief0 = BeliefState.from_truth(
            cfg, truth, np.random.default_rng([cfg.run.seed, scene, 202]))
        base = build_base_gains(
            cfg, truth, np.random.default_rng([cfg.run.seed, scene, 303]))

        for radius in args.radii_m:
            belief, _ = gate._belief_for_radius(
                cfg, truth, belief0, args.target, float(radius), scene)
            belief_geometry = belief.as_geometry(truth)
            receiver_values = []
            for rx in receivers:
                observation, _ = gate._pair(
                    cfg, truth, belief_geometry, base, gate_args, scene, rx, 0)
                truth_sources = _target_sources(observation.targets, args.target)
                belief_sources = _target_sources(
                    observation.targets_belief, args.target)
                truth_dictionary = target_dictionary(
                    cfg, truth_sources, covariance_expanded=False)
                belief_dictionary = target_dictionary(cfg, belief_sources)
                eta, energy = _retention(truth_dictionary, belief_dictionary)
                receiver_values.append((eta, energy))

            for count in (1, 2, 3, 4, 5, 6):
                subset_values = []
                for subset in itertools.combinations(range(6), count):
                    numerator = sum(receiver_values[j][0] * receiver_values[j][1]
                                    for j in subset)
                    denominator = sum(receiver_values[j][1] for j in subset)
                    subset_values.append(numerator / max(denominator, 1e-30))
                rows.append({
                    "scene": scene,
                    "error_radius_m": float(radius),
                    "normalized_error": float(radius / args.prior_sigma_m),
                    "receiver_count": count,
                    "retention_mean_over_subsets": float(np.mean(subset_values)),
                    "retention_sd_over_subsets": float(np.std(subset_values)),
                    "minimum_uav_separation_m": float(separation),
                })

    summary = []
    for radius in args.radii_m:
        for count in (1, 2, 3, 4, 5, 6):
            block = [r for r in rows
                     if r["error_radius_m"] == float(radius)
                     and r["receiver_count"] == count]
            values = np.asarray([r["retention_mean_over_subsets"] for r in block])
            subset_sd = np.asarray([r["retention_sd_over_subsets"] for r in block])
            summary.append({
                "error_radius_m": float(radius),
                "normalized_error": float(radius / args.prior_sigma_m),
                "receiver_count": count,
                "retention_mean": float(np.mean(values)),
                "retention_ci95_low": float(np.quantile(values, 0.025)),
                "retention_ci95_high": float(np.quantile(values, 0.975)),
                "mean_subset_sd": float(np.mean(subset_sd)),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": {
            "scenes": args.scenes,
            "radii_m": list(args.radii_m),
            "prior_sigma_m": args.prior_sigma_m,
            "target": args.target,
            "receivers": 6,
            "min_uav_separation_m": args.min_uav_separation_m,
            "metric": "truth-energy-weighted subspace retention",
            "scope": "mechanism diagnostic; not a detection-performance claim",
        },
        "summary": summary,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
