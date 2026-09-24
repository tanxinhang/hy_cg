#!/usr/bin/env python3
"""Screen disjoint same-CPI pilot MAP before detector integration."""
from __future__ import annotations

import argparse
import csv
import json
from copy import copy
from dataclasses import replace
from pathlib import Path

import numpy as np

from isac_sim.receiver import cancellation as cx
from tools import gate_crossfit_hierarchical_map as single
from tools import run_tpuic_receiver_benchmark as bench
from tools.pilot_map_detection import pilot_statistic


def _protected_dictionary(obs):
    basis = obs.basis_belief
    if basis is None or not basis.shape[1]:
        return obs.X
    return obs.X - basis @ (basis.conj().T @ obs.X)


def run(args):
    cfg = single._cfg(argparse.Namespace(
        out=args.source, master_seed=args.master_seed))
    rows = []
    for scene in args.scenes:
        truth, belief, base0 = bench._scene(
            cfg, scene, args.source / "scenes" / f"scene_{scene:05d}.npz")
        permutation = None
        for boost in args.boost_db:
            pair_args = copy(args)
            base = bench._boost_direct(base0, boost)
            pairs = [single._pair(
                cfg, truth, belief, base, pair_args, scene, look
            ) for look in range(2)]
            sources = cx.refine_direct_dd_joint_gn(
                cfg, [pairs[1][0]], max_nfev=args.gn_max_nfev)
            held = cx.apply_direct_dd(cfg, pairs[0][0], sources)
            sources_h0 = cx.refine_direct_dd_joint_gn(
                cfg, [pairs[1][1]], max_nfev=args.gn_max_nfev)
            held_h0 = cx.apply_direct_dd(cfg, pairs[0][1], sources_h0)
            if permutation is None:
                permutation = np.random.default_rng(
                    [args.seed, scene, args.receiver, args.target]
                ).permutation(held.y.size)
            self_result = cx.cancellation_arms(
                cfg, held, weak_target=args.target)["tp_uic_stage1"]
            self_dictionary = _protected_dictionary(held)
            for fraction in args.pilot_fractions:
                count = int(round(float(fraction) * held.y.size))
                pilot = np.sort(permutation[:count])
                sensing = np.sort(permutation[count:])
                estimate = cx.fit_pilot_map(
                    held, pilot, cfg.cancellation.prior_variance)
                residual, factor = cx.apply_pilot_map(held, estimate, sensing)
                direct_left = held.x_direct[sensing] - held.X[sensing] @ estimate.h_hat
                oracle = cx.fit_pilot_map(
                    replace(held, y=held.x_direct), pilot,
                    cfg.cancellation.prior_variance)
                oracle_left = held.x_direct[sensing] - held.X[sensing] @ oracle.h_hat
                self_left = (
                    held.x_direct[sensing]
                    - self_dictionary[sensing] @ self_result.h_hat)
                input_energy = max(float(np.vdot(
                    held.x_direct[sensing], held.x_direct[sensing]).real),
                    np.finfo(float).tiny)
                target_total = max(float(np.vdot(
                    held.s_target, held.s_target).real), np.finfo(float).tiny)
                rows.append({
                    "scene": scene, "boost_db": boost,
                    "pilot_fraction": fraction, "pilot_rows": count,
                    "sensing_rows": sensing.size,
                    "pilot_direct_fraction": float(
                        np.vdot(direct_left, direct_left).real / input_energy),
                    "oracle_pilot_direct_fraction": float(
                        np.vdot(oracle_left, oracle_left).real / input_energy),
                    "self_fit_direct_fraction": float(
                        np.vdot(self_left, self_left).real / input_energy),
                    "posterior_trace": float(np.trace(estimate.covariance).real),
                    "coefficient_error_trace": float(
                        np.linalg.norm(factor, "fro") ** 2),
                    "sensing_target_energy_fraction": float(
                        np.vdot(held.s_target[sensing], held.s_target[sensing]).real
                        / target_total),
                    "output_residual_power": float(np.vdot(residual, residual).real),
                    "h1_statistic": pilot_statistic(
                        cfg, held, pilot, sensing, args.target),
                    "h0_statistic": pilot_statistic(
                        cfg, held_h0, pilot, sensing, args.target),
                    "oracle_h1_statistic": pilot_statistic(
                        cfg, held, pilot, sensing, args.target, True),
                    "oracle_h0_statistic": pilot_statistic(
                        cfg, held_h0, pilot, sensing, args.target, True),
                })
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for boost in args.boost_db:
        summary[str(boost)] = {}
        for fraction in args.pilot_fractions:
            group = [r for r in rows if r["boost_db"] == boost
                     and r["pilot_fraction"] == fraction]
            summary[str(boost)][str(fraction)] = {
                key: float(np.mean([row[key] for row in group]))
                for key in rows[0] if key not in {
                    "scene", "boost_db", "pilot_fraction"}}
            summary[str(boost)][str(fraction)]["auc"] = bench._auc(
                [row["h0_statistic"] for row in group],
                [row["h1_statistic"] for row in group])
            summary[str(boost)][str(fraction)]["oracle_direct_auc"] = bench._auc(
                [row["oracle_h0_statistic"] for row in group],
                [row["oracle_h1_statistic"] for row in group])
    payload = {"protocol": vars(args) | {"source": str(args.source),
        "out": str(args.out)}, "summary": summary}
    (args.out / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", type=lambda v: tuple(map(int, v.split(","))),
                        default=(2, 3, 4, 5))
    parser.add_argument("--boost-db", type=lambda v: tuple(map(float, v.split(","))),
                        default=(10.0, 30.0, 50.0))
    parser.add_argument("--pilot-fractions",
                        type=lambda v: tuple(map(float, v.split(","))),
                        default=(0.125, 0.25, 0.5))
    parser.add_argument("--receiver", type=int, default=1)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--master-seed", type=int, default=20261010)
    parser.add_argument("--gn-max-nfev", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261012)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
