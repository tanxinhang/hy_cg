#!/usr/bin/env python3
"""Screen 1/2/4-reference coefficient MAP without changing the detector."""
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


def _projected_dictionary(obs):
    basis = obs.basis_belief
    if basis is None or not basis.shape[1]:
        return obs.X
    return obs.X - basis @ (basis.conj().T @ obs.X)


def run(args):
    cfg = single._cfg(argparse.Namespace(
        out=args.source, master_seed=args.master_seed))
    rows = []
    max_reference = max(args.reference_counts)
    for scene in args.scenes:
        truth, belief, base0 = bench._scene(
            cfg, scene, args.source / "scenes" / f"scene_{scene:05d}.npz")
        for boost in args.boost_db:
            base = bench._boost_direct(base0, boost)
            pair_args = copy(args)
            pairs = [single._pair(
                cfg, truth, belief, base, pair_args, scene, look
            ) for look in range(max_reference + 1)]
            raw_references = [pair[1] for pair in pairs[:max_reference]]
            raw_sensing = pairs[max_reference][1]
            for count in args.reference_counts:
                sources = cx.refine_direct_dd_joint_gn(
                    cfg, raw_references[:count], max_nfev=args.gn_max_nfev)
                references = [cx.apply_direct_dd(cfg, obs, sources)
                              for obs in raw_references[:count]]
                sensing = cx.apply_direct_dd(cfg, raw_sensing, sources)
                estimate = cx.fit_reference_map(
                    references, cfg.cancellation.prior_variance)
                residual, factor = cx.apply_reference_map(sensing, estimate)
                dictionary = _projected_dictionary(sensing)
                direct_left = sensing.x_direct - dictionary @ estimate.h_hat
                oracle_refs = [replace(obs, y=obs.x_direct) for obs in references]
                oracle = cx.fit_reference_map(
                    oracle_refs, cfg.cancellation.prior_variance)
                oracle_left = sensing.x_direct - dictionary @ oracle.h_hat
                sensing_oracle = cx.fit_reference_map(
                    [replace(sensing, y=sensing.x_direct)],
                    cfg.cancellation.prior_variance)
                coefficient_scale = max(
                    float(np.linalg.norm(sensing_oracle.h_hat)),
                    np.finfo(float).tiny)
                coherence_scale = max(
                    float(np.linalg.norm(oracle.h_hat)) * coefficient_scale,
                    np.finfo(float).tiny)
                self_result = cx.cancellation_arms(
                    cfg, sensing, weak_target=args.target)["tp_uic_stage1"]
                rows.append({
                    "scene": scene, "boost_db": boost,
                    "n_reference": count,
                    "reference_direct_residual": float(
                        np.vdot(direct_left, direct_left).real),
                    "oracle_reference_direct_residual": float(
                        np.vdot(oracle_left, oracle_left).real),
                    "self_fit_structural_residual": self_result.i_res_structural,
                    "self_fit_noise_energy": self_result.i_res_estimate,
                    "posterior_trace": float(np.trace(estimate.covariance).real),
                    "coefficient_error_trace": float(
                        np.linalg.norm(factor, "fro") ** 2),
                    "oracle_coefficient_relative_error": float(
                        np.linalg.norm(oracle.h_hat - sensing_oracle.h_hat)
                        / coefficient_scale),
                    "oracle_coefficient_coherence": float(abs(
                        np.vdot(oracle.h_hat, sensing_oracle.h_hat))
                        / coherence_scale),
                    "dictionary_relative_change": float(
                        np.linalg.norm(references[0].X - sensing.X)
                        / max(np.linalg.norm(sensing.X), np.finfo(float).tiny)),
                    "output_residual_power": float(np.vdot(residual, residual).real),
                })
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for boost in args.boost_db:
        summary[str(boost)] = {}
        for count in args.reference_counts:
            group = [r for r in rows if r["boost_db"] == boost
                     and r["n_reference"] == count]
            summary[str(boost)][str(count)] = {
                key: float(np.mean([row[key] for row in group]))
                for key in rows[0] if key not in {"scene", "boost_db", "n_reference"}}
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
    parser.add_argument("--reference-counts",
                        type=lambda v: tuple(map(int, v.split(","))), default=(1, 2, 4))
    parser.add_argument("--receiver", type=int, default=1)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--master-seed", type=int, default=20261010)
    parser.add_argument("--gn-max-nfev", type=int, default=4)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
