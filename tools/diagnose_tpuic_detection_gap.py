#!/usr/bin/env python3
"""Diagnose why TP-UIC cancellation depth does not improve detection information.

This is a mechanism probe, not a Monte-Carlo benchmark.  It fixes one
receiver/target pair, rebuilds paired observations across a few scenes, and
reports the exact GLRT information interface for every receiver arm.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_tpuic_receiver_benchmark as bench
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.sensing.model import radar_hardware_gain


ARMS = ("no_ic", "plain_ls", "protected_ls", "tp_uic_full", "perfect_channel")


def _cfg(args):
    return bench._make_cfg(SimpleNamespace(
        area_xy=args.area_xy, uavs=args.uavs, targets_count=args.targets_count,
        target_rcs=args.target_rcs, total_power_w=1.0, sensing_fraction=0.8,
        master_seed=args.master_seed, max_protected_targets=args.targets_count,
        direct_dd_sigma=args.direct_dd_sigma, belief_pos_sigma=20.0,
        belief_vel_sigma=3.0, m_rx=args.m_rx,
    ))


def _observation(cfg, truth, belief, base, receiver, target, scene, realisation):
    m = int(cfg.scale.M)
    power = np.full(m, float(cfg.radio.P_default) * float(cfg.radio.rho))
    rng = np.random.default_rng([
        int(cfg.run.seed), int(scene), int(receiver), int(target),
        int(realisation), 404,
    ])
    return cx.build_observation_pair(
        cfg, truth, belief.as_geometry(truth), base, int(receiver), rng=rng,
        sense_power=power, radiated_power=power,
        processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)),
        exclude_target=int(target), active_mask=np.ones(m, dtype=bool),
        weak_index=int(target), share_noise=False,
    )[0]


def _row(cfg, obs, arm, results, scene, realisation, receiver, target, boost):
    result = results[arm]
    model = gl.residual_model(cfg, obs, arm, results)
    out = gl.target_conditioned_glrt(
        cfg, obs, result, model, target=target, dictionary="belief",
        centre_only=True,
    )
    info = gl.detection_information(cfg, obs, result, model, target)
    cov = model.cov
    elevated = cov.sigma2 + cov.lam if cov.lam.size else np.zeros(0)
    return {
        "scene": scene,
        "realisation": realisation,
        "receiver": receiver,
        "target": target,
        "boost_db": boost,
        "arm": arm,
        "conflict_index": bench.interference_target_conflict(obs, target),
        "direct_inr_db": bench.direct_inr_db(obs),
        "kappa_db": result.kappa_db,
        "eta_survive_q": result.eta_survive_q,
        "model_rank": model.rank,
        "cov_rank": cov.rank,
        "cov_trace_over_noise": cov.trace / (cov.sigma2 * obs.y.size),
        "cov_max_eig_over_noise": (
            float(np.max(elevated) / cov.sigma2) if elevated.size else 1.0
        ),
        "transfer_energy": info.transfer_energy,
        "ncp_unit": info.value,
        "identifiable_fraction": info.identifiable_fraction,
        "statistic": out.statistic,
        "residual_power": out.residual_power,
        "whitened_residual_power": out.whitened_residual_power,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", type=int, default=6)
    p.add_argument("--realisations", type=int, default=2)
    p.add_argument("--receiver", type=int, default=0)
    p.add_argument("--target", type=int, default=1)
    p.add_argument("--boost-db", type=float, default=30.0)
    p.add_argument("--master-seed", type=int, default=20260922)
    p.add_argument("--uavs", type=int, default=6)
    p.add_argument("--targets-count", type=int, default=3)
    p.add_argument("--target-rcs", type=float, default=0.05)
    p.add_argument("--area-xy", type=float, default=800.0)
    p.add_argument("--m-rx", type=int, default=4)
    p.add_argument("--direct-dd-sigma", type=float, default=0.10)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(args)
    rows = []
    for scene in range(args.scenes):
        truth, belief, base0 = bench._scene(
            cfg, scene, out / "scenes" / f"scene_{scene:05d}.npz"
        )
        base = bench._boost_direct(base0, args.boost_db)
        for realisation in range(args.realisations):
            obs = _observation(
                cfg, truth, belief, base, args.receiver, args.target,
                scene, realisation,
            )
            results = cx.cancellation_arms(
                cfg, obs, weak_target=int(args.target)
            )
            rows.extend(_row(
                cfg, obs, arm, results, scene, realisation, args.receiver,
                args.target, args.boost_db,
            ) for arm in ARMS)

    with (out / "records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            key: float(np.median([float(r[key]) for r in selected]))
            for key in (
                "conflict_index", "direct_inr_db", "kappa_db",
                "eta_survive_q", "cov_trace_over_noise",
                "cov_max_eig_over_noise", "transfer_energy", "ncp_unit",
                "identifiable_fraction", "residual_power",
                "whitened_residual_power",
            )
        }
    base_ncp = summary["plain_ls"]["ncp_unit"]
    oracle_ncp = summary["perfect_channel"]["ncp_unit"]
    for arm in ARMS:
        summary[arm]["delta_ncp_vs_plain_ls"] = summary[arm]["ncp_unit"] - base_ncp
        summary[arm]["oracle_gap_ncp"] = oracle_ncp - summary[arm]["ncp_unit"]

    payload = {"protocol": vars(args), "summary": summary}
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
