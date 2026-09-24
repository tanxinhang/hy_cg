#!/usr/bin/env python3
"""Matched runtime/objective benchmark for grid and GN shared-DD MAP solvers."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from isac_sim.receiver import cancellation as cx
from isac_sim.receiver.cancellation.joint_dd_map import (
    _profiled_loss, _projected_looks)
from tools import gate_crossfit_hierarchical_map as gate
from tools import run_tpuic_receiver_benchmark as bench


def _objective(cfg, obs, sources):
    value = _profiled_loss(cfg, sources, _projected_looks([obs]))
    sl = float(cfg.cancellation.direct_estimation_sigma_delay_bins)
    sk = float(cfg.cancellation.direct_estimation_sigma_doppler_bins)
    for initial, source in zip(obs.direct_est, sources):
        value += ((source.delay_bin - initial.delay_bin) / sl) ** 2
        value += ((source.doppler_bin - initial.doppler_bin) / sk) ** 2
    return float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--boost-db", type=float, default=50.0)
    parser.add_argument("--master-seed", type=int, default=20261009)
    parser.add_argument("--gn-max-nfev", type=int, default=24)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = gate._cfg(SimpleNamespace(out=args.out, master_seed=args.master_seed))
    rows = []
    for scene in range(args.scenes):
        truth, belief, base0 = bench._scene(
            cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz")
        base = bench._boost_direct(base0, args.boost_db)
        pair = gate._pair(cfg, truth, belief, base,
                          SimpleNamespace(receiver=0, target=1), scene, 0)
        for hypothesis, obs in (("h1", pair[0]), ("h0", pair[1])):
            solved = {}
            solvers = (
                ("grid", lambda: cx.refine_direct_dd_joint(cfg, [obs])),
                ("gn", lambda: cx.refine_direct_dd_joint_gn(
                    cfg, [obs], max_nfev=args.gn_max_nfev)),
            )
            for name, function in solvers:
                started = time.perf_counter()
                sources = function()
                seconds = time.perf_counter() - started
                solved[name] = sources
                rows.append({"scene": scene, "hypothesis": hypothesis,
                             "solver": name, "seconds": seconds,
                             "objective": _objective(cfg, obs, sources)})
            delta = []
            for left, right in zip(solved["grid"], solved["gn"]):
                delta.extend([(left.doppler_bin - right.doppler_bin) / 0.1,
                              (left.delay_bin - right.delay_bin) / 0.1])
            rows[-1]["grid_gn_offset_rms_sigma"] = float(
                np.sqrt(np.mean(np.square(delta))))
    aggregate = {}
    for name in ("grid", "gn"):
        selected = [r for r in rows if r["solver"] == name]
        aggregate[name] = {
            "median_seconds": float(np.median([r["seconds"] for r in selected])),
            "median_objective": float(np.median([r["objective"] for r in selected])),
        }
    aggregate["speedup_grid_over_gn"] = (
        aggregate["grid"]["median_seconds"] / aggregate["gn"]["median_seconds"])
    payload = {"protocol": dict(vars(args)), "aggregate": aggregate, "rows": rows}
    payload["protocol"]["out"] = str(payload["protocol"]["out"])
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
