#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the communication-side and sensing-side interference models.

Communication SINR counts only *active* transmitters (active_tx_mask) and uses
the raw channel gain.  Sensing SINR uses a *static* full-concurrency sum with
several residual-suppression factors.  This script quantifies the asymmetry.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np

from isac_sim.core.config import Config, apply_overrides
from isac_sim.sensing.model import (
    build_base_gains,
    compute_link_tables,
    generate_geometry,
    noise_power,
)

PAPER = {
    "geometry.uav_speed_min": 30,
    "geometry.target_speed_min": 50,
    "geometry.target_speed_max": 90,
    "comm.interference_model": "active_set",
}


def db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, float), 1e-300))


def main():
    cfg = overrides = apply_overrides(Config(), PAPER)
    r = cfg.radio
    M, Q = cfg.scale.M, cfg.scale.Q
    n0 = noise_power(cfg)
    P = r.P_default
    P_sense = r.rho * P
    P_comm = (1.0 - r.rho) * P

    print("=" * 78)
    print("COMM vs SENSING INTERFERENCE MODEL")
    print("=" * 78)
    print(f"  n0 = {n0:.4e} W,  P = {P} W,  P_sense = {P_sense} W,  P_comm = {P_comm} W")
    print(f"  residual_self_factor       = {r.residual_self_factor:g}")
    print(f"  residual_direct_factor     = {r.residual_direct_factor:g}")
    print(f"  residual_multi_uav_factor  = {r.residual_multi_uav_factor:g}")
    print(f"  comm_leakage_from_sensing  = {cfg.comm.comm_leakage_from_sensing:g}")
    print(f"  comm_direct_leakage_factor = {cfg.comm.comm_direct_leakage_factor:g}")
    print(f"  interference_model         = {cfg.comm.interference_model}")
    print()

    # collect per-trial quantities on a common link set
    rows = {k: [] for k in ["if_comm", "rs_self", "rs_direct", "rs_multi", "dg"]}
    for t in range(10):
        rng = np.random.default_rng([cfg.run.seed, t])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        eye = ~np.eye(M, dtype=bool)
        dg = base.direct_gain
        # communication interference, full concurrency (no active gate)
        interf = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                if i == j or not base.edge_mask[i, j]:
                    continue
                s = 0.0
                for k in range(M):
                    if k == i or k == j:
                        continue
                    s += (P_comm + cfg.comm.comm_leakage_from_sensing * P_sense) * dg[k, j]
                interf[i, j] = s
            # sensing residual components for the same (k!=i,j) interferer set
        rs_self = r.residual_self_factor * P
        rs_direct = np.zeros((M, M))
        rs_multi = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                if i == j or not base.edge_mask[i, j]:
                    continue
                sd = 0.0
                sm = 0.0
                for k in range(M):
                    if k == i or k == j:
                        continue
                    sd += r.residual_direct_factor * P * dg[k, j]
                    sm += r.residual_multi_uav_factor * P_sense * dg[k, j]
                rs_direct[i, j] = sd
                rs_multi[i, j] = sm
        rows["if_comm"].append(interf[eye])
        rows["rs_self"].append(np.full(eye.sum(), rs_self))
        rows["rs_direct"].append(rs_direct[eye])
        rows["rs_multi"].append(rs_multi[eye])
        rows["dg"].append(dg[eye])

    agg = {k: np.concatenate(v) for k, v in rows.items()}

    def line(name, arr):
        a = np.asarray(arr, float)
        print(f"  {name:26s} p50={np.median(a):10.4e}  mean={a.mean():10.4e}  "
              f"median INR vs n0 = {np.median(a)/n0:9.4g} ({db(np.median(a)/n0):7.2f} dB)")

    line("COMM interference", agg["if_comm"])
    line("SENS residual_self", agg["rs_self"])
    line("SENS residual_direct", agg["rs_direct"])
    line("SENS residual_multi", agg["rs_multi"])
    print()
    peak = db(np.median(agg["if_comm"]) / n0)
    sens = db((np.median(agg["rs_self"]) + np.median(agg["rs_direct"]) + np.median(agg["rs_multi"])) / n0)
    print(f"  comm interference-to-noise  median = {peak:7.2f} dB")
    print(f"  sens interference-to-noise  median = {sens:7.2f} dB")
    print(f"  => comm interference is {10**((peak-sens)/10):.4g}x larger")
    print()
    print(f"  residual_direct_factor = {r.residual_direct_factor:g} "
          f"-> {db(r.residual_direct_factor):.1f} dB direct-path suppression assumed")
    print(f"  so sensing 'sees' 100 times less interference by MODELING CHOICE, not physics")


if __name__ == "__main__":
    main()
