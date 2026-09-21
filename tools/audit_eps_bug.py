#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the EPS-degradation bug and rank the performance loss sources.

The sensing-SINR denominators in model.compute_link_tables are
    gamma = signal / (n0 + residual_total + EPS)
with EPS = 1e-12 while the noise power n0 ~ 3.8e-14, i.e. the "guard" is 26x
larger than the physical noise floor.  This script quantifies the resulting
systematic SINR suppression and its effect on P_D.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np

import isac_sim.sensing.model as M
from isac_sim.core.config import Config, apply_overrides
from isac_sim.sensing.model import (
    build_base_gains,
    compute_link_tables,
    generate_geometry,
    noise_power,
)
from experiments.flow.simulate import run_simulation

PAPER = {
    "geometry.uav_speed_min": 30,
    "geometry.target_speed_min": 50,
    "geometry.target_speed_max": 90,
    "comm.interference_model": "active_set",
}


def db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, float), 1e-300))


def sinr_stats(cfg, tag):
    g, c = [], []
    for t in range(20):
        rng = np.random.default_rng([cfg.run.seed, t])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tb = compute_link_tables(cfg, base)
        v = base.valid_dd & base.edge_mask[..., None]
        g.append(tb.gamma_sense[v])
        eye = ~np.eye(cfg.scale.M, dtype=bool)
        c.append(tb.chi_comm[eye])
    g = np.concatenate(g)
    c = np.concatenate(c)
    print(f"  [{tag}] gamma^s dB: p50={np.percentile(db(g),50):7.2f}  "
          f"mean={db(g).mean():7.2f}   chi: p50={np.percentile(c,50):.3f} "
          f"mean={c.mean():.3f}")


def main():
    cfg = apply_overrides(Config(), PAPER)
    n0 = noise_power(cfg)
    print("=" * 80)
    print("EPS BUG VERIFICATION")
    print("=" * 80)
    print(f"  noise power n0        = {n0:.4e} W")
    print(f"  guard EPS             = {M.EPS:.4e}")
    print(f"  EPS / n0              = {M.EPS/n0:.2f}  -> SINR suppressed by "
          f"{db(1+M.EPS/n0):.2f} dB")
    print()

    cfg.run.num_mc = 100
    cfg.run.verbose = False

    M.EPS = 1e-12
    sinr_stats(cfg, "EPS=1e-12 (current)")
    s_bad = run_simulation(cfg, methods=["proposed_lagrangian", "all_neighbor"])

    M.EPS = 1e-30
    sinr_stats(cfg, "EPS=1e-30 (fixed)  ")
    s_fix = run_simulation(cfg, methods=["proposed_lagrangian", "all_neighbor"])

    print()
    print(f"{'':22s} {'P_D (EPS=1e-12)':>16s} {'P_D (fixed)':>13s} {'delta':>9s}")
    for m in ["proposed_lagrangian", "all_neighbor"]:
        a, b = s_bad[m]["P_D"], s_fix[m]["P_D"]
        print(f"  {m:20s} {a:16.4f} {b:13.4f} {b-a:+9.4f}")
    print()
    print("  D_mean:")
    for m in ["proposed_lagrangian", "all_neighbor"]:
        a, b = s_bad[m]["D_mean"], s_fix[m]["D_mean"]
        print(f"  {m:20s} {a:16.3f} {b:13.3f}  x{b/max(a,1e-9):.1f}")

    # restore
    M.EPS = 1e-12
    print()
    print("  (EPS restored to 1e-12; no source file was modified by this script)")


if __name__ == "__main__":
    main()
