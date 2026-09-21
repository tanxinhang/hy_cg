#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Counterfactual audit: which mechanism actually caps P_D?

Runs the proposed selector on the paper operating point, then relaxes one
mechanism at a time and measures the P_D ceiling each relaxation buys.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from isac_sim.core.config import Config, apply_overrides
from experiments.flow.simulate import run_simulation

PAPER = {
    "geometry.uav_speed_min": 30,
    "geometry.target_speed_min": 50,
    "geometry.target_speed_max": 90,
    "comm.interference_model": "active_set",
}

CASES = [
    ("baseline (paper)", {}),
    ("+ perfect reporting (chi=1)", {"detect.enable_comm_error_pollution": False}),
    ("+ Tx power x10 (+10 dB)", {"radio.P_default": 10.0}),
    ("+ no DD fractional loss", {"dd.enable_dd_fractional_penalty": False}),
    ("+ no DD collision penalty", {"dd.enable_dd_collision_penalty": False}),
    ("+ ideal comm AND power x10",
     {"detect.enable_comm_error_pollution": False, "radio.P_default": 10.0}),
    ("+ power x100 (+20 dB)", {"radio.P_default": 100.0}),
    ("+ all ideal (power x10, chi=1, no DD)",
     {"detect.enable_comm_error_pollution": False, "radio.P_default": 10.0,
      "dd.enable_dd_fractional_penalty": False, "dd.enable_dd_collision_penalty": False}),
]


def main():
    mc = int(os.environ.get("AUDIT_MC", "200"))
    eps = float(os.environ.get("AUDIT_EPS", "1e-12"))
    import isac_sim.sensing.model as M
    M.EPS = eps
    print("=" * 78)
    print(f"COUNTERFACTUAL AUDIT (proposed_lagrangian, MC={mc}, active_set, EPS={eps:g})")
    print("=" * 78)
    print(f"{'case':38s} {'P_D':>7s} {'D_mean':>8s} {'T_ms':>8s} {'links':>7s}")
    base_pd = None
    for name, ov in CASES:
        cfg = apply_overrides(apply_overrides(Config(), PAPER), ov)
        cfg.run.num_mc = mc
        cfg.run.verbose = False
        s = run_simulation(cfg, methods=["proposed_lagrangian"])["proposed_lagrangian"]
        pd = s["P_D"]
        if base_pd is None:
            base_pd = pd
        delta = f"{pd - base_pd:+.4f}"
        print(f"{name:38s} {pd:7.4f} {s['D_mean']:8.3f} {s['T_mean_ms']:8.2f} "
              f"{s['selected_links_mean']:7.1f}   {delta}")
    print()
    print("Note: chi=1 is emulated with enable_comm_error_pollution=False (the")
    print("      fusion sees clean statistics), i.e. an infinite-reliability report.")


if __name__ == "__main__":
    main()
