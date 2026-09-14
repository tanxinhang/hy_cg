#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit where the detection performance is lost.

Quantifies, on the paper's operating point (M=15, Q=10, active-set
interference), the chain
    raw sensing SINR -> DD fractional loss -> reporting pollution -> fused D -> P_D
and reports the marginal contribution of each mechanism.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from isac_sim.config import Config, apply_overrides
from isac_sim.model import (
    build_base_gains,
    compute_link_tables,
    generate_geometry,
    noise_power,
    pd_from_deflection,
    threshold_from_pfa,
)

PAPER = {
    "geometry.uav_speed_min": 30,
    "geometry.target_speed_min": 50,
    "geometry.target_speed_max": 90,
    "comm.interference_model": "active_set",
}


def collect(cfg, n_trials=20):
    M, Q = cfg.scale.M, cfg.scale.Q
    g_gamma, g_eta, g_raw, g_chi, g_tg, g_rinr = [], [], [], [], [], []
    D_all = []
    for t in range(n_trials):
        rng = np.random.default_rng([cfg.run.seed, t])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tb = compute_link_tables(cfg, base)
        v = base.valid_dd & base.edge_mask[..., None]
        g_gamma.append(tb.gamma_sense[v])
        g_raw.append(tb.raw_gamma_sense[v])
        g_eta.append(base.dd_frac_loss[v])
        g_tg.append(base.target_gain[v])
        eye = ~np.eye(M, dtype=bool)
        g_chi.append(tb.chi_comm[eye])
        g_rinr.append(tb.rinr[eye])
        # fused D on all valid single links (upper bound per target)
        from isac_sim.fusion import deflection_for_links
        Dq = np.zeros(Q)
        for q in range(Q):
            links = [(i, j) for i in range(M) for j in range(M)
                     if i != j and base.edge_mask[i, j] and base.valid_dd[i, j, q]]
            if links:
                Dq[q] = deflection_for_links(cfg, tb, q, links, weight_mode="deflection", base=base)
        D_all.append(Dq)
    out = {k: np.concatenate(v) for k, v in
           dict(gamma=g_gamma, raw=g_raw, eta=g_eta, tg=g_tg, chi=g_chi, rinr=g_rinr).items()}
    out["D_per_target"] = np.vstack(D_all)
    return out


def pct_stats(a, name, nd=4):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    print(f"  {name:28s} p10={np.percentile(a,10):{nd}g}  "
          f"p50={np.percentile(a,50):{nd}g}  p90={np.percentile(a,90):{nd}g}  "
          f"mean={a.mean():{nd}g}")


def main():
    cfg = apply_overrides(Config(), PAPER)
    print("=" * 78)
    print("PERFORMANCE AUDIT  (M=15, Q=10, active_set, MC=20)")
    print("=" * 78)

    d = collect(cfg, 20)
    print("\n[1] Link budget over DD-valid links")
    pct_stats(d["raw"], "raw sensing SINR")
    pct_stats(d["gamma"], "sensing SINR gamma^s")
    pct_stats(d["eta"], "DD fractional loss eta^DD")
    pct_stats(d["tg"], "bistatic target gain g^s")
    pct_stats(d["chi"], "reporting chi")
    pct_stats(d["rinr"], "residual-to-noise RINR")

    eta = d["eta"]
    print(f"\n  eta^DD mean = {eta.mean():.4f}  ->  {10*np.log10(max(eta.mean(),1e-12)):.2f} dB"
          f"   fraction < 0.5: {(eta<0.5).mean():.3f}")
    print(f"  L_fraction (N=64) mean = {1/64:.4f}  ->  {10*np.log10(1/64):.2f} dB (est. leakage floor)")

    # [2] detection curve
    print("\n[2] Detection curve P_D = Q(eta - sqrt(D))")
    thr = threshold_from_pfa(cfg)
    print(f"  threshold eta = {thr:.4f} (P_FA target {cfg.detect.Pfa_target})")
    for D in [0.5, 1, 2, 3, 5, 8, 12, 20, 40]:
        print(f"    D={D:6.1f} -> P_D = {float(pd_from_deflection(cfg, float(D))):.4f}")

    # [3] achieved D
    Dp = d["D_per_target"]
    print("\n[3] Achieved fused deflection (all feasi    ble links, upper bound)")
    print(f"  mean over targets = {Dp.mean():.3f};  median = {np.median(Dp):.3f}")
    print(f"  fraction of targets with D >= D_min=3: {(Dp>=3).mean():.3f}")
    print(f"  fraction of targets with D >= 8.6 (P_D=0.9): {(Dp>=8.6).mean():.3f}")
    print(f"  implied P_D (all-links upper bound) = "
          f"{float(np.mean(pd_from_deflection(cfg, Dp))):.4f}")

    # [4] mechanism marginal contribution: multiplicative loss factors
    print("\n[4] Multiplicative factors on the effective mean delta")
    chi = d["chi"]
    print(f"  reporting pollution factor E[chi] = {chi.mean():.4f}  "
          f"(delta_eff = chi * delta -> variance of D scales by chi^2)")
    print(f"  => D loss from reporting: {10*np.log10(max(chi.mean()**2,1e-12)):.2f} dB "
          f"({(1-chi.mean()**2)*100:.1f}% of D)")
    print(f"  DD loss factor eta^DD: {eta.mean():.4f} -> "
          f"{10*np.log10(max(eta.mean(),1e-12)):.2f} dB on SINR")
    n0 = noise_power(cfg)
    print(f"  noise power N0 = {n0:.3e} W;  P_sense = {cfg.radio.rho*cfg.radio.P_default:.3f} W")


if __name__ == "__main__":
    main()
