#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Link-budget dB breakdown of the bistatic sensing link.

Explains why the median sensing SINR sits far below 0 dB by decomposing
    P_sense + g^s + G_proc + eta^DD + eta^col - N0
in dB, per component, over DD-valid links.
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
    bandwidth,
    compute_link_tables,
    generate_geometry,
    noise_power,
    radar_hardware_gain,
    wavelength,
)

PAPER = {
    "geometry.uav_speed_min": 30,
    "geometry.target_speed_min": 50,
    "geometry.target_speed_max": 90,
    "comm.interference_model": "active_set",
}


def db(x, floor=1e-300):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), floor))


def main():
    cfg = apply_overrides(Config(), PAPER)
    M, Q = cfg.scale.M, cfg.scale.Q
    G_proc = cfg.waveform.N * cfg.waveform.L
    n0 = noise_power(cfg)
    lam = wavelength(cfg)

    rows = {k: [] for k in
            ["P_sense", "g_target", "G_hw", "G_proc", "eta_dd", "eta_col", "N0",
             "snr", "gamma", "d_iq", "d_jq", "rcs"]}
    for t in range(20):
        rng = np.random.default_rng([cfg.run.seed, t])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tb = compute_link_tables(cfg, base)
        v = base.valid_dd & base.edge_mask[..., None]
        idx = np.argwhere(v)                       # (n, 3) = i,j,q
        if len(idx) == 0:
            continue
        i, j, q = idx[:, 0], idx[:, 1], idx[:, 2]
        rows["P_sense"].append(np.full(len(i), cfg.radio.rho * cfg.radio.P_default))
        rows["g_target"].append(base.target_gain[i, j, q])
        rows["G_hw"].append(np.full(len(i), radar_hardware_gain(cfg)))
        rows["G_proc"].append(np.full(len(i), float(G_proc)))
        rows["eta_dd"].append(base.dd_frac_loss[i, j, q])
        rows["eta_col"].append(1.0 / np.maximum(base.dd_collision_count[i, j, q], 1.0))
        rows["N0"].append(np.full(len(i), n0))
        rows["gamma"].append(tb.gamma_sense[i, j, q])
        rows["snr"].append(tb.raw_gamma_sense[i, j, q])
        rows["d_iq"].append(base.d_uav_tgt[i, q])
        rows["d_jq"].append(base.d_uav_tgt[j, q])
        # recover the RCS realisation from target_gain
        d_iq = np.maximum(base.d_uav_tgt[i, q], 1.0)
        d_jq = np.maximum(base.d_uav_tgt[j, q], 1.0)
        rows["rcs"].append(base.target_gain[i, j, q] * (4 * np.pi) ** 3 * d_iq ** 2 * d_jq ** 2 / lam ** 2)

    agg = {k: np.concatenate(v) for k, v in rows.items()}

    def line(name, arr_db, extra=""):
        a = np.asarray(arr_db, dtype=float)
        print(f"  {name:22s} p10={np.percentile(a,10):8.2f}  "
              f"p50={np.percentile(a,50):8.2f}  p90={np.percentile(a,90):8.2f}  "
              f"mean={a.mean():8.2f} dB   {extra}")

    print("=" * 84)
    print("BISTATIC SENSING LINK BUDGET (dB), DD-valid links, MC=20")
    print("=" * 84)
    print(f"  lambda = {lam:.4f} m,  G_proc = N*L = {G_proc},  N0 = {n0:.3e} W")
    print()
    line("P_sense", db(agg["P_sense"]))
    line("bistatic gain g^s", db(agg["g_target"]))
    line("radar hardware G_hw", db(agg["G_hw"]))
    line("processing gain", db(agg["G_proc"]))
    line("DD fractional eta", db(agg["eta_dd"]))
    line("DD collision eta", db(agg["eta_col"]))
    line("- noise N0", -db(agg["N0"]))
    print()
    snr_db = (db(agg["P_sense"]) + db(agg["g_target"]) + db(agg["G_hw"]) + db(agg["G_proc"])
              + db(agg["eta_dd"]) + db(agg["eta_col"]) - db(agg["N0"]))
    line("=> SNR (raw, no resid.)", snr_db)
    line("=> gamma^s (with resid.)", db(agg["gamma"]))
    print()
    line("dist i->q (m, 20log10)", 20 * np.log10(agg["d_iq"]))
    line("dist j->q (m, 20log10)", 20 * np.log10(agg["d_jq"]))
    line("RCS realisation", db(agg["rcs"]))
    print()
    print(f"  RCS: mean={agg['rcs'].mean():.1f} m^2, p10={np.percentile(agg['rcs'],10):.1f}, "
          f"p90={np.percentile(agg['rcs'],90):.1f}  -> dynamic range "
          f"{10*np.log10(np.percentile(agg['rcs'],90)/max(np.percentile(agg['rcs'],10),1e-9)):.1f} dB")
    print(f"  gamma^s < 0 dB fraction: {(db(agg['gamma']) < 0).mean():.3f}")
    print(f"  gamma^s < -10 dB fraction: {(db(agg['gamma']) < -10).mean():.3f}")
    print(f"  gamma^s > 0 dB fraction: {(db(agg['gamma']) > 0).mean():.3f}")
    print()
    print("  Reference: a monostatic radar needs the SAME round-trip d^-4; here the")
    print("  bistatic product d_iq^2*d_jq^2 is doubly penalised at 4000x4000 m scale.")


if __name__ == "__main__":
    main()
