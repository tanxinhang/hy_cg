# RETIRED PREMISE (2026-09-20): this script swept / read the config field
# `interference.direct_cancellation_db` (kappa_dc).
# That field was DELETED: it asserted a fixed 40 dB direct-path cancellation with no
# receiver implementation behind it while propping up the whole SINR denominator.
# Direct-path cancellation is now only ever a MEASURED TP-UIC residual.  Running this
# script as-is will fail on the missing attribute -- kept as historical evidence only.
"""Quantify what kappa_dc (direct-path cancellation) actually buys.

The question this answers: is ``interference.direct_cancellation_db = 40`` a
free performance gain, or is it the amount of interference suppression the
bistatic geometry physically requires?

Two measurements, both without running Monte-Carlo detection:

1. ``near_far_db`` -- the ratio of the illuminators' direct-path field at each
   sensing receiver to the *processed* target echo (echo already includes the
   ``N*L`` processing gain).  This is a purely geometric/RCS quantity; it does
   not depend on kappa.  It is the number of dB the direct path must be
   cancelled by before the echo is even level with it.

2. ``sense_sinr_db`` as a function of kappa -- where the curve flattens is the
   point at which the direct path stops being the dominant residual.

Usage::

    python tools/probe_kappa_necessity.py --area 600 --rcs 0.1 --trials 20
    python tools/probe_kappa_necessity.py --area 4000 --rcs 50 --trials 20
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    denominator_guard,
    generate_geometry,
    noise_power,
    radar_hardware_gain,
)

KAPPAS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0)
SEED = 10919


def build_config(area: float, rcs: float, kappa: float, preset: str):
    cfg = apply_preset(Config(), preset)
    return apply_overrides(
        cfg,
        {
            "geometry.area_xy": float(area),
            "detect.target_rcs": float(rcs),
            "interference.direct_cancellation_db": float(kappa),
            "run.seed": SEED,
            "run.verbose": False,
        },
    )


def valid_mask(cfg, base):
    """The (i, j, q) support used by the sensing block of model.py."""
    M, _, q_count = base.target_gain.shape
    keep = np.broadcast_to(~np.eye(M, dtype=bool)[:, :, None],
                           (M, M, q_count)).copy()
    if cfg.dd.use_otfs_bin_validity:
        keep &= base.valid_dd
    return keep


def measure(cfg, base, kappa: float) -> dict:
    """Median sensing SINR / echo / field / near-far for one kappa."""
    M = cfg.scale.M
    r = cfg.radio
    n0 = noise_power(cfg)
    eps = denominator_guard(cfg, n0)
    kappa_factor = 10.0 ** (-float(kappa) / 10.0)
    g_proc = cfg.waveform.N * cfg.waveform.L
    g_hw = radar_hardware_gain(cfg)

    rho = 0.6
    p_default = 1.0
    p_sense = np.full(M, rho * p_default)
    p_rad = p_sense  # orthogonal reporting model

    field = p_rad @ base.direct_gain          # (M,) direct-path field at each j
    residual = r.residual_self_factor * p_default + kappa_factor * field
    denom = n0 + residual + eps

    coll = 1.0 / (np.maximum(base.dd_collision_count, 1.0) ** cfg.dd.dd_collision_alpha)
    dd = base.eta_fine if cfg.dd.enable_dd_fractional_penalty else 1.0
    signal = p_sense[:, None, None] * base.target_gain * g_proc * g_hw * coll * dd

    sinr = signal / denom[None, :, None]
    keep = valid_mask(cfg, base)
    vals = sinr[keep]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return {}

    echo = signal[keep]
    echo = echo[echo > 0]
    f_pos = field[field > 0]
    echo_over_n0 = float(10 * math.log10(np.median(echo) / (n0 + eps)))
    residual_over_n0 = float(
        10 * math.log10(np.median(residual[residual > 0]) / (n0 + eps))
    )
    return {
        "sense_sinr_db": float(10 * math.log10(np.median(vals))),
        "echo_db": float(10 * math.log10(np.median(echo))),
        "echo_over_n0_db": echo_over_n0,
        "field_db": float(10 * math.log10(np.median(f_pos))) if f_pos.size else float("nan"),
        "near_far_db": (
            float(10 * math.log10(np.median(f_pos) / np.median(echo)))
            if f_pos.size else float("nan")
        ),
        "residual_over_n0_db": residual_over_n0,
        "direct_over_echo_db": residual_over_n0 - echo_over_n0,
    }


def cross_check(cfg, base) -> float | None:
    """Sensing-SINR median from the *production* table builder.

    The analytic branch above is a closed-form replication used because it is
    cheap to sweep.  Before any of its numbers are quoted, they must be checked
    against the real ``compute_link_tables`` path that the simulator uses.
    """
    from isac_sim.sensing.model import compute_link_tables

    tb = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
    keep = valid_mask(cfg, base)
    vals = tb.gamma_sense[keep]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(10 * np.log10(np.median(vals))) if vals.size else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--preset", default="small-uav-compact-800m")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default="results_kappa_necessity")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    checks = []
    for trial in range(int(args.trials)):
        rng = np.random.default_rng([SEED, trial])
        # geometry does not depend on kappa; build it once per trial
        cfg0 = build_config(args.area, args.rcs, 40.0, args.preset)
        geom = generate_geometry(cfg0, rng)
        base = build_base_gains(cfg0, geom, rng)
        for kappa in KAPPAS:
            cfg = build_config(args.area, args.rcs, kappa, args.preset)
            m = measure(cfg, base, kappa)
            if not m:
                continue
            prod = cross_check(cfg, base)
            if prod is not None:
                m["sense_sinr_prod_db"] = prod
                checks.append(
                    {"trial": trial, "kappa_db": kappa,
                     "analytic_db": round(m["sense_sinr_db"], 4),
                     "production_db": round(prod, 4),
                     "delta_db": round(m["sense_sinr_db"] - prod, 4)}
                )
            rows.append({"trial": trial, "area_m": args.area,
                         "rcs_m2": args.rcs, "kappa_db": kappa, **m})

    if checks:
        cpath = out_dir / "analytic_vs_production.csv"
        with open(cpath, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(checks[0]))
            w.writeheader()
            w.writerows(checks)
        worst = max(abs(c["delta_db"]) for c in checks)
        print(f"analytic-vs-production check -> {cpath}  (max |delta| = {worst:.3f} dB)")
        print("  the SINR column below is the PRODUCTION table value; the "
              "analytic branch is kept only as a cheap cross-check.")
        print()

    path = out_dir / "kappa_necessity.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {path}  ({len(rows)} rows)")
    print()
    print(f"geometry: area={args.area} m, RCS={args.rcs} m^2, "
          f"preset={args.preset}, trials={args.trials}")
    print()
    hdr = (f"{'kappa_dB':>9s} {'SINR_prod':>10s} {'SINR_anal':>10s} "
           f"{'direct-echo':>12s} {'resid/n0':>9s} {'echo/n0':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for kappa in KAPPAS:
        sub = [r for r in rows if r["kappa_db"] == kappa]
        if not sub:
            continue
        med = lambda key: float(np.median([r[key] for r in sub]))  # noqa: E731
        sp = med("sense_sinr_prod_db") if "sense_sinr_prod_db" in sub[0] else float("nan")
        print(f"{kappa:9.0f} {sp:10.2f} {med('sense_sinr_db'):10.2f} "
              f"{med('direct_over_echo_db'):12.2f} "
              f"{med('residual_over_n0_db'):9.2f} {med('echo_over_n0_db'):8.2f}")
    nf = float(np.median([r["near_far_db"] for r in rows]))
    print()
    print(f"near_far_db (direct field / processed echo) = {nf:.2f} dB")
    print("  -> this is the cancellation the geometry demands before the echo")
    print("     is even level with the residual direct path.")


if __name__ == "__main__":
    main()
