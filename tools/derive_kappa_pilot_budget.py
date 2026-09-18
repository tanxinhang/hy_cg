"""Derive the sensing direct-path cancellation depth from a cooperative reference budget.

Motivation
----------
``interference.direct_cancellation_db`` (called ``kappa_dc`` internally) was the
weakest claim in the paper: a single constant that sets the entire operating
level, documented only as "40 dB" in a supplement table, with no derivation.
This tool removes that weakness by tying the constant to a *system parameter
that a reviewer can already reason about*: the cooperative reference budget.

Derivation
----------
Cooperative ISAC knows every illuminator's waveform ``s_i[n]`` (the payload is
shared over the control plane, so in a cooperative network the whole frame --
not only the designated pilots -- is a valid reference).  The sensing receiver
therefore observes

    y_j[n] = sum_i sqrt(P_i) h_ij s_i[n]  +  echo  +  w_j[n]

and the direct-path coefficients ``h_ij`` are estimated by a least-squares /
maximum-likelihood fit against the known reference over ``N_p`` resource
elements at per-element SNR ``gamma_p``.  For orthogonal unit-power reference
symbols the LS estimate is a sample mean, so its error variance is

    Var( h_hat - h ) = sigma^2 / (N_p P_p) = 1 / (N_p gamma_p)          (1)

in units where the channel gain is normalised out.  Reconstructing and
subtracting the direct path leaves a residual direct power that is
``1/(N_p gamma_p)`` of the direct power, hence

    kappa_dB = 10 log10( N_p * gamma_p )                                (2)

Equation (2) is *channel-estimation limited*: it is the depth set by the
reference budget.  It cannot grow without bound -- the analogue/RF chain
(PA non-linearity, phase noise, I/Q imbalance, ADC dynamic range, DAC
quantisation) leaves a floor that more pilots cannot remove.  Writing that
ceiling ``kappa_hw``,

    kappa_dB = min( 10 log10(N_p * gamma_p) ,  kappa_hw )               (3)

``kappa_hw`` is anchored on published measurements (see KAPPA_DERIVATION.md):

  * up to 60 dB of *digital-domain* cancellation in full-duplex ISAC
    (Liu et al., IEEE JSAC 41(9), 2023),
  * 35.3 dB measured in a photonics-assisted SIC front end
    (Yu et al., Opt. Express 32(23), 2024),
  * 10-20 dB of *net* SNR improvement after adaptive direct-signal
    cancellation on real DVB-T data (Brustad, FFI-Report 2014).

The default is therefore not free: 40 dB corresponds to ``N_p gamma_p = 1e4``,
e.g. 64 reference elements at 20 dB -- 1.6 % of this waveform's 4096 elements.

What this tool measures
-----------------------
1. The analytic table ``kappa(N_p, gamma_p)`` from (2)/(3).
2. The *requirement* implied by the concrete scenario, taken from the
   **production** link-table builder: the value ``kappa*`` at which the
   residual direct path drops level with the processed target echo.  The
   decomposition uses the identity

       residual/echo = rinr / ( gamma_sense * (1 + rinr) )             (4)

   which follows from ``gamma_sense = S/(N0+I+eps)`` and ``rinr = I/(N0+eps)``
   and therefore needs no analytic replication of the model.
3. The reference budget ``N_p gamma_p = 10^(kappa*/10)`` the scenario demands.

Usage::

    python tools/derive_kappa_pilot_budget.py --area 600 --rcs 0.1 --trials 6
    python tools/derive_kappa_pilot_budget.py --area 4000 --rcs 50 --trials 6
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

from isac_sim.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)

SEED = 10919

# Cancellation depths swept to locate the scenario requirement.  Bracketing
# kappa* is the point, so the grid is fine near the expected crossing.
KAPPAS = (0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0)

# Physical ceiling of the cancellation chain, in dB.  Anchored on published
# digital-domain full-duplex ISAC results (Liu et al., JSAC 2023).
HW_CEIL_DB = 60.0

PILOT_LENGTHS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
PILOT_SNRS_DB = (10.0, 15.0, 20.0, 25.0, 30.0)


def kappa_from_pilot(n_pilot: float, gamma_p_db: float, hw_ceil: float = HW_CEIL_DB):
    """Cancellation depth from a reference budget, capped by the hardware floor."""
    est = 10.0 * math.log10(n_pilot * 10.0 ** (gamma_p_db / 10.0))
    return est, min(est, hw_ceil)


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
    m, _, q_count = base.target_gain.shape
    keep = np.broadcast_to(
        ~np.eye(m, dtype=bool)[:, :, None], (m, m, q_count)
    ).copy()
    if cfg.dd.use_otfs_bin_validity:
        keep &= base.valid_dd
    return keep


def production_metrics(cfg, base, keep) -> dict | None:
    """Residual-over-echo and SINR from the production link-table builder.

    Uses (4), so no analytic replication of the interference model is involved.
    """
    tables = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
    gamma = np.asarray(tables.gamma_sense, dtype=float)
    rinr = np.asarray(tables.rinr, dtype=float)

    # rinr is a per-link interference-to-noise ratio (shape (M,M) or (M,M,Q));
    # broadcast it onto the per-(i,j,q) sensing support so the ratio in (4) can
    # be formed elementwise.
    if rinr.shape != gamma.shape:
        if rinr.ndim == 2:                    # (i, j) -> (i, j, 1)
            rinr = rinr[:, :, None]
        elif rinr.ndim == 1:                  # (j,)  -> (1, j, 1)
            rinr = rinr[None, :, None]
        rinr = np.broadcast_to(rinr, gamma.shape)

    g = gamma[keep]
    r = rinr[keep]
    good = np.isfinite(g) & np.isfinite(r) & (g > 0) & (r > 0)
    if not good.any():
        return None
    g, r = g[good], r[good]

    echo_over_n0 = g * (1.0 + r)
    resid_over_echo = r / (g * (1.0 + r))
    return {
        "sense_sinr_db": float(10.0 * np.log10(np.median(g))),
        "residual_over_echo_db": float(10.0 * np.log10(np.median(resid_over_echo))),
        "residual_over_n0_db": float(10.0 * np.log10(np.median(r))),
        "echo_over_n0_db": float(10.0 * np.log10(np.median(echo_over_n0))),
    }


def crossing(kappas, values, level=0.0):
    """First ascending crossing of ``level`` by linear interpolation."""
    for k0, k1, v0, v1 in zip(kappas, kappas[1:], values, values[1:]):
        if (v0 - level) * (v1 - level) <= 0 and v1 != v0:
            return k0 + (k1 - k0) * (level - v0) / (v1 - v0)
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--preset", default="small-uav-compact-800m")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--hw-ceil-db", type=float, default=HW_CEIL_DB)
    ap.add_argument("--out", default="results_kappa_pilot_budget")
    args = ap.parse_args()

    hw_ceil = float(args.hw_ceil_db)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- table
    table = []
    for n_pilot in PILOT_LENGTHS:
        for snr in PILOT_SNRS_DB:
            est, capped = kappa_from_pilot(n_pilot, snr, hw_ceil)
            table.append(
                {
                    "n_reference_elements": n_pilot,
                    "reference_snr_db": snr,
                    "kappa_from_budget_db": round(est, 2),
                    "kappa_effective_db": round(capped, 2),
                    "hardware_limited": int(est > hw_ceil),
                }
            )
    tpath = out_dir / "pilot_budget_table.csv"
    with open(tpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    # -------------------------------------------------- scenario requirement
    rows = []
    for trial in range(int(args.trials)):
        rng = np.random.default_rng([SEED, trial])
        cfg0 = build_config(args.area, args.rcs, 40.0, args.preset)
        geom = generate_geometry(cfg0, rng)
        base = build_base_gains(cfg0, geom, rng)
        keep = valid_mask(cfg0, base)
        for kappa in KAPPAS:
            cfg = build_config(args.area, args.rcs, kappa, args.preset)
            m = production_metrics(cfg, base, keep)
            if not m:
                continue
            rows.append(
                {
                    "trial": trial,
                    "area_m": args.area,
                    "rcs_m2": args.rcs,
                    "kappa_db": kappa,
                    **m,
                }
            )
    if not rows:
        raise SystemExit("no valid sensing links produced; check the preset")

    rpath = out_dir / "scenario_requirement.csv"
    with open(rpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    med = {}
    for kappa in KAPPAS:
        sub = [r for r in rows if r["kappa_db"] == kappa]
        if sub:
            med[kappa] = {
                k: float(np.median([r[k] for r in sub]))
                for k in ("sense_sinr_db", "residual_over_echo_db",
                          "residual_over_n0_db", "echo_over_n0_db")
            }
    ks = sorted(med)
    residual = [med[k]["residual_over_echo_db"] for k in ks]
    kappa_star = crossing(ks, residual, 0.0)
    budget_star = 10.0 ** (kappa_star / 10.0) if math.isfinite(kappa_star) else float("nan")

    # ----------------------------------------------------------------- log
    print(f"wrote {tpath}  ({len(table)} rows)")
    print(f"wrote {rpath}  ({len(rows)} rows)")
    print()
    print(f"geometry: area={args.area} m, RCS={args.rcs} m^2, "
          f"preset={args.preset}, trials={args.trials}")
    print(f"hardware ceiling assumed: kappa_hw = {hw_ceil:.0f} dB")
    print()

    print("=== (1) analytic: kappa = min(10*log10(N_p * gamma_p), kappa_hw) ===")
    hdr = "  N_p      " + "".join(f"{s:>9.0f}dB" for s in PILOT_SNRS_DB)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for n_pilot in PILOT_LENGTHS:
        cells = []
        for snr in PILOT_SNRS_DB:
            _, capped = kappa_from_pilot(n_pilot, snr, hw_ceil)
            cells.append(f"{capped:>11.1f}")
        print(f"  {n_pilot:<8d} " + "".join(cells))
    print("  (values are dB; a '*' marks the hardware-limited region)")
    print()

    print("=== (2) production link tables: what the scenario requires ===")
    hdr2 = (f"  {'kappa':>6s} {'sense_SINR':>11s} {'resid/echo':>11s} "
            f"{'resid/n0':>9s} {'echo/n0':>8s}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for k in ks:
        v = med[k]
        print(f"  {k:6.0f} {v['sense_sinr_db']:11.2f} "
              f"{v['residual_over_echo_db']:11.2f} "
              f"{v['residual_over_n0_db']:9.2f} {v['echo_over_n0_db']:8.2f}")
    print()

    print("=== (3) cross-check: budget <-> requirement ===")
    print(f"  scenario requirement kappa* (residual == processed echo): "
          f"{kappa_star:.2f} dB" if math.isfinite(kappa_star)
          else "  scenario requirement kappa*: not bracketed by the grid")
    if math.isfinite(kappa_star):
        print(f"  -> needs a reference budget N_p * gamma_p = 10^(kappa*/10) "
              f"= {budget_star:,.0f}")
        for n_pilot in PILOT_LENGTHS:
            need_db = kappa_star - 10.0 * math.log10(n_pilot)
            if 0 < need_db <= 40:
                frac = 100.0 * n_pilot / 4096.0
                print(f"     {n_pilot:5d} reference elements "
                      f"({frac:5.1f}% of the 4096-element frame) "
                      f"needs {need_db:5.1f} dB per-element SNR")
        if kappa_star > hw_ceil:
            print(f"  !! requirement exceeds the {hw_ceil:.0f} dB hardware "
                  "ceiling: geometry, not estimation, is the binding constraint")
        else:
            print(f"  the requirement sits BELOW the {hw_ceil:.0f} dB hardware "
                  "ceiling -> estimable from a cooperative reference budget")
    print()

    # The comparison that matters for the paper: the shipped default.
    default_row = med.get(40.0)
    if default_row:
        print("=== (4) the shipped default, in budget terms ===")
        print(f"  direct_cancellation_db = 40 dB  <=>  N_p * gamma_p = 1e4")
        print(f"  measured residual-over-echo there: "
              f"{default_row['residual_over_echo_db']:+.2f} dB")
        n_at_20 = 10.0 ** (40.0 / 10.0) / 10.0 ** (20.0 / 10.0)
        print(f"  equivalent budget at 20 dB per element: {n_at_20:.0f} "
              f"reference elements = {100.0 * n_at_20 / 4096.0:.2f}% of the "
              "4096-element frame")
        if math.isfinite(kappa_star):
            print(f"  the default is {kappa_star - 40.0:+.2f} dB relative to "
                  "the scenario requirement")


if __name__ == "__main__":
    main()
