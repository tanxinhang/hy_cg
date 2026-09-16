"""Which power/coordination axis actually moves the sensing SINR?

Claim under test
----------------
Under ``interference.coupling='shared_spectrum'`` the sensing interference
field at receiving node j is built from *every* UAV's total radiated power::

    I_sense[j] = sum_k (P_sense[k] + P_comm[k]) * G_direct[k, j]

Because ``P_sense + P_comm == P_default`` holds per UAV, the sensing/report
split ``rho`` *cancels out of the field*.  The wanted echo does not::

    echo[i] = P_sense[i] * target_gain[i, j, q] * G_proc * G_hw
            = rho_i * P_default * target_gain[i, j, q] * G_proc * G_hw

Two knobs that both look like "power" therefore behave in opposite ways:

* ``radio.P_default`` scales echo and field together.  Once the direct-path
  residual dominates the receiver noise -- which it does by ~40 dB in this
  geometry -- the sensing SINR is flat in it.
* ``rho_i`` scales the echo alone.  The sensing SINR is linear in it, so the
  sensing/report split *is* a coordination lever.

A third axis, ``interference.sense_gate_by_active_tx``, decides whether the
*schedule* can touch the field at all.  With it off, ``I_sense`` is a constant
of the geometry and no selection can avoid an interferer; with it on, the
active transmitter set gates the field.

Model scope
-----------
Block A sweeps ``rho x P_default`` with no gating.  Block B fixes the released
operating point and sweeps the gating axis with an "active fraction", i.e. the
share of UAVs that are transmitting concurrently.  Block B is deliberately an
*optimistic* bound on scheduling: the silent UAVs are removed from the
interference field only, while every illuminator is still allowed to provide
its echo.  A real scheduler would also lose that illuminator's observation.

The analytic expression is cross-checked against ``compute_link_tables`` on the
released operating point, so a mismatch here means this file is wrong rather
than the model.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isac_sim.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    denominator_guard,
    generate_geometry,
    noise_power,
    radar_hardware_gain,
)

SEED = 10919
RHOS = (0.2, 0.4, 0.6, 0.8, 0.95)
P_DEFAULTS = (1.0, 2.0)
MODELS = ("active_set", "orthogonal")
FRACTIONS = (1.0, 0.5)

FIELDS = ["block", "area_m", "rcs_m2", "trial", "model", "gate", "rho",
          "p_default", "active_frac", "sense_sinr_db", "echo_db",
          "field_db", "near_far_db", "residual_db"]


def build_config(area, rcs, seed):
    cfg = apply_preset(Config(), "target-local-v1")
    return apply_overrides(cfg, {
        "detect.comm_error_model": "gaussian_replacement",
        "selector.score_mode": "detector_pd",
        "selector.max_remote_reports": 8,
        "geometry.area_xy": float(area),
        "detect.target_rcs": float(rcs),
        "run.seed": int(seed),
        "run.verbose": False,
    })


def valid_mask(cfg, base):
    """The exact (i, j, q) support used by the sensing block of model.py.

    The sensing loop skips ``i == j`` and, when ``dd.use_otfs_bin_validity`` is
    set, invalid DD bins.  It does *not* consult ``edge_mask`` -- that gate
    belongs to the communication link, not to the bistatic observation.
    """
    M, _, q_count = base.target_gain.shape
    keep = np.broadcast_to(~np.eye(M, dtype=bool)[:, :, None],
                           (M, M, q_count)).copy()
    if cfg.dd.use_otfs_bin_validity:
        keep &= base.valid_dd
    return keep


def analytic_sinr(cfg, base, rho, p_default, model, gate, active_frac):
    """Median sensing SINR over valid bistatic links, replicating model.py."""
    M = cfg.scale.M
    r = cfg.radio
    ic = cfg.interference
    n0 = noise_power(cfg)
    eps = denominator_guard(cfg, n0)
    kappa = 10.0 ** (-ic.direct_cancellation_db / 10.0)
    g_proc = cfg.waveform.N * cfg.waveform.L
    g_hw = radar_hardware_gain(cfg)

    rho_vec = np.full(M, float(rho))
    p = np.full(M, float(p_default))
    p_sense = rho_vec * p
    p_comm = (1.0 - rho_vec) * p

    if model == "orthogonal":
        p_rad = p_sense
    else:
        p_rad = p_sense + p_comm

    if gate and active_frac < 1.0:
        # Deterministic mask so the sweep is reproducible without extra RNG.
        keep = max(1, int(round(active_frac * M)))
        mask = np.zeros(M, dtype=bool)
        mask[:keep] = True
        p_rad = p_rad * mask

    field = p_rad @ base.direct_gain
    residual = r.residual_self_factor * p + kappa * field
    denom = n0 + residual + eps

    coll = 1.0 / (np.maximum(base.dd_collision_count, 1.0) ** cfg.dd.dd_collision_alpha)
    # The audit protocol and ``power_joint`` both evaluate the C2F-refined DD
    # gain (``eta_fine``), not the coarse ``dd_frac_loss``.  Using the coarse
    # array here silently costs ~0.9 dB, which is why the production builder is
    # cross-checked below.
    dd = base.eta_fine if cfg.dd.enable_dd_fractional_penalty else 1.0
    signal = (p_sense[:, None, None] * base.target_gain
              * g_proc * g_hw * coll * dd)

    sinr = signal / denom[None, :, None]
    keep_mask = valid_mask(cfg, base)
    vals = sinr[keep_mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return None
    echo = signal[keep_mask]
    echo = echo[echo > 0]
    return {
        "sense_sinr_db": float(10 * np.log10(np.median(vals))),
        "echo_db": float(10 * np.log10(np.median(echo))),
        "field_db": float(10 * np.log10(np.median(field[field > 0]))),
        "near_far_db": float(10 * np.log10(np.median(field[field > 0])
                                           / np.median(echo))),
        "residual_db": float(10 * np.log10(np.median(residual[residual > 0]))),
    }


def cross_check(cfg, base, model):
    """Compare the analytic value against the production table builder.

    ``cfg`` must be switched to the same ``comm.interference_model`` the
    analytic branch emulates, otherwise this compares two different physical
    hypotheses instead of two implementations of one hypothesis.
    """
    from isac_sim.config import apply_overrides
    probe = apply_overrides(cfg, {"comm.interference_model": model})
    tb = compute_link_tables(probe, base, dd_gain=base.eta_fine)
    keep = valid_mask(probe, base)
    vals = tb.gamma_sense[keep]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(10 * np.log10(np.median(vals))) if vals.size else None


def run_trial(cfg, area, rcs, trial, rows):
    rng = np.random.default_rng([SEED, trial])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)

    for model in MODELS:
        for p_default in P_DEFAULTS:
            for rho in RHOS:
                res = analytic_sinr(cfg, base, rho, p_default, model,
                                    gate=False, active_frac=1.0)
                if res:
                    rows.append({"block": "A", "area_m": area, "rcs_m2": rcs,
                                 "trial": trial, "model": model, "gate": False,
                                 "rho": rho, "p_default": p_default,
                                 "active_frac": 1.0, **res})

    for gate in (False, True):
        for frac in FRACTIONS:
            if not gate and frac < 1.0:
                continue
            res = analytic_sinr(cfg, base, 0.8, 1.0, "active_set",
                                gate=gate, active_frac=frac)
            if res:
                rows.append({"block": "B", "area_m": area, "rcs_m2": rcs,
                             "trial": trial, "model": "active_set",
                             "gate": gate, "rho": 0.8, "p_default": 1.0,
                             "active_frac": frac, **res})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", type=int, default=8)
    parser.add_argument("--areas", type=float, nargs="+", default=[400.0, 600.0])
    parser.add_argument("--rcs", type=float, nargs="+", default=[0.05])
    parser.add_argument("--out", type=Path,
                        default=Path("results_v1_rho_coordination"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    if (args.out / "protocol.json").exists():
        raise SystemExit("Use a fresh output directory")

    rows = []
    checks = []
    for area in args.areas:
        for rcs in args.rcs:
            cfg = build_config(area, rcs, SEED)
            if area == args.areas[0] and rcs == args.rcs[0]:
                rng = np.random.default_rng([SEED, 0])
                base0 = build_base_gains(cfg, generate_geometry(cfg, rng), rng)
                for model in MODELS:
                    real = cross_check(cfg, base0, model)
                    ana = analytic_sinr(cfg, base0, 0.8, 1.0, model,
                                        gate=False, active_frac=1.0)
                    checks.append({
                        "gate": f"analytic_vs_table[{model}]",
                        "sense_sinr_db_table": real,
                        "sense_sinr_db_analytic": ana["sense_sinr_db"],
                        "delta_db": (ana["sense_sinr_db"] - real
                                     if real is not None else None),
                    })
            for trial in range(args.mc):
                run_trial(cfg, area, rcs, trial, rows)
            print(f"done area={area:g} rcs={rcs:g}", flush=True)

    with (args.out / "rho_coordination.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    protocol = {
        "question": ("Does the sensing/report power split rho move the sensing "
                     "SINR, given that the shared interference field sums "
                     "P_sense + P_comm = P_default per UAV?"),
        "seed": SEED, "mc": args.mc, "areas": args.areas, "rcs": args.rcs,
        "rhos": list(RHOS), "p_defaults": list(P_DEFAULTS),
        "models": list(MODELS), "active_fractions": list(FRACTIONS),
        "note": ("Block B is an optimistic scheduling bound: silent UAVs are "
                 "removed from the field only, their echo contribution is "
                 "retained."),
        "cross_check": checks,
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=1))

    summary = {}
    for block in ("A", "B"):
        sel = [r for r in rows if r["block"] == block]
        keys = ({"model", "p_default", "rho"} if block == "A"
                else {"gate", "active_frac"})
        grouped = {}
        for r in sel:
            key = tuple(sorted((k, r[k]) for k in keys))
            grouped.setdefault(key, []).append(r["sense_sinr_db"])
        for key, vals in sorted(grouped.items(), key=lambda kv: str(kv[0])):
            summary[f"block{block}:" + ",".join(f"{k}={v}" for k, v in key)] = {
                "n": len(vals),
                "sense_sinr_db": float(np.mean(vals)),
            }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))

    print(json.dumps(checks, indent=1))
    for key, val in summary.items():
        print(f"{key:58s} n={val['n']:3d} sense_sinr={val['sense_sinr_db']:7.2f} dB")


if __name__ == "__main__":
    main()
