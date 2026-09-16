"""Which knobs are physically able to close the low-RCS sensing gap?

Motivation
----------
The low-RCS sweeps measured *what each knob buys in P_D*, but not *why the
ranking is what it is*.  Reading the sensing-SINR assembly in ``model.py``:

    signal   = rho * P * target_gain * G_proc * G_hw * capture
    residual = f_self * P + kappa_dc * I_sense_field(P) + multi * P
    gamma    = signal / ((n0 + residual + eps) * (1 + INR))

shows the levers split into two families:

  * **Multiplicative, never saturating**: everything that only enters the
    numerator -- ``G_proc`` (= N*L, or ``detect.sensing_processing_gain``),
    ``G_hw`` (``radio.radar_net_gain_db``), RCS, geometry.  A factor ``m``
    here multiplies gamma by exactly ``m``.
  * **Saturating**: ``radio.P_default`` and ``radio.noise_figure_db``, because
    the residual interference scales with transmitted power.  A factor ``m``
    on power gives, exactly,

        gamma' / gamma = m * (1 + rinr) / (1 + m * rinr),   rinr = residual/(n0+eps)

    which tends to ``1 + 1/rinr`` -- i.e. nothing once ``rinr >> 1``.

So the whole question "how should we optimise performance" reduces to one
number per link: ``rinr = LinkTables.rinr``.  This probe measures it once,
then converts *every* candidate lever analytically, because the multiplicative
family needs no re-simulation and the saturating family is closed-form.

Design
------
One Monte-Carlo pass records, per selected link, the sensing ``gamma``, the
``rinr`` and the ``n_looks``.  Afterwards every lever is applied in closed
form and the shortfall (extra gamma-dB needed to lift the weak target to
``detect.weak_pd_required``) is recomputed with the same LLR moments as
``tools/probe_lowrcs_shortfall.py``, so the numbers are comparable to the
already-published shortfall table.

Note on the P_D reading: the shortfall machinery predicts P_D from a Gaussian
(diagonal-covariance) approximation of the fused LLR.  It is a *bound*, not a
Monte-Carlo result, and it is optimistic about ``corr``.  It is used here only
to rank levers and to size the gap, never to claim a detection number.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isac_sim.belief import BeliefState  # noqa: E402
from isac_sim.config import apply_overrides, validate_config  # noqa: E402
from isac_sim.llr import llr_delta, llr_var0, llr_var1  # noqa: E402
from isac_sim.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
    noise_power,
    denominator_guard,
    radar_hardware_gain,
)
from isac_sim.reporting import assign_fusion_nodes  # noqa: E402
from isac_sim.selection import select_c2f_adaptive  # noqa: E402
from tools.audit_v1_exact_budget import config  # noqa: E402
from tools.audit_v1_lowrcs_sweep import REPORT_CAP, SEED  # noqa: E402


def d_prime(gammas, n_looks, z):
    delta = float(sum(llr_delta(g, n_looks) for g in gammas))
    var0 = float(sum(llr_var0(g, n_looks) for g in gammas))
    var1 = float(sum(llr_var1(g, n_looks) for g in gammas))
    if var1 <= 0.0:
        return float("-inf")
    return (delta - z * np.sqrt(var0)) / np.sqrt(var1)


def required_gain(gammas, n_looks, z, target_pd):
    """Smallest scalar multiplier on every gamma reaching ``target_pd``."""
    want = norm.ppf(target_pd)
    if d_prime(gammas, n_looks, z) >= want:
        return 1.0
    lo, hi = 1.0, 1.0
    for _ in range(60):
        hi *= 2.0
        if d_prime([g * hi for g in gammas], n_looks, z) >= want:
            break
    else:
        return float("inf")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if d_prime([g * mid for g in gammas], n_looks, z) >= want:
            hi = mid
        else:
            lo = mid
    return hi


def job(spec):
    area, rcs, trial = spec
    cfg = apply_overrides(
        config(SEED, REPORT_CAP),
        {"geometry.area_xy": float(area), "detect.target_rcs": float(rcs)})
    validate_config(cfg)
    z = float(norm.ppf(1.0 - cfg.detect.Pfa_target))
    rng = np.random.default_rng([SEED, trial])
    truth = generate_geometry(cfg, rng)
    truth_base = build_base_gains(cfg, truth, rng)
    belief = BeliefState.from_truth(cfg, truth, rng)
    geom = belief.as_geometry(truth)
    base = build_base_gains(cfg, geom, rng, channel=truth_base, rcs_view="mean")
    coarse = compute_link_tables(cfg, base)
    # Direct-link cancellation share of the residual interference.  Rescaling
    # ``direct_cancellation_db`` scales ONLY the direct term, so a second
    # evaluation 20 dB deeper isolates how much of ``rinr`` it carries:
    #     rinr - rinr_hi = (D/Dn) * (1 - 10^-2)
    # This is exact up to the model's own arithmetic and needs no assumption
    # about the relative size of the self / multi-UAV terms.
    kappa = float(cfg.interference.direct_cancellation_db)
    cfg_hi = apply_overrides(
        cfg, {"interference.direct_cancellation_db": kappa + 20.0})
    coarse_hi = compute_link_tables(cfg_hi, base)
    rinr_hi = np.asarray(coarse_hi.rinr, dtype=float)

    plan = assign_fusion_nodes(cfg, base, coarse, geom)
    chosen, _d, _stats = select_c2f_adaptive(cfg, base, coarse, plan)

    gamma = np.asarray(coarse.gamma_sense, dtype=float)
    rinr = np.asarray(coarse.rinr, dtype=float)
    n0 = float(noise_power(cfg))
    eps = float(denominator_guard(cfg, n0))

    out = {"area_m": float(area), "rcs_m2": float(rcs), "trial": int(trial),
           "n0": n0, "eps_den": eps, "n_looks": int(cfg.detect.n_looks),
           "g_hw": float(radar_hardware_gain(cfg)),
           "g_proc": float(cfg.waveform.N * cfg.waveform.L),
           "p_default": float(cfg.radio.P_default),
           "noise_figure_db": float(cfg.radio.noise_figure_db),
           "kappa_dc_db": kappa,
           "targets": []}
    for q in range(cfg.scale.Q):
        links = chosen.get(q, [])
        gs = [float(gamma[i, j, q]) for (i, j) in links]
        rs = [float(rinr[i, j]) for (i, j) in links]
        ds = [max(float(rinr[i, j]) - float(rinr_hi[i, j]), 0.0) / 0.99
              for (i, j) in links]
        if not gs:
            out["targets"].append({"target": q, "n_links": 0, "gamma": [],
                                   "rinr": [], "dshare": []})
            continue
        out["targets"].append({
            "target": int(q), "n_links": len(gs), "gamma": gs, "rinr": rs,
            "dshare": ds,
            "required_gain_db": float(10.0 * np.log10(
                required_gain(gs, int(cfg.detect.n_looks), z,
                              cfg.detect.weak_pd_required)))
            if np.isfinite(required_gain(gs, int(cfg.detect.n_looks), z,
                                         cfg.detect.weak_pd_required))
            else float("inf"),
        })
    return out


def apply_steps(gammas, rinr, dshare, n_looks, steps, d_n, n0,
                kappa_base=40.0):
    """Apply an ordered list of levers in closed form.

    State is carried as explicit physical quantities rather than as a single
    mixed ``tot``: ``d_cur`` is the *current* noise floor (``n0 + eps``, which
    the noise-figure lever changes) and ``res`` is the residual interference in
    absolute terms (which the power and cancellation levers change).  Composing
    levers on a single ``tot`` silently breaks as soon as ``d_cur`` moves.

      * ``gain``  (G_proc / G_hw / RCS / geometry): pure numerator;
      * ``power`` : signal and residual both scale, noise does not;
      * ``noise`` : ``n0 -> k*n0``, residual untouched;
      * ``kappa`` : only the direct-cancellation term scales;
      * ``looks`` : does not touch ``gamma``, only the LLR moments.
    """
    gs = list(gammas)
    d_cur = d_n
    res = [d_n * r for r in rinr]
    direct = [ds * d_n for ds in dshare]     # absolute direct-path residual
    n_looks = int(n_looks)
    for kind, value in steps:
        if kind == "gain":
            gs = [g * value for g in gs]
        elif kind == "power":
            gs = [g * value * (d_cur + r) / (d_cur + value * r)
                  for g, r in zip(gs, res)]
            res = [value * r for r in res]
        elif kind == "noise":
            old = d_cur
            d_cur = value * n0 + (d_n - n0)
            gs = [g * (old + r) / (d_cur + r) for g, r in zip(gs, res)]
        elif kind == "kappa":
            # ``direct_cancellation_db`` is a CANCELLATION DEPTH, so the
            # residual scales as 10^(-dB/10): deeper cancellation means a
            # smaller kappa, hence the negative exponent.
            ratio = 10.0 ** (-(value - kappa_base) / 10.0)
            gs = [g * (d_cur + r) / (d_cur + r - dr * (1.0 - ratio))
                  for g, r, dr in zip(gs, res, direct)]
            res = [r - dr * (1.0 - ratio) for r, dr in zip(res, direct)]
        elif kind == "looks":
            n_looks = int(value)
        else:
            raise ValueError(kind)
    return gs, n_looks


def _gap_for_lever(records, steps, z, target_pd):
    """Per-target gap, aggregated exactly like ``probe_lowrcs_shortfall``.

    The simulator defines the system metric as *mean P_D per target, then the
    worst target*.  The matching aggregation here is therefore: mean the
    required gain over trials **within** a target, then take the maximum over
    targets.  Taking a trial-level maximum instead would report a
    several-dB-larger, non-comparable number.
    """
    per_key_target = {}
    per_key_all = {}
    for r in records:
        d_n = r["n0"] + r["eps_den"]
        for t in r["targets"]:
            if not t["gamma"]:
                continue
            gs, n_looks = apply_steps(t["gamma"], t["rinr"], t["dshare"],
                                      r["n_looks"], steps, d_n, r["n0"],
                                      kappa_base=r["kappa_dc_db"])
            g = required_gain(gs, n_looks, z, target_pd)
            db = 10.0 * np.log10(g) if np.isfinite(g) else np.inf
            key = (r["area_m"], r["rcs_m2"])
            per_key_target.setdefault(key, {}).setdefault(
                t["target"], []).append(db)
            per_key_all.setdefault(key, []).append(db)
    worst, median, trial_max = {}, {}, {}
    for (a, c), by_target in per_key_target.items():
        per_target_mean = [float(np.mean(v)) for v in by_target.values()]
        label = f"{a:.0f}/{c}"
        worst[label] = float(max(per_target_mean))
        median[label] = float(np.median(per_target_mean))
        trial_max[label] = float(np.max(per_key_all[(a, c)]))
    return worst, median, trial_max


def evaluate(records, levers, z, target_pd):
    """Apply each lever to every recorded target and report the closing gap."""
    rows = []
    base_row = {"lever": "base", "kind": "base", "value": 1.0}
    w, m, tm = _gap_for_lever(records, [], z, target_pd)
    base_row.update({"gap_worst_db": w, "gap_median_db": m,
                     "gap_trial_max_db": tm})
    for name, steps in levers:
        if not steps:
            continue
        w, m, tm = _gap_for_lever(records, steps, z, target_pd)
        rows.append({"lever": name,
                     "kind": "+".join(k for k, _ in steps),
                     "value": [v for _, v in steps],
                     "gap_worst_db": w, "gap_median_db": m,
                     "gap_trial_max_db": tm})
    return base_row, rows


LEVERS = [
    # (label, [(kind, value), ...])  -- applied in order
    # --- numerator-only family: never saturates -------------------------
    ("G_proc x2   (sensing_processing_gain = 2*N*L)", [("gain", 2.0)]),
    ("G_proc x4   (= 128x128 effective)", [("gain", 4.0)]),
    ("G_proc x16  (= 256x256 effective)", [("gain", 16.0)]),
    ("G_proc x64", [("gain", 64.0)]),
    ("G_hw +10 dB (radar_net_gain_db)", [("gain", 10.0 ** 0.5)]),
    ("G_hw +15 dB", [("gain", 10.0 ** 0.75)]),
    ("G_hw +20 dB", [("gain", 10.0)]),
    ("RCS x4 (+6 dB)", [("gain", 4.0)]),
    ("RCS x10 (+10 dB)", [("gain", 10.0)]),
    # --- saturating family: power / noise figure ------------------------
    ("P_default x2   (+3 dB)", [("power", 2.0)]),
    ("P_default x10  (+10 dB)", [("power", 10.0)]),
    ("P_default x100 (+20 dB)", [("power", 100.0)]),
    ("NF 7 -> 3 dB", [("noise", 0.4)]),
    ("NF 7 -> 0 dB", [("noise", 0.2)]),
    # --- CPI length: enters the LLR moments as sqrt(L), not L -----------
    ("n_looks 16 -> 64", [("looks", 64)]),
    ("n_looks 16 -> 256", [("looks", 256)]),
    # --- direct-path cancellation: the only lever that cuts the denominator
    ("kappa_dc 40 -> 50 dB", [("kappa", 50.0)]),
    ("kappa_dc 40 -> 60 dB", [("kappa", 60.0)]),
    ("kappa_dc 40 -> 80 dB", [("kappa", 80.0)]),
    # --- combinations: does removing interference revive the power lever?
    ("kappa 60 + P x100", [("kappa", 60.0), ("power", 100.0)]),
    ("kappa 60 + P x100 + NF 3", [("kappa", 60.0), ("noise", 0.4),
                                  ("power", 100.0)]),
    ("kappa 60 + NF 3", [("kappa", 60.0), ("noise", 0.4)]),
    ("kappa 60 + n_looks 256", [("kappa", 60.0), ("looks", 256)]),
    ("G_proc x4 + kappa 60", [("gain", 4.0), ("kappa", 60.0)]),
    ("G_proc x4 + kappa 60 + n_looks 64", [("gain", 4.0), ("kappa", 60.0),
                                           ("looks", 64)]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--areas", type=float, nargs="+", default=[400.0])
    parser.add_argument("--rcs", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2])
    parser.add_argument("--out", type=Path,
                        default=Path("results_v1_lever_closure"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)

    specs = [(a, c, t) for a in args.areas for c in args.rcs
             for t in range(args.mc)]
    records = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(job, s) for s in specs]
        for index, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if index % 20 == 0:
                print(f"{index}/{len(specs)}", flush=True)

    with (args.out / "links.json").open("w", encoding="utf8") as handle:
        json.dump(records, handle)

    # ---- regime diagnostic: how interference-limited are the chosen links?
    print("\n" + "=" * 96)
    print("REGIME DIAGNOSTIC  rinr = residual_interference / (n0 + eps)")
    print("=" * 96)
    r0 = records[0]
    print(f"  n0 = {r0['n0']:.4g}   eps_den = {r0['eps_den']:.4g}   "
          f"eps/n0 = {r0['eps_den'] / r0['n0']:.2f}")
    print(f"  G_proc = {r0['g_proc']:.0f}   G_hw = {r0['g_hw']:.4g}   "
          f"P_default = {r0['p_default']}   NF = {r0['noise_figure_db']} dB")
    by_key = {}
    for r in records:
        for t in r["targets"]:
            if t["rinr"]:
                by_key.setdefault((r["area_m"], r["rcs_m2"]), []).extend(t["rinr"])
    print(f"\n  {'area/rcs':>12s} {'n_links':>8s} {'rinr_med_dB':>12s} "
          f"{'rinr_p90_dB':>12s} {'regime':>16s}")
    for k in sorted(by_key):
        vals = np.asarray(by_key[k], dtype=float)
        med = float(np.median(vals))
        p90 = float(np.percentile(vals, 90))
        regime = ("interference-limited" if med > 1.0
                  else "noise-limited" if med < 0.1 else "mixed")
        print(f"  {k[0]:.0f} m / {k[1]:>5} {'':>8s} "
              f"{10 * np.log10(max(med, 1e-300)):>12.2f} "
              f"{10 * np.log10(max(p90, 1e-300)):>12.2f} {regime:>16s}")
    print(f"\n  n_links = {np.mean([t['n_links'] for r in records for t in r['targets']]):.2f}"
          f" per target   (rinr > 1 means the residual interference beats the")
    print("  receiver noise: power and noise figure buy (almost) nothing there)")

    ds_all = [d for r in records for t in r["targets"] for d in t["dshare"]]
    if ds_all:
        ds = np.asarray(ds_all, dtype=float)
        print(f"\n  direct-cancellation share of the residual interference "
              f"(kappa_dc = {records[0]['kappa_dc_db']:.0f} dB):")
        print(f"    median {np.median(ds):.4f}   mean {ds.mean():.4f}   "
              f"p10 {np.percentile(ds, 10):.4f}   p90 {np.percentile(ds, 90):.4f}")

    # ---- lever closure table
    cfg0 = config(SEED, REPORT_CAP)
    z = float(norm.ppf(1.0 - cfg0.detect.Pfa_target))
    target_pd = float(cfg0.detect.weak_pd_required)
    base_row, rows = evaluate(records, LEVERS, z, target_pd)

    print("\n" + "=" * 96)
    print(f"BUDGET CLOSURE  gamma-dB still needed to lift the WEAKEST target "
          f"to P_D = {target_pd:.2f}")
    print("  (analytic LLR-moment bound; 0 means the weak target is closed)")
    print("=" * 96)
    keys = sorted(base_row["gap_worst_db"])
    hdr = f"  {'lever':<44s}" + "".join(f"{k:>12s}" for k in keys)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in [base_row] + rows:
        cells = "".join(
            f"{min(row['gap_worst_db'][k], 999.9):>12.2f}" for k in keys)
        print(f"  {row['lever']:<44s}{cells}")
    print("\n  Reading: the number is the extra per-link sensing-SINR gain (dB)")
    print("  that would still be required after the lever is applied. A lever")
    print("  that removes ~all of it closes the physics; nothing else does.")

    with (args.out / "closure.json").open("w", encoding="utf8") as handle:
        json.dump({"base": base_row, "levers": rows,
                   "weak_pd_required": target_pd}, handle, indent=1)
    print(f"\n  wrote {args.out / 'closure.json'}")


if __name__ == "__main__":
    main()
