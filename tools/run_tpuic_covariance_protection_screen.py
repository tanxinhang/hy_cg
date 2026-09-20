"""Paired receiver-only screen for covariance-inflated TP-UIC protection.

The truth, belief, channel and receiver random stream are identical between
the hard tangent baseline and the covariance-protection candidate.  The screen
is fixed at n_cpi=1 and reports lower-tail cancellation and genuine per-target
echo survival; it does not use detector outcomes to tune the receiver.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim import cancellation as cx  # noqa: E402
from isac_sim.belief import BeliefState  # noqa: E402
from isac_sim.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.model import build_base_gains, generate_geometry, radar_hardware_gain  # noqa: E402


def config(args, enabled: bool) -> Config:
    return apply_overrides(apply_preset(Config(), args.preset), {
        "scale.M": int(args.m),
        "scale.Q": int(args.q),
        "geometry.area_xy": float(args.area),
        "detect.target_rcs": float(args.rcs),
        "run.seed": int(args.seed),
        "run.verbose": False,
        "cancellation.enable": True,
        "cancellation.n_cpi": 1,
        "cancellation.max_protected_targets": min(
            int(args.max_protected_targets), int(args.q)
        ),
        "cancellation.covariance_protection": bool(enabled),
        "aperture.enable": bool(args.m_rx > 1),
        "aperture.m_rx": int(args.m_rx),
    })


def world(cfg: Config, trial: int):
    rng = np.random.default_rng([cfg.run.seed, int(trial)])
    truth = generate_geometry(cfg, rng)
    base_truth = build_base_gains(cfg, truth, rng)
    belief = BeliefState.from_truth(cfg, truth, rng)
    geom_belief = belief.as_geometry(truth)
    base_belief = build_base_gains(
        cfg, geom_belief, rng, channel=base_truth,
        rcs_view=cfg.prior.scheduler_rcs.lower(),
    )
    return truth, geom_belief, base_truth, base_belief


def measure(cfg, trial, truth, geom_belief, base_truth, base_belief, arm):
    m = int(cfg.scale.M)
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default, dtype=float)
    ctx = cx.ReceiverContext.from_trial(
        cfg, truth, geom_belief, base_truth, base_belief=base_belief,
        sense_power=sense, radiated_power=sense,
        processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)), arm=arm,
    )
    return cx.measure_receiver_context(
        ctx,
        rng=np.random.default_rng([cfg.run.seed, 6_000_000 + int(trial)]),
        all_targets=True,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--m-rx", type=int, default=4)
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--trial-start", type=int, default=0)
    ap.add_argument("--arm", default="tp_uic_full")
    ap.add_argument("--max-protected-targets", type=int, default=3)
    ap.add_argument("--out", default="results_tpuic_covariance_protection")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    started = time.time()
    for offset, trial in enumerate(range(
        int(args.trial_start), int(args.trial_start) + int(args.trials)
    )):
        cfg0 = config(args, False)
        truth, geom_belief, base_truth, base_belief = world(cfg0, trial)
        for enabled in (False, True):
            cfg = config(args, enabled)
            got = measure(
                cfg, trial, truth, geom_belief, base_truth, base_belief, args.arm
            )
            kappa = np.asarray(got.kappa_db, dtype=float)
            kappa = kappa[np.isfinite(kappa)]
            eta = got.as_retention(source="per_target", per_target=args.q)
            rows.append({
                "trial": int(trial),
                "arm": str(args.arm),
                "covariance_protection": bool(enabled),
                "kappa_median_db": float(np.median(kappa)),
                "kappa_p10_db": float(np.quantile(kappa, 0.10)),
                "kappa_min_db": float(np.min(kappa)),
                "eta_median": float(np.median(eta)),
                "eta_p10": float(np.quantile(eta, 0.10)),
                "eta_min": float(np.min(eta)),
                "noise_enhance_median_db": float(np.median(got.noise_enhance_db)),
                "n_cpi": 1,
            })
        print("trial %d/%d [%.1f s]" % (
            offset + 1, args.trials, time.time() - started
        ), flush=True)

    path = os.path.join(args.out, "tpuic_covariance_protection.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    saved = vars(args).copy()
    saved.update(effective_n_cpi=1, cpi_scaling_in_scope=False)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(saved, fh, indent=2, sort_keys=True)

    print("\n%-12s %10s %10s %10s %10s %10s" % (
        "protection", "kappa", "kappa_p10", "kappa_min", "eta_p10", "eta_min"
    ))
    for enabled in (False, True):
        sub = [r for r in rows if r["covariance_protection"] is enabled]
        med = lambda key: float(np.median([r[key] for r in sub]))
        print("%-12s %10.2f %10.2f %10.2f %10.4f %10.4f" % (
            "covariance" if enabled else "tangent",
            med("kappa_median_db"), med("kappa_p10_db"), med("kappa_min_db"),
            med("eta_p10"), med("eta_min"),
        ))
    print("\nwrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
