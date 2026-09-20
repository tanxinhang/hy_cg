"""Paired TP-UIC prior screen at fixed n_cpi=1.

This is a receiver-only screening experiment.  It changes the interference
coefficient prior, never the truth geometry or detector outcome, and reports the
lower tail of cancellation and per-target echo survival.  A prior value is not
promoted from this screen alone: surviving candidates must subsequently pass
conditional H0/H1 and CFAR tests on an independent holdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim import cancellation as cx  # noqa: E402
from isac_sim.belief import BeliefState  # noqa: E402
from isac_sim.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.model import build_base_gains, generate_geometry, radar_hardware_gain  # noqa: E402


def build_config(args, prior_variance: float) -> Config:
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
        "cancellation.prior_variance": float(prior_variance),
        "aperture.enable": bool(args.m_rx > 1),
        "aperture.m_rx": int(args.m_rx),
    })


def measure(cfg: Config, trial: int, arm: str):
    rng = np.random.default_rng([cfg.run.seed, int(trial)])
    truth = generate_geometry(cfg, rng)
    base_truth = build_base_gains(cfg, truth, rng)
    belief = BeliefState.from_truth(cfg, truth, rng)
    geom_belief = belief.as_geometry(truth)
    base_belief = build_base_gains(
        cfg, geom_belief, rng, channel=base_truth,
        rcs_view=cfg.prior.scheduler_rcs.lower(),
    )
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
        rng=np.random.default_rng([cfg.run.seed, 4_000_000 + int(trial)]),
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
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--trial-start", type=int, default=0)
    ap.add_argument("--prior", default="0.1,0.3,1,3,10")
    ap.add_argument("--arm", default="tp_uic_full")
    ap.add_argument("--max-protected-targets", type=int, default=3)
    ap.add_argument("--out", default="results_tpuic_prior_screen")
    args = ap.parse_args(argv)

    priors = [float(value) for value in args.prior.split(",")]
    if not priors or any(not math.isfinite(value) or value <= 0.0 for value in priors):
        raise ValueError("--prior must contain finite positive values")

    os.makedirs(args.out, exist_ok=True)
    rows = []
    started = time.time()
    for offset, trial in enumerate(range(
        int(args.trial_start), int(args.trial_start) + int(args.trials)
    )):
        for prior in priors:
            cfg = build_config(args, prior)
            got = measure(cfg, trial, args.arm)
            kappa = np.asarray(got.kappa_db, dtype=float)
            kappa = kappa[np.isfinite(kappa)]
            eta = got.as_retention(source="per_target", per_target=args.q)
            rows.append({
                "trial": int(trial),
                "arm": str(args.arm),
                "prior_variance": float(prior),
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

    path = os.path.join(args.out, "tpuic_prior_screen.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    saved = vars(args).copy()
    saved.update(effective_n_cpi=1, cpi_scaling_in_scope=False)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(saved, fh, indent=2, sort_keys=True)

    print("\n%-10s %10s %10s %10s %10s" % (
        "prior", "kappa", "kappa_p10", "eta_p10", "eta_min"
    ))
    for prior in priors:
        sub = [row for row in rows if row["prior_variance"] == prior]
        med = lambda key: float(np.median([row[key] for row in sub]))
        print("%-10g %10.2f %10.2f %10.4f %10.4f" % (
            prior, med("kappa_median_db"), med("kappa_p10_db"),
            med("eta_p10"), med("eta_min"),
        ))
    print("\nwrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
