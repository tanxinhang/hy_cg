"""Audit analytic TP-UIC residual-power moments against receiver realizations.

The geometry, belief, channel and n_cpi=1 budget are frozen.  Only direct-path
phases and receiver noise are redrawn.  This isolates the realization risk that
the planning capability certificate must cover.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim import cancellation as cx  # noqa: E402
from isac_sim.belief import BeliefState  # noqa: E402
from isac_sim.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.model import build_base_gains, generate_geometry, radar_hardware_gain  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--receiver", type=int, default=0)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--tail-probability", type=float, default=0.20)
    ap.add_argument("--arm", default="tp_uic_full")
    ap.add_argument("--out", default="results_tpuic_residual_risk_audit")
    args = ap.parse_args(argv)
    if args.reps < 2:
        ap.error("--reps must be at least 2")
    if not 0.0 < args.tail_probability < 1.0:
        ap.error("--tail-probability must lie in (0,1)")

    cfg = apply_overrides(apply_preset(Config(), "paper-canonical"), {
        "scale.M": 6,
        "scale.Q": 3,
        "geometry.area_xy": 600.0,
        "detect.target_rcs": 0.1,
        "run.seed": 2026,
        "run.verbose": False,
        "cancellation.enable": True,
        "cancellation.n_cpi": 1,
        "cancellation.max_protected_targets": 3,
        "aperture.enable": True,
        "aperture.m_rx": 4,
    })
    if not 0 <= args.receiver < cfg.scale.M:
        ap.error("--receiver is outside the configured fleet")
    rng = np.random.default_rng([cfg.run.seed, int(args.trial)])
    truth = generate_geometry(cfg, rng)
    base_truth = build_base_gains(cfg, truth, rng)
    belief = BeliefState.from_truth(cfg, truth, rng)
    geom_belief = belief.as_geometry(truth)
    base_belief = build_base_gains(
        cfg, geom_belief, rng, channel=base_truth,
        rcs_view=cfg.prior.scheduler_rcs.lower(),
    )
    sense = np.full(cfg.scale.M, cfg.radio.rho * cfg.radio.P_default)
    ctx = cx.ReceiverContext.from_trial(
        cfg, truth, geom_belief, base_truth, base_belief=base_belief,
        sense_power=sense, radiated_power=sense,
        processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)), arm=args.arm,
    )

    rows = []
    j = int(args.receiver)
    for rep in range(int(args.reps)):
        got = cx.measure_receiver_context(
            ctx,
            rng=np.random.default_rng([cfg.run.seed, 9_000_000, args.trial, rep]),
            receivers=[j],
            residual_quantile_probability=1.0 - args.tail_probability,
        )
        mean = float(got.i_res_moment_mean[j])
        variance = float(got.i_res_pred_var[j])
        upper = mean + np.sqrt(
            (1.0 - args.tail_probability) / args.tail_probability * variance
        )
        prior_upper = float(
            got.as_prior_quantile_fraction()[j] * got.i_in_pred[j]
        )
        rows.append({
            "rep": rep,
            "i_res": float(got.i_res[j]),
            "moment_mean": mean,
            "moment_variance": variance,
            "cantelli_upper": float(upper),
            "covered": bool(got.i_res[j] <= upper),
            "prior_upper": float(prior_upper),
            "prior_covered": bool(got.i_res[j] <= prior_upper),
            "n_cpi": 1,
        })

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "residual_risk.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    values = np.asarray([row["i_res"] for row in rows], dtype=float)
    mean = float(rows[0]["moment_mean"])
    variance = float(rows[0]["moment_variance"])
    summary = {
        "empirical_mean_over_analytic": float(np.mean(values) / mean),
        "empirical_variance_over_analytic": float(np.var(values) / variance),
        "coverage": float(np.mean([row["covered"] for row in rows])),
        "prior_quantile_coverage": float(np.mean([
            row["prior_covered"] for row in rows
        ])),
        "required_coverage": float(1.0 - args.tail_probability),
        "reps": int(args.reps),
        "trial": int(args.trial),
        "receiver": j,
        "arm": str(args.arm),
        "n_cpi": 1,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("wrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
