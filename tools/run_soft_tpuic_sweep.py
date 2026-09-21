"""Paired DD-only soft-TP-UIC sweep at fixed n_cpi=1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim.receiver import cancellation as cx  # noqa: E402
from isac_sim.scenario.belief import BeliefState  # noqa: E402
from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.sensing.model import build_base_gains, generate_geometry, radar_hardware_gain  # noqa: E402


def config(args, mu: float, *, adaptive: bool = False) -> Config:
    return apply_overrides(apply_preset(Config(), args.preset), {
        "scale.M": args.m,
        "scale.Q": args.q,
        "geometry.area_xy": args.area,
        "detect.target_rcs": args.rcs,
        "run.seed": args.seed,
        "run.verbose": False,
        "cancellation.enable": True,
        "cancellation.n_cpi": 1,
        "cancellation.max_protected_targets": min(args.max_protected_targets, args.q),
        "cancellation.soft_protection_mu": float(mu),
        "cancellation.adaptive_soft_enable": bool(adaptive),
        "cancellation.adaptive_soft_risk_slack": float(args.adaptive_risk_slack),
        "aperture.enable": bool(args.m_rx > 1),
        "aperture.m_rx": int(args.m_rx),
    })


def one(cfg: Config, trial: int, arm: str):
    rng = np.random.default_rng([cfg.run.seed, trial])
    truth = generate_geometry(cfg, rng)
    base_truth = build_base_gains(cfg, truth, rng)
    belief = BeliefState.from_truth(cfg, truth, rng)
    geom_belief = belief.as_geometry(truth)
    base_belief = build_base_gains(
        cfg, geom_belief, rng, channel=base_truth,
        rcs_view=cfg.prior.scheduler_rcs.lower(),
    )
    m = int(cfg.scale.M)
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    ctx = cx.ReceiverContext.from_trial(
        cfg, truth, geom_belief, base_truth, base_belief=base_belief,
        sense_power=sense, radiated_power=sense,
        processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)), arm=arm,
    )
    return cx.measure_receiver_context(
        ctx, rng=np.random.default_rng([cfg.run.seed, 3_000_000 + trial]),
        all_targets=True, residual_quantile_probability=0.80,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--m-rx", type=int, default=1)
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--trial-start", type=int, default=0)
    ap.add_argument("--mu", default="0,0.01,0.1,1,10,100")
    ap.add_argument("--max-protected-targets", type=int, default=3)
    ap.add_argument("--adaptive-risk-slack", type=float, default=0.0)
    ap.add_argument(
        "--adaptive-only", action="store_true",
        help="run only the paired hard/adaptive arms for cell-level audits",
    )
    ap.add_argument("--out", default="results_soft_tpuic_sweep")
    args = ap.parse_args(argv)
    if not 0.0 <= args.adaptive_risk_slack <= 1.0:
        ap.error("--adaptive-risk-slack must lie in [0, 1]")
    mus = [float(v) for v in args.mu.split(",")]
    os.makedirs(args.out, exist_ok=True)
    rows = []
    cell_rows = []
    started = time.time()
    for trial in range(args.trial_start, args.trial_start + args.trials):
        arm_specs = [
            ("hard", float("nan"), "tp_uic_full"),
            ("targeted", float("nan"), "targeted_tpuic_full"),
            ("adaptive", float("nan"), "adaptive_soft_tpuic"),
        ] + [
            ("soft", value, "soft_tpuic") for value in mus
        ]
        if args.adaptive_only:
            arm_specs = [arm_specs[0], arm_specs[2]]
        for label, mu, arm in arm_specs:
            cfg = config(
                args,
                0.0 if label in ("hard", "targeted", "adaptive") else mu,
                adaptive=(label == "adaptive"),
            )
            got = one(cfg, trial, arm)
            eta = got.as_retention(source="per_target", per_target=args.q)
            finite_kappa = got.kappa_db[np.isfinite(got.kappa_db)]
            rows.append({
                "trial": trial,
                "receiver": label,
                "mu": mu,
                "kappa_median_db": float(np.median(finite_kappa)),
                "kappa_p10_db": float(np.quantile(finite_kappa, 0.10)),
                "eta_median": float(np.median(eta)),
                "eta_p10": float(np.quantile(eta, 0.10)),
                "eta_min": float(np.min(eta)),
                "selected_mu_median": float(np.nanmedian(got.selected_soft_mu_by_target))
                if np.any(np.isfinite(got.selected_soft_mu_by_target)) else float("nan"),
                "hard_fallback_fraction": float(np.mean(
                    ~np.isfinite(got.selected_soft_mu_by_target)
                )) if label == "adaptive" else float("nan"),
                "n_cpi": 1,
            })
            for receiver in range(int(args.m)):
                for target in range(int(args.q)):
                    cell_rows.append({
                        "trial": trial,
                        "receiver_arm": label,
                        "receiver": receiver,
                        "target": target,
                        "measured_fraction": float(
                            got.fraction_by_target[receiver, target]
                        ),
                        "predicted_fraction": float(
                            got.predicted_fraction_by_target[receiver, target]
                        ),
                        "prior_q80_fraction": float(
                            got.prior_quantile_fraction_by_target[receiver, target]
                        ),
                        "realised_retention": float(
                            got.eta_survive_by_target[receiver, target]
                        ),
                        "risk_retention": float(
                            got.risk_retention_by_target[receiver, target]
                        ),
                        "selected_mu": float(
                            got.selected_soft_mu_by_target[receiver, target]
                        ),
                    })
        print("trial %d/%d [%.1f s]" % (
            trial - args.trial_start + 1, args.trials, time.time() - started
        ), flush=True)

    path = os.path.join(args.out, "soft_tpuic_sweep.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    cell_path = os.path.join(args.out, "soft_tpuic_cells.csv")
    with open(cell_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cell_rows[0]))
        writer.writeheader(); writer.writerows(cell_rows)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as fh:
        saved = vars(args).copy(); saved["effective_n_cpi"] = 1
        saved["cpi_scaling_in_scope"] = False
        json.dump(saved, fh, indent=2, sort_keys=True)

    print("\n%-8s %8s %10s %10s %10s" % (
        "arm", "mu", "kappa", "eta_p10", "eta_min"
    ))
    keys = [
        ("hard", float("nan")), ("targeted", float("nan")),
        ("adaptive", float("nan")),
    ] + [
        ("soft", mu) for mu in mus
    ]
    for label, mu in keys:
        sub = [r for r in rows if r["receiver"] == label and (
            label in ("hard", "targeted", "adaptive") or float(r["mu"]) == mu
        )]
        if not sub:
            continue
        med = lambda key: float(np.median([r[key] for r in sub]))
        print("%-8s %8s %10.2f %10.4f %10.4f" % (
            label, "-" if label in ("hard", "targeted", "adaptive") else "%g" % mu,
            med("kappa_median_db"), med("eta_p10"), med("eta_min"),
        ))
    print("\nwrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
