#!/usr/bin/env python3
"""Coarse, declared net-radar-gain sweep for the small-UAV scenarios.

This is an algorithmic sensitivity experiment, not an antenna specification.
It varies only the net desired-echo gain and never invents a Tx/Rx split.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from isac_sim.core.config import Config, apply_overrides, apply_preset, validate_config  # noqa: E402
from experiments.flow.simulate import run_simulation  # noqa: E402


METHOD = "proposed_c2f_adaptive_pd"
RULES = (
    ("pd_lookahead", "capacitated_pd_lookahead", 2),
    ("nearest", "nearest_target_capacitated", 2),
)


def experiment_config(preset: str, gain_db: float, mc: int, seed: int,
                      report_budget: int = 2) -> Config:
    cfg = apply_preset(Config(), preset)
    cfg = apply_overrides(cfg, {
        "radio.radar_net_gain_db": gain_db,
        "detect.comm_error_model": "erasure",
        "selector.score_mode": "detector_pd",
        "selector.use_delay_price": False,
        "selector.lambda_c": 0.0,
        "selector.require_local_anchor": True,
        "selector.max_local_observations_per_target": 1,
        "selector.max_remote_reports": report_budget,
        "selector.max_observations_per_receiver": 2,
        "selector.max_observations_per_fusion_uav": 2,
        "fusion.max_targets_per_uav": 1,
        "run.num_mc": mc,
        "run.seed": seed,
        "run.verbose": False,
    })
    validate_config(cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="small-uav-nominal-s2",
                        choices=("small-uav-compact-800m", "small-uav-dense-s1",
                                 "small-uav-nominal-s2", "small-uav-sparse-s3"))
    parser.add_argument("--gains", type=float, nargs="+", default=[0, 10, 20, 30])
    parser.add_argument("--mc", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--include-local-only", action="store_true",
                        help="also run PD-lookahead with K_remote=0")
    parser.add_argument("--out", type=Path,
                        default=Path("results_radar_net_gain_sweep.csv"))
    args = parser.parse_args()

    rows = []
    rules = list(RULES)
    if args.include_local_only:
        rules.insert(0, ("local_only", "capacitated_pd_lookahead", 0))
    for gain_db in args.gains:
        for label, rule, report_budget in rules:
            cfg = experiment_config(
                args.preset, gain_db, args.mc, args.seed, report_budget
            )
            cfg.fusion.rule = rule
            summary = run_simulation(cfg, methods=[METHOD])[METHOD]
            row = {
                "preset": args.preset,
                "net_gain_db": gain_db,
                "fusion_rule": label,
                "report_budget": report_budget,
                "mc": args.mc,
                "seed": args.seed,
                "P_D": summary["P_D"],
                "P_D_weak": summary["P_D_weak"],
                "remote_reports": summary["selected_links_mean"],
                "selected_observations": summary["selected_observations_mean"],
                "all_targets_satisfied_prob": summary["all_targets_satisfied_prob"],
                "actual_worst_target_P_D": summary["actual_worst_target_P_D"],
                "selector_score_evaluations": summary["selector_score_evaluations_mean"],
            }
            rows.append(row)
            print(
                f"gain={gain_db:5.1f} dB  rule={label:12s}  "
                f"P_D={row['P_D']:.3f}  weak={row['P_D_weak']:.3f}  "
                f"reports={row['remote_reports']:.2f}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
