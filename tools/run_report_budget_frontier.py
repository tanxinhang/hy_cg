#!/usr/bin/env python3
"""Report-budget Pareto frontier under one frozen physical scenario."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from isac_sim.config import Config, apply_overrides, apply_preset, validate_config  # noqa: E402
from isac_sim.simulate import run_simulation  # noqa: E402


METHOD = "proposed_c2f_adaptive_pd"
RULES = (
    ("pd_lookahead", "capacitated_pd_lookahead"),
    ("nearest", "nearest_target_capacitated"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="small-uav-compact-800m")
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--net-gain-db", type=float, default=0.0)
    parser.add_argument("--mc", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path,
                        default=Path("results_report_budget_frontier.csv"))
    args = parser.parse_args()

    rows = []
    for budget in args.budgets:
        for label, rule in RULES:
            cfg = apply_preset(Config(), args.preset)
            cfg = apply_overrides(cfg, {
                "radio.radar_net_gain_db": args.net_gain_db,
                "detect.comm_error_model": "erasure",
                "selector.score_mode": "detector_pd",
                "selector.use_delay_price": False,
                "selector.lambda_c": 0.0,
                "selector.require_local_anchor": True,
                "selector.max_local_observations_per_target": 1,
                "selector.max_remote_reports": budget,
                "selector.max_observations_per_receiver": 2,
                "selector.max_observations_per_fusion_uav": 2,
                "fusion.max_targets_per_uav": 1,
                "fusion.rule": rule,
                "run.num_mc": args.mc,
                "run.seed": args.seed,
                "run.verbose": False,
            })
            validate_config(cfg)
            summary = run_simulation(cfg, methods=[METHOD])[METHOD]
            row = {
                "preset": args.preset,
                "net_gain_db": args.net_gain_db,
                "report_budget": budget,
                "fusion_rule": label,
                "mc": args.mc,
                "seed": args.seed,
                "P_D": summary["P_D"],
                "P_D_weak": summary["P_D_weak"],
                "actual_reports": summary["selected_links_mean"],
                "selected_observations": summary["selected_observations_mean"],
                "all_targets_satisfied_prob": summary["all_targets_satisfied_prob"],
                "actual_worst_target_P_D": summary["actual_worst_target_P_D"],
            }
            rows.append(row)
            print(
                f"K={budget:2d}  rule={label:12s}  P_D={row['P_D']:.3f}  "
                f"weak={row['P_D_weak']:.3f}  reports={row['actual_reports']:.2f}",
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
