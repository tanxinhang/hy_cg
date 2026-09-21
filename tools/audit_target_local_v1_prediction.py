#!/usr/bin/env python3
"""Stress target-local V1 over tracker prediction-error levels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from experiments.app.report import scalar_summary_row, write_rows_csv  # noqa: E402
from experiments.flow.simulate import run_simulation  # noqa: E402


METHODS = [
    "proposed_c2f_adaptive_pd",
    "exact_marginal_greedy",
    "sense_sinr",
]
ERROR_GRID = [
    ("perfect", 0.0, 0.0),
    ("mild", 50.0, 5.0),
    ("nominal", 150.0, 15.0),
    ("high", 300.0, 25.0),
    ("severe", 500.0, 40.0),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results_target_local_v1" / "prediction-stress",
    )
    args = parser.parse_args()
    if args.mc < 1:
        parser.error("--mc must be at least one")
    if args.workers < 1:
        parser.error("--workers must be at least one")

    base = apply_preset(Config(), "target-local-v1")
    rows: list[dict[str, object]] = []
    resolved_configs: dict[str, object] = {}
    for label, sigma_pos, sigma_vel in ERROR_GRID:
        cfg = apply_overrides(
            base,
            {
                "prior.belief_sigma_pos_m": sigma_pos,
                "prior.belief_sigma_vel_mps": sigma_vel,
                "run.num_mc": args.mc,
                "run.seed": args.seed,
                "run.workers": args.workers,
                "run.verbose": False,
            },
        )
        print(
            f"prediction stress {label}: sigma_pos={sigma_pos:g} m, "
            f"sigma_vel={sigma_vel:g} m/s",
            flush=True,
        )
        summary = run_simulation(
            cfg,
            methods=METHODS,
            paired_reference="proposed_c2f_adaptive_pd",
        )
        for method in METHODS:
            rows.append(
                scalar_summary_row(
                    summary,
                    method,
                    {
                        "experiment": "target_local_v1_prediction_stress",
                        "error_level": label,
                        "belief_sigma_pos_m": sigma_pos,
                        "belief_sigma_vel_mps": sigma_vel,
                    },
                )
            )
        resolved_configs[label] = asdict(cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    write_rows_csv(rows, args.out / "prediction_stress.csv")
    (args.out / "configs.json").write_text(
        json.dumps(resolved_configs, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved prediction-stress audit under {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
