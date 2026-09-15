#!/usr/bin/env python3
"""Run the V1.2 joint-bundle comparison and small-system oracle audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isac_sim.bundle_master import joint_bundle_column_generation  # noqa: E402
from isac_sim.config import Config, apply_overrides, apply_preset, validate_config  # noqa: E402
from isac_sim.model import build_base_gains, compute_link_tables, generate_geometry  # noqa: E402
from isac_sim.oracle import joint_fusion_selection_oracle, lexicographic_gap_components  # noqa: E402
from isac_sim.report import scalar_summary_row, write_rows_csv  # noqa: E402
from isac_sim.simulate import run_simulation  # noqa: E402


METHODS = [
    "joint_bundle_cg",
    "fixed_fusion_bundle",
    "proposed_c2f_adaptive_pd",
    "sense_sinr_budgeted",
    "local_only_bundle",
]


def configured_v12(args: argparse.Namespace) -> Config:
    cfg = apply_preset(Config(), args.preset)
    cfg = apply_overrides(cfg, {
        "detect.comm_error_model": "erasure",
        "selector.use_delay_price": False,
        "selector.lambda_c": 0.0,
        "selector.require_local_anchor": False,
        "selector.max_local_observations_per_target": 1,
        "selector.max_remote_reports": args.remote_reports,
        "selector.max_observations_per_receiver": args.receiver_capacity,
        "selector.max_observations_per_fusion_uav": args.fusion_capacity,
        "fusion.max_targets_per_uav": args.target_capacity,
        "fusion.rule": "capacitated_value",
        "run.num_mc": args.mc,
        "run.seed": args.seed,
        "run.workers": args.workers,
        "run.verbose": False,
    })
    validate_config(cfg)
    return cfg


def run_main(args: argparse.Namespace, out: Path) -> None:
    cfg = configured_v12(args)
    started = time.perf_counter()
    summary = run_simulation(
        cfg, methods=METHODS, paired_reference="joint_bundle_cg"
    )
    rows = [
        scalar_summary_row(summary, method, {
            "experiment": "joint_bundle_v1.2_main",
            "preset": args.preset,
            "mc": args.mc,
            "seed": args.seed,
            "wall_time_s": time.perf_counter() - started,
        })
        for method in METHODS
    ]
    main_dir = out / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(rows, main_dir / "main.csv")
    (main_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8"
    )
    for method in METHODS:
        item = summary[method]
        print(
            f"{method:28s} PD={item['P_D']:.4f} "
            f"weak={item['P_D_weak']:.4f} PFA={item['P_FA']:.4f} "
            f"reports={item['selected_links_mean']:.2f}"
        )


def run_oracle(args: argparse.Namespace, out: Path) -> None:
    cfg = configured_v12(args)
    cfg = apply_overrides(cfg, {
        "scale.M": 3,
        "scale.Q": 2,
        "dd.use_otfs_bin_validity": False,
        "fusion.max_targets_per_uav": 1,
        "selector.max_links_per_target": 2,
        "selector.max_total_links": 3,
        "selector.max_local_observations_per_target": 1,
        "selector.max_remote_reports": 1,
        "selector.max_observations_per_receiver": 2,
        "selector.max_observations_per_fusion_uav": 2,
        "selector.bundle_shortlist_per_type": 20,
        "selector.bundle_exact_pricing_max_candidates": 20,
    })
    rows: list[dict[str, float | int]] = []
    for trial in range(args.oracle_trials):
        rng = np.random.default_rng([cfg.run.seed, 88001, trial])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        candidate = joint_bundle_column_generation(cfg, base, tables)
        _, _, oracle = joint_fusion_selection_oracle(cfg, base, tables)
        gaps = lexicographic_gap_components(candidate.objective, oracle)
        rows.append({
            "trial": trial,
            **candidate.objective,
            **gaps,
            "column_count": candidate.column_count,
            "pricing_iterations": candidate.pricing_iterations,
            "lp_worst_deficit_bound": candidate.lp_worst_deficit_bound,
        })
        print(
            f"oracle {trial + 1:3d}/{args.oracle_trials}: "
            f"exact={int(gaps['exact_lexicographic_match'])} "
            f"columns={candidate.column_count}"
        )
    oracle_dir = out / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    with (oracle_dir / "oracle_gap.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (oracle_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="small-uav-compact-800m")
    parser.add_argument("--mode", choices=("main", "oracle", "all"), default="all")
    parser.add_argument("--mc", type=int, default=20)
    parser.add_argument("--oracle-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--remote-reports", type=int, default=2)
    parser.add_argument("--receiver-capacity", type=int, default=2)
    parser.add_argument("--fusion-capacity", type=int, default=2)
    parser.add_argument("--target-capacity", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "results_joint_bundle_v12")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode in {"main", "all"}: run_main(args, args.out)
    if args.mode in {"oracle", "all"}: run_oracle(args, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
