#!/usr/bin/env python3
"""Run the V1.3 RCS-robust joint-bundle screening and scenario audit."""

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

from isac_sim.bundle_master import (  # noqa: E402
    joint_bundle_column_generation,
    rcs_robust_bundle_column_generation,
)
from isac_sim.config import Config, apply_overrides, apply_preset, validate_config  # noqa: E402
from isac_sim.fusion import predicted_pd_for_links  # noqa: E402
from isac_sim.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
    rescale_sensing_tables_for_rcs,
)
from isac_sim.report import scalar_summary_row, write_rows_csv  # noqa: E402
from isac_sim.simulate import (  # noqa: E402
    run_method_on_trial,
    run_simulation,
    summarize,
    weakest_belief_target,
)


METHODS = [
    "rcs_robust_bundle_cg",
    "joint_bundle_cg",
    "fixed_fusion_bundle",
    "local_only_bundle",
]


def configured(args: argparse.Namespace) -> Config:
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
        "prior.rcs_lower_factor": args.rcs_lower_factor,
        "run.num_mc": args.mc,
        "run.seed": args.seed,
        "run.workers": args.workers,
        "run.verbose": False,
    })
    validate_config(cfg)
    return cfg


def run_main(cfg: Config, args: argparse.Namespace, out: Path) -> None:
    started = time.perf_counter()
    summary = run_simulation(
        cfg, methods=METHODS, paired_reference="rcs_robust_bundle_cg"
    )
    rows = [
        scalar_summary_row(summary, method, {
            "experiment": "rcs_robust_v1.3_main",
            "preset": args.preset,
            "mc": args.mc,
            "seed": args.seed,
            "rcs_mean_m2": cfg.detect.target_rcs,
            "rcs_lower_m2": (
                cfg.detect.target_rcs * cfg.prior.rcs_lower_factor
            ),
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
            f"paired={item.get('paired_reference_delta_P_D', float('nan')):+.4f}"
        )


def run_scenario_audit(
    cfg: Config, args: argparse.Namespace, out: Path
) -> None:
    factors = tuple(float(item) for item in args.audit_factors.split(","))
    if not factors or any((not np.isfinite(x)) or x <= 0.0 for x in factors):
        raise ValueError("audit factors must be positive comma-separated numbers")
    rows: list[dict[str, float | int | str]] = []
    for trial in range(args.audit_trials):
        rng = np.random.default_rng([cfg.run.seed, 93001, trial])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng, rcs_view="mean")
        coarse = compute_link_tables(cfg, base)
        fine = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
        results = {
            "rcs_robust_bundle_cg": rcs_robust_bundle_column_generation(
                cfg, base, coarse, value_tables=fine
            ),
            "joint_bundle_cg": joint_bundle_column_generation(
                cfg, base, coarse, value_tables=fine
            ),
        }
        for factor in factors:
            scenario = rescale_sensing_tables_for_rcs(cfg, fine, factor)
            for method, result in results.items():
                pds = np.asarray([
                    predicted_pd_for_links(
                        cfg,
                        scenario,
                        q,
                        result.selected.get(q, []),
                        weight_mode="deflection",
                        plan=result.plan,
                        base=base,
                    ) if result.selected.get(q, []) else 0.0
                    for q in range(cfg.scale.Q)
                ])
                deficits = np.maximum(cfg.detect.pd_required - pds, 0.0)
                rows.append({
                    "trial": trial,
                    "method": method,
                    "rcs_factor": factor,
                    "rcs_m2": cfg.detect.target_rcs * factor,
                    "minimum_predicted_pd": float(np.min(pds)),
                    "mean_predicted_pd": float(np.mean(pds)),
                    "worst_detection_deficit": float(np.max(deficits)),
                    "total_detection_deficit": float(np.sum(deficits)),
                    "remote_reports": result.objective["remote_reports"],
                    "processing_load": result.objective["processing_load"],
                })
        print(f"audit {trial + 1:3d}/{args.audit_trials}")
    audit_dir = out / "scenario_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "scenario_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_low_rcs_stress_mc(
    cfg: Config, args: argparse.Namespace, out: Path
) -> None:
    """Evaluate nominal and robust decisions against a hidden low-RCS truth."""
    truth_cfg = apply_overrides(cfg, {
        "detect.target_rcs": cfg.detect.target_rcs * args.stress_rcs_factor,
        "detect.rcs_model": "mean",
    })
    methods = ("rcs_robust_bundle_cg", "joint_bundle_cg")
    results = {method: [] for method in methods}
    for trial in range(args.stress_trials):
        rng = np.random.default_rng([cfg.run.seed, 94001, trial])
        geom = generate_geometry(cfg, rng)
        base_truth = build_base_gains(
            truth_cfg, geom, rng, rcs_view="mean"
        )
        tables_truth = compute_link_tables(truth_cfg, base_truth)
        base_scheduler = build_base_gains(
            cfg, geom, rng, channel=base_truth, rcs_view="mean"
        )
        tables_scheduler = compute_link_tables(cfg, base_scheduler)
        fine_scheduler = compute_link_tables(
            cfg, base_scheduler, dd_gain=base_scheduler.eta_fine
        )
        weak = weakest_belief_target(cfg, base_scheduler, tables_scheduler)
        for method in methods:
            results[method].append(run_method_on_trial(
                cfg,
                base_scheduler,
                tables_scheduler,
                method,
                trial,
                c2f_tables=fine_scheduler,
                eval_base=base_truth,
                eval_tables=tables_truth,
                weak_target_index=weak,
            ))
        print(f"stress MC {trial + 1:3d}/{args.stress_trials}")

    summary = {method: summarize(items, cfg) for method, items in results.items()}
    robust = results["rcs_robust_bundle_cg"]
    nominal = results["joint_bundle_cg"]
    diffs = np.asarray([
        (a.detected - b.detected) / max(a.total_targets, 1)
        for a, b in zip(robust, nominal)
    ])
    paired_mean = float(np.mean(diffs))
    paired_half = (
        1.96 * float(np.std(diffs, ddof=1)) / np.sqrt(diffs.size)
        if diffs.size > 1 else 0.0
    )
    summary["rcs_robust_bundle_cg"].update({
        "paired_reference_method": "joint_bundle_cg",
        "paired_reference_delta_P_D": paired_mean,
        "paired_reference_delta_ci95_low": paired_mean - paired_half,
        "paired_reference_delta_ci95_high": paired_mean + paired_half,
    })
    summary["joint_bundle_cg"].update({
        "paired_reference_method": "joint_bundle_cg",
        "paired_reference_delta_P_D": 0.0,
        "paired_reference_delta_ci95_low": 0.0,
        "paired_reference_delta_ci95_high": 0.0,
    })
    rows = []
    for method in methods:
        rows.append(scalar_summary_row(summary, method, {
            "experiment": "rcs_robust_v1.3_hidden_low_rcs",
            "preset": args.preset,
            "trials": args.stress_trials,
            "scheduler_rcs_m2": cfg.detect.target_rcs,
            "truth_rcs_m2": truth_cfg.detect.target_rcs,
        }))
    stress_dir = out / "hidden_low_rcs_mc"
    stress_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(rows, stress_dir / "hidden_low_rcs_mc.csv")
    print(
        "hidden-low-RCS robust-minus-nominal "
        f"delta={paired_mean:+.4f} "
        f"CI=[{paired_mean - paired_half:+.4f}, {paired_mean + paired_half:+.4f}]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="small-uav-compact-800m")
    parser.add_argument(
        "--mode", choices=("main", "audit", "stress", "all"), default="all"
    )
    parser.add_argument("--mc", type=int, default=20)
    parser.add_argument("--audit-trials", type=int, default=20)
    parser.add_argument("--audit-factors", default="0.5,1.0,2.0")
    parser.add_argument("--stress-trials", type=int, default=20)
    parser.add_argument("--stress-rcs-factor", type=float, default=0.5)
    parser.add_argument("--rcs-lower-factor", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    # V1.3 removes the exogenous global report-count ceiling. Report overhead
    # remains the third lexicographic objective, after both reliability tiers.
    parser.add_argument("--remote-reports", type=int, default=-1)
    parser.add_argument("--receiver-capacity", type=int, default=2)
    parser.add_argument("--fusion-capacity", type=int, default=2)
    parser.add_argument("--target-capacity", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results_rcs_robust_v13"
    )
    args = parser.parse_args()
    cfg = configured(args)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode in {"main", "all"}:
        run_main(cfg, args, args.out)
    if args.mode in {"audit", "all"}:
        run_scenario_audit(cfg, args, args.out)
    if args.mode in {"stress", "all"}:
        run_low_rcs_stress_mc(cfg, args, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
