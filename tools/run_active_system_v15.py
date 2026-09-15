#!/usr/bin/env python3
"""Run the V1.5-System global active-column master diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isac_sim.active_information import (  # noqa: E402
    configured_sensing_modes,
    evaluate_active_detection,
)
from isac_sim.active_statistics import paired_cluster_summary  # noqa: E402
from isac_sim.active_system import (  # noqa: E402
    generate_active_columns,
    solve_global_active_master,
)
from isac_sim.config import Config, apply_overrides, apply_preset, validate_config  # noqa: E402
from isac_sim.model import build_base_gains, compute_link_tables, generate_geometry  # noqa: E402


def implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "isac_sim").glob("*.py")) + [Path(__file__).resolve()]:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def configured(args: argparse.Namespace) -> Config:
    cfg = apply_preset(Config(), args.preset)
    cfg = apply_overrides(cfg, {
        "scale.M": args.uavs,
        "scale.Q": args.targets,
        "prior.belief_mode": False,
        "dd.use_otfs_bin_validity": False,
        "active_sensing.enable": True,
        "active_sensing.energy_budget_per_target": args.target_energy,
        "active_sensing.energy_budget_per_uav": args.uav_energy,
        "active_sensing.max_candidates_per_pair": args.candidate_limit,
        "active_sensing.max_tx_observations_per_uav": args.tx_cap,
        "detect.comm_error_model": "erasure",
        "detect.soft_stat_model": "llr",
        "fusion.mode": "explicit",
        "fusion.max_targets_per_uav": args.fusion_target_cap,
        "fusion.cpu_rate_cycles_per_s": args.cpu_budget / 0.01,
        "fusion.processing_window_s": 0.01,
        "selector.max_links_per_target": args.max_observations,
        "selector.max_observations_per_receiver": args.receiver_cap,
        "selector.max_observations_per_fusion_uav": args.fusion_observation_cap,
        "selector.max_remote_reports": args.report_cap,
        "selector.max_total_links": args.total_observation_cap,
        "run.seed": args.seed,
        "run.verbose": False,
    })
    validate_config(cfg)
    return cfg


def run(cfg: Config, args: argparse.Namespace) -> list[dict[str, object]]:
    modes = configured_sensing_modes(cfg)
    nominal = min(modes, key=lambda mode: (
        abs(mode.power_scale - 1.0), abs(mode.looks - 16), not mode.refined
    ))
    rows: list[dict[str, object]] = []
    total = args.seed_count * args.trials
    for seed_index in range(args.seed_count):
        experiment_seed = cfg.run.seed + 104729 * seed_index
        for trial in range(args.trials):
            rng = np.random.default_rng([experiment_seed, 15515, trial])
            geom = generate_geometry(cfg, rng)
            base = build_base_gains(cfg, geom, rng, rcs_view="mean")
            coarse = compute_link_tables(cfg, base)
            refined = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
            columns = generate_active_columns(
                cfg, base, coarse, refined, modes=modes,
                candidate_limit=args.candidate_limit,
                calibration_samples=args.calibration_samples,
                evaluation_samples=args.evaluation_samples,
                seed=experiment_seed + 1009 * trial,
            )
            fixed_columns = [column for column in columns if all(
                observation.mode.power_scale == nominal.power_scale
                and observation.mode.looks == nominal.looks
                and observation.mode.refined == nominal.refined
                for observation in column.observations
            )]
            results = {
                "fixed_nominal_global": solve_global_active_master(cfg, fixed_columns),
                "active_modes_global": solve_global_active_master(cfg, columns),
            }
            reference = np.full(
                cfg.scale.M, max(mode.power_scale for mode in modes), dtype=float
            )
            envelope_coarse = compute_link_tables(
                cfg, base, sensing_power_scale_by_uav=reference
            )
            envelope_refined = compute_link_tables(
                cfg, base, dd_gain=base.eta_fine,
                sensing_power_scale_by_uav=reference,
            )
            for method, result in results.items():
                heldout = [
                    evaluate_active_detection(
                        cfg, base, envelope_coarse, envelope_refined,
                        column.target, column.fusion, column.observations,
                        calibration_samples=args.calibration_samples,
                        evaluation_samples=args.validation_samples,
                        seed=experiment_seed + 1009 * trial + 0x51A7E,
                        transmitter_reference_scales=reference,
                    )
                    for column in result.columns
                ]
                row: dict[str, object] = {
                    "method": method,
                    "experiment_seed": experiment_seed,
                    "trial": trial,
                    "candidate_columns": result.candidate_column_count,
                    "exact_over_columns": result.exact_over_columns,
                    "power_externality_model": result.power_externality_model,
                    **result.objective,
                    "design_worst_pd": result.objective["worst_pd"],
                    "design_mean_pd": result.objective["mean_pd"],
                    "heldout_worst_pd": float(min(item.worst_pd for item in heldout)),
                    "heldout_mean_pd": float(np.mean([
                        item.worst_pd for item in heldout
                    ])),
                    "heldout_scenario_pfa_mean": float(np.mean([
                        value for item in heldout for value in item.scenario_pfa
                    ])),
                    "heldout_max_pfa": float(max(item.worst_pfa for item in heldout)),
                    "columns": json.dumps([
                        {
                            "target": column.target,
                            "fusion": column.fusion,
                            "robust_pd": column.robust_pd,
                            "robust_information": column.robust_information,
                            "bundle": [
                                [obs.link[0], obs.link[1], obs.mode.name]
                                for obs in column.observations
                            ],
                        }
                        for column in result.columns
                    ], separators=(",", ":")),
                }
                rows.append(row)
            completed = seed_index * args.trials + trial + 1
            print(f"active-system {completed:3d}/{total} columns={len(columns)}")
    return rows


def summarize(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    summary: dict[str, object] = {}
    for method in ("fixed_nominal_global", "active_modes_global"):
        group = [row for row in rows if row["method"] == method]
        summary[method] = {
            "runs": len(group),
            "heldout_worst_pd_mean": float(np.mean([
                float(row["heldout_worst_pd"]) for row in group
            ])),
            "heldout_mean_pd_mean": float(np.mean([
                float(row["heldout_mean_pd"]) for row in group
            ])),
            "heldout_max_pfa_mean": float(np.mean([
                float(row["heldout_max_pfa"]) for row in group
            ])),
            "heldout_scenario_pfa_mean": float(np.mean([
                float(row["heldout_scenario_pfa_mean"]) for row in group
            ])),
            "energy_mean": float(np.mean([float(row["total_energy"]) for row in group])),
            "cpu_cycles_mean": float(np.mean([float(row["cpu_cycles"]) for row in group])),
            "remote_reports_mean": float(np.mean([float(row["remote_reports"]) for row in group])),
        }
    fixed = [row for row in rows if row["method"] == "fixed_nominal_global"]
    active = [row for row in rows if row["method"] == "active_modes_global"]
    clusters = [f"{row['experiment_seed']}:{row['trial']}" for row in fixed]
    summary["paired_active_minus_fixed"] = {
        "worst_pd": paired_cluster_summary(
            [float(row["heldout_worst_pd"]) for row in active],
            [float(row["heldout_worst_pd"]) for row in fixed], clusters,
            bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        ),
        "mean_pd": paired_cluster_summary(
            [float(row["heldout_mean_pd"]) for row in active],
            [float(row["heldout_mean_pd"]) for row in fixed], clusters,
            bootstrap_samples=args.bootstrap_samples, seed=args.seed + 1,
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="small-uav-compact-800m")
    parser.add_argument("--uavs", type=int, default=3)
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20250915)
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--max-observations", type=int, default=3)
    parser.add_argument("--target-energy", type=float, default=48.0)
    parser.add_argument("--uav-energy", type=float, default=96.0)
    parser.add_argument("--tx-cap", type=int, default=4)
    parser.add_argument("--receiver-cap", type=int, default=4)
    parser.add_argument("--fusion-target-cap", type=int, default=2)
    parser.add_argument("--fusion-observation-cap", type=int, default=6)
    parser.add_argument("--report-cap", type=int, default=8)
    parser.add_argument("--total-observation-cap", type=int, default=6)
    parser.add_argument("--cpu-budget", type=float, default=220.0)
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument("--evaluation-samples", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=8192)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("results_active_system_v15"))
    args = parser.parse_args()
    cfg = configured(args)
    scientific = {key: value for key, value in vars(args).items() if key != "out"}
    manifest = {
        "artifact": "active_system_v15",
        "implementation_version": "1.5-system.2",
        "implementation_digest": implementation_digest(),
        "scope": "global active-column master with full-load interference envelope",
        "arguments": scientific,
        "config": asdict(cfg),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    run_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    run_dir = args.out / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest["run_id"] = run_id
    manifest_path = run_dir / "active_system_v15.config.json"
    rendered = json.dumps(manifest, indent=2, sort_keys=True)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"refusing to overwrite mismatched manifest: {manifest_path}")
    manifest_path.write_text(rendered, encoding="utf-8")
    rows = run(cfg, args)
    with (run_dir / "active_system_v15_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows, args)
    (run_dir / "active_system_v15_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"run directory: {run_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
