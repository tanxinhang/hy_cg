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
from isac_sim.coherent_oracle import (  # noqa: E402
    evaluate_coherent_tx_detection_oracle,
)
from isac_sim.scientific_gates import lower_tail_detection_summary  # noqa: E402
from isac_sim.active_system import (  # noqa: E402
    evaluate_active_transport_headroom,
    generate_active_columns,
    screen_fusion_candidates,
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
    overrides = {
        "scale.M": args.uavs,
        "scale.Q": args.targets,
        "prior.belief_mode": False,
        "dd.use_otfs_bin_validity": False,
        "active_sensing.enable": True,
        "active_sensing.energy_budget_per_target": args.target_energy,
        "active_sensing.energy_budget_per_uav": args.uav_energy,
        "active_sensing.max_candidates_per_pair": (
            args.pricing_candidate_limit
            if args.pricing_candidate_limit is not None
            else args.candidate_limit
        ),
        "active_sensing.candidate_strategy": args.candidate_strategy,
        "active_sensing.complete_pool_max_links": args.complete_pool_max_links,
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
    }
    if args.target_rcs is not None:
        overrides["detect.target_rcs"] = args.target_rcs
    if args.area_xy is not None:
        overrides["geometry.area_xy"] = args.area_xy
    if args.radar_net_gain_db is not None:
        overrides["radio.radar_net_gain_db"] = args.radar_net_gain_db
    if args.rescue_looks is not None:
        rescue_looks = args.rescue_looks
        rescue_power = args.rescue_power_scale
        if args.rescue_control == "looks_only":
            rescue_power = 1.0
        elif args.rescue_control == "power_only":
            rescue_looks = 16
        elif args.rescue_control == "baseline":
            rescue_power = 1.0
            rescue_looks = 16
        overrides["active_sensing.mode_names"] = (
            "eco", "nominal", "intensive", "rescue"
        )
        overrides["active_sensing.power_scales"] = (
            0.5, 1.0, 1.25, rescue_power
        )
        overrides["active_sensing.looks"] = (8, 16, 32, rescue_looks)
        overrides["active_sensing.refined"] = (False, True, True, True)
    cfg = apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def run(cfg: Config, args: argparse.Namespace) -> list[dict[str, object]]:
    modes = configured_sensing_modes(cfg)
    nominal = min(modes, key=lambda mode: (
        abs(mode.power_scale - 1.0), abs(mode.looks - 16), not mode.refined
    ))
    rows: list[dict[str, object]] = []
    holdout_angles = (
        tuple(float(value) for value in args.holdout_aspect_angles)
        if args.holdout_aspect_angles else None
    )
    total = args.seed_count * args.trials
    for seed_index in range(args.seed_count):
        experiment_seed = cfg.run.seed + 104729 * seed_index
        for trial in range(args.trials):
            rng = np.random.default_rng([experiment_seed, 15515, trial])
            geom = generate_geometry(cfg, rng)
            base = build_base_gains(cfg, geom, rng, rcs_view="mean")
            coarse = compute_link_tables(cfg, base)
            refined = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
            allowed_fusions = None
            fixed_allowed_fusions = None
            if args.fusion_candidate_limit is not None:
                allowed_fusions = screen_fusion_candidates(
                    cfg,
                    base,
                    coarse,
                    refined,
                    modes=modes,
                    information_limit=args.fusion_candidate_limit,
                )
                fixed_allowed_fusions = screen_fusion_candidates(
                    cfg,
                    base,
                    coarse,
                    refined,
                    modes=(nominal,),
                    information_limit=args.fusion_candidate_limit,
                )
            columns = generate_active_columns(
                cfg, base, coarse, refined, modes=modes,
                candidate_limit=args.candidate_limit,
                allowed_fusions=allowed_fusions,
                calibration_samples=args.calibration_samples,
                evaluation_samples=args.evaluation_samples,
                seed=experiment_seed + 1009 * trial,
                transport_mode=args.transport_mode,
            )
            fixed_columns = generate_active_columns(
                cfg, base, coarse, refined, modes=(nominal,),
                candidate_limit=args.candidate_limit,
                allowed_fusions=fixed_allowed_fusions,
                calibration_samples=args.calibration_samples,
                evaluation_samples=args.evaluation_samples,
                seed=experiment_seed + 1009 * trial,
                transport_mode=args.transport_mode,
            )
            results = {
                "fixed_nominal_global": solve_global_active_master(
                    cfg, fixed_columns, pd_design_target=args.design_pd_target
                ),
                "active_modes_global": solve_global_active_master(
                    cfg, columns, pd_design_target=args.design_pd_target
                ),
            }
            active_reference = np.full(
                cfg.scale.M, max(mode.power_scale for mode in modes), dtype=float
            )
            fixed_reference = np.full(
                cfg.scale.M, nominal.power_scale, dtype=float
            )
            active_envelope_coarse = compute_link_tables(
                cfg, base, sensing_power_scale_by_uav=active_reference
            )
            active_envelope_refined = compute_link_tables(
                cfg, base, dd_gain=base.eta_fine,
                sensing_power_scale_by_uav=active_reference,
            )
            fixed_envelope_coarse = compute_link_tables(
                cfg, base, sensing_power_scale_by_uav=fixed_reference
            )
            fixed_envelope_refined = compute_link_tables(
                cfg, base, dd_gain=base.eta_fine,
                sensing_power_scale_by_uav=fixed_reference,
            )
            for method, result in results.items():
                if method == "active_modes_global":
                    reference = active_reference
                    envelope_coarse = active_envelope_coarse
                    envelope_refined = active_envelope_refined
                else:
                    reference = fixed_reference
                    envelope_coarse = fixed_envelope_coarse
                    envelope_refined = fixed_envelope_refined
                headroom = None
                if args.lossless_report_headroom:
                    headroom = [
                        evaluate_active_transport_headroom(
                            cfg, base, envelope_coarse, envelope_refined,
                            column.target, column.fusion, column.observations,
                            reference,
                            calibration_samples=args.calibration_samples,
                            evaluation_samples=args.validation_samples,
                            seed=experiment_seed + 1009 * trial + 0x51A7E,
                            transport_mode=args.transport_mode,
                            aspect_angles_deg=holdout_angles,
                        )
                        for column in result.columns
                    ]
                    heldout = [item.current for item in headroom]
                    heldout_worst_pd = [item.robust_pd for item in heldout]
                    heldout_pfa = [item.scenario_pfa for item in heldout]
                else:
                    heldout = [
                        evaluate_active_detection(
                            cfg, base, envelope_coarse, envelope_refined,
                            column.target, column.fusion, column.observations,
                            calibration_samples=args.calibration_samples,
                            evaluation_samples=args.validation_samples,
                            seed=experiment_seed + 1009 * trial + 0x51A7E,
                            transmitter_reference_scales=reference,
                            transport_mode=args.transport_mode,
                            aspect_angles_deg=holdout_angles,
                        )
                        for column in result.columns
                    ]
                    heldout_worst_pd = [item.worst_pd for item in heldout]
                    heldout_pfa = [item.scenario_pfa for item in heldout]
                lossless_worst = (
                    [item.lossless_report.robust_pd for item in headroom]
                    if headroom is not None else []
                )
                coherent_heldout = (
                    [
                        evaluate_coherent_tx_detection_oracle(
                            cfg, base, envelope_coarse, envelope_refined,
                            column.target, column.fusion, column.observations,
                            calibration_samples=args.calibration_samples,
                            evaluation_samples=args.validation_samples,
                            seed=experiment_seed + 1009 * trial + 0xC0E2E17,
                            transmitter_reference_scales=reference,
                            phase_error_std_rad=np.deg2rad(
                                args.coherent_phase_error_deg
                            ),
                            transport_mode=args.transport_mode,
                            aspect_angles_deg=holdout_angles,
                        )
                        for column in result.columns
                    ]
                    if args.coherent_tx_oracle else []
                )
                coherent_worst = [item.worst_pd for item in coherent_heldout]
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
                    "target_rcs_m2": float(cfg.detect.target_rcs),
                    "fusion_candidate_limit": args.fusion_candidate_limit,
                    "transport_mode": args.transport_mode,
                    "geometry_role": args.geometry_role,
                    "design_aspect_angles_deg": json.dumps(
                        cfg.active_sensing.aspect_angles_deg
                    ),
                    "evaluation_aspect_angles_deg": json.dumps(
                        holdout_angles or cfg.active_sensing.aspect_angles_deg
                    ),
                    "heldout_worst_pd": float(min(heldout_worst_pd)),
                    "heldout_mean_pd": float(np.mean(heldout_worst_pd)),
                    "heldout_scenario_pfa_mean": float(np.mean([
                        value for values in heldout_pfa for value in values
                    ])),
                    "heldout_max_pfa": float(max(max(values) for values in heldout_pfa)),
                    "lossless_heldout_worst_pd": (
                        float(min(lossless_worst)) if lossless_worst else ""
                    ),
                    "lossless_heldout_mean_pd": (
                        float(np.mean(lossless_worst)) if lossless_worst else ""
                    ),
                    "lossless_worst_pd_headroom": (
                        float(min(lossless_worst) - min(heldout_worst_pd))
                        if lossless_worst else ""
                    ),
                    "coherent_tx_oracle": bool(args.coherent_tx_oracle),
                    "coherent_phase_error_deg": (
                        float(args.coherent_phase_error_deg)
                        if args.coherent_tx_oracle else ""
                    ),
                    "coherent_oracle_worst_pd": (
                        float(min(coherent_worst)) if coherent_worst else ""
                    ),
                    "coherent_oracle_mean_pd": (
                        float(np.mean(coherent_worst)) if coherent_worst else ""
                    ),
                    "coherent_oracle_worst_pd_gain": (
                        float(min(coherent_worst) - min(heldout_worst_pd))
                        if coherent_worst else ""
                    ),
                    "columns": json.dumps([
                        {
                            "target": column.target,
                            "fusion": column.fusion,
                            "robust_pd": column.robust_pd,
                            "robust_information": column.robust_information,
                            "generated_information": column.scenario_generated_information,
                            "network_evidence_loss": column.network_evidence_loss,
                            "robust_evidence_retention": column.robust_evidence_retention,
                            "receiver_cpu_cycles": column.receiver_cpu_cycles,
                            "fusion_cpu_cycles": column.fusion_cpu_cycles,
                            "local_aggregation_cpu_cycles": (
                                column.local_aggregation_cpu_cycles
                            ),
                            "fusion_inputs": column.fusion_inputs,
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
            "network_evidence_loss_mean": float(np.mean([
                float(row["network_evidence_loss"]) for row in group
            ])),
            "worst_evidence_retention_mean": float(np.mean([
                float(row["worst_evidence_retention"]) for row in group
            ])),
            "max_uav_cpu_cycles_mean": float(np.mean([
                float(row["max_uav_cpu_cycles"]) for row in group
            ])),
            "geometry_tail": lower_tail_detection_summary(
                [float(row["heldout_worst_pd"]) for row in group],
                args.geometry_quantile,
            ),
        }
        if args.lossless_report_headroom:
            summary[method]["lossless_heldout_worst_pd_mean"] = float(np.mean([
                float(row["lossless_heldout_worst_pd"]) for row in group
            ]))
            summary[method]["lossless_worst_pd_headroom_mean"] = float(np.mean([
                float(row["lossless_worst_pd_headroom"]) for row in group
            ]))
        if args.coherent_tx_oracle:
            summary[method]["coherent_oracle_worst_pd_mean"] = float(np.mean([
                float(row["coherent_oracle_worst_pd"]) for row in group
            ]))
            summary[method]["coherent_oracle_worst_pd_gain_mean"] = float(np.mean([
                float(row["coherent_oracle_worst_pd_gain"]) for row in group
            ]))
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
    parser.add_argument("--pricing-candidate-limit", type=int)
    parser.add_argument("--fusion-candidate-limit", type=int)
    parser.add_argument("--target-rcs", type=float)
    parser.add_argument("--radar-net-gain-db", type=float)
    parser.add_argument("--area-xy", type=float)
    parser.add_argument(
        "--geometry-role", choices=("development", "holdout"),
        default="development",
    )
    parser.add_argument("--geometry-quantile", type=float, default=0.05)
    parser.add_argument("--holdout-aspect-angles", nargs="*", type=float)
    parser.add_argument(
        "--candidate-strategy", choices=("scenario_union", "robust_singleton", "full"),
        default="scenario_union",
    )
    parser.add_argument("--complete-pool-max-links", type=int, default=12)
    parser.add_argument("--design-pd-target", type=float)
    parser.add_argument("--rescue-looks", type=int)
    parser.add_argument("--rescue-power-scale", type=float, default=1.25)
    parser.add_argument(
        "--rescue-control",
        choices=("combined", "looks_only", "power_only", "baseline"),
        default="combined",
    )
    parser.add_argument("--lossless-report-headroom", action="store_true")
    parser.add_argument(
        "--transport-mode",
        choices=("direct_llr", "receiver_local_llr"),
        default="direct_llr",
    )
    parser.add_argument("--coherent-tx-oracle", action="store_true")
    parser.add_argument("--coherent-phase-error-deg", type=float, default=0.0)
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
        "implementation_version": "mechanism-stable.1",
        "release_class": "v1.6-mechanism-stable-generalization-pending",
        "implementation_digest": implementation_digest(),
        "scope": (
            "global active-column master with full-load interference envelope, "
            "factorial resource controls, and separate design/holdout aspects"
        ),
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
