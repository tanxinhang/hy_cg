#!/usr/bin/env python3
"""Run the V1.4 exact-LLR detector ablation and fusion-headroom diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isac_sim.config import Config, apply_overrides, apply_preset, validate_config  # noqa: E402
from isac_sim.fusion_headroom import fusion_headroom_diagnostic  # noqa: E402
from isac_sim.model import build_base_gains, compute_link_tables, generate_geometry  # noqa: E402
from isac_sim.reporting import assign_fusion_nodes  # noqa: E402
from isac_sim.report import scalar_summary_row, write_rows_csv  # noqa: E402
from isac_sim.simulate import run_simulation  # noqa: E402


def _write_artifact_manifest(
    out: Path,
    artifact: str,
    cfg: Config,
    parameters: dict[str, object] | None = None,
) -> None:
    """Write one immutable-by-name configuration record per result artifact."""
    payload = {
        "artifact": artifact,
        "parameters": parameters or {},
        "config": asdict(cfg),
    }
    (out / f"{artifact}.config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def configured(args: argparse.Namespace) -> Config:
    cfg = apply_preset(Config(), args.preset)
    cfg = apply_overrides(cfg, {
        "detect.comm_error_model": "erasure",
        "detect.fused_calibration_samples": args.calibration_samples,
        "fusion.mode": "explicit",
        "fusion.rule": "capacitated_value",
        "fusion.max_targets_per_uav": -1,
        "fusion.cpu_rate_cycles_per_s": args.cpu_rate,
        "fusion.processing_window_s": args.processing_window_ms * 1e-3,
        "fusion.cpu_fixed_cycles": args.cpu_c0,
        "fusion.cpu_per_observation_cycles": args.cpu_c1,
        "fusion.cpu_cubic_cycles": args.cpu_c2,
        "selector.max_links_per_target": args.k_safe,
        "selector.max_remote_reports": -1,
        "selector.max_observations_per_fusion_uav": -1,
        "selector.bundle_shortlist_per_type": args.shortlist_per_type,
        "selector.use_delay_price": False,
        "selector.lambda_c": 0.0,
        "run.num_mc": args.mc,
        "run.seed": args.seed,
        "run.workers": args.workers,
        "run.verbose": False,
    })
    validate_config(cfg)
    return cfg


def run_cpu_dynamic_main(cfg: Config, out: Path) -> None:
    methods = [
        "rcs_robust_bundle_cg",
        "joint_bundle_cg",
        "fixed_fusion_bundle",
        "local_only_bundle",
    ]
    _write_artifact_manifest(
        out, "cpu_dynamic_main", cfg,
        {"methods": methods, "paired_reference": "rcs_robust_bundle_cg"},
    )
    summary = run_simulation(
        cfg, methods=methods, paired_reference="rcs_robust_bundle_cg"
    )
    rows = [
        scalar_summary_row(summary, method, {
            "experiment": "v1.4_rcs_robust_cpu_dynamic_bundle",
            "cpu_budget_cycles": (
                cfg.fusion.cpu_rate_cycles_per_s * cfg.fusion.processing_window_s
            ),
        })
        for method in methods
    ]
    write_rows_csv(rows, out / "cpu_dynamic_main.csv")
    for method in methods:
        item = summary[method]
        print(
            f"{method:28s} PD={item['P_D']:.4f} PFA={item['P_FA']:.4f} "
            f"obs={item['selected_observations_mean']:.2f} "
            f"cpu_u={item['cpu_capacity_max_utilization_mean']:.3f}"
        )


def run_detector_ablation(cfg: Config, detector_k: int, out: Path) -> None:
    # Hold the selected observation count fixed while isolating the detector.
    cfg = apply_overrides(cfg, {
        "selector.max_links_per_target": detector_k,
        "selector.max_total_links": detector_k * cfg.scale.Q,
    })
    validate_config(cfg)
    methods = ["joint_bundle_cg", "joint_bundle_cg_exact_llr"]
    _write_artifact_manifest(
        out, "detector_ablation", cfg,
        {"detector_k": detector_k, "methods": methods},
    )
    summary = run_simulation(
        cfg, methods=methods, paired_reference="joint_bundle_cg"
    )
    rows = [
        scalar_summary_row(summary, method, {
            "experiment": "v1.4_exact_llr_detector_ablation",
            "selection_policy": "shared_joint_bundle_cg",
            "detector_k": detector_k,
        })
        for method in methods
    ]
    write_rows_csv(rows, out / "detector_ablation.csv")
    for method in methods:
        item = summary[method]
        print(
            f"{method:29s} PD={item['P_D']:.4f} PFA={item['P_FA']:.4f} "
            f"paired={item.get('paired_reference_delta_P_D', float('nan')):+.4f}"
        )


def run_headroom(cfg: Config, args: argparse.Namespace, out: Path) -> None:
    _write_artifact_manifest(out, "fusion_headroom", cfg, {
        "headroom_trials": args.headroom_trials,
        "k_safe": args.k_safe,
        "shortlist_per_type": args.shortlist_per_type,
    })
    rows: list[dict[str, object]] = []
    for trial in range(args.headroom_trials):
        rng = np.random.default_rng([cfg.run.seed, 14141, trial])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng, rcs_view="mean")
        coarse_tables = compute_link_tables(cfg, base)
        # Match the main simulation: the fixed fusion assignment only sees the
        # coarse scheduler view, while detector headroom is evaluated with the
        # configured refined receiver statistics.
        fixed = assign_fusion_nodes(cfg, base, coarse_tables, geom)
        tables = (
            compute_link_tables(cfg, base, dd_gain=base.eta_fine)
            if cfg.refine.enable or cfg.refine.apply_to_all
            else coarse_tables
        )
        diagnostics = fusion_headroom_diagnostic(
            cfg,
            base,
            tables,
            fixed,
            k_safe=args.k_safe,
            shortlist_per_type=args.shortlist_per_type,
        )
        for item in diagnostics:
            row = asdict(item)
            row.update({
                "trial": trial,
                "g_observation": item.observation_headroom,
                "g_communication": item.communication_headroom,
                "g_fusion_location": item.fusion_location_headroom,
                "k_safe": args.k_safe,
                "shortlist_per_type": args.shortlist_per_type,
            })
            for key in ("links_local", "links_k2", "links_ksafe"):
                row[key] = json.dumps(row[key])
            rows.append(row)
        print(f"headroom {trial + 1:3d}/{args.headroom_trials}")

    with (out / "fusion_headroom_targets.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        "pd_local", "pd_best_k2", "pd_best_ksafe", "pd_perfect_comm_ksafe",
        "pd_fixed_k2", "g_observation", "g_communication", "g_fusion_location",
    ]
    summary = {
        key: {
            "mean": float(np.mean([float(row[key]) for row in rows])),
            "median": float(np.median([float(row[key]) for row in rows])),
            "p10": float(np.quantile([float(row[key]) for row in rows], 0.10)),
            "p90": float(np.quantile([float(row[key]) for row in rows], 0.90)),
        }
        for key in metrics
    }
    (out / "fusion_headroom_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_correlation_sensitivity(
    cfg: Config, args: argparse.Namespace, out: Path
) -> None:
    levels = [float(value) for value in args.corr_levels.split(",")]
    if not levels or any(value < 0.0 or value > 1.0 for value in levels):
        raise ValueError("correlation levels must lie in [0,1]")
    method = "rcs_robust_bundle_cg"
    rows: list[dict[str, object]] = []
    trial_pd: dict[float, np.ndarray] = {}
    effective_configs = _correlation_configs(cfg, levels, args.corr_mc)
    _write_correlation_manifest(out, cfg, method, levels, effective_configs)
    for level, corr_cfg in effective_configs:
        records: list[dict[str, object]] = []
        summary = run_simulation(
            corr_cfg, methods=[method], trial_records=records
        )
        rows.append(scalar_summary_row(summary, method, {
            "experiment": "v1.4_correlation_sensitivity",
            "rho_total": level,
            "rho_tx": corr_cfg.corr.rho_tx,
            "rho_rx": corr_cfg.corr.rho_rx,
            "rho_target": corr_cfg.corr.rho_target,
            "rho_dd": corr_cfg.corr.rho_dd,
        }))
        ordered = sorted(records, key=lambda row: int(row["trial"]))
        trial_pd[level] = np.asarray(
            [float(row["detected_fraction"]) for row in ordered], dtype=float
        )
        item = summary[method]
        print(
            f"rho={level:.2f} PD={item['P_D']:.4f} "
            f"PFA={item['P_FA']:.4f} obs={item['selected_observations_mean']:.2f}"
        )

    reference = levels[0]
    for row, level in zip(rows, levels):
        diff = trial_pd[level] - trial_pd[reference]
        half = (
            1.96 * float(np.std(diff, ddof=1)) / np.sqrt(diff.size)
            if diff.size > 1 else 0.0
        )
        row["paired_delta_vs_first_rho"] = float(np.mean(diff))
        row["paired_delta_vs_first_rho_ci95_low"] = float(np.mean(diff) - half)
        row["paired_delta_vs_first_rho_ci95_high"] = float(np.mean(diff) + half)
    write_rows_csv(rows, out / "correlation_sensitivity.csv")


def _correlation_configs(
    cfg: Config, levels: list[float], corr_mc: int
) -> list[tuple[float, Config]]:
    effective_configs: list[tuple[float, Config]] = []
    for level in levels:
        # Preserve the declared structural composition while varying total
        # correlation mass: tx/rx/target/DD = 0.3/0.3/0.2/0.2.
        corr_cfg = apply_overrides(cfg, {
            "corr.enable": True,
            "corr.rho_tx": 0.30 * level,
            "corr.rho_rx": 0.30 * level,
            "corr.rho_target": 0.20 * level,
            "corr.rho_dd": 0.20 * level,
            "run.num_mc": corr_mc,
        })
        validate_config(corr_cfg)
        effective_configs.append((level, corr_cfg))
    return effective_configs


def _write_correlation_manifest(
    out: Path,
    cfg: Config,
    method: str,
    levels: list[float],
    effective_configs: list[tuple[float, Config]],
) -> None:
    _write_artifact_manifest(out, "correlation_sensitivity", cfg, {
        "method": method,
        "levels": levels,
        "effective_configs": [
            {"rho_total": level, "config": asdict(corr_cfg)}
            for level, corr_cfg in effective_configs
        ],
    })


def write_requested_manifests(
    cfg: Config, args: argparse.Namespace, out: Path
) -> None:
    """Materialize artifact-specific manifests without rerunning simulations."""
    if args.mode in {"main", "all"}:
        methods = [
            "rcs_robust_bundle_cg", "joint_bundle_cg",
            "fixed_fusion_bundle", "local_only_bundle",
        ]
        _write_artifact_manifest(
            out, "cpu_dynamic_main", cfg,
            {"methods": methods, "paired_reference": "rcs_robust_bundle_cg"},
        )
    if args.mode in {"detector", "all"}:
        detector_cfg = apply_overrides(cfg, {
            "selector.max_links_per_target": args.detector_k,
            "selector.max_total_links": args.detector_k * cfg.scale.Q,
        })
        validate_config(detector_cfg)
        _write_artifact_manifest(out, "detector_ablation", detector_cfg, {
            "detector_k": args.detector_k,
            "methods": ["joint_bundle_cg", "joint_bundle_cg_exact_llr"],
        })
    if args.mode in {"headroom", "all"}:
        _write_artifact_manifest(out, "fusion_headroom", cfg, {
            "headroom_trials": args.headroom_trials,
            "k_safe": args.k_safe,
            "shortlist_per_type": args.shortlist_per_type,
        })
    if args.mode in {"correlation", "all"}:
        levels = [float(value) for value in args.corr_levels.split(",")]
        if not levels or any(value < 0.0 or value > 1.0 for value in levels):
            raise ValueError("correlation levels must lie in [0,1]")
        effective = _correlation_configs(cfg, levels, args.corr_mc)
        _write_correlation_manifest(
            out, cfg, "rcs_robust_bundle_cg", levels, effective
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="small-uav-compact-800m")
    parser.add_argument(
        "--mode",
        choices=("main", "detector", "headroom", "correlation", "all"),
        default="all",
    )
    parser.add_argument("--mc", type=int, default=20)
    parser.add_argument("--headroom-trials", type=int, default=3)
    parser.add_argument("--k-safe", type=int, default=6)
    parser.add_argument("--detector-k", type=int, default=2)
    parser.add_argument("--shortlist-per-type", type=int, default=3)
    parser.add_argument("--calibration-samples", type=int, default=2048)
    parser.add_argument("--cpu-rate", type=float, default=1e9)
    parser.add_argument("--processing-window-ms", type=float, default=20.0)
    parser.add_argument("--cpu-c0", type=float, default=1e6)
    parser.add_argument("--cpu-c1", type=float, default=2e6)
    parser.add_argument("--cpu-c2", type=float, default=0.0)
    parser.add_argument("--corr-mc", type=int, default=20)
    parser.add_argument("--corr-levels", default="0,0.3,0.6")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "results_fusion_headroom_v14")
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="write artifact-specific effective configuration files and exit",
    )
    args = parser.parse_args()
    cfg = configured(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.manifest_only:
        write_requested_manifests(cfg, args, args.out)
        return 0
    if args.mode in {"main", "all"}:
        run_cpu_dynamic_main(cfg, args.out)
    if args.mode in {"detector", "all"}:
        run_detector_ablation(cfg, args.detector_k, args.out)
    if args.mode in {"headroom", "all"}:
        run_headroom(cfg, args, args.out)
    if args.mode in {"correlation", "all"}:
        run_correlation_sensitivity(cfg, args, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
