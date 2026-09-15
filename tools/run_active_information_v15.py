#!/usr/bin/env python3
"""Run the V1.5 active-information pricing comparison.

This experiment isolates the new observation-design layer.  Both methods use
the same geometry, fusion assignment, aspect scenarios, reporting reliability,
link cap, and normalized energy budget.  ``fixed_nominal`` may only use the
nominal acquisition mode; ``active_modes`` jointly chooses links and modes.
The reported metric is received KL information, not a replacement for the
end-to-end detection probability used by the frozen V1.4 main experiment.
"""

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
    SensingMode,
    configured_sensing_modes,
    evaluate_active_detection,
    price_active_information_bundle,
)
from isac_sim.active_statistics import paired_cluster_summary  # noqa: E402
from isac_sim.config import (  # noqa: E402
    Config,
    apply_overrides,
    apply_preset,
    validate_config,
)
from isac_sim.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from isac_sim.reporting import assign_fusion_nodes  # noqa: E402


def implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "isac_sim").glob("*.py")) + [Path(__file__).resolve()]:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def configured(args: argparse.Namespace) -> Config:
    cfg = apply_preset(Config(), args.preset)
    cfg = apply_overrides(cfg, {
        "active_sensing.enable": True,
        "active_sensing.energy_budget_per_target": args.energy_budget,
        "active_sensing.max_candidates_per_pair": args.shortlist,
        "active_sensing.branch_node_limit": args.node_limit,
        "active_sensing.information_metric": args.metric,
        "active_sensing.aspect_floor": args.aspect_floor,
        "detect.comm_error_model": "erasure",
        "fusion.mode": "explicit",
        "fusion.rule": "capacitated_value",
        "fusion.max_targets_per_uav": -1,
        "selector.max_links_per_target": args.max_observations,
        "selector.max_remote_reports": -1,
        "run.seed": args.seed,
        "run.verbose": False,
    })
    validate_config(cfg)
    return cfg


def _row(method: str, experiment_seed: int, trial: int, result: object,
         detection: object) -> dict[str, object]:
    observations = getattr(result, "observations")
    return {
        "method": method,
        "experiment_seed": experiment_seed,
        "trial": trial,
        "target": getattr(result, "target"),
        "fusion": getattr(result, "fusion"),
        "robust_information": getattr(result, "robust_information"),
        "mean_scenario_information": float(
            np.mean(getattr(result, "scenario_information"))
        ),
        "objective": getattr(result, "objective"),
        "energy": getattr(result, "energy"),
        "observations": len(observations),
        "remote_reports": getattr(result, "remote_reports"),
        "explored_nodes": getattr(result, "explored_nodes"),
        "exact": getattr(result, "exact"),
        "upper_bound": getattr(result, "upper_bound"),
        "certificate_gap": getattr(result, "certificate_gap"),
        "worst_pd": getattr(detection, "worst_pd"),
        "worst_pfa": getattr(detection, "worst_pfa"),
        "mean_pfa": float(np.mean(getattr(detection, "scenario_pfa"))),
        "mean_pd": float(np.mean(getattr(detection, "scenario_pd"))),
        "scenario_pd": json.dumps(
            getattr(detection, "scenario_pd"), separators=(",", ":")
        ),
        "scenario_pfa": json.dumps(
            getattr(detection, "scenario_pfa"), separators=(",", ":")
        ),
        "scenario_information": json.dumps(
            getattr(result, "scenario_information"), separators=(",", ":")
        ),
        "bundle": json.dumps([
            {
                "tx": obs.link[0],
                "rx": obs.link[1],
                "mode": obs.mode.name,
                "power_scale": obs.mode.power_scale,
                "looks": obs.mode.looks,
                "refined": obs.mode.refined,
            }
            for obs in observations
        ], separators=(",", ":")),
    }


def run(cfg: Config, trials: int, seed_count: int,
        calibration_samples: int, evaluation_samples: int) -> list[dict[str, object]]:
    modes = configured_sensing_modes(cfg)
    nominal = min(
        modes,
        key=lambda mode: (
            abs(mode.power_scale - 1.0),
            abs(mode.looks - 16),
            mode.refined is False,
        ),
    )
    mode_families = {
        "fixed_nominal": (nominal,),
        "power_only": tuple(
            SensingMode(f"power_{mode.power_scale:g}", mode.power_scale,
                        nominal.looks, nominal.refined)
            for mode in modes
        ),
        "looks_only": tuple(
            SensingMode(f"looks_{mode.looks}", nominal.power_scale,
                        mode.looks, nominal.refined)
            for mode in modes
        ),
        "refinement_only": (
            SensingMode("coarse", nominal.power_scale, nominal.looks, False),
            SensingMode("refined", nominal.power_scale, nominal.looks, True),
        ),
        "active_modes": modes,
    }
    rows: list[dict[str, object]] = []
    for seed_index in range(seed_count):
        experiment_seed = int(cfg.run.seed + 104729 * seed_index)
        for trial in range(trials):
            rng = np.random.default_rng([experiment_seed, 15150, trial])
            geom = generate_geometry(cfg, rng)
            base = build_base_gains(cfg, geom, rng, rcs_view="mean")
            coarse = compute_link_tables(cfg, base)
            refined = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
            fusion_plan = assign_fusion_nodes(cfg, base, coarse, geom)
            for q in range(cfg.scale.Q):
                fusion = int(fusion_plan.f_q[q])
                active = price_active_information_bundle(
                    cfg, base, coarse, refined, q, fusion, modes=modes
                )
                results = {"active_modes": active}
                for method, family in mode_families.items():
                    if method == "active_modes":
                        continue
                    results[method] = price_active_information_bundle(
                        cfg, base, coarse, refined, q, fusion, modes=family,
                        candidate_links=active.candidate_links,
                    )
                detector_seed = experiment_seed + 1009 * trial
                for method in mode_families:
                    result = results[method]
                    detection = evaluate_active_detection(
                        cfg, base, coarse, refined, q, fusion,
                        result.observations,
                        calibration_samples=calibration_samples,
                        evaluation_samples=evaluation_samples,
                        seed=detector_seed,
                    )
                    rows.append(_row(
                        method, experiment_seed, trial, result, detection
                    ))
            completed = seed_index * trials + trial + 1
            print(f"active-information {completed:3d}/{seed_count * trials}")
    return rows


def summarize(rows: list[dict[str, object]], *, epsilon: float,
              weak_threshold: float, bootstrap_samples: int,
              seed: int) -> dict[str, object]:
    summary: dict[str, object] = {}
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        summary[method] = {
            "target_instances": len(group),
            "robust_information_mean": float(np.mean([
                float(row["robust_information"]) for row in group
            ])),
            "robust_information_median": float(np.median([
                float(row["robust_information"]) for row in group
            ])),
            "energy_mean": float(np.mean([float(row["energy"]) for row in group])),
            "observations_mean": float(np.mean([
                float(row["observations"]) for row in group
            ])),
            "exact_rate": float(np.mean([bool(row["exact"]) for row in group])),
            "certificate_gap_max": float(max(
                float(row["certificate_gap"]) for row in group
            )),
            "worst_pd_mean": float(np.mean([
                float(row["worst_pd"]) for row in group
            ])),
            "worst_pfa_mean": float(np.mean([
                float(row["worst_pfa"]) for row in group
            ])),
            "scenario_pfa_mean": float(np.mean([
                float(row["mean_pfa"]) for row in group
            ])),
        }
    keyed = {
        (row["method"], row["experiment_seed"], row["trial"], row["target"]): row
        for row in rows
    }
    comparisons: dict[str, object] = {}
    fixed_rows = [row for row in rows if row["method"] == "fixed_nominal"]
    clusters = [f"{row['experiment_seed']}:{row['trial']}" for row in fixed_rows]
    for method in methods:
        if method == "fixed_nominal":
            continue
        paired = [keyed[(method, row["experiment_seed"], row["trial"], row["target"])]
                  for row in fixed_rows]
        information = paired_cluster_summary(
            [float(row["robust_information"]) for row in paired],
            [float(row["robust_information"]) for row in fixed_rows],
            clusters, epsilon=epsilon, weak_threshold=weak_threshold,
            bootstrap_samples=bootstrap_samples, seed=seed,
        )
        detection = paired_cluster_summary(
            [float(row["worst_pd"]) for row in paired],
            [float(row["worst_pd"]) for row in fixed_rows],
            clusters, epsilon=epsilon, weak_threshold=1.0,
            bootstrap_samples=bootstrap_samples, seed=seed + 1,
        )
        comparisons[method] = {
            "information": information,
            "worst_pd": detection,
        }
    summary["paired_vs_fixed_nominal"] = comparisons
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="small-uav-compact-800m")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20250915)
    parser.add_argument("--energy-budget", type=float, default=64.0)
    parser.add_argument("--max-observations", type=int, default=4)
    parser.add_argument("--shortlist", type=int, default=6)
    parser.add_argument("--node-limit", type=int, default=200_000)
    parser.add_argument("--metric", choices=("forward_kl", "jeffreys"),
                        default="forward_kl")
    parser.add_argument("--aspect-floor", type=float, default=0.20)
    parser.add_argument("--calibration-samples", type=int, default=2048)
    parser.add_argument("--evaluation-samples", type=int, default=4096)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--gain-epsilon", type=float, default=1e-6)
    parser.add_argument("--weak-information-threshold", type=float, default=1.0)
    parser.add_argument("--out", type=Path,
                        default=Path("results_active_information_v15"))
    args = parser.parse_args()
    if args.trials <= 0 or args.seed_count <= 0:
        parser.error("--trials and --seed-count must be positive")

    cfg = configured(args)
    scientific_arguments = {
        key: value for key, value in vars(args).items() if key != "out"
    }
    manifest = {
        "artifact": "active_information_v15",
        "implementation_version": "1.5-system.2",
        "implementation_digest": implementation_digest(),
        "scope": "pair-level information pricing and mixed-mode P_D consequence",
        "arguments": scientific_arguments,
        "config": asdict(cfg),
    }
    canonical_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    run_id = hashlib.sha256(canonical_manifest.encode("utf-8")).hexdigest()[:12]
    run_dir = args.out / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest["run_id"] = run_id
    manifest_path = run_dir / "active_information_v15.config.json"
    rendered_manifest = json.dumps(manifest, indent=2, sort_keys=True)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != rendered_manifest:
        raise RuntimeError(f"refusing to overwrite mismatched manifest: {manifest_path}")
    manifest_path.write_text(rendered_manifest, encoding="utf-8")

    rows = run(
        cfg, args.trials, args.seed_count,
        args.calibration_samples, args.evaluation_samples,
    )
    with (run_dir / "active_information_v15_targets.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(
        rows, epsilon=args.gain_epsilon,
        weak_threshold=args.weak_information_threshold,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    (run_dir / "active_information_v15_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"run directory: {run_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
