#!/usr/bin/env python3
"""Preregistered RCS sensitivity scan for V1.2 without headline selection."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isac_sim.config import Config, apply_overrides, apply_preset, validate_config  # noqa: E402
from isac_sim.report import scalar_summary_row, write_rows_csv  # noqa: E402
from isac_sim.simulate import run_simulation  # noqa: E402


METHODS = [
    "joint_bundle_cg",
    "fixed_fusion_bundle",
    "proposed_c2f_adaptive_pd",
    "local_only_bundle",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rcs", type=float, nargs="+",
                        default=[0.025, 0.05, 0.1, 0.2])
    parser.add_argument("--mc", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results_joint_bundle_v12_rcs")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    configs: dict[str, object] = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for index, rcs in enumerate(args.rcs, start=1):
        cfg = apply_preset(Config(), "small-uav-compact-800m")
        cfg = apply_overrides(cfg, {
            "detect.target_rcs": float(rcs),
            "detect.comm_error_model": "erasure",
            "selector.use_delay_price": False,
            "selector.lambda_c": 0.0,
            "selector.require_local_anchor": False,
            "selector.max_local_observations_per_target": 1,
            "selector.max_remote_reports": 2,
            "selector.max_observations_per_receiver": 2,
            "selector.max_observations_per_fusion_uav": 2,
            "fusion.max_targets_per_uav": 1,
            "fusion.rule": "capacitated_value",
            "run.num_mc": args.mc,
            "run.seed": args.seed,
            "run.workers": args.workers,
            "run.verbose": False,
        })
        validate_config(cfg)
        print(f"[{index}/{len(args.rcs)}] RCS={rcs:g} m^2", flush=True)
        started = time.perf_counter()
        summary = run_simulation(
            cfg, methods=METHODS, paired_reference="joint_bundle_cg"
        )
        elapsed = time.perf_counter() - started
        for method in METHODS:
            rows.append(scalar_summary_row(summary, method, {
                "experiment": "joint_bundle_v1.2_rcs_sensitivity",
                "target_rcs_m2": rcs,
                "rcs_dbsm": 10.0 * math.log10(rcs),
                "mc": args.mc,
                "seed": args.seed,
                "wall_time_s": elapsed,
            }))
        configs[f"rcs_{rcs:g}"] = asdict(cfg)
        write_rows_csv(rows, args.out / "rcs_sensitivity.csv")
        (args.out / "configs.json").write_text(
            json.dumps(configs, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"  joint={summary['joint_bundle_cg']['P_D']:.3f}, "
            f"local={summary['local_only_bundle']['P_D']:.3f}, "
            f"fixed={summary['fixed_fusion_bundle']['P_D']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
