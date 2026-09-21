#!/usr/bin/env python3
"""Run the V1 overview sweeps without modifying the frozen main preset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from experiments.app.report import scalar_summary_row, write_rows_csv  # noqa: E402
from experiments.flow.simulate import run_simulation  # noqa: E402


METHOD = "proposed_c2f_adaptive_pd"

SWEEPS: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = {
    "fusion-rule": [
        ("max_in_rate", {"fusion.rule": "max_in_rate"}, {"fusion_rule": "max_in_rate"}),
        ("max_min_rate", {"fusion.rule": "max_min_rate"}, {"fusion_rule": "max_min_rate"}),
        ("nearest_target", {"fusion.rule": "nearest_target"}, {"fusion_rule": "nearest_target"}),
    ],
    "communication": [
        (f"Rmin_{r/1e6:g}Mbps", {"comm.R_min": r}, {"R_min_mbps": r / 1e6})
        for r in (0.1e6, 0.2e6, 0.5e6, 1.0e6, 2.0e6)
    ],
    "geometry": [
        (f"area_{area:g}m", {"geometry.area_xy": area}, {"area_xy_m": area})
        for area in (2500.0, 4000.0, 5500.0)
    ],
    "lambda": [
        (f"lambda_{lam:g}", {"selector.lambda_c": lam}, {"lambda_c": lam})
        for lam in (0.0, 0.0025, 0.005, 0.01, 0.02)
    ],
    "blocklength": [
        (f"n_{n}", {"comm.n_block": n}, {"n_block": n})
        for n in (1024, 1536, 2048, 3072)
    ],
    "scale": [
        (f"M_{m}_Q_10", {"scale.M": m, "scale.Q": 10}, {"M": m, "Q": 10})
        for m in (10, 15, 20)
    ],
}


def run_sweep(
    name: str,
    entries: list[tuple[str, dict[str, Any], dict[str, Any]]],
    base: Config,
    out_root: Path,
) -> None:
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    configs: dict[str, Any] = {}
    for index, (label, overrides, axes) in enumerate(entries, start=1):
        cfg = apply_overrides(base, overrides)
        print(f"[{name} {index}/{len(entries)}] {label}", flush=True)
        started = time.perf_counter()
        summary = run_simulation(cfg, methods=[METHOD])
        elapsed = time.perf_counter() - started
        row = scalar_summary_row(
            summary,
            METHOD,
            {
                "experiment": f"target_local_v1_{name.replace('-', '_')}",
                "condition": label,
                **axes,
                "wall_time_s": elapsed,
                "seconds_per_trial": elapsed / cfg.run.num_mc,
            },
        )
        rows.append(row)
        configs[label] = asdict(cfg)
        # Checkpoint after every condition so a long suite can be resumed or
        # inspected even if a later condition fails.
        write_rows_csv(rows, out_dir / f"{name}.csv")
        (out_dir / "configs.json").write_text(
            json.dumps(configs, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"  P_D={summary[METHOD]['P_D']:.4f}, "
            f"reports={summary[METHOD]['selected_links_mean']:.2f}, "
            f"delay={summary[METHOD]['T_mean_ms']:.3f} ms, "
            f"wall={elapsed:.1f} s",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results_target_local_v1" / "overview"
    )
    parser.add_argument("--only", nargs="+", choices=sorted(SWEEPS), default=None)
    args = parser.parse_args()
    if args.mc < 1:
        parser.error("--mc must be at least one")
    if args.workers < 1:
        parser.error("--workers must be at least one")

    base = apply_preset(Config(), "target-local-v1")
    base = apply_overrides(
        base,
        {
            "run.num_mc": args.mc,
            "run.seed": args.seed,
            "run.workers": args.workers,
            "run.verbose": False,
        },
    )
    selected = args.only if args.only is not None else list(SWEEPS)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preset": "target-local-v1",
        "method": METHOD,
        "mc_per_condition": args.mc,
        "seed": args.seed,
        "workers": args.workers,
        "sweeps": selected,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    for name in selected:
        run_sweep(name, SWEEPS[name], base, args.out)
    print(f"V1 overview sweeps saved under {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
