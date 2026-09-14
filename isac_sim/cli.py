"""Command-line interface.

The v10 prototype exposed ~30 flags, most of them a per-experiment
``--<mode>-csv`` / ``--<mode>-plot-prefix`` pair.  Here the whole surface is:

    --mode      which experiment to run
    --mc/--seed Monte-Carlo size and reproducibility
    --out       one output directory; paths inside follow a fixed convention
    --set       any configuration field, by dotted path, repeatable
    --config    a JSON file with the same dotted/nested keys
    --values    override a sweep grid, repeatable-free
    --axis      the robustness axis (only meaningful for --mode robustness)

Adding an experiment never adds a flag again.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import (
    PRESETS,
    Config,
    apply_overrides,
    apply_preset,
    default_config,
    iter_leaf_paths,
    validate_config,
)
from .experiments import EXPERIMENTS, EXPERIMENT_NAMES, VALUE_MODES, prior_sweep, robustness
from .plotting import PLOTTERS, plot_main_comparison
from .report import print_summary, write_csv, write_main_summary_latex, write_rows_csv
from .simulate import run_simulation
from .selection import METHODS

DESCRIPTIONS: Dict[str, str] = {
    "main": "single-run comparison of the proposed selector against all baselines",
    "lambda-sweep": "sweep the communication price lambda_c of the proposed selector",
    "ablation": "ablate each mechanism of the proposed selector",
    "fair-ablation": "ablate the mechanisms under a shared global link budget",
    "dd-ablation": "measure each OTFS delay-Doppler mechanism's contribution",
    "comm-sweep": "sweep the minimum communication rate R_min",
    "robustness": "sweep one robustness axis (use --axis)",
    "c2f": "compare coarse / C2F / full local refinement (paper Section 4)",
    "prior-sweep": "measure P_D degradation under target-state prior uncertainty",
    "waveform-check": "compare analytic eta^c / eta^loc against the OTFS PSF",
    "waveform-detection": "recompute empirical P_D/P_FA from noisy OTFS matched-filter outputs",
    "waveform-detection-grid": "stress-test calibrated P_D/P_FA over random fractional-DD offsets",
    "oracle-gap": "exact small-scale optimality gap of the greedy selector",
    "runtime": "per-selection wall-clock benchmark (greedy vs oracle vs baselines)",
    "belief-mismatch": "P_D degradation under tracker belief vs truth mismatch",
    "fbl-sweep": "finite-blocklength reliability-latency trade-off of the reporting links",
    "correlation-ablation": "correlation-aware vs independence-assuming fusion",
    "submodularity": "finite-instance diminishing-returns and curvature audit",
    "same-objective-gap": "greedy-vs-oracle gap on the identical task objective",
    "interference-consistency": "communication/sensing interference coupling and direct-path cancellation sweep",
}

ROBUSTNESS_AXES = ["comm_model", "error_sigma", "residual_direct", "direct_cancellation"]


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="isac_sim",
        description="Simplified Lagrangian-guided DOTFS-ISAC cooperative-sensing simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "  python -m isac_sim\n"
            "  python -m isac_sim --mc 200 --out runs/paper\n"
            "  python -m isac_sim --mode ablation --mc 100\n"
            "  python -m isac_sim --mode comm-sweep --values 2e5 1e6 2e6\n"
            "  python -m isac_sim --mode robustness --axis error_sigma\n"
            "  python -m isac_sim --set selector.lambda_c=0.02 --set radio.rho=0.7\n"
        ),
    )
    p.add_argument("--mode", choices=["main"] + EXPERIMENT_NAMES, default="main",
                   help="which experiment to run (default: main)")
    p.add_argument("--mc", "--num-mc", dest="mc", type=int, default=None,
                   help="number of Monte-Carlo trials")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--workers", type=int, default=None,
                   help="parallel Monte-Carlo worker processes (default: 1)")
    p.add_argument("--out", type=Path, default=Path("runs"),
                   help="root output directory (default: ./runs)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any configuration field by dotted path; repeatable")
    p.add_argument("--config", type=Path, default=None,
                   help="JSON file with configuration overrides (nested or dotted keys)")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="coherent named model bundle applied before --config/--set")
    p.add_argument("--values", type=float, nargs="+", default=None,
                   help="override the sweep grid of the selected experiment")
    p.add_argument("--axis", choices=ROBUSTNESS_AXES, default="comm_model",
                   help="robustness axis (only used with --mode robustness)")
    p.add_argument("--no-plots", action="store_true", help="skip figure generation")
    p.add_argument("--quiet", action="store_true", help="only print the final summary")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=None,
                   help="method subset for --mode main")
    p.add_argument("--paired-reference", choices=METHODS, default=None,
                   help="method used as the paired P_D contrast reference in --mode main")
    p.add_argument("--list-modes", action="store_true", help="list available modes and exit")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Configuration assembly
# --------------------------------------------------------------------------
def _flatten(node: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested JSON object into dotted ``key=value`` pairs."""
    out: Dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    else:
        out[prefix.rstrip(".")] = node
    return out


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        overrides[key.strip()] = value.strip()
    return overrides


def build_config(args: argparse.Namespace) -> Config:
    cfg = default_config()
    if args.preset is not None:
        cfg = apply_preset(cfg, args.preset)

    overrides: Dict[str, Any] = {}
    if args.config is not None:
        overrides.update(_flatten(json.loads(Path(args.config).read_text(encoding="utf-8"))))
    overrides.update(parse_overrides(args.overrides))

    if args.mc is not None:
        overrides["run.num_mc"] = args.mc
    if args.seed is not None:
        overrides["run.seed"] = args.seed
    if args.workers is not None:
        overrides["run.workers"] = args.workers
    if args.quiet:
        overrides["run.verbose"] = False

    cfg = apply_overrides(cfg, overrides) if overrides else cfg
    validate_config(cfg)
    return cfg


def diff_from_default(cfg: Config) -> List[Tuple[str, Any]]:
    """Return the non-default fields, so a run is self-documenting."""
    base = dict(iter_leaf_paths(default_config()))
    return [(path, value) for path, value in iter_leaf_paths(cfg) if base.get(path) != value]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_modes:
        print("Available modes:\n")
        for name in ["main"] + EXPERIMENT_NAMES:
            print(f"  {name:<15} {DESCRIPTIONS[name]}")
        return

    cfg = build_config(args)
    out_dir = Path(args.out) / args.mode
    fig_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"mode: {args.mode}  |  MC: {cfg.run.num_mc}  |  seed: {cfg.run.seed}")
    changed = diff_from_default(cfg)
    if changed:
        print("non-default settings: " + ", ".join(f"{k}={v}" for k, v in changed))
    print(f"output: {out_dir}")

    if args.mode == "main":
        trial_records: List[Dict[str, Any]] = []
        summary = run_simulation(
            cfg,
            methods=args.methods,
            paired_reference=args.paired_reference,
            trial_records=trial_records,
        )
        print_summary(summary)
        write_csv(summary, out_dir / "main.csv")
        write_rows_csv(trial_records, out_dir / "trials.csv")
        write_main_summary_latex(summary, out_dir / "main_summary.tex")
        if not args.no_plots:
            plot_main_comparison(summary, fig_dir)
        print(f"\nSaved: {out_dir / 'main.csv'}, {out_dir / 'trials.csv'}, "
              f"{out_dir / 'main_summary.tex'}"
              + ("" if args.no_plots else f", figures in {fig_dir}"))
        return

    runner = EXPERIMENTS[args.mode]
    if args.mode == "robustness":
        rows = robustness(cfg, args.axis, args.values)
    elif args.mode == "prior-sweep":
        # The prior grid is a list of dicts, not a flat float list, so the
        # ``--values`` flag is not the right input here.  Users wanting to
        # customise the grid should edit ``experiments.PRIOR_GRID``.
        rows = prior_sweep(cfg)
    elif args.values is not None and args.mode in VALUE_MODES:
        rows = runner(cfg, args.values)  # type: ignore[call-arg]
    elif args.values is not None:
        raise SystemExit(f"--values is not supported for --mode {args.mode}")
    else:
        rows = runner(cfg)

    write_rows_csv(rows, out_dir / f"{args.mode}.csv")
    print(f"\nCSV saved to: {out_dir / f'{args.mode}.csv'}")
    if not args.no_plots:
        PLOTTERS[args.mode](rows, fig_dir)
        print(f"Figures saved to: {fig_dir}")
