#!/usr/bin/env python3
"""Run the frozen target-local V1 contract tests and evaluation suite.

This runner never writes to ``results/`` or ``results_release/`` by default.
It fixes the V1 preset, method roster, paired reference, and seed so that a
candidate result cannot accidentally be presented under the paper-canonical
protocol.

The archived label erasure at source 69f3300 denoted Gaussian replacement.
Pin the current semantic name here; current true erasure is a different model.

Suites (``--suite``)
--------------------
``main``              MC=1000 three-method head-to-head -> ``<out>/main``
``full-refinement``   MC=200 adaptive-C2F vs full refinement -> ``<out>/full-refinement-pd``
``overview``          MC=100 six diagnostic sweeps -> ``<out>/overview``
``prediction-stress`` MC=100 tracker-mismatch grid -> ``<out>/prediction-stress``
``all``               every suite above, i.e. the complete V1 release tree

Only ``main`` feeds ``figs/fig2_v1_main_comparison.pdf`` and the headline table;
``overview`` feeds ``figs/fig3_v1_fusion_ablation.pdf`` and the supplement
operating-envelope figure.  Both consume detection counts, so both must be
regenerated whenever the detector's sampling path changes.

Scenario re-anchoring
---------------------
``--set`` forwards extra dotted-path overrides to ``run_isac_sim.py`` for the
``main`` and ``full-refinement`` suites.  It exists so the same frozen preset can
be evaluated under a *different declared scenario* - e.g. the 500-800 m /
RCS 0.05-0.2 m^2 band in ``LOW_RCS_SCENARIO_500_800.md`` - without editing the
preset, which is part of the release identity.  With no ``--set`` the command
lines are unchanged and therefore bit-exact against the archived V1 tree.
``overview`` and ``prediction-stress`` drive their own scripts and reject
``--set`` rather than ignoring it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "run_isac_sim.py"
METHODS = [
    "proposed_c2f_adaptive_pd",
    "exact_marginal_greedy",
    "sense_sinr",
]
FULL_REFINEMENT_METHODS = [
    "proposed_c2f_adaptive_pd",
    "proposed_c2f_full_pd",
]
SUITES = ("main", "full-refinement", "overview", "prediction-stress")


def checked_run(cmd: list[str]) -> None:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def assert_preset_semantics() -> None:
    """Guard the frozen detector model before spending compute on a suite.

    ``target-local-v1`` deliberately does not restate ``detect.comm_error_model``
    in its preset table; it inherits the dataclass default.  The archived V1
    configs record the historical label ``erasure``, which at source 69f3300
    meant Gaussian replacement.  If the dataclass default ever moves, every V1
    number silently changes physics instead of drifting by sampling noise.
    Fail loudly here rather than 40 minutes later.
    """
    sys.path.insert(0, str(ROOT))
    from isac_sim.config import Config, apply_preset  # noqa: E402

    cfg = apply_preset(Config(), "target-local-v1")
    expected = "gaussian_replacement"
    if cfg.detect.comm_error_model != expected:
        raise SystemExit(
            f"target-local-v1 resolves detect.comm_error_model="
            f"{cfg.detect.comm_error_model!r}, expected {expected!r}. "
            "The archived V1 tree was produced under Gaussian replacement; "
            "fix the preset before rerunning."
        )
    print(
        f"[guard] target-local-v1: comm_error_model={expected}, "
        f"interference_model={cfg.comm.interference_model}, "
        f"rho={cfg.radio.rho}, target_rcs={cfg.detect.target_rcs}",
        flush=True,
    )


def run_main(out: Path, mc: int, seed: int, workers: int, plots: bool,
             extra: list[str]) -> None:
    cmd = [
        sys.executable,
        str(RUN),
        "--mode",
        "main",
        "--preset",
        "target-local-v1",
        "--set",
        "detect.comm_error_model=gaussian_replacement",
        "--mc",
        str(mc),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--methods",
        *METHODS,
        "--paired-reference",
        "proposed_c2f_adaptive_pd",
        "--out",
        str(out),
        "--quiet",
    ]
    for item in extra:
        cmd += ["--set", item]
    if not plots:
        cmd.append("--no-plots")
    checked_run(cmd)


def normalize_flat_mode_dir(root: Path, mode: str) -> None:
    """Collapse ``<root>/<mode>/{main.csv,...}`` into ``<root>/{main.csv,...}``.

    ``run_isac_sim.py`` always nests its output one level under the requested
    ``--out``.  The archived ``full-refinement-pd`` directory predates that
    convention and stores ``main.csv`` at its top level, so flatten it back to
    keep this release tree layout-compatible with its archive and with
    ``SUBMISSION_TRACEABILITY.md``.
    """
    nested = root / mode
    if not nested.is_dir():
        return
    for item in sorted(nested.iterdir()):
        if item.name == "figs":
            continue
        target = root / item.name
        if target.exists():
            target.unlink()
        item.replace(target)
    figs = nested / "figs"
    if figs.is_dir() and not any(figs.iterdir()):
        figs.rmdir()
    try:
        nested.rmdir()
    except OSError:
        pass


def run_full_refinement(out: Path, mc: int, seed: int, workers: int,
                        plots: bool, extra: list[str]) -> None:
    cmd = [
        sys.executable,
        str(RUN),
        "--mode",
        "main",
        "--preset",
        "target-local-v1",
        "--set",
        "detect.comm_error_model=gaussian_replacement",
        "--mc",
        str(mc),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--methods",
        *FULL_REFINEMENT_METHODS,
        "--paired-reference",
        "proposed_c2f_adaptive_pd",
        "--out",
        str(out),
        "--quiet",
    ]
    for item in extra:
        cmd += ["--set", item]
    if not plots:
        cmd.append("--no-plots")
    checked_run(cmd)


def run_overview(out: Path, mc: int, seed: int, workers: int) -> None:
    checked_run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_target_local_v1_overview.py"),
            "--mc",
            str(mc),
            "--seed",
            str(seed),
            "--workers",
            str(workers),
            "--out",
            str(out / "overview"),
        ]
    )


def run_prediction_stress(out: Path, mc: int, seed: int, workers: int) -> None:
    checked_run(
        [
            sys.executable,
            str(ROOT / "tools" / "audit_target_local_v1_prediction.py"),
            "--mc",
            str(mc),
            "--seed",
            str(seed),
            "--workers",
            str(workers),
            "--out",
            str(out / "prediction-stress"),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES + ("all",), default="main",
                        help="which part of the V1 release tree to regenerate")
    parser.add_argument("--mc", type=int, default=1000,
                        help="Monte-Carlo trials for --suite main")
    parser.add_argument("--full-refinement-mc", type=int, default=200)
    parser.add_argument("--overview-mc", type=int, default=100)
    parser.add_argument("--prediction-mc", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE",
        help="scenario override forwarded to run_isac_sim.py (main and "
             "full-refinement suites only); repeatable",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results_target_local_v1"
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    for name in ("mc", "full_refinement_mc", "overview_mc", "prediction_mc"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least one")
    if args.workers < 1:
        parser.error("--workers must be at least one")

    suites = SUITES if args.suite == "all" else (args.suite,)
    plots = not args.no_plots
    if args.overrides and set(suites) - {"main", "full-refinement"}:
        parser.error("--set only applies to the main and full-refinement "
                     "suites; overview and prediction-stress drive their own "
                     "scripts and would silently ignore it")

    if not args.skip_tests:
        checked_run(
            [
                sys.executable,
                "-m",
                "unittest",
                "tests.test_target_local_v1",
                "tests.test_canonical_consistency",
            ]
        )

    assert_preset_semantics()

    if "main" in suites:
        run_main(args.out, args.mc, args.seed, args.workers, plots,
                 args.overrides)
    if "full-refinement" in suites:
        fr_out = args.out / "full-refinement-pd"
        run_full_refinement(fr_out, args.full_refinement_mc, args.seed,
                            args.workers, plots, args.overrides)
        normalize_flat_mode_dir(fr_out, "main")
    if "overview" in suites:
        run_overview(args.out, args.overview_mc, args.seed, args.workers)
    if "prediction-stress" in suites:
        run_prediction_stress(args.out, args.prediction_mc, args.seed,
                              args.workers)

    print(f"Target-local V1 suite {args.suite!r} saved under {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
