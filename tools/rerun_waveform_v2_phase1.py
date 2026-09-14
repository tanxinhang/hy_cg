#!/usr/bin/env python3
"""Run the isolated phase-1 waveform-calibration candidate against frozen V1.

Outputs are written under ``results_waveform_v2_phase1`` by default.  The
runner never writes into the V1 evidence tree or manuscript figure directory.
Waveform-impairment magnitudes must be supplied explicitly; their defaults are
zero because no physical nominal values have yet been validated.
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
    "proposed_c2f_adaptive_pd_calibrated",
]


def checked_run(cmd: list[str]) -> None:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results_waveform_v2_phase1"
    )
    parser.add_argument("--clutter-inr", type=float, default=0.0)
    parser.add_argument("--multipath-inr", type=float, default=0.0)
    parser.add_argument("--unresolved-target-inr", type=float, default=0.0)
    parser.add_argument("--sync-delay-bins", type=float, default=0.0)
    parser.add_argument("--sync-doppler-bins", type=float, default=0.0)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--record-runtime", action="store_true")
    args = parser.parse_args()

    if args.mc < 1:
        parser.error("--mc must be at least one")
    if args.workers < 1:
        parser.error("--workers must be at least one")
    for name in ("clutter_inr", "multipath_inr", "unresolved_target_inr"):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")

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

    settings = {
        "waveform_impairments.clutter_inr": args.clutter_inr,
        "waveform_impairments.multipath_inr": args.multipath_inr,
        "waveform_impairments.unresolved_target_inr": args.unresolved_target_inr,
        "waveform_impairments.sync_delay_bins": args.sync_delay_bins,
        "waveform_impairments.sync_doppler_bins": args.sync_doppler_bins,
    }
    cmd = [
        sys.executable,
        str(RUN),
        "--mode",
        "main",
        "--preset",
        "target-local-waveform-v2-phase1",
        "--mc",
        str(args.mc),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--methods",
        *METHODS,
        "--paired-reference",
        "proposed_c2f_adaptive_pd",
        "--out",
        str(args.out),
        "--quiet",
    ]
    for key, value in settings.items():
        cmd.extend(("--set", f"{key}={value}"))
    if args.record_runtime:
        cmd.extend(("--set", "run.record_runtime=true"))
    if args.no_plots:
        cmd.append("--no-plots")
    checked_run(cmd)
    print(f"Waveform phase-1 evaluation saved under {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
