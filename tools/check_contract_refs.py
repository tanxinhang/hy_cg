#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify that every code reference in the three-layer contract still resolves.

The contract's traceability claim ("every manuscript equation points at exactly
one place in the code") is only true while the line numbers are true.  They are
not self-maintaining: a 30-line insertion in ``model.py`` silently invalidates
every reference below it, and nothing in the test suite notices.

This tool makes the claim checkable.  Each entry says *what* should be found in
a range -- a symbol name, not a line number -- so a drifted reference fails
loudly instead of pointing at an unrelated line.

Usage
-----
    python tools/check_contract_refs.py            # verify, exit 1 on any miss
    python tools/check_contract_refs.py --list     # print the table it checks
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Ref:
    """One contract reference: a file, a line span, and the tokens it must hold."""

    where: str  # the equation / row label, for the report
    path: str
    start: int
    end: int
    tokens: tuple[str, ...]


# --------------------------------------------------------------------------
# Theory layer (V1_STABLE_RELEASE.md section 2.1)
# --------------------------------------------------------------------------
THEORY: tuple[Ref, ...] = (
    Ref("eq:comm_sinr_rate", "isac_sim/model.py", 805, 830, ("gamma_comm", "rate[i, j]")),
    Ref("eq:fbl_reliability", "isac_sim/fbl.py", 120, 160, ("def chi_from_gamma",)),
    Ref("eq:bistatic_delay_doppler", "isac_sim/model.py", 350, 410, ("delay_bin", "doppler_bin")),
    Ref("eq:local_dd_energy", "isac_sim/dd.py", 1, 287, ("eta_local", "eta_fine_array")),
    Ref("eq:sensing_sinr", "isac_sim/model.py", 830, 879, ("gamma_sense",)),
    Ref("eq:local_llr", "isac_sim/llr.py", 80, 115, ("def llr_delta", "def llr_var1")),
    Ref("eq:received_moments", "isac_sim/soft_channel.py", 30, 95, ("def _mix", "received_moments")),
    Ref("eq:detector_prediction", "isac_sim/fusion.py", 180, 260, ("def predicted_pd_for_links",)),
    Ref("eq:nearest_target_fusion", "isac_sim/reporting.py", 130, 260, ("def assign_fusion_nodes",)),
    Ref("eq:feasible_candidate_set", "isac_sim/selection.py", 150, 210, ("def feasible_links_for_target",)),
    Ref("eq:fair_sensing_utility", "isac_sim/fusion.py", 420, 500, ("def selection_utility",)),
    Ref("eq:task_objective", "isac_sim/selection.py", 150, 175, ("link_cost_ms",)),
    Ref("eq:c2f_complexity", "isac_sim/selection.py", 650, 700, ("def select_c2f_adaptive",)),
    Ref("detector threshold (shared)", "isac_sim/fusion.py", 300, 380, ("def calibrated_fused_threshold",)),
    Ref("detector threshold call site", "isac_sim/simulate.py", 200, 220, ("calibrated_fused_threshold",)),
)

# --------------------------------------------------------------------------
# Model layer (section 2.2)
# --------------------------------------------------------------------------
MODEL: tuple[Ref, ...] = (
    Ref("coupling=shared_spectrum", "isac_sim/model.py", 625, 675, ("shared_spectrum",)),
    Ref("residual direct/multi", "isac_sim/model.py", 875, 900, ("residual_direct", "residual_multi")),
    Ref("direct-path cancellation depth", "isac_sim/model.py", 760, 800, ("kappa_dc", "direct_cancellation_db")),
    Ref("denominator_guard", "isac_sim/model.py", 55, 75, ("def denominator_guard",)),
    Ref("radar_hardware_gain", "isac_sim/model.py", 38, 58, ("def radar_hardware_gain",)),
    Ref("threshold_from_pfa", "isac_sim/model.py", 105, 125, ("def threshold_from_pfa",)),
)

# --------------------------------------------------------------------------
# Algorithm layer (section 2.3), including the coordination addition of rev. 2
# --------------------------------------------------------------------------
ALGORITHM: tuple[Ref, ...] = (
    Ref("fusion placement", "isac_sim/reporting.py", 130, 260, ("def assign_fusion_nodes",)),
    Ref("greedy_lagrangian", "isac_sim/selection.py", 329, 350, ("def _greedy_lagrangian",)),
    Ref("c2f selector", "isac_sim/selection.py", 650, 700, ("def select_c2f_adaptive",)),
    Ref("coordination: illuminator_mask", "isac_sim/coordination.py", 50, 90, ("def illuminator_mask",)),
    Ref("coordination: mask covers schedule", "isac_sim/coordination.py", 85, 120, ("def require_mask_covers_schedule",)),
    Ref("coordination: fixed point", "isac_sim/coordination.py", 165, 260, ("def select_with_coordination",)),
    Ref("coordination: config section", "isac_sim/config.py", 706, 745, ("class Coordination",)),
    Ref("coordination: release-path fixed point", "isac_sim/simulate.py", 615, 640, ("COORDINATION_C2F_METHODS",)),
    Ref("coordination: reported diagnostics", "isac_sim/simulate.py", 1500, 1545, ("coordination_rounds_mean",)),
    Ref("coordination: eval tables under own mask", "isac_sim/simulate.py", 705, 735, ("def _coordination_eval_tables",)),
    Ref("coordination: mask gating in link tables", "isac_sim/model.py", 690, 715, ("gate_echo", "active_tx_mask")),
)

GROUPS = (("theory", THEORY), ("model", MODEL), ("algorithm", ALGORITHM))


def _check(ref: Ref) -> list[str]:
    path = ROOT / ref.path
    if not path.exists():
        return [f"{ref.where}: file missing: {ref.path}"]
    lines = path.read_text(encoding="utf-8").splitlines()
    if ref.start < 1 or ref.end > len(lines):
        return [
            f"{ref.where}: {ref.path}:{ref.start}-{ref.end} out of range "
            f"(file has {len(lines)} lines)"
        ]
    window = "\n".join(lines[ref.start - 1: ref.end])
    missing = [t for t in ref.tokens if t not in window]
    if missing:
        # Say *where the tokens went*.  Ranges are part of the contract (a
        # symbol present somewhere else in the file is not the contract), so the
        # checker must not silently re-anchor -- but making the next person
        # bisect the file by hand for a number the checker already computed is
        # gratuitous.  Measured on the 2026-09-19 edit: a one-line insertion in
        # ``compute_link_tables`` moved three refs at once.
        moved = {
            t: [i + 1 for i, line in enumerate(lines) if t in line][:4]
            for t in missing
        }
        return [
            f"{ref.where}: {ref.path}:{ref.start}-{ref.end} does not contain "
            f"{missing} -- the reference has drifted. Now at {moved}"
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the checked table")
    args = parser.parse_args(argv)

    if args.list:
        for name, group in GROUPS:
            print(f"== {name}")
            for ref in group:
                print(f"   {ref.path}:{ref.start}-{ref.end}  {ref.where}")
        return 0

    problems: list[str] = []
    total = 0
    for name, group in GROUPS:
        misses = 0
        for ref in group:
            total += 1
            found = _check(ref)
            problems.extend(found)
            misses += len(found)
        print(f"{name:10s}: {len(group) - misses}/{len(group)} references resolve")

    for problem in problems:
        print(f"MISS {problem}")
    print(f"RESULT: {'CLEAN' if not problems else f'{len(problems)} DRIFTED'} "
          f"({total} references checked)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
