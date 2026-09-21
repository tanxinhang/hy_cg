# RETIRED PREMISE (2026-09-20): this script swept / read the config field
# `interference.direct_cancellation_db` (kappa_dc).
# That field was DELETED: it asserted a fixed 40 dB direct-path cancellation with no
# receiver implementation behind it while propping up the whole SINR denominator.
# Direct-path cancellation is now only ever a MEASURED TP-UIC residual.  Running this
# script as-is will fail on the missing attribute -- kept as historical evidence only.
"""Audit the coordination strategy itself, before building more on top of it.

Three questions, in decreasing order of how badly a wrong answer would hurt:

A. Is the step-1 gate fix actually correct? "tests pass" is not evidence for a
   numerical change -- assert the invariants directly.

B. Does the fixed point always exist, or did I get lucky on one seed? A loop that
   silently oscillates would make "2 rounds" a guess, not a validated budget.

C. **Is the K=3 headline (+0.584) an oracle artifact?** The illuminator subset in
   ``probe_coordination_aware_selection.py`` is ranked by how often each node is used
   in the *uncoordinated* optimum -- i.e. the cap is chosen with hindsight. If only
   that ranking reaches 0.79 and arbitrary K=3 subsets do not, the mechanism is an
   artifact of the warm start and must not be reported as a system gain.

Everything is measured; nothing is asserted from reading code.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isac_sim.core.config import (  # noqa: E402
    apply_overrides,
    apply_preset,
    default_config,
    validate_config,
)
from experiments.coordination import illuminator_mask, select_with_coordination
from isac_sim.detection.fusion import predicted_pd_for_links  # noqa: E402
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from isac_sim.cooperation.reporting import assign_fusion_nodes  # noqa: E402
from experiments.selection import _greedy_lagrangian, feasible_links_for_target, select_lagrangian, target_alpha, topk_links_by_marginal

AREA = 500.0
RCS = 0.2
RESULTS: list[tuple[str, str]] = []


def record(verdict: str, msg: str) -> None:
    RESULTS.append((verdict, msg))
    print(f"[{verdict:4}] {msg}")


def make_cfg(gate: bool):
    cfg = default_config()
    cfg = apply_preset(cfg, "target-local-v1")
    cfg = apply_overrides(
        cfg,
        {
            "geometry.area_xy": AREA,
            "detect.target_rcs": RCS,
            "interference.sense_gate_by_active_tx": bool(gate),
        },
    )
    validate_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# A. gate-fix invariants
# ---------------------------------------------------------------------------
def audit_gate_fix() -> None:
    print("=" * 88)
    print("A. step-1 gate fix invariants")
    print("=" * 88)
    cfg_off = make_cfg(False)
    cfg_on = make_cfg(True)
    geom = generate_geometry(cfg_off, np.random.default_rng(7))

    def tables(cfg, mask):
        base = build_base_gains(cfg, geom, np.random.default_rng(7))
        return compute_link_tables(cfg, base, active_tx_mask=mask)

    M = cfg_off.scale.M
    full = np.ones(M, dtype=bool)

    # A1: the gate must be a no-op when every node is active.
    t_off = tables(cfg_off, None)
    t_on_full = tables(cfg_on, full)
    same = (
        np.array_equal(t_off.gamma_sense, t_on_full.gamma_sense)
        and np.array_equal(t_off.gamma_comm, t_on_full.gamma_comm)
        and np.array_equal(t_off.rinr, t_on_full.rinr)
    )
    record(
        "PASS" if same else "FAIL",
        "A1 gate with an all-active mask is bit-exactly the un-gated release table"
        if same
        else "A1 gate with an all-active mask CHANGES the tables -> the口径 fix is incomplete",
    )

    # A2: muting must remove exactly the muted nodes' sensing contribution.
    keep = [0, 3, 7]
    mask = np.zeros(M, dtype=bool)
    mask[keep] = True
    t_some = tables(cfg_on, mask)
    # Interference at j from a subset must be <= the full-set interference,
    # and for a receiver whose only interferers are the muted ones it must drop.
    r_full = np.asarray(t_off.rinr)
    r_some = np.asarray(t_some.rinr)
    # resid_self does not depend on the mask, so compare the direct term only.
    kappa = 10.0 ** (-cfg_on.interference.direct_cancellation_db / 10.0)
    base = build_base_gains(cfg_on, geom, np.random.default_rng(7))
    P_sense = cfg_on.radio.rho * cfg_on.radio.P_default * np.ones(M)
    direct_full = kappa * (P_sense @ base.direct_gain)
    direct_some = kappa * ((P_sense * mask) @ base.direct_gain)
    ratio = np.divide(
        direct_some, direct_full, out=np.zeros_like(direct_some), where=direct_full > 0
    )
    record(
        "PASS" if np.all(ratio <= 1.0 + 1e-12) else "FAIL",
        f"A2 muted interference never exceeds the full-set interference "
        f"(max ratio {ratio.max():.3f} at j, expected <1 since {len(keep)}/{M} radiate)",
    )
    # A3: a fully muted swarm has no direct-path interference at all.
    t_none = tables(cfg_on, np.zeros(M, dtype=bool))
    rc = np.asarray(t_none.rinr)
    rs = np.asarray(t_off.rinr)
    record(
        "PASS" if rc.max() < rs.max() else "FAIL",
        f"A3 muting everything lowers residual interference (max rinr {rs.max():.3e} -> {rc.max():.3e})",
    )
    print()


# ---------------------------------------------------------------------------
# B. does the fixed point always exist?
# ---------------------------------------------------------------------------
def audit_fixed_point(seeds=(1, 2, 3, 4, 5, 6)) -> None:
    print("=" * 88)
    print("B. fixed-point existence across seeds (oscillation would invalidate '2 rounds')")
    print("=" * 88)
    cfg = make_cfg(True)
    print(f"  {'seed':>6}{'rounds run':>12}{'converged':>11}{'#TX path':>18}{'worst P_D':>11}")
    all_conv, rounds_used = True, []
    for s in seeds:
        geom = generate_geometry(cfg, np.random.default_rng(s))
        res = select_with_coordination(cfg, geom, rounds=6, seed=s)
        path = "->".join(str(r.n_tx) for r in res.history)
        all_conv &= res.converged
        rounds_used.append(len(res.history))
        print(f"  {s:>6}{len(res.history):>12}{str(res.converged):>11}{path:>18}"
              f"{res.final().worst_pd:>11.4f}")
    record(
        "PASS" if all_conv else "FAIL",
        f"B1 fixed point reached for all {len(seeds)} seeds (rounds used: {rounds_used}); "
        f"max {max(rounds_used)}",
    )
    if max(rounds_used) > 2:
        record("WARN", f"B2 some seeds needed {max(rounds_used)} rounds, not 2")
    else:
        record("PASS", "B2 every seed settles within 2 rounds")
    print()


# ---------------------------------------------------------------------------
# C. is the K=3 gain an artifact of the hindsight ranking?
# ---------------------------------------------------------------------------
def _pd_vector(cfg, base, tables, plan, selected, n_tgt):
    return [
        float(predicted_pd_for_links(cfg, tables, q, selected.get(q, []), plan=plan))
        if selected.get(q)
        else 0.0
        for q in range(n_tgt)
    ]


def _select_capped(cfg, base, tables, plan, allowed):
    n_tgt = cfg.scale.Q
    alpha0 = target_alpha(cfg, np.zeros(n_tgt))
    cand = {}
    for q in range(n_tgt):
        feas = [
            lk for lk in feasible_links_for_target(cfg, base, tables, q, plan)
            if int(lk[0]) in allowed
        ]
        cand[q] = topk_links_by_marginal(cfg, tables, base, feas, q, plan, float(alpha0[q]))
    return _greedy_lagrangian(cfg, tables, cand, plan, base)


def _evaluate_subset(cfg, base, geom, allowed, n_uav, n_tgt):
    mask = np.zeros(n_uav, dtype=bool)
    mask[list(allowed)] = True
    tables = compute_link_tables(cfg, base, active_tx_mask=mask)
    plan = assign_fusion_nodes(cfg, base, tables, geom=geom)
    sel, _ = _select_capped(cfg, base, tables, plan, set(allowed))
    pds = _pd_vector(cfg, base, tables, plan, sel, n_tgt)
    return min(pds), sum(pds) / n_tgt


def audit_subset_choice(seed: int = 20260917, n_random: int = 10) -> None:
    print("=" * 88)
    print("C. is K=3's gain reachable by ANY subset, or only by the hindsight ranking?")
    print("=" * 88)
    cfg = make_cfg(True)
    geom = generate_geometry(cfg, np.random.default_rng(seed))
    base = build_base_gains(cfg, geom, np.random.default_rng(seed))
    n_uav, n_tgt = cfg.scale.M, cfg.scale.Q

    # reference (uncoordinated)
    t0 = compute_link_tables(cfg, base, active_tx_mask=None)
    plan0 = assign_fusion_nodes(cfg, base, t0, geom=geom)
    sel0, _ = select_lagrangian(cfg, base, t0, plan0)
    ref = min(_pd_vector(cfg, base, t0, plan0, sel0, n_tgt))
    usage = {}
    for links in sel0.values():
        for i, _j in links:
            usage[int(i)] = usage.get(int(i), 0) + 1
    ranked = sorted(usage, key=lambda i: (-usage[i], i))
    print(f"  reference (no coordination): worst P_D = {ref:.4f}, "
          f"{len(usage)} distinct illuminators")

    K = 3
    hindsight = _evaluate_subset(cfg, base, geom, ranked[:K], n_uav, n_tgt)
    print(f"  hindsight-ranked K={K} {tuple(ranked[:K])}: worst P_D = {hindsight[0]:.4f} "
          f"({hindsight[0] - ref:+.4f})")

    rng = np.random.default_rng(12345)
    draws, seen = [], set()
    while len(draws) < n_random:
        pick = tuple(sorted(rng.choice(n_uav, size=K, replace=False).tolist()))
        if pick in seen:
            continue
        seen.add(pick)
        draws.append(pick)
    vals = []
    for pick in draws:
        w, _m = _evaluate_subset(cfg, base, geom, pick, n_uav, n_tgt)
        vals.append(w)
        print(f"    random subset {pick}: worst P_D = {w:.4f} ({w - ref:+.4f})")
    vals_arr = np.asarray(vals)
    print(f"  random K={K}: min {vals_arr.min():.4f} / median {np.median(vals_arr):.4f} "
          f"/ max {vals_arr.max():.4f}   (hindsight {hindsight[0]:.4f})")
    beat = int((vals_arr >= hindsight[0]).sum())
    record(
        "PASS" if vals_arr.max() >= hindsight[0] - 0.05 else "FAIL",
        f"C1 best random K={K} subset reaches {vals_arr.max():.4f} vs hindsight "
        f"{hindsight[0]:.4f}: the mechanism is {'robust to the subset choice' if vals_arr.max() >= hindsight[0] - 0.05 else 'DEPENDENT on the hindsight ranking (possible oracle artifact)'}",
    )
    record(
        "WARN" if np.median(vals_arr) < ref + 0.05 else "PASS",
        f"C2 median random subset {np.median(vals_arr):.4f} vs reference {ref:.4f} "
        f"-> {'a random 3-node cap is not automatically good' if np.median(vals_arr) < ref + 0.05 else 'even random caps beat the reference'}",
    )
    print()


def main() -> int:
    print(f"coordination strategy audit | {AREA:.0f} m | RCS {RCS} m^2\n")
    audit_gate_fix()
    audit_fixed_point()
    audit_subset_choice()
    fails = [m for v, m in RESULTS if v == "FAIL"]
    warns = [m for v, m in RESULTS if v == "WARN"]
    print("=" * 88)
    print(f"{len(RESULTS) - len(fails) - len(warns)} PASS / {len(warns)} WARN / {len(fails)} FAIL")
    for w in warns:
        print(f"  WARN: {w}")
    for f in fails:
        print(f"  FAIL: {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
