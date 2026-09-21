"""Why simple coordination only buys +0.026, and what would fix it.

Measured facts (500 m / RCS 0.2, multi-target selection):
  * single link in isolation, only its illuminator radiating -> worst P_D 0.953 (+0.78)
  * real selection, union of all illuminators           -> 10 of 15 nodes must radiate
                                                           -> worst P_D 0.195 (+0.026)
  * round-robin time-sharing                            -> net NEGATIVE

So the blocker is not physics, it is the OBJECTIVE: ``select_lagrangian`` maximises
each target's detection independently and never asks how many distinct nodes it is
waking up. Every new illuminator re-introduces its own direct-path leakage.

This probe measures the frontier that the objective should be navigating: cap the
number of simultaneously radiating nodes at K, restrict the candidate pool to links
illuminated by those K nodes, and see what detection is still achievable.
If a small K retains most of the performance, a sparsity-aware selection objective
is worth building; if not, the coordination route is capped and should be reported
as such.
"""

from __future__ import annotations

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
from isac_sim.detection.fusion import predicted_pd_for_links  # noqa: E402
from isac_sim.sensing.model import compute_link_tables, build_base_gains, generate_geometry  # noqa: E402
from isac_sim.cooperation.reporting import assign_fusion_nodes  # noqa: E402
from experiments.selection import _greedy_lagrangian, feasible_links_for_target, select_lagrangian, target_alpha, topk_links_by_marginal

AREA = 500.0
RCS = 0.2
PD_REQ = 0.95
SEED = 20260917


def make_cfg():
    cfg = default_config()
    cfg = apply_preset(cfg, "target-local-v1")
    cfg = apply_overrides(
        cfg,
        {
            "geometry.area_xy": AREA,
            "detect.target_rcs": RCS,
            "interference.sense_gate_by_active_tx": True,
        },
    )
    validate_config(cfg)
    return cfg


def worst_pd(cfg, tables, selected, plan, n_tgt):
    pds = [
        float(predicted_pd_for_links(cfg, tables, q, selected.get(q, []), plan=plan))
        if selected.get(q)
        else 0.0
        for q in range(n_tgt)
    ]
    return min(pds), sum(pds) / n_tgt, pds


def select_with_illuminator_cap(cfg, base, tables, plan, allowed: set[int]):
    """Standard greedy, but only links illuminated by ``allowed`` may enter."""
    n_tgt = cfg.scale.Q
    alpha0 = target_alpha(cfg, np.zeros(n_tgt))
    candidates = {}
    for q in range(n_tgt):
        feas = [
            lk for lk in feasible_links_for_target(cfg, base, tables, q, plan)
            if int(lk[0]) in allowed
        ]
        candidates[q] = topk_links_by_marginal(
            cfg, tables, base, feas, q, plan, float(alpha0[q])
        )
    return _greedy_lagrangian(cfg, tables, candidates, plan, base)


def main() -> int:
    cfg = make_cfg()
    geom = generate_geometry(cfg, np.random.default_rng(SEED))
    base = build_base_gains(cfg, geom, np.random.default_rng(SEED))
    n_uav, n_tgt = cfg.scale.M, cfg.scale.Q

    # ---- un-coordinated reference ----------------------------------------
    t0 = compute_link_tables(cfg, base, active_tx_mask=None)
    plan0 = assign_fusion_nodes(cfg, base, t0, geom=geom)
    sel0, _ = select_lagrangian(cfg, base, t0, plan0)
    w0, m0, _ = worst_pd(cfg, t0, sel0, plan0, n_tgt)
    used = sorted({int(i) for links in sel0.values() for (i, _j) in links})
    use_count = {i: 0 for i in used}
    for links in sel0.values():
        for i, _j in links:
            use_count[int(i)] = use_count.get(int(i), 0) + 1
    # rank illuminators by how many selected links they serve
    ranked = sorted(used, key=lambda i: (-use_count.get(i, 0), i))

    print("=" * 92)
    print(f"illuminator-cap frontier | {AREA:.0f} m | RCS {RCS} m^2")
    print("=" * 92)
    print(f"un-coordinated reference: worst P_D = {w0:.4f}, mean = {m0:.4f}, "
          f"{len(used)} distinct illuminators used by {sum(len(v) for v in sel0.values())} links")
    print(f"illuminator usage (node: #links): "
          f"{ {i: use_count.get(i, 0) for i in ranked} }")
    print()
    print(f"  {'K':>4}{'#TX':>6}{'#links':>8}{'worst P_D':>11}{'mean P_D':>10}{'sat':>6}"
          f"{'vs ref':>10}")

    rows = []
    for K in (2, 3, 4, 5, 6, 8, 10, 12, 15):
        allowed = set(ranked[:K]) if K < len(ranked) else set(range(n_uav))
        mask = np.zeros(n_uav, dtype=bool)
        mask[list(allowed)] = True
        tables = compute_link_tables(cfg, base, active_tx_mask=mask)
        plan = assign_fusion_nodes(cfg, base, tables, geom=geom)
        sel, _ = select_with_illuminator_cap(cfg, base, tables, plan, allowed)
        w, m, pds = worst_pd(cfg, tables, sel, plan, n_tgt)
        n_sat = int(sum(p >= PD_REQ for p in pds))
        n_links = sum(len(v) for v in sel.values())
        rows.append((K, int(mask.sum()), n_links, w, m))
        print(f"  {K:>4}{int(mask.sum()):>6}{n_links:>8}{w:>11.4f}{m:>10.4f}{n_sat:>6}"
              f"{w - w0:>+10.4f}")

    best = max(rows, key=lambda r: r[3])
    print()
    print(f"best cap: K={best[0]} -> worst P_D {best[3]:.4f} ({best[3] - w0:+.4f} vs reference), "
          f"using {best[1]} radiators")
    print("read: if a small K keeps most of the detection, a sparsity-aware objective")
    print("      can convert that into a real system gain; if the frontier collapses")
    print("      immediately, the coordination route is capped by link availability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
