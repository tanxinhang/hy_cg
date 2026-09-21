"""Does feeding the TX mask back into selection actually pay off?

Locked scenario: 500 m / RCS 0.2 m^2.

The single-link probe (``probe_coordination_gain.py``) showed that silencing the
non-participating nodes is worth +0.78 P_D on the worst target. But that measured
**one** link in isolation. In the real multi-target selection the mask is the union
of the illuminators of *all* selected links, which may cover most of the swarm --
in which case coordination buys nothing. This probe answers that, and it is the
question that decides whether the route is worth implementing for real.

Also reports how many rounds the fixed point needs, using the only sound criterion:
keep giving it rounds and see whether the answer still changes.
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
from experiments.coordination import select_with_coordination
from isac_sim.sensing.model import generate_geometry  # noqa: E402

AREA = 500.0
RCS = 0.2
PD_REQ = 0.95


def make_cfg(coordinated: bool):
    cfg = default_config()
    cfg = apply_preset(cfg, "target-local-v1")
    cfg = apply_overrides(
        cfg,
        {
            "geometry.area_xy": AREA,
            "detect.target_rcs": RCS,
            "interference.sense_gate_by_active_tx": bool(coordinated),
        },
    )
    validate_config(cfg)
    return cfg


def report(tag: str, res) -> None:
    print(f"--- {tag} ---")
    print(f"  {'round':>6}{'#TX':>6}{'#links':>8}{'worst P_D':>11}{'mean P_D':>10}{'sat':>6}  note")
    base = res.baseline()
    for rec in res.history:
        n_sat = int(sum(p >= PD_REQ for p in rec.per_target_pd))
        note = []
        if rec.round_index == 0:
            note.append("uncoordinated baseline")
        if rec.fixed_point:
            note.append("FIXED POINT")
        print(f"  {rec.round_index:>6}{rec.n_tx:>6}{rec.n_links:>8}"
              f"{rec.worst_pd:>11.4f}{rec.mean_pd:>10.4f}{n_sat:>6}  {'; '.join(note)}")
    fin = res.final()
    print(f"  => worst P_D {base.worst_pd:.4f} -> {fin.worst_pd:.4f} "
          f"({fin.worst_pd - base.worst_pd:+.4f})")
    print(f"  => radiators {base.n_tx} -> {fin.n_tx} of {len(base.mask)}")
    print()


def grouped_tradeoff(cfg_on, geom, groups_list=(1, 2, 5, 10)) -> None:
    """The real question: can the isolated +0.78 be reached in a multi-target system?

    In one CPI with every target observed simultaneously, the mask is the union of
    *all* selected illuminators, so little can be silenced. To get a sparse mask you
    must sense targets in *separate time slots* -- which costs looks, and looks is
    itself a strong lever. This function measures that trade directly.

    The link choice is held fixed at the un-coordinated selection so that only the
    interference and the look budget vary (stated so the ablation stays honest).
    """
    from isac_sim.detection.fusion import predicted_pd_for_links
    from experiments.coordination import illuminator_mask
    from isac_sim.sensing.model import compute_link_tables, build_base_gains
    from isac_sim.cooperation.reporting import assign_fusion_nodes
    from experiments.selection import select_lagrangian

    n_uav = cfg_on.scale.M
    n_tgt = cfg_on.scale.Q
    base = build_base_gains(cfg_on, geom, np.random.default_rng(20260917))

    # baseline selection, un-coordinated, at the full look budget
    t0 = compute_link_tables(cfg_on, base, active_tx_mask=None)
    plan0 = assign_fusion_nodes(cfg_on, base, t0, geom=geom)
    selected, _ = select_lagrangian(cfg_on, base, t0, plan0)
    pd_base = [
        float(predicted_pd_for_links(cfg_on, t0, q, selected.get(q, []), plan=plan0))
        if selected.get(q) else 0.0
        for q in range(n_tgt)
    ]

    print("=" * 92)
    print("grouped coordination trade-off (link choice frozen; only interference + looks vary)")
    print("=" * 92)
    print(f"  baseline: all {n_uav} radiators, L=16, worst P_D = {min(pd_base):.4f}")

    from isac_sim.core.config import apply_overrides

    for n_groups in groups_list:
        looks = max(int(round(16 / n_groups)), 1)
        cfg_s = apply_overrides(cfg_on, {"detect.n_looks": looks})
        all_pd = [0.0] * n_tgt
        tx_seen = []
        for g in range(n_groups):
            group = [q for q in range(n_tgt) if q % n_groups == g]
            members = sorted({int(i) for q in group for (i, _j) in selected.get(q, [])})
            mask = illuminator_mask({q: selected.get(q, []) for q in group}, n_uav)
            tabs = compute_link_tables(cfg_s, base, active_tx_mask=mask)
            plan = assign_fusion_nodes(cfg_s, base, tabs, geom=geom)
            tx_seen.append(int(mask.sum()))
            for q in group:
                if selected.get(q):
                    all_pd[q] = float(
                        predicted_pd_for_links(cfg_s, tabs, q, selected[q], plan=plan)
                    )
        print(f"  slots={n_groups:>2}  looks/target={looks:>2}  radiators/slot={tx_seen}  "
              f"worst P_D={min(all_pd):.4f}  mean={sum(all_pd) / n_tgt:.4f}")
    print("  read: a row beats the baseline only if the interference drop outweighs")
    print("        the sqrt(looks) loss from splitting the CPI.")
    print()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    print("=" * 92)
    print(f"coordination fixed point | {AREA:.0f} m | RCS {RCS} m^2 | multi-target selection")
    print("=" * 92)

    # --- control: the gate ON but with mask=None must equal the release path ---
    cfg_on = make_cfg(True)
    res_on = select_with_coordination(cfg_on, generate_geometry(cfg_on, np.random.default_rng(1)),
                                      rounds=args.rounds, seed=args.seed)
    report(f"coordinated gate ON (rounds budget {args.rounds})", res_on)

    # --- how many rounds does it really need? -------------------------------
    print("convergence check -- keep giving it rounds and see whether the answer moves:")
    print(f"  {'rounds':>7}{'#TX':>6}{'worst P_D':>11}{'mean P_D':>10}{'converged':>11}")
    answers = []
    for r in (1, 2, 3, 4):
        res = select_with_coordination(cfg_on, generate_geometry(cfg_on, np.random.default_rng(1)),
                                       rounds=r, seed=args.seed)
        fin = res.final()
        answers.append((r, fin.n_tx, fin.worst_pd, fin.mean_pd, res.converged))
        print(f"  {r:>7}{fin.n_tx:>6}{fin.worst_pd:>11.4f}{fin.mean_pd:>10.4f}"
              f"{str(res.converged):>11}")

    stable = len({round(a[2], 6) for a in answers if a[0] >= 2}) == 1
    print(f"  => worst P_D stable across round budgets >= 2: {stable}")
    print("     (rounds=1 is the un-fed-back baseline; from 2 on the answer must stop")
    print("      moving, otherwise the fixed point is not what we think it is.)")

    grouped_tradeoff(cfg_on, generate_geometry(cfg_on, np.random.default_rng(1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
