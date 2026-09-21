"""Step 3: the ONE-SHOT sparsity-aware selection, no feedback iteration.

Why one-shot and not the feedback loop (``probe_coordination_fixedpoint.py``):
audited over 8 seeds the feedback map settled after 2/4/2/3/2/3 rounds, entered a
real cycle on one seed and failed to settle in 6 rounds on another. A method whose
fixed point does not provably exist cannot be the main mechanism. The audited
robust fact is the other one: **capping the number of radiating nodes** lifts the
worst-target P_D from 0.205 to ~0.79, and a *random* 3-node subset already gets
+0.19..+0.67 -- i.e. the cap itself does the work, not which nodes are chosen.

So this probe measures the honest one-shot pipeline:

    1. select links with a TX price (``selector.tx_penalty``) or a hard cap
       (``selector.max_tx_nodes``) on the un-coordinated tables;
    2. derive the mask from the illuminators of that selection;
    3. rebuild the tables under the realised mask and evaluate P_D there.

The selector never sees the reduced interference, so its model is conservative:
the realised performance can only be better than what it optimised against. That
is the price of dropping the non-convergent iteration, and it is worth paying.

Reported across seeds, because the effect is a worst-case/variance effect, not a
mean effect -- a single seed cannot distinguish the two.
"""

from __future__ import annotations

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
from experiments.coordination import illuminator_mask
from isac_sim.detection.fusion import predicted_pd_for_links  # noqa: E402
from isac_sim.sensing.model import compute_link_tables, build_base_gains, generate_geometry  # noqa: E402
from isac_sim.cooperation.reporting import assign_fusion_nodes  # noqa: E402
from experiments.selection import select_lagrangian

AREA = 500.0
RCS = 0.2
PD_REQ = 0.95
SEEDS = (20260917, 101, 202, 303)


def make_cfg(tx_penalty: float = 0.0, max_tx: int | None = None):
    cfg = default_config()
    cfg = apply_preset(cfg, "target-local-v1")
    cfg = apply_overrides(
        cfg,
        {
            "geometry.area_xy": AREA,
            "detect.target_rcs": RCS,
            "interference.sense_gate_by_active_tx": True,
            "selector.tx_penalty": tx_penalty,
            "selector.max_tx_nodes": max_tx,
        },
    )
    validate_config(cfg)
    return cfg


def run_one(cfg, seed: int, coordinate: bool = True):
    """One-shot pipeline. Returns (worst_pd, mean_pd, n_tx, n_links, per_target).

    ``coordinate=False`` is the TRUE uncoordinated reference: every node radiates,
    no mask is derived. Without it the "knobs off" row is not a baseline at all --
    it is the post-hoc-masking variant, because it still silences the nodes its own
    selection did not use.
    """
    n_uav, n_tgt = cfg.scale.M, cfg.scale.Q
    geom = generate_geometry(cfg, np.random.default_rng(seed))
    base = build_base_gains(cfg, geom, np.random.default_rng(seed))

    # 1. select on the un-coordinated tables
    t0 = compute_link_tables(cfg, base, active_tx_mask=None)
    plan0 = assign_fusion_nodes(cfg, base, t0, geom=geom)
    selected, _ = select_lagrangian(cfg, base, t0, plan0)

    # 2. derive the mask from the selection (or keep everyone radiating)
    mask = illuminator_mask(selected, n_uav) if coordinate else None

    # 3. evaluate under the realised mask
    tabs = compute_link_tables(cfg, base, active_tx_mask=mask)
    plan = assign_fusion_nodes(cfg, base, tabs, geom=geom)
    pds = [
        float(predicted_pd_for_links(cfg, tabs, q, selected.get(q, []), plan=plan))
        if selected.get(q)
        else 0.0
        for q in range(n_tgt)
    ]
    n_tx = n_uav if mask is None else int(mask.sum())
    return min(pds), sum(pds) / len(pds), n_tx, sum(len(v) for v in selected.values()), pds


def sweep(label: str, settings, rows, coordinate: bool = True) -> None:
    print("=" * 96)
    print(label)
    print("=" * 96)
    print(f"  {'setting':>16}{'#TX':>6}{'#links':>8}{'worst P_D (mean over seeds)':>29}"
          f"{'min':>9}{'max':>9}{'std':>8}")
    for name, kwargs in settings:
        if name == "REFERENCE (no coordination)":
            res = [run_one(make_cfg(), s, coordinate=False) for s in SEEDS]
        else:
            try:
                res = [run_one(make_cfg(**kwargs), s) for s in SEEDS]
            except ValueError as exc:
                print(f"  {name:>16}  -> rejected: {str(exc)[:70]}...")
                continue
        w = np.array([r[0] for r in res])
        tx = int(np.mean([r[2] for r in res]))
        nl = int(np.mean([r[3] for r in res]))
        rows.append((name, w.mean(), w.min(), w.max(), w.std(), tx, nl))
        print(f"  {name:>16}{tx:>6}{nl:>8}{w.mean():>29.4f}{w.min():>9.4f}"
              f"{w.max():>9.4f}{w.std():>8.4f}")
    print()


def main() -> int:
    rows: list = []
    print(f"sparse (one-shot) selection | {AREA:.0f} m | RCS {RCS} m^2 | seeds {SEEDS}\n")

    sweep(
        "hard cap on the number of radiating nodes (selector.max_tx_nodes)",
        [("REFERENCE (no coordination)", {}),
         ("mask only, no cap", {}), ("K=1", {"max_tx": 1}), ("K=2", {"max_tx": 2}),
         ("K=3", {"max_tx": 3}), ("K=4", {"max_tx": 4}), ("K=6", {"max_tx": 6}),
         ("K=8", {"max_tx": 8})],
        rows,
    )
    sweep(
        "soft price on waking a NEW radiator (selector.tx_penalty)",
        [(f"penalty={p:g}", {"tx_penalty": p}) for p in (0.01, 0.05, 0.2, 1.0, 5.0)],
        rows,
    )

    base = rows[0]
    print("=" * 96)
    print("verdict")
    print("=" * 96)
    best_mean = max(rows[1:], key=lambda r: r[1])
    best_worst = max(rows[1:], key=lambda r: r[2])
    print(f"  baseline        : mean {base[1]:.4f}  worst-seed {base[2]:.4f}  std {base[4]:.4f}  "
          f"#TX {base[5]}")
    print(f"  best mean       : {best_mean[0]} -> mean {best_mean[1]:.4f} "
          f"(+{best_mean[1] - base[1]:.4f}), #TX {best_mean[5]}")
    print(f"  best worst-seed : {best_worst[0]} -> {best_worst[2]:.4f} "
          f"(+{best_worst[2] - base[2]:.4f}), #TX {best_worst[5]}")
    print(f"  std compression : {base[4]:.4f} -> {best_mean[4]:.4f} "
          f"({base[4] / max(best_mean[4], 1e-9):.1f}x)")
    print("  note: the selector optimises against the UN-coordinated tables, so these")
    print("        realised numbers are conservative, not optimistic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
