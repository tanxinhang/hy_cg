"""Does the coordination price need the coarse stage, or is the fine replay enough?

Locked scenario: 500 m footprint, RCS 0.2 m^2, V1 release 口径 (``target-local-v1``),
coordination gate on.  Decision question: ``selector.max_tx_nodes`` / ``tx_penalty``
are only read by ``_greedy_lagrangian``, while the release entry
``select_c2f_adaptive`` first runs an unpriced coarse rollout that chooses the fine
candidate pool.  Should the price be wired into that rollout too?

Two designs are compared at matched cap and matched radiating count, on several
deployment seeds (a single seed is not enough: the frontier is non-monotone in K).
Both are measured with the *release* selector on *gated* tables, so neither is
handicapped by F1 (the selector not seeing the mask) any more -- that defect is what
made the earlier "cap makes things worse" reading (0.2050 -> 0.1472) uninterpretable.

  FINE   select freely under the cap (``selector.max_tx_nodes=K``), derive the
         illuminator mask from the result, feed it back until the mask repeats.
         The cap binds on the fine replay only; the coarse rollout is unpriced.
  PLAN   decide the radiating set first: pick the K nodes that served the most
         links in the un-coordinated schedule, gate the tables with them, then
         select among their links.  One shot, no fixed point.  This is the
         frontier ``probe_coordination_aware_selection`` measured.

If PLAN is systematically ahead of FINE at matched #TX, the coarse rollout is the
bottleneck and the price belongs there.  If they are on the same frontier, the fine
replay already does the job and the extra wiring buys nothing.

Every arm is scored on the **full fine table** (``dd_gain=base.eta_fine``), which is
what ``simulate.py`` feeds the detector (``c2f_tables``).  Scoring on the coarse
``dd_frac_loss`` table instead costs the release selector ~0.15 worst-P_D at #TX=3,
because it selected against the fine one -- a 口径 mismatch that silently penalises
exactly the method under test.

Read the ordering, not the absolute values: ``predicted_pd_for_links`` is ~24 %
biased low in the deep low-SNR regime.

    python tools/probe_price_stage.py                       # 4 seeds
    python tools/probe_price_stage.py --seeds 20260917      # one seed
"""

from __future__ import annotations

import argparse
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
from isac_sim.detection.fusion import predicted_pd_for_links  # noqa: E402
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from isac_sim.cooperation.reporting import assign_fusion_nodes  # noqa: E402
from experiments.selection import select_c2f_adaptive

AREA = 500.0
RCS = 0.2
PD_REQ = 0.95
DEFAULT_SEEDS = (20260917, 101, 202, 303)
KS = (2, 3, 4, 5, 6, 8)
# Pre-registered decision rule: wire the price into the coarse rollout only if the
# plan-first frontier beats the fine-capped one by more than this, on average over
# seeds and caps, at equal radiating count.
DECISION_GAP = 0.05


def make_cfg(**extra):
    cfg = default_config()
    cfg = apply_preset(cfg, "target-local-v1")
    overrides = {
        "geometry.area_xy": AREA,
        "detect.target_rcs": RCS,
        "interference.sense_gate_by_active_tx": True,
    }
    overrides.update(extra)
    cfg = apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def fine_tables(cfg, base, mask):
    """The table the release pipeline scores on.

    ``simulate.py`` evaluates detection on ``c2f_tables`` -- the **full**
    ``eta_fine`` DD table (``dd_gain=base.eta_fine``), not the coarse
    ``dd_frac_loss`` table the selector's shortlist stage uses. Scoring a
    schedule on the coarse table understates the release selector by ~0.15
    worst-P_D at #TX=3 (it selected against the fine one), so every arm here is
    measured on the fine table, gated by its own mask.
    """
    return compute_link_tables(
        cfg, base, dd_gain=base.eta_fine, active_tx_mask=mask
    )


def evaluate(cfg, tables, selected, plan):
    """Worst / mean predicted P_D, saturated-target count, per-target vector."""
    pds = [
        float(predicted_pd_for_links(cfg, tables, q, selected.get(q, []), plan=plan))
        if selected.get(q)
        else 0.0
        for q in range(cfg.scale.Q)
    ]
    return min(pds), float(np.mean(pds)), int(sum(p >= PD_REQ for p in pds)), pds


def run_seed(seed: int, verbose: bool):
    cfg = make_cfg()
    geom = generate_geometry(cfg, np.random.default_rng(seed))
    base = build_base_gains(cfg, geom, np.random.default_rng(seed))

    # --- un-coordinated reference (also the usage ranking PLAN starts from) ----
    t0 = compute_link_tables(cfg, base, active_tx_mask=None)
    plan0 = assign_fusion_nodes(cfg, base, t0, geom=geom)
    sel0, _D0, _s0 = select_c2f_adaptive(cfg, base, t0, plan=plan0)
    tf0 = fine_tables(cfg, base, None)
    plan_f0 = assign_fusion_nodes(cfg, base, tf0, geom=geom)
    w0, m0, sat0, _ = evaluate(cfg, tf0, sel0, plan_f0)
    usage: dict[int, int] = {}
    for links in sel0.values():
        for i, _j in links:
            usage[int(i)] = usage.get(int(i), 0) + 1
    ranked = sorted(usage, key=lambda i: (-usage[i], i))
    n_tx0 = len(ranked)

    res_unc = select_with_coordination(cfg, geom, rounds=4, seed=seed)
    fin = res_unc.final()
    tf_unc = fine_tables(cfg, base, fin.mask_used)
    plan_unc = assign_fusion_nodes(cfg, base, tf_unc, geom=geom)
    w_unc, m_unc, _sat_unc, _ = evaluate(cfg, tf_unc, res_unc.selected, plan_unc)

    if verbose:
        print(f"\nseed {seed}: un-coordinated worst P_D {w0:.4f} mean {m0:.4f} sat {sat0}/{cfg.scale.Q}"
              f" #TX {n_tx0} #links {sum(len(v) for v in sel0.values())}")
        print(f"          mask-only fixed point worst P_D {w_unc:.4f} mean {m_unc:.4f}"
              f" #TX {int(fin.mask_used.sum())} #links {fin.n_links}"
              f" rounds {len(res_unc.history)} converged {res_unc.converged}")
        print(f"  {'K':>3} | {'FINE (cap in the fine replay)':^38} | {'PLAN (radiating set first)':^36} | {'gap':>8}")
        print(f"  {'':>3} | {'#TX':>4}{'#links':>7}{'worst P_D':>11}{'mean':>9}{'sat':>6}"
              f" | {'#TX':>4}{'#links':>7}{'worst P_D':>11}{'mean':>9}{'sat':>6} | {'P-F':>8}")

    rows = []
    fine_pts: dict[int, list[float]] = {}
    plan_pts: dict[int, list[float]] = {}
    for K in KS:
        # --- FINE: cap the fine replay, feed the derived mask back -----------
        cfg_fine = make_cfg(**{"selector.max_tx_nodes": K})
        res = select_with_coordination(cfg_fine, geom, rounds=4, seed=seed)
        n_tx_f = int(res.final().mask_used.sum()) if res.final().mask_used is not None else cfg.scale.M
        nl_f = res.final().n_links
        # Scored on the fine table (pipeline 口径), not on the coarse table the
        # selector's shortlist stage optimises against.
        tf_f = fine_tables(cfg, base, res.final().mask_used)
        plan_f = assign_fusion_nodes(cfg, base, tf_f, geom=geom)
        wf, mf, sat_f, _ = evaluate(cfg, tf_f, res.selected, plan_f)
        # --- PLAN: fix the radiating set from the un-coordinated ranking -----
        mask = np.zeros(cfg.scale.M, dtype=bool)
        mask[ranked[:K]] = True
        tables_p = compute_link_tables(cfg, base, active_tx_mask=mask)
        plan_p = assign_fusion_nodes(cfg, base, tables_p, geom=geom)
        sel_p, _Dp, _sp = select_c2f_adaptive(
            cfg, base, tables_p, plan=plan_p, active_tx_mask=mask
        )
        tf_p = fine_tables(cfg, base, mask)
        plan_fp = assign_fusion_nodes(cfg, base, tf_p, geom=geom)
        wp, mp, sat_p, _ = evaluate(cfg, tf_p, sel_p, plan_fp)
        n_tx_p = len({int(i) for links in sel_p.values() for (i, _j) in links})
        nl_p = sum(len(v) for v in sel_p.values())
        rows.append((K, n_tx_f, wf, n_tx_p, wp))
        fine_pts.setdefault(n_tx_f, []).append(wf)
        plan_pts.setdefault(n_tx_p, []).append(wp)
        if verbose:
            print(f"  {K:>3} | {n_tx_f:>4}{nl_f:>7}{wf:>11.4f}{mf:>9.4f}{sat_f:>6}"
                  f" | {n_tx_p:>4}{nl_p:>7}{wp:>11.4f}{mp:>9.4f}{sat_p:>6} | {wp - wf:>+8.4f}")

    # --- the soft price, wired the same (fine-only) way ----------------------
    # The decision is about WHERE the price is read, not about its shape, so the
    # soft price is swept too: if its points land on the same frontier the answer
    # does not depend on using the hard cap.
    soft = []
    for pen in (0.05, 0.1, 0.2):
        cfg_pen = make_cfg(**{"selector.tx_penalty": pen})
        res = select_with_coordination(cfg_pen, geom, rounds=4, seed=seed)
        tf_pen = fine_tables(cfg, base, res.final().mask_used)
        plan_pen = assign_fusion_nodes(cfg, base, tf_pen, geom=geom)
        wp_post, mp_post, sat_post, _ = evaluate(cfg, tf_pen, res.selected, plan_pen)
        n_tx_pen = int(res.final().mask_used.sum()) if res.final().mask_used is not None else cfg.scale.M
        soft.append((pen, n_tx_pen, wp_post))
        fine_pts.setdefault(n_tx_pen, []).append(wp_post)
    if verbose:
        print("  soft price (fine-only, same fixed point): "
              + "  ".join(f"pen={p}: #TX={n} worst {w:.4f}" for p, n, w in soft))
    return rows, fine_pts, plan_pts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("=" * 108)
    print(f"price-stage probe | {AREA:.0f} m | RCS {RCS} m^2 | release selector | "
          f"seeds {seeds}")
    print("=" * 108)

    per_seed = {}
    fine_pts: dict[int, list[float]] = {}
    plan_pts: dict[int, list[float]] = {}
    for i, seed in enumerate(seeds):
        rows, fp, pp = run_seed(seed, verbose=(i == 0))
        per_seed[seed] = rows
        for n, vals in fp.items():
            fine_pts.setdefault(n, []).extend(vals)
        for n, vals in pp.items():
            plan_pts.setdefault(n, []).extend(vals)

    print("\n" + "=" * 108)
    print("matched-#TX comparison (PLAN - FINE), pooled over seeds, caps AND soft prices")
    print("=" * 108)
    matched: dict[int, list[float]] = {}
    for n in sorted(set(fine_pts) & set(plan_pts)):
        f = float(np.mean(fine_pts[n]))
        p = float(np.mean(plan_pts[n]))
        matched[n] = [p - f]
        print(f"  #TX={n:>2}  FINE {f:.4f} (n={len(fine_pts[n])})   "
              f"PLAN {p:.4f} (n={len(plan_pts[n])})   gap {p - f:+.4f}")
    per_seed_matched: list[float] = []
    for seed, rows in per_seed.items():
        pairs = {}
        for K, n_tx_f, wf, n_tx_p, wp in rows:
            pairs.setdefault((n_tx_f, "F"), []).append(wf)
            pairs.setdefault((n_tx_p, "P"), []).append(wp)
        for n in sorted({k[0] for k in pairs}):
            f = np.mean(pairs.get((n, "F"), [np.nan]))
            p = np.mean(pairs.get((n, "P"), [np.nan]))
            if np.isfinite(f) and np.isfinite(p):
                per_seed_matched.append(p - f)
    gaps = np.asarray(per_seed_matched)
    print()
    print(f"  per-seed-on-matched-#TX gap (cap arms): mean {gaps.mean():+.4f}  "
          f"median {np.median(gaps):+.4f}  n={gaps.size}  "
          f"PLAN ahead on {int((gaps > 0).sum())}/{gaps.size}")
    # Matched-K comparison as a control: same cap, not necessarily same #TX.
    per_k = np.asarray([[row[4] - row[2] for row in rows] for rows in per_seed.values()])
    print(f"  matched-K mean gap  = {per_k.mean():+.4f}"
          f"  (per K {dict(zip(KS, np.round(per_k.mean(axis=0), 4)))})")
    pooled = np.asarray(
        [float(np.mean(plan_pts[n])) - float(np.mean(fine_pts[n])) for n in matched]
    )
    print(f"  pooled matched-#TX mean gap = {pooled.mean():+.4f}")
    print()
    print(f"  decision threshold {DECISION_GAP:+.2f}  =>  "
          + ("WIRE the price into the coarse rollout"
             if pooled.mean() > DECISION_GAP else
             "DO NOT wire the price into the coarse rollout: at matched #TX the "
             "fine-capped design sits on the same frontier"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
