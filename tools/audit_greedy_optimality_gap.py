"""Optimizer-layer diagnostic: how far is the release greedy from optimal?

Two independent measurements, because neither alone is conclusive:

1. **Small-instance optimality gap.**  ``theory.same_objective_oracle``
   exhaustively maximises the *same* objective ``F`` the greedy optimises, so
   ``F_opt - F_greedy`` is a true optimality gap (not a heuristic estimate).
   Only affordable for tiny instances -- the oracle refuses trees above 2e8.
   Reported as a reference, not extrapolated to the release scale.

2. **Full-scale local-search headroom.**  At the release scale (M=15, Q=10) the
   oracle is impossible, so the question "is the greedy leaving anything on the
   table?" is answered directly: run a (drop, add) / swap local search on top of
   the greedy schedule and report how much ``F`` and worst-target P_D move.
   Two pools are compared so the two sources of loss stay separable:
     * ``shortlist`` -- the pool the greedy itself saw.  Any gain here is pure
       *optimizer* loss (path dependence of the single-pass greedy).
     * ``full``      -- every feasible link.  Gains here additionally include
       what the C2F shortlist truncation threw away.

Nothing here writes into the released path: the local search is a prototype
living in this file.  It only becomes part of the library if the measured
headroom justifies it, and even then behind a default-off config key.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
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
from isac_sim.detection.fusion import (  # noqa: E402
    deflection_for_links,
    predicted_pd_for_links,
    selection_utility,
    selection_utility_from_pd,
)
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from isac_sim.cooperation.reporting import assign_fusion_nodes  # noqa: E402
from experiments.selection import feasible_links_for_target, link_cost_ms, local_cap_allows, processing_caps_allow, remote_cap_allows, select_c2f_adaptive, select_lagrangian
from audits.theory import same_objective_oracle, task_objective  # noqa: E402

Link = tuple[int, int]


def make_cfg(area: float, rcs: float, M: int, Q: int, extra: dict | None = None):
    cfg = apply_preset(default_config(), "target-local-v1")
    ov = {
        "geometry.area_xy": float(area),
        "detect.target_rcs": float(rcs),
        "scale.M": int(M),
        "scale.Q": int(Q),
    }
    if extra:
        ov.update(extra)
    cfg = apply_overrides(cfg, ov)
    validate_config(cfg)
    return cfg


def build_instance(cfg, seed: int):
    rng = np.random.default_rng(seed)
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    tables = compute_link_tables(cfg, base)
    plan = assign_fusion_nodes(cfg, base, tables, geom=geom)
    return geom, base, tables, plan


def feasible(cfg, base, tables, plan, q):
    return feasible_links_for_target(cfg, base, tables, q, plan)


def admits(cfg, selected, link, q, plan) -> bool:
    return (
        local_cap_allows(cfg, selected[q], link, q, plan)
        and remote_cap_allows(cfg, selected, link, q, plan)
        and processing_caps_allow(cfg, selected, link, q, plan)
    )


def pd_summary(cfg, tables, plan, base, selected) -> tuple[float, float]:
    pds = []
    for q in range(cfg.scale.Q):
        links = selected.get(q, [])
        pds.append(
            float(predicted_pd_for_links(cfg, tables, q, links, plan=plan, base=base))
            if links
            else 0.0
        )
    return float(np.mean(pds)), float(np.min(pds))


class _Incremental:
    """Incremental evaluation of the greedy objective ``F``.

    ``task_objective`` recomputes every target's deflection on each call, which
    makes a swap neighbourhood cost O(Q) per candidate.  Only the mutated
    target's entry changes, so this keeps the deflection vector and the delay
    price and re-evaluates just those -- same arithmetic, ~Q times fewer calls.
    Cross-checked against :func:`task_objective` by the caller.
    """

    def __init__(self, cfg, tables, base, plan, selected, objective: str = "F"):
        self.cfg = cfg
        self.tables = tables
        self.base = base
        self.plan = plan
        self.objective = objective
        self.detector_aligned = cfg.selector.score_mode.lower() == "detector_pd"
        # The predicted P_D is always maintained: ``worst_pd`` scores on it, and
        # every arm reports it.
        self.always_pd = True
        self.Q = cfg.scale.Q
        self.sel = {q: list(selected.get(q, [])) for q in range(self.Q)}
        self.D = np.zeros(self.Q)
        self.pd = np.zeros(self.Q)
        self.cost = np.zeros(self.Q)
        for q in range(self.Q):
            self._refresh(q)

    def _links_cost(self, q, links) -> float:
        if not self.cfg.selector.use_delay_price:
            return 0.0
        return float(
            self.cfg.selector.lambda_c
            * sum(link_cost_ms(self.cfg, self.tables, q, l, self.plan) for l in links)
        )

    def _refresh(self, q) -> None:
        links = self.sel[q]
        self.D[q] = float(
            deflection_for_links(
                self.cfg, self.tables, q, links,
                weight_mode="deflection", plan=self.plan, base=self.base,
            )
        )
        if self.detector_aligned or self.always_pd:
            self.pd[q] = float(
                predicted_pd_for_links(
                    self.cfg, self.tables, q, links,
                    weight_mode="deflection", plan=self.plan, base=self.base,
                )
            )
        self.cost[q] = self._links_cost(q, links)

    def F(self) -> float:
        """The paper objective ``F`` (always reported, whatever is optimised)."""
        if self.detector_aligned:
            u = selection_utility_from_pd(self.cfg, self.D, self.pd)
        else:
            u = selection_utility(self.cfg, self.D)
        return float(u) - float(self.cost.sum())

    def value(self) -> float:
        """The quantity the local search actually maximises."""
        if self.objective == "worst_pd":
            # Lexicographic on the reported statistic: the worst target first,
            # with a tiny mean term so ties break towards overall detection.
            return float(np.min(self.pd)) + 1e-6 * float(np.mean(self.pd))
        return self.F()

    def try_replace(self, q, links) -> float:
        """Swap in ``links`` for target ``q``, keep it iff ``F`` improves."""
        prev_links, prev_D, prev_pd, prev_cost = (
            self.sel[q], self.D[q], self.pd[q], self.cost[q],
        )
        self.sel[q] = list(links)
        self._refresh(q)
        new_v = self.value()
        if new_v > self._cur_v + 1e-12:
            self._cur_v = new_v
            return new_v
        self.sel[q] = prev_links
        self.D[q], self.pd[q], self.cost[q] = prev_D, prev_pd, prev_cost
        return self._cur_v

    def start(self, v0: float) -> None:
        self._cur_v = float(v0)


def local_search(
    cfg,
    tables,
    base,
    plan,
    selected: dict[int, list[Link]],
    pool: dict[int, list[Link]],
    max_passes: int = 6,
    objective: str = "F",
) -> tuple[dict[int, list[Link]], float, float, int]:
    """(drop, add) / swap improvement on the exact greedy objective.

    Neighbourhood, per target: replace one selected link by one unselected
    candidate, add one candidate when a slot is free, and drop one link when
    removing it raises ``F`` (possible: the delay price can make a link net
    negative once the target is already served).
    """
    st = _Incremental(cfg, tables, base, plan, selected, objective=objective)
    F0 = float(task_objective(cfg, tables, selected, plan, base))
    st.start(st.value())
    evals = 0
    K_per = cfg.selector.max_links_per_target
    K_tot = cfg.selector.max_total_links
    for _ in range(int(max_passes)):
        before = st.value()
        for q in range(cfg.scale.Q):
            cands = list(pool.get(q, []))
            if not cands:
                continue
            # --- drop ---
            for link in list(st.sel[q]):
                st.try_replace(q, [l for l in st.sel[q] if l != link])
                evals += 1
            # --- add ---
            if sum(len(v) for v in st.sel.values()) >= K_tot:
                pass
            else:
                for link in cands:
                    if link in st.sel[q] or len(st.sel[q]) >= K_per:
                        continue
                    if not admits(cfg, st.sel, link, q, plan):
                        continue
                    st.try_replace(q, list(st.sel[q]) + [link])
                    evals += 1
            # --- swap (same target) ---
            for old in list(st.sel[q]):
                for new in cands:
                    if new in st.sel[q]:
                        continue
                    trial = [new if l == old else l for l in st.sel[q]]
                    trial_sel = {qq: list(st.sel[qq]) for qq in st.sel}
                    trial_sel[q] = trial
                    if not admits(cfg, trial_sel, new, q, plan):
                        continue
                    st.try_replace(q, trial)
                    evals += 1
        if st.value() <= before + 1e-12:
            break
    cur = {q: list(v) for q, v in st.sel.items()}
    cur_F = float(task_objective(cfg, tables, cur, plan, base))
    return cur, F0, cur_F, evals


def shortlist_pool(cfg, base, tables, plan, selected) -> dict[int, list[Link]]:
    """The pool the greedy saw, approximated by its own choice plus neighbours.

    ``select_c2f_adaptive`` does not expose its shortlist, so the shortlist pool
    is reconstructed as the greedy's own selection unioned with the per-target
    top-ranked feasible links.  This is deliberately *narrower* than the full
    pool; gains on it are attributable to the optimizer, not to truncation.
    """
    return {
        q: list(dict.fromkeys(list(selected.get(q, [])) + feasible(cfg, base, tables, plan, q)))
        [: max(cfg.selector.max_links_per_target * 4, 8)]
        for q in range(cfg.scale.Q)
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=20260918)
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--M", type=int, default=15)
    ap.add_argument("--Q", type=int, default=10)
    ap.add_argument("--max-passes", type=int, default=6)
    ap.add_argument(
        "--objective",
        default="F",
        choices=["F", "worst_pd"],
        help="what the local search maximises: the paper objective F, or the "
        "reported statistic (worst-target predicted P_D)",
    )
    ap.add_argument("--small-M", type=int, default=3)
    ap.add_argument("--small-Q", type=int, default=3)
    ap.add_argument("--small-trials", type=int, default=6)
    ap.add_argument("--small-max-links", type=int, default=2)
    ap.add_argument("--out", default="results_greedy_optimality_gap")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. full-scale local-search headroom ----------------
    cfg = make_cfg(args.area, args.rcs, args.M, args.Q)
    rows = []
    t0 = time.time()
    for t in range(int(args.trials)):
        seed = int(args.seed0) + t
        geom, base, tables, plan = build_instance(cfg, seed)
        sel, _D, _stats = select_c2f_adaptive(cfg, base, tables, plan=plan)
        F_g = float(task_objective(cfg, tables, sel, plan, base))
        mean_g, worst_g = pd_summary(cfg, tables, plan, base, sel)

        pool_s = shortlist_pool(cfg, base, tables, plan, sel)
        pool_f = {q: feasible(cfg, base, tables, plan, q) for q in range(cfg.scale.Q)}

        sel_s, F0_s, F_s, ev_s = local_search(
            cfg, tables, base, plan, sel, pool_s, args.max_passes, args.objective
        )
        sel_f, F0_f, F_f, ev_f = local_search(
            cfg, tables, base, plan, sel, pool_f, args.max_passes, args.objective
        )
        mean_s, worst_s = pd_summary(cfg, tables, plan, base, sel_s)
        mean_f, worst_f = pd_summary(cfg, tables, plan, base, sel_f)

        rows.append(
            {
                "trial": t,
                "seed": seed,
                "F_greedy": F_g,
                "F_swap_shortlist": F_s,
                "F_swap_full": F_f,
                "dF_shortlist": F_s - F_g,
                "dF_full": F_f - F_g,
                "rel_gap_shortlist": (F_s - F_g) / abs(F_g) if abs(F_g) > 1e-9 else 0.0,
                "rel_gap_full": (F_f - F_g) / abs(F_g) if abs(F_g) > 1e-9 else 0.0,
                "pd_mean_greedy": mean_g,
                "pd_worst_greedy": worst_g,
                "pd_mean_swap_full": mean_f,
                "pd_worst_swap_full": worst_f,
                "d_pd_worst_full": worst_f - worst_g,
                "links_greedy": int(sum(len(v) for v in sel.values())),
                "links_swap_full": int(sum(len(v) for v in sel_f.values())),
                "evals_full": ev_f,
            }
        )
        print(
            f"trial {t:3d}  F {F_g:8.4f} -> short {F_s:8.4f} ({F_s - F_g:+7.4f})"
            f" | full {F_f:8.4f} ({F_f - F_g:+7.4f})"
            f"  worstPD {worst_g:.4f} -> {worst_f:.4f}",
            flush=True,
        )

    with open(out_dir / f"local_search_{args.objective}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    d_s = np.array([r["dF_shortlist"] for r in rows])
    d_f = np.array([r["dF_full"] for r in rows])
    dw = np.array([r["d_pd_worst_full"] for r in rows])
    print("\n=== full-scale local search (greedy -> local optimum) ===")
    print(f"  objective maximised : {args.objective}")
    print(f"  scale M={args.M} Q={args.Q}, area={args.area} m, RCS={args.rcs}, trials={len(rows)}")
    print(f"  dF shortlist pool : mean {d_s.mean():+.5f}  median {np.median(d_s):+.5f}  max {d_s.max():+.5f}")
    print(f"  dF full pool      : mean {d_f.mean():+.5f}  median {np.median(d_f):+.5f}  max {d_f.max():+.5f}")
    print(f"  d worst P_D (full): mean {dw.mean():+.5f}  median {np.median(dw):+.5f}  max {dw.max():+.5f}")
    print(f"  trials improved   : {(d_f > 1e-9).sum()}/{len(rows)}")
    print(f"  elapsed {time.time() - t0:.1f}s")

    # ---------------- 2. small-instance exact optimality gap -------------
    small = []
    scfg = make_cfg(
        args.area, args.rcs, args.small_M, args.small_Q,
        extra={"selector.max_links_per_target": int(args.small_max_links)},
    )
    for t in range(min(int(args.trials), int(args.small_trials))):
        seed = int(args.seed0) + t
        geom, base, tables, plan = build_instance(scfg, seed)
        sel_g, _D, _s = select_c2f_adaptive(scfg, base, tables, plan=plan)
        F_g = float(task_objective(scfg, tables, sel_g, plan, base))
        sel_l, _ = select_lagrangian(scfg, base, tables, plan)
        F_l = float(task_objective(scfg, tables, sel_l, plan, base))
        try:
            _sel_o, F_o = same_objective_oracle(scfg, base, tables, plan)
        except RuntimeError as exc:
            print(f"  oracle skipped (trial {t}): {exc}")
            continue
        small.append(
            {
                "trial": t,
                "seed": seed,
                "F_optimal": F_o,
                "F_c2f": F_g,
                "F_lagrangian": F_l,
                "gap_c2f": F_o - F_g,
                "gap_lagrangian": F_o - F_l,
                "rel_gap_c2f": (F_o - F_g) / abs(F_o) if abs(F_o) > 1e-9 else 0.0,
            }
        )
    if small:
        with open(out_dir / "oracle_gap.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(small[0]))
            w.writeheader()
            w.writerows(small)
        g = np.array([r["rel_gap_c2f"] for r in small])
        print("\n=== small-instance exact gap (oracle, same objective) ===")
        print(f"  scale M={args.small_M} Q={args.small_Q}, trials={len(small)}")
        print(f"  relative gap: mean {g.mean():.4%}  median {np.median(g):.4%}  max {g.max():.4%}")
        print("  (reference only: the oracle is intractable at the release scale)")

    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
