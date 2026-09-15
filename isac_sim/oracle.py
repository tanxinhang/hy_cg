"""Exact small-scale oracle for the proposed link-selection problem.

The proposed selector is a *greedy* heuristic: it commits, per iteration,
the link with the largest positive Lagrangian marginal gain
``alpha_q * DeltaD - lambda_c * delay``.  To say anything rigorous about how
good that heuristic is, we need an exact reference on the same objective.

Because the fused deflection ``D_q`` is nonlinear in the link set (the fusion
weights are ``w_k propto mu_k / sigma_k^2``), the only honest oracle is
exhaustive search over link subsets under the resource budget.  This module
implements that search for the small ``(M, Q)`` regimes where it is
tractable (``M=3..4``, ``Q=3``).  It is the ``isac_sim`` counterpart of
``CodeCg/gate_otfs_collision/assignment.py: assignment_oracle_small``, adapted
to the soft-information-fusion objective.

The objective maximised here is the *detection-quality* quantity

    Obj(S) = sum_q D_q(S_q)

under the per-target ``max_links_per_target`` and global ``max_total_links``
budgets.  This strips the delay-price ``lambda_c`` and the priority weights
``alpha_q``, which are *budget-allocation* heuristics rather than *quality*
measures, so the greedy-vs-oracle gap isolates exactly how much detection
quality the heuristic leaves on the table.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

from .config import Config, Link
from .fusion import deflection_for_links
from .model import BaseGains, LinkTables
from .selection import feasible_links_for_target
from .reporting import ReportingPlan, is_local_observation, report_dest


def oracle_exhaustive(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    plan: "object | None" = None,
) -> Tuple[Dict[int, List[Link]], float]:
    """Exhaustively maximise ``sum_q D_q`` under the link budgets.

    Returns ``(selected, obj)`` where ``selected[q]`` is the optimal link set
    for target ``q`` and ``obj`` is the achieved ``sum_q D_q``.

    Complexity is ``prod_q C(|E_q|, <=K_q)`` with budget pruning; only call
    this on small ``(M, Q)``.  A guard raises if the search tree is too large.
    """
    Q = cfg.scale.Q
    K_per = cfg.selector.max_links_per_target
    K_tot = cfg.selector.max_total_links

    q_candidates: List[List[Link]] = [
        feasible_links_for_target(cfg, base, tables, q, plan) for q in range(Q)
    ]

    # Guard against accidental exponential blow-ups.
    tree_size = 1
    for cand in q_candidates:
        n = min(K_per, len(cand))
        n_ways = sum(_comb(len(cand), r) for r in range(0, n + 1))
        tree_size *= n_ways
    if tree_size > 200_000_000:
        raise RuntimeError(
            f"oracle search tree too large ({tree_size}); "
            f"use M<=4 and Q<=3 (current M={cfg.scale.M}, Q={cfg.scale.Q})"
        )

    best_selected: Dict[int, List[Link]] = {q: [] for q in range(Q)}
    best_obj = -np.inf

    def _dfs(q: int, selected: Dict[int, List[Link]], total_links: int) -> None:
        nonlocal best_obj, best_selected
        if q == Q:
            if isinstance(plan, ReportingPlan) and not _selection_respects_hard_budgets(
                cfg, selected, plan
            ):
                return
            obj = sum(
                deflection_for_links(cfg, tables, qq, selected[qq], weight_mode="deflection", plan=plan, base=base)
                for qq in range(Q)
            )
            if obj > best_obj:
                best_obj = float(obj)
                best_selected = {qq: list(selected[qq]) for qq in range(Q)}
            return

        cand = q_candidates[q]
        for r in range(0, min(K_per, len(cand)) + 1):
            if total_links + r > K_tot:
                break  # larger r only exceeds the budget further
            for subset in itertools.combinations(cand, r):
                selected[q] = list(subset)
                _dfs(q + 1, selected, total_links + r)
        selected[q] = []

    _dfs(0, {q: [] for q in range(Q)}, 0)
    return best_selected, float(best_obj)


def _comb(n: int, r: int) -> int:
    """Number of r-combinations of n items (integer, exact)."""
    import math

    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def greedy_objective(
    cfg: Config,
    tables: LinkTables,
    selected: Dict[int, List[Link]],
    plan: "object | None" = None,
    base: BaseGains | None = None,
) -> float:
    """The same ``sum_q D_q`` objective evaluated on a greedy selection."""
    return float(
        sum(
            deflection_for_links(cfg, tables, q, selected.get(q, []), weight_mode="deflection", plan=plan, base=base)
            for q in range(cfg.scale.Q)
        )
    )


def lexicographic_objective_vector(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    plan: ReportingPlan,
    selected: Dict[int, List[Link]],
) -> Dict[str, float]:
    """Evaluate one feasible plan on the joint oracle's objective vector.

    ``base`` and ``tables`` must be the same belief-side quantities available
    to the scheduler.  The four returned components are intentionally kept
    separate; callers must not collapse them into a weighted scalar gap.
    """
    from .fusion import predicted_pd_for_links

    if not _selection_respects_hard_budgets(cfg, selected, plan):
        raise ValueError("candidate plan violates a hard resource budget")
    pd = np.asarray([
        predicted_pd_for_links(
            cfg,
            tables,
            q,
            selected.get(q, []),
            weight_mode="deflection",
            plan=plan,
            base=base,
        ) if selected.get(q, []) else 0.0
        for q in range(cfg.scale.Q)
    ])
    deficits = np.maximum(cfg.detect.pd_required - pd, 0.0)
    remote = sum(
        not is_local_observation(plan, link, q)
        for q, links in selected.items() for link in links
    )
    processing = sum(len(links) for links in selected.values())
    return {
        "worst_detection_deficit": float(np.max(deficits)),
        "total_detection_deficit": float(np.sum(deficits)),
        "remote_reports": float(remote),
        "processing_load": float(processing),
    }


def lexicographic_gap_components(
    candidate: Dict[str, float],
    oracle: Dict[str, float],
    *,
    atol: float = 1e-12,
) -> Dict[str, float]:
    """Return component-wise candidate-minus-oracle gaps and exact-match flag."""
    names = (
        "worst_detection_deficit",
        "total_detection_deficit",
        "remote_reports",
        "processing_load",
    )
    gaps = {
        f"delta_{name}": float(candidate[name] - oracle[name])
        for name in names
    }
    gaps["exact_lexicographic_match"] = float(
        all(abs(gaps[f"delta_{name}"]) <= atol for name in names)
    )
    return gaps


def joint_fusion_selection_oracle(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
) -> Tuple[ReportingPlan, Dict[int, List[Link]], Dict[str, float]]:
    """Exact small-system lexicographic oracle over fusion and observations.

    The objective is, in order: minimum worst-target detection deficit, minimum
    total detection deficit, minimum number of remote reports, and minimum
    processing load.  It contains no scalarization weights.  The caller must
    pass the scheduler's belief-side ``base`` and ``tables``; a truth-side call
    is a different, clairvoyant bound.  This routine is deliberately guarded
    for small ``M,Q`` validation cases only.
    """
    Q, M = cfg.scale.Q, cfg.scale.M
    K_per = int(cfg.selector.max_links_per_target)
    K_tot = int(cfg.selector.max_total_links)
    target_cap = int(cfg.fusion.max_targets_per_uav)
    if target_cap < 0:
        target_cap = Q

    assignment_count = M ** Q
    if assignment_count > 100_000:
        raise RuntimeError(
            f"joint oracle assignment tree too large ({assignment_count}); "
            "use a small M,Q validation case"
        )

    best_key: tuple[float, float, int, int] | None = None
    best_plan = ReportingPlan(mode="explicit", f_q=np.full(Q, -1, dtype=int))
    best_selected: Dict[int, List[Link]] = {q: [] for q in range(Q)}

    for assignment in itertools.product(range(M), repeat=Q):
        counts = np.bincount(np.asarray(assignment, dtype=int), minlength=M)
        if np.any(counts > target_cap):
            continue
        plan = ReportingPlan(mode="explicit", f_q=np.asarray(assignment, dtype=int))
        candidates = [
            feasible_links_for_target(cfg, base, tables, q, plan)
            for q in range(Q)
        ]
        ways = 1
        for cand in candidates:
            ways *= sum(_comb(len(cand), r) for r in range(min(K_per, len(cand)) + 1))
        if ways * assignment_count > 200_000_000:
            raise RuntimeError(
                "joint oracle selection tree too large; reduce M, Q, or observation caps"
            )

        def visit(q: int, selected: Dict[int, List[Link]], total: int) -> None:
            nonlocal best_key, best_plan, best_selected
            if q == Q:
                if not _selection_respects_hard_budgets(cfg, selected, plan):
                    return
                objective = lexicographic_objective_vector(
                    cfg, base, tables, plan, selected
                )
                key = (
                    objective["worst_detection_deficit"],
                    objective["total_detection_deficit"],
                    int(objective["remote_reports"]),
                    int(objective["processing_load"]),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_plan = ReportingPlan(mode="explicit", f_q=plan.f_q.copy())
                    best_selected = {qq: list(selected[qq]) for qq in range(Q)}
                return

            for r in range(min(K_per, len(candidates[q])) + 1):
                if total + r > K_tot:
                    break
                for subset in itertools.combinations(candidates[q], r):
                    selected[q] = list(subset)
                    visit(q + 1, selected, total + r)
            selected[q] = []

        visit(0, {q: [] for q in range(Q)}, 0)

    if best_key is None:
        raise ValueError("joint oracle found no feasible fusion/selection plan")
    return best_plan, best_selected, {
        "worst_detection_deficit": float(best_key[0]),
        "total_detection_deficit": float(best_key[1]),
        "remote_reports": float(best_key[2]),
        "processing_load": float(best_key[3]),
    }


def _selection_respects_hard_budgets(
    cfg: Config,
    selected: Dict[int, List[Link]],
    plan: ReportingPlan,
) -> bool:
    """Check the report, receiver, and fusion capacities of a complete plan."""
    remote = sum(
        not is_local_observation(plan, link, q)
        for q, links in selected.items() for link in links
    )
    if cfg.selector.max_remote_reports >= 0 and remote > cfg.selector.max_remote_reports:
        return False

    rx_counts = np.zeros(cfg.scale.M, dtype=int)
    fusion_counts = np.zeros(cfg.scale.M, dtype=int)
    for q, links in selected.items():
        for link in links:
            rx_counts[link[1]] += 1
            fusion_counts[report_dest(plan, link, q)] += 1
    rx_cap = cfg.selector.max_observations_per_receiver
    fusion_cap = cfg.selector.max_observations_per_fusion_uav
    return not (
        (rx_cap >= 0 and np.any(rx_counts > rx_cap))
        or (fusion_cap >= 0 and np.any(fusion_counts > fusion_cap))
    )
