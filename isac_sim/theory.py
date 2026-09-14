"""Structural interpretation and small-scale audits for the greedy selector.

The reviewers' sharpest question about the proposed method was *"why greedy?"*.
A greedy heuristic is only defensible when its objective has structure.  This
module provides that structure, in three parts:

1. **Same-objective oracle.**  The paper-canonical greedy rule evaluates the
   exact marginal of the fair sensing potential

       U(P_D) = -Q tau log sum_q exp(-min(P_D,q,P_D,req)/tau)
                - mu/(2 P_D,req) sum_q [P_D,req-P_D,q]_+^2

   minus linear reporting cost.  The historical ``alpha_q * DeltaD`` rule is
   retained as a first-order ablation, not presented as the exact objective.

       F(S) = U(D(S)) - lambda_c * sum_{l in S} cost_l .

   ``task_objective`` evaluates exactly that ``F``, and
   ``same_objective_oracle`` exhaustively maximises it on a small instance, so
   the reported gap is on the *same* function the greedy optimises (unlike the
   older detection-only oracle, which dropped ``lambda_c`` and ``alpha_q``).

2. **Submodularity audit.**  The independent-observation special case has an
   additive deflection.  General correlation, the fair multi-target potential,
   and a positive reporting price do not inherit a blanket theorem.  The audit
   therefore checks the *implemented task objective* empirically and reports
   violations without promoting a finite sample to a proof.

3. **Curvature diagnostic.**  Curvature is reported only as an empirical shape
   diagnostic.  The per-target plus total link caps form a matroid-style
   constraint, for which ``1/(1+c)`` is the relevant classical reference under
   assumptions that need not hold here.  It is not labelled a guarantee.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

from .config import Config, Link
from .fusion import (
    deflection_for_links,
    predicted_pd_for_links,
    selection_utility,
    selection_utility_from_pd,
)
from .model import BaseGains, LinkTables
from .selection import feasible_links_for_target, link_cost_ms


# ==========================================================================
# The objective the greedy actually optimises
# ==========================================================================
def task_objective(
    cfg: Config,
    tables: LinkTables,
    selected: Dict[int, List[Link]],
    plan: "object | None" = None,
    base: BaseGains | None = None,
) -> float:
    """Exact configured sensing utility minus linear reporting cost."""
    s = cfg.selector
    D = np.zeros(cfg.scale.Q, dtype=float)
    pd = np.zeros(cfg.scale.Q, dtype=float)
    total_cost = 0.0
    for q in range(cfg.scale.Q):
        links = selected.get(q, [])
        D[q] = deflection_for_links(
            cfg, tables, q, links, weight_mode="deflection", plan=plan, base=base
        )
        if s.score_mode.lower() == "detector_pd":
            pd[q] = predicted_pd_for_links(
                cfg, tables, q, links, weight_mode="deflection", plan=plan, base=base
            )
        if s.use_delay_price:
            for link in links:
                total_cost += s.lambda_c * link_cost_ms(cfg, tables, q, link, plan)
    utility = (
        selection_utility_from_pd(cfg, D, pd)
        if s.score_mode.lower() == "detector_pd"
        else selection_utility(cfg, D)
    )
    return utility - total_cost


def same_objective_oracle(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    plan: "object | None" = None,
) -> Tuple[Dict[int, List[Link]], float]:
    """Exhaustively maximise :func:`task_objective` under the link budgets.

    This is the *same-objective* counterpart of
    :func:`isac_sim.oracle.oracle_exhaustive` (which maximised ``sum_q D_q``).
    """
    Q = cfg.scale.Q
    K_per = cfg.selector.max_links_per_target
    K_tot = cfg.selector.max_total_links

    q_candidates: List[List[Link]] = [
        feasible_links_for_target(cfg, base, tables, q, plan) for q in range(Q)
    ]

    tree_size = 1
    for cand in q_candidates:
        n = min(K_per, len(cand))
        tree_size *= sum(_comb(len(cand), r) for r in range(0, n + 1))
    if tree_size > 200_000_000:
        raise RuntimeError(f"same-objective oracle tree too large ({tree_size}); use M<=4, Q<=3")

    best_selected: Dict[int, List[Link]] = {q: [] for q in range(Q)}
    best_obj = -np.inf

    def _dfs(q: int, selected: Dict[int, List[Link]], total_links: int) -> None:
        nonlocal best_obj, best_selected
        if q == Q:
            obj = task_objective(cfg, tables, selected, plan, base)
            if obj > best_obj:
                best_obj = float(obj)
                best_selected = {qq: list(selected[qq]) for qq in range(Q)}
            return
        cand = q_candidates[q]
        for r in range(0, min(K_per, len(cand)) + 1):
            if total_links + r > K_tot:
                break
            for subset in itertools.combinations(cand, r):
                selected[q] = list(subset)
                _dfs(q + 1, selected, total_links + r)
        selected[q] = []

    _dfs(0, {q: [] for q in range(Q)}, 0)
    return best_selected, float(best_obj)


def _comb(n: int, r: int) -> int:
    import math

    return math.comb(n, r) if 0 <= r <= n else 0


# ==========================================================================
# Submodularity audit + curvature
# ==========================================================================
def _ground_elements(
    cfg: Config, base: BaseGains, tables: LinkTables, plan: "object | None"
) -> List[tuple[int, Link]]:
    return [
        (q, link)
        for q in range(cfg.scale.Q)
        for link in feasible_links_for_target(cfg, base, tables, q, plan)
    ]


def _selection_from_elements(
    cfg: Config, elements: List[tuple[int, Link]]
) -> Dict[int, List[Link]]:
    selected: Dict[int, List[Link]] = {q: [] for q in range(cfg.scale.Q)}
    for q, link in elements:
        selected[q].append(link)
    return selected


def _audited_value(cfg, tables, elements, plan, base) -> float:
    return task_objective(
        cfg, tables, _selection_from_elements(cfg, elements), plan=plan, base=base
    )


def submodularity_audit(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    plan: "object | None" = None,
    n_samples: int = 500,
    seed: int = 0,
) -> Dict[str, float]:
    """Empirical diminishing-returns check on the implemented task objective.

    For random ``A subset B`` and ``e notin B``, submodularity requires
    ``F(A u e) - F(A) >= F(B u e) - F(B)``.  Returns the fraction of violations
    and the average (and minimum) marginal-gain ratio.
    """
    rng = np.random.default_rng(seed)
    pool = _ground_elements(cfg, base, tables, plan)
    monotone_violations = 0
    submod_violations = 0
    ratios: List[float] = []
    n_checks = 0

    for _ in range(n_samples):
        n_pool = len(pool)
        if n_pool < 3:
            continue
        k = int(rng.integers(1, n_pool))
        b_idx = rng.choice(n_pool, size=k, replace=False)
        B = [pool[int(x)] for x in b_idx]
        A_size = int(rng.integers(0, k))
        a_idx = rng.choice(k, size=A_size, replace=False) if A_size else []
        A = [B[int(x)] for x in a_idx]
        rest = [e for e in pool if e not in set(B)]
        if not rest:
            continue
        e = [rest[int(rng.integers(0, len(rest)))]]

        F_A = _audited_value(cfg, tables, A, plan, base)
        F_Ae = _audited_value(cfg, tables, A + e, plan, base)
        F_B = _audited_value(cfg, tables, B, plan, base)
        F_Be = _audited_value(cfg, tables, B + e, plan, base)

        n_checks += 1
        if F_Ae < F_A - 1e-12 or F_Be < F_B - 1e-12:
            monotone_violations += 1
        mg_A = F_Ae - F_A
        mg_B = F_Be - F_B
        if mg_A < mg_B - 1e-9:
            submod_violations += 1
        if mg_B > 1e-9:
            ratios.append(float(mg_A / mg_B))

    if n_checks == 0:
        return {"n_checks": 0.0, "monotone_violation_rate": 0.0,
                "submodularity_violation_rate": 0.0, "curvature": 0.0,
                "matroid_reference_bound": 1.0}

    curvature = float(empirical_curvature(cfg, base, tables, plan))
    return {
        "n_checks": float(n_checks),
        "monotone_violation_rate": float(monotone_violations / n_checks),
        "submodularity_violation_rate": float(submod_violations / n_checks),
        "mean_marginal_ratio": float(np.mean(ratios)) if ratios else 1.0,
        "min_marginal_ratio": float(np.min(ratios)) if ratios else 1.0,
        "curvature": curvature,
        "matroid_reference_bound": float(curvature_reference_bound(curvature)),
    }


def empirical_curvature(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    plan: "object | None" = None,
) -> float:
    r"""Empirical curvature of the normalized implemented task objective.

    For a monotone submodular ``F`` with ground set ``V``,

        c = 1 - min_e ( F(V) - F(V \ {e}) ) / F({e}).

    ``c = 0`` indicates modular behavior and ``c -> 1`` strong diminishing
    returns.  The diagnostic is computed on the finite audited ground set and
    is not promoted to a theorem for correlated observations.
    """
    V = _ground_elements(cfg, base, tables, plan)
    if len(V) < 2:
        return 0.0
    F0 = _audited_value(cfg, tables, [], plan, base)
    F_V = _audited_value(cfg, tables, V, plan, base) - F0
    c = 0.0
    n = 0
    for e in V:
        F_e = _audited_value(cfg, tables, [e], plan, base) - F0
        if F_e <= 1e-12:
            continue
        V_no_e = [x for x in V if x != e]
        last_marginal = F_V - (_audited_value(cfg, tables, V_no_e, plan, base) - F0)
        c = max(c, 1.0 - last_marginal / F_e)
        n += 1
    return float(np.clip(c, 0.0, 1.0)) if n else 0.0


def curvature_reference_bound(curvature: float) -> float:
    """Classical ``1/(1+c)`` matroid reference, not a guarantee for this task."""
    c = float(np.clip(curvature, 0.0, 1.0))
    return 1.0 / (1.0 + c)


def greedy_guarantee(curvature: float) -> float:
    """Backward-compatible alias for the non-claiming curvature reference."""
    return curvature_reference_bound(curvature)
