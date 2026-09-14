"""Structural guarantees for the greedy link selector.

The reviewers' sharpest question about the proposed method was *"why greedy?"*.
A greedy heuristic is only defensible when its objective has structure.  This
module provides that structure, in three parts:

1. **Same-objective oracle.**  The greedy rule

       score = alpha_q * DeltaD - lambda_c * cost

   with ``alpha_q = d P_D / d D_q`` is precisely the *first-order* greedy step
   on the scalar objective

       F(S) = sum_q P_D(D_q(S_q)) - lambda_c * sum_{l in S} cost_l .

   ``task_objective`` evaluates exactly that ``F``, and
   ``same_objective_oracle`` exhaustively maximises it on a small instance, so
   the reported gap is on the *same* function the greedy optimises (unlike the
   older detection-only oracle, which dropped ``lambda_c`` and ``alpha_q``).

2. **Submodularity audit.**  ``P_D(D)`` is a monotone concave function of the
   deflection ``D``, and with deflection-optimal weights and independent
   observations ``D_q`` is *additive* (``sum_l delta_l^2/sigma_l^2``).  A
   monotone concave function of a modular function is submodular, so ``F``
   should be submodular (monotone) when the link cost is linear.  The audit
   verifies this empirically: it samples set pairs ``A subset B`` and elements
   ``e notin B`` and counts diminishing-returns violations.

3. **Curvature bound.**  For a monotone submodular ``F`` with total curvature
   ``c in [0, 1]``, the cost-agnostic greedy achieves

       F(S_greedy) >= (1/c)(1 - e^{-c}) * F(S*),

   which interpolates between the ``(1 - 1/e)`` guarantee at ``c = 1`` and
   exact optimality at ``c = 0`` (modular).  ``empirical_curvature`` estimates
   ``c`` on the single-target ground set.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

from .config import Config, Link
from .fusion import deflection_for_links
from .model import BaseGains, LinkTables, pd_from_deflection
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
    """``sum_q P_D(D_q) - lambda_c * sum_l cost_l`` -- the greedy's surrogate."""
    s = cfg.selector
    total = 0.0
    for q in range(cfg.scale.Q):
        links = selected.get(q, [])
        D = deflection_for_links(cfg, tables, q, links, weight_mode="deflection", plan=plan, base=base)
        total += float(pd_from_deflection(cfg, D))
        if s.use_delay_price:
            for link in links:
                total -= s.lambda_c * link_cost_ms(cfg, tables, q, link, plan)
    return total


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
def _single_target_value(
    cfg: Config, tables, q: int, subset: List[Link], plan, base
) -> float:
    D = deflection_for_links(cfg, tables, q, subset, weight_mode="deflection", plan=plan, base=base)
    return float(pd_from_deflection(cfg, D))


def submodularity_audit(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    plan: "object | None" = None,
    n_samples: int = 500,
    seed: int = 0,
) -> Dict[str, float]:
    """Empirical diminishing-returns check on the per-target P_D objective.

    For random ``A subset B`` and ``e notin B``, submodularity requires
    ``F(A u e) - F(A) >= F(B u e) - F(B)``.  Returns the fraction of violations
    and the average (and minimum) marginal-gain ratio.
    """
    rng = np.random.default_rng(seed)
    Q = cfg.scale.Q
    monotone_violations = 0
    submod_violations = 0
    ratios: List[float] = []
    n_checks = 0

    for _ in range(n_samples):
        q = int(rng.integers(0, Q))
        cand = feasible_links_for_target(cfg, base, tables, q, plan)
        if len(cand) < 3:
            continue
        pool = list(cand)
        n_pool = len(pool)
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

        F_A = _single_target_value(cfg, tables, q, A, plan, base)
        F_Ae = _single_target_value(cfg, tables, q, A + e, plan, base)
        F_B = _single_target_value(cfg, tables, q, B, plan, base)
        F_Be = _single_target_value(cfg, tables, q, B + e, plan, base)

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
                "submodularity_violation_rate": 0.0, "curvature": 0.0, "greedy_guarantee": 1.0}

    curvature = float(empirical_curvature(cfg, base, tables, plan))
    return {
        "n_checks": float(n_checks),
        "monotone_violation_rate": float(monotone_violations / n_checks),
        "submodularity_violation_rate": float(submod_violations / n_checks),
        "mean_marginal_ratio": float(np.mean(ratios)) if ratios else 1.0,
        "min_marginal_ratio": float(np.min(ratios)) if ratios else 1.0,
        "curvature": curvature,
        "greedy_guarantee": float(greedy_guarantee(curvature)),
    }


def empirical_curvature(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    plan: "object | None" = None,
) -> float:
    r"""Total curvature of the single-target P_D objective, averaged over targets.

    For a monotone submodular ``F`` with ground set ``V``,

        c = 1 - min_e ( F(V) - F(V \ {e}) ) / F({e}).

    ``c = 0`` iff ``F`` is modular (greedy is then exact); ``c -> 1`` gives the
    worst-case ``1 - 1/e`` guarantee.  Computed exactly over each target's
    candidate list -- tractable because ``M`` is small in the audit regime.
    """
    Q = cfg.scale.Q
    total_c = 0.0
    n = 0
    for q in range(Q):
        V = feasible_links_for_target(cfg, base, tables, q, plan)
        if len(V) < 2:
            continue
        F_V = _single_target_value(cfg, tables, q, V, plan, base)
        c_q = 0.0
        for e in V:
            Fe = _single_target_value(cfg, tables, q, [e], plan, base)
            if Fe <= 1e-12:
                continue
            V_no_e = [x for x in V if x != e]
            F_Vne = _single_target_value(cfg, tables, q, V_no_e, plan, base)
            c_q = max(c_q, 1.0 - (F_V - F_Vne) / Fe)
        total_c += float(np.clip(c_q, 0.0, 1.0))
        n += 1
    return total_c / n if n else 0.0


def greedy_guarantee(curvature: float) -> float:
    """``(1/c)(1 - e^{-c})`` approximation factor for curvature ``c``."""
    c = float(np.clip(curvature, 1e-9, 1.0))
    if c < 1e-9:
        return 1.0
    return (1.0 - np.exp(-c)) / c
