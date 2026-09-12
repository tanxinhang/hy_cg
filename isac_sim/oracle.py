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


def oracle_exhaustive(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
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
        feasible_links_for_target(cfg, base, tables, q) for q in range(Q)
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
            obj = sum(
                deflection_for_links(cfg, tables, qq, selected[qq], weight_mode="deflection")
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
) -> float:
    """The same ``sum_q D_q`` objective evaluated on a greedy selection."""
    return float(
        sum(
            deflection_for_links(cfg, tables, q, selected.get(q, []), weight_mode="deflection")
            for q in range(cfg.scale.Q)
        )
    )