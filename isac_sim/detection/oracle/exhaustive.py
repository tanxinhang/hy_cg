"""exhaustive（自 ``isac_sim/detection/oracle.py`` 拆出）。"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.detection.fusion import deflection_for_links
from isac_sim.sensing.model import BaseGains, LinkTables
from isac_sim.cooperation.primitives import feasible_links_for_target
from isac_sim.cooperation.reporting import ReportingPlan, is_local_observation, report_dest

from isac_sim.detection.oracle.helpers import _comb, _selection_respects_hard_budgets


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
