"""joint（自 ``isac_sim/detection/oracle.py`` 拆出）。"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains, LinkTables
from isac_sim.cooperation.primitives import feasible_links_for_target
from isac_sim.cooperation.reporting import ReportingPlan, is_local_observation, report_dest

from isac_sim.detection.oracle.helpers import _comb, _selection_respects_hard_budgets
from isac_sim.detection.oracle.lexicographic import lexicographic_objective_vector


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
