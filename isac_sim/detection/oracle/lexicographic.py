"""lexicographic（自 ``isac_sim/detection/oracle.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains, LinkTables
from isac_sim.cooperation.reporting import ReportingPlan, is_local_observation, report_dest

from isac_sim.detection.oracle.helpers import _selection_respects_hard_budgets


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
    from isac_sim.detection.fusion import predicted_pd_for_links

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
