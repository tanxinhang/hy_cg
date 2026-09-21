"""helpers（自 ``isac_sim/detection/oracle.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.cooperation.reporting import ReportingPlan, is_local_observation, report_dest


def _comb(n: int, r: int) -> int:
    """Number of r-combinations of n items (integer, exact)."""
    import math

    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def _selection_respects_hard_budgets(
    cfg: Config,
    selected: Dict[int, List[Link]],
    plan: ReportingPlan,
) -> bool:
    """Check every hard observation/report capacity of a complete plan.

    This predicate is shared by the fixed-plan and joint oracles.  It must
    therefore describe the same feasible set as the production selectors;
    otherwise an oracle gap can silently compare two different problems.
    """
    local_cap = int(cfg.selector.max_local_observations_per_target)
    if local_cap >= 0:
        for q, links in selected.items():
            local_count = sum(
                is_local_observation(plan, link, q) for link in links
            )
            if local_count > local_cap:
                return False

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
