"""caps（自 ``isac_sim/cooperation/primitives.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from isac_sim.core.config import Config, Link
from isac_sim.cooperation.reporting import is_local_observation, report_dest, report_rate


def local_cap_allows(
        cfg: Config,
        selected: List[Link],
        link: Link,
        q: int,
        plan: "object | None",
) -> bool:
    """Whether ``link`` respects the per-target local-evidence audit cap."""
    if not is_local_observation(plan, link, q):
        return True
    cap = int(cfg.selector.max_local_observations_per_target)
    if cap < 0:
        return True
    used = sum(is_local_observation(plan, chosen, q) for chosen in selected)
    return used < cap


def remote_cap_allows(
        cfg: Config,
        selected: Dict[int, List[Link]],
        link: Link,
        q: int,
        plan: "object | None",
) -> bool:
    """Whether adding ``link`` respects the global remote-report hard cap."""
    if is_local_observation(plan, link, q):
        return True
    cap = int(cfg.selector.max_remote_reports)
    if cap < 0:
        return True
    used = sum(
        not is_local_observation(plan, chosen, qq)
        for qq, links in selected.items()
        for chosen in links
    )
    return used < cap


def processing_caps_allow(
        cfg: Config,
        selected: Dict[int, List[Link]],
        link: Link,
        q: int,
        plan: "object | None",
) -> bool:
    """Whether adding an observation respects receiver and fusion capacities."""
    rx_cap = int(cfg.selector.max_observations_per_receiver)
    if rx_cap >= 0:
        receiver = int(link[1])
        rx_used = sum(
            int(chosen[1]) == receiver
            for links in selected.values()
            for chosen in links
        )
        if rx_used >= rx_cap:
            return False

    fusion_cap = int(cfg.selector.max_observations_per_fusion_uav)
    if fusion_cap >= 0:
        destination = report_dest(plan, link, q)
        fusion_used = sum(
            report_dest(plan, chosen, qq) == destination
            for qq, links in selected.items()
            for chosen in links
        )
        if fusion_used >= fusion_cap:
            return False
    return True
