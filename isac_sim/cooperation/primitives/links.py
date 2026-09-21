"""links（自 ``isac_sim/cooperation/primitives.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from isac_sim.core.config import Config, Link
from isac_sim.cooperation.reporting import is_local_observation, report_dest, report_rate
from isac_sim.sensing.model import BaseGains, EPS, LinkTables


def link_delay_s(cfg: Config, tables: LinkTables, q: int, link: Link, plan: "object | None" = None) -> float:
    """Latency of one report for target ``q`` on ``link``.

    The statistic is produced at the receiving UAV ``j`` and reported to the
    destination (``i`` in the legacy architecture, ``f_q`` in the explicit
    one); the latency honours ``cfg.comm.latency_model``.
    """
    from isac_sim.sensing.fbl import report_latency_s

    if is_local_observation(plan, link, q):
        return 0.0
    return report_latency_s(cfg, report_rate(tables, plan, link, q))


def link_cost_ms(cfg: Config, tables: LinkTables, q: int, link: Link, plan: "object | None" = None) -> float:
    return 1e3 * link_delay_s(cfg, tables, q, link, plan)


def feasible_links_for_target(
        cfg: Config, base: BaseGains, tables: LinkTables, q: int, plan: "object | None" = None
) -> List[Link]:
    """Links that satisfy range, DD-validity and communication constraints.

    The communication constraint is judged on the *reporting* leg ``j -> dest``
    (``dest`` = ``i`` legacy, ``f_q`` explicit), never on the sensing pair.
    """
    links: List[Link] = []
    for i in range(cfg.scale.M):
        for j in range(cfg.scale.M):
            if i == j:
                continue
            if cfg.dd.use_otfs_bin_validity and not base.valid_dd[i, j, q]:
                continue
            dest = report_dest(plan, (i, j), q)
            if dest < 0:
                continue
            if dest == j:
                links.append((i, j))
                continue
            if not base.edge_mask[j, dest]:
                continue
            if not tables.feasible_comm[dest, j]:
                continue
            links.append((i, j))
    return links


def sensing_only_links_for_target(cfg: Config, base: BaseGains, q: int) -> List[Link]:
    """Links that are usable for sensing only (communication constraint ignored)."""
    links: List[Link] = []
    for i in range(cfg.scale.M):
        for j in range(cfg.scale.M):
            if i == j:
                continue
            if cfg.dd.use_otfs_bin_validity and not base.valid_dd[i, j, q]:
                continue
            links.append((i, j))
    return links
