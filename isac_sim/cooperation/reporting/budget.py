"""budget（自 ``isac_sim/cooperation/reporting.py`` 拆出）。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains, Geometry, LinkTables


EPS = 1e-12


@dataclass
class ReportingPlan:
    """Per-target fusion UAV assignment.

    ``mode``      -- ``"tx"`` (legacy, destination is the sensing initiator) or
                     ``"explicit"`` (destination is ``f_q[q]``).
    ``f_q``       -- ``(Q,)`` integer array of fusion-UAV indices; ``-1`` when
                     the legacy mode is active.
    """

    mode: str = "tx"
    f_q: np.ndarray | None = None

    def destination(self, link: Link, q: int) -> int:
        """UAV that receives the report produced for target ``q`` on ``link``."""
        if self.mode == "explicit" and self.f_q is not None:
            return int(self.f_q[q])
        return int(link[0])  # legacy: report back to the sensing initiator i

    def is_explicit(self) -> bool:
        return self.mode == "explicit" and self.f_q is not None


def report_dest(plan: ReportingPlan | None, link: Link, q: int) -> int:
    """UAV that receives the report for target ``q`` on ``link``.

    ``plan=None`` means the legacy architecture: the report goes back to the
    sensing initiator ``i``.
    """
    if plan is None:
        return int(link[0])
    return int(plan.destination(link, q))


def is_local_observation(plan: ReportingPlan | None, link: Link, q: int) -> bool:
    """Return whether the echo receiver is already the fusion destination."""
    return int(link[1]) == report_dest(plan, link, q)


def report_rate(tables: LinkTables, plan: ReportingPlan | None, link: Link, q: int) -> float:
    """Reporting rate, with local evidence represented by an infinite-rate leg."""
    if is_local_observation(plan, link, q):
        return float("inf")
    return float(tables.rate[link[1], report_dest(plan, link, q)])


def report_chi(tables: LinkTables, plan: ReportingPlan | None, link: Link, q: int) -> float:
    """Packet-success probability; local evidence arrives with probability one."""
    if is_local_observation(plan, link, q):
        return 1.0
    return float(tables.chi_comm[link[1], report_dest(plan, link, q)])


def report_gamma(tables: LinkTables, plan: ReportingPlan | None, link: Link, q: int) -> float:
    """Reporting SINR, with local evidence represented by an infinite-SINR leg."""
    if is_local_observation(plan, link, q):
        return float("inf")
    return float(tables.gamma_comm[link[1], report_dest(plan, link, q)])
