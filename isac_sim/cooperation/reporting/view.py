"""view（自 ``isac_sim/cooperation/reporting.py`` 拆出）。"""

from __future__ import annotations

from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains, Geometry, LinkTables

from isac_sim.cooperation.reporting.budget import ReportingPlan, is_local_observation, report_chi, report_gamma, report_rate


class ReportingView:
    """Reporting-side quantities of a link, resolved through a plan.

    Wraps the ``(M, M)`` communication tables so that the rest of the code
    never has to know whether the destination is ``i`` or ``f_q``.
    """

    def __init__(self, cfg: Config, base: BaseGains, tables: LinkTables, plan: ReportingPlan):
        self.cfg = cfg
        self.base = base
        self.tables = tables
        self.plan = plan

    # -- elementary accessors -------------------------------------------
    def rate(self, link: Link, q: int) -> float:
        return report_rate(self.tables, self.plan, link, q)

    def chi(self, link: Link, q: int) -> float:
        return report_chi(self.tables, self.plan, link, q)

    def gamma(self, link: Link, q: int) -> float:
        return report_gamma(self.tables, self.plan, link, q)

    def feasible(self, link: Link, q: int) -> bool:
        """Range + rate + (optional) reliability feasibility of the report."""
        j = link[1]
        m = self.plan.destination(link, q)
        if m < 0:
            return False
        if m == j:
            return True
        if not self.base.edge_mask[j, m]:
            return False
        # NOTE: ``feasible_comm[a, b]`` records the feasibility of the b -> a
        # leg, so the j -> m leg is stored at ``feasible_comm[m, j]``.
        return bool(self.tables.feasible_comm[m, j])

    def latency_s(self, link: Link, q: int) -> float:
        from isac_sim.sensing.fbl import report_latency_s

        if is_local_observation(self.plan, link, q):
            return 0.0
        return report_latency_s(self.cfg, self.rate(link, q))

    def transmitter(self, link: Link, q: int) -> int:
        """The UAV that actually transmits the report (used by ``active_set``)."""
        return int(link[1])
