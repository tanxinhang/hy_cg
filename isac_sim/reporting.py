"""Reporting architecture: where the soft statistics of a target meet.

The sensing chain is ``i -> q -> j``: UAV ``i`` illuminates, UAV ``j`` receives
the echo and therefore *produces* the soft statistic.  The statistic then has
to travel somewhere to be fused.

Legacy model (``fusion.mode = "tx"``)
-------------------------------------
The statistic is reported back to the sensing initiator ``i``.  That reuses the
ordered sensing pair ``(i, j)`` as the reporting relation, which is the
shortcut the reviewers attacked: the same ``(i, j)`` simultaneously means "a
bistatic geometry" and "a communication link", so ``chi_ij``, ``R_ij`` and the
reporting delay are properties of a pair that has no physical reason to carry
them.

Explicit model (``fusion.mode = "explicit"``)
---------------------------------------------
Each target ``q`` is assigned an explicit fusion UAV ``f_q`` and *every*
statistic about that target is reported ``j -> f_q``.  The triplet becomes

    (i, j, q; f_q)

and the communication quantities that matter are

    chi_{j f_q},   R_{j f_q},   T_{j f_q},

none of which has anything to do with ``i``.  The sensing topology and the
reporting topology are then fully decoupled, which is also the structure used
by the networked-sensing literature (sensing nodes / backhaul / fusion centre).

Crucially this also closes the hole that the previous fix left open: with
``j -> i``, three selected links ``(i1,j1,q), (i2,j2,q), (i3,j3,q)`` deliver
their statistics to three *different* UAVs, yet the fusion step silently
assumed they had already met.  With a single ``f_q`` they genuinely do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import Config, Link
from .model import BaseGains, Geometry, LinkTables

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


def slot_schedule(
    cfg: Config,
    selected: Dict[int, List[Link]],
    plan: "ReportingPlan | None" = None,
) -> List[List[tuple]]:
    """Conflict-graph colouring of the selected reporting legs.

    Returns a list of slots; each slot is a list of ``(q, i, j, dest)`` reports
    that can be transmitted concurrently.  Two reports conflict when they share
    the same transmitter ``j`` (a UAV cannot send two reports at once) or the
    same destination ``dest`` (a receiver decodes one report at a time in a
    plain FDMA-free model).  Greedy first-fit colouring is used.

    This is the MAC model that makes the reporting interference and the latency
    describe *one* system: only reports inside the same slot co-interfere, and
    the slots run back to back.
    """
    reports: List[tuple] = [
        (q, i, j, report_dest(plan, (i, j), q))
        for q, links in selected.items() for (i, j) in links
    ]
    slots: List[List[tuple]] = []
    for (q, i, j, dest) in reports:
        placed = False
        for slot in slots:
            if all(d != dest and j != jj for (_, _, jj, d) in slot):
                slot.append((q, i, j, dest))
                placed = True
                break
        if not placed:
            slots.append([(q, i, j, dest)])
    return slots


def assign_fusion_nodes(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    geom: "Geometry | None" = None,
) -> ReportingPlan:
    """Pick one fusion UAV per target according to ``cfg.fusion``.

    Returns a :class:`ReportingPlan`.  With ``cfg.fusion.mode = "tx"`` the plan
    is a no-op wrapper that reproduces the legacy destination, so every frozen
    result stays bit-exact.
    """
    Q = cfg.scale.Q
    M = cfg.scale.M
    if cfg.fusion.mode.lower() != "explicit":
        return ReportingPlan(mode="tx", f_q=None)

    rule = cfg.fusion.rule.lower()
    f_q = np.full(Q, -1, dtype=int)

    for q in range(Q):
        # Candidate receivers: every UAV that can act as the echo receiver of
        # some feasible sensing pair for this target.
        cand_rx = [
            j for j in range(M)
            if any(
                base.edge_mask[i, j] and (not cfg.dd.use_otfs_bin_validity or base.valid_dd[i, j, q])
                for i in range(M) if i != j
            )
        ]
        if not cand_rx:
            f_q[q] = -1
            continue

        best_m, best_score = -1, -np.inf
        for m in range(M):
            rates = np.array([
                tables.rate[j, m] if base.edge_mask[j, m] else 0.0 for j in cand_rx
            ])
            reachable = np.array([base.edge_mask[j, m] for j in cand_rx], dtype=bool)
            if not np.any(reachable):
                continue
            if rule == "max_min_rate":
                score = float(np.min(rates[reachable]))
            elif rule == "nearest_centroid":
                if geom is None:
                    score = 0.0
                else:
                    score = -float(np.linalg.norm(geom.p_uav[m] - geom.p_tgt[q]))
            else:  # "max_in_rate"
                score = float(np.sum(rates[reachable]))
            if score > best_score:
                best_score, best_m = score, m
        f_q[q] = best_m

    return ReportingPlan(mode="explicit", f_q=f_q)


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
        return float(self.tables.rate[link[1], self.plan.destination(link, q)])

    def chi(self, link: Link, q: int) -> float:
        return float(self.tables.chi_comm[link[1], self.plan.destination(link, q)])

    def gamma(self, link: Link, q: int) -> float:
        return float(self.tables.gamma_comm[link[1], self.plan.destination(link, q)])

    def feasible(self, link: Link, q: int) -> bool:
        """Range + rate + (optional) reliability feasibility of the report."""
        j = link[1]
        m = self.plan.destination(link, q)
        if m < 0 or m == j or not self.base.edge_mask[j, m]:
            return False
        # NOTE: ``feasible_comm[a, b]`` records the feasibility of the b -> a
        # leg, so the j -> m leg is stored at ``feasible_comm[m, j]``.
        return bool(self.tables.feasible_comm[m, j])

    def latency_s(self, link: Link, q: int) -> float:
        from .fbl import report_latency_s

        return report_latency_s(self.cfg, self.rate(link, q))

    def transmitter(self, link: Link, q: int) -> int:
        """The UAV that actually transmits the report (used by ``active_set``)."""
        return int(link[1])
