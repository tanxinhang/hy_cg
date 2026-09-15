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
        if not is_local_observation(plan, (i, j), q)
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
    supported_rules = {
        "max_in_rate", "max_min_rate", "nearest_target", "nearest_centroid",
        "nearest_target_capacitated",
        "capacitated_value", "capacitated_pd_lookahead",
    }
    if rule not in supported_rules:
        raise ValueError(
            f"Unknown fusion.rule={cfg.fusion.rule!r}; expected one of "
            f"{sorted(supported_rules)}"
        )
    if rule in {"nearest_target", "nearest_centroid", "nearest_target_capacitated"}:
        if geom is None:
            raise ValueError(
                f"fusion.rule={cfg.fusion.rule!r} requires predicted target geometry"
            )
        if geom.p_uav.shape != (M, 3) or geom.p_tgt.shape != (Q, 3):
            raise ValueError(
                "predicted geometry shape does not match scale.M/scale.Q: "
                f"p_uav={geom.p_uav.shape}, p_tgt={geom.p_tgt.shape}, M={M}, Q={Q}"
            )
        if not np.all(np.isfinite(geom.p_uav)) or not np.all(np.isfinite(geom.p_tgt)):
            raise ValueError("predicted fusion-planning geometry must be finite")
    if rule == "capacitated_value":
        return _assign_capacitated_value(cfg, base, tables)
    if rule == "capacitated_pd_lookahead":
        return _assign_capacitated_pd_lookahead(cfg, base, tables)
    if rule == "nearest_target_capacitated":
        values = -np.linalg.norm(
            geom.p_tgt[:, None, :] - geom.p_uav[None, :, :], axis=2
        )
        return ReportingPlan(
            mode="explicit", f_q=_solve_capacity_assignment(cfg, values)
        )

    f_q = np.full(Q, -1, dtype=int)

    for q in range(Q):
        # Candidate receivers: every UAV that can act as the echo receiver of
        # some feasible sensing pair for this target.
        cand_rx = [
            j for j in range(M)
            if any(
                not cfg.dd.use_otfs_bin_validity or base.valid_dd[i, j, q]
                for i in range(M) if i != j
            )
        ]
        if not cand_rx:
            f_q[q] = -1
            continue

        best_m, best_score = -1, -np.inf
        for m in range(M):
            rates = np.array([
                tables.rate[j, m] if j != m and base.edge_mask[j, m] else 0.0
                for j in cand_rx
            ])
            reachable = np.array([
                (j == m) or base.edge_mask[j, m] for j in cand_rx
            ], dtype=bool)
            if not np.any(reachable):
                continue
            # A receiver's own sensing observation is local evidence, not a
            # zero-rate reporting leg.  It therefore establishes receiver
            # eligibility but must not depress max-min reporting rate or add
            # fictitious capacity to the max-in-rate score.
            remote_reachable = np.array([
                j != m and base.edge_mask[j, m] for j in cand_rx
            ], dtype=bool)
            if rule == "max_min_rate":
                score = (
                    float(np.min(rates[remote_reachable]))
                    if np.any(remote_reachable) else 0.0
                )
            elif rule in {"nearest_target", "nearest_centroid"}:
                # In belief mode ``geom`` is the predicted geometry, not the
                # current-CPI target truth.  This locality rule thus avoids
                # truth leakage while distributing target-specific fusion
                # traffic across the fleet.
                score = -float(np.linalg.norm(geom.p_uav[m] - geom.p_tgt[q]))
            else:  # validated "max_in_rate"
                score = float(np.sum(rates[remote_reachable]))
            if score > best_score:
                best_score, best_m = score, m
        f_q[q] = best_m

    return ReportingPlan(mode="explicit", f_q=f_q)


def _assign_capacitated_value(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
) -> ReportingPlan:
    """Solve the target--fusion assignment under a hard target capacity.

    For each pair ``(q,f)``, the value is the sum of the strongest feasible
    singleton deflections, capped by the per-target observation budget.  This
    proxy accounts for local evidence, report feasibility, packet reliability,
    and sensing quality without introducing a hand-tuned scalar price.  The
    fixed-utility capacitated assignment subproblem is solved exactly by
    expanding each UAV into identical capacity slots and applying linear-sum
    assignment inside the optimal bottleneck tier. Observation selection
    remains the fast inner problem, so this is a decomposition rather than an
    exact solution of the joint sensing problem.
    """
    from .fusion import deflection_for_links

    Q, M = cfg.scale.Q, cfg.scale.M
    values = np.full((Q, M), -np.inf, dtype=float)
    per_target_budget = max(int(cfg.selector.max_links_per_target), 0)
    for q in range(Q):
        for f in range(M):
            plan = ReportingPlan(mode="explicit", f_q=np.full(Q, f, dtype=int))
            candidates: List[Link] = []
            for i in range(M):
                for j in range(M):
                    if i == j:
                        continue
                    if cfg.dd.use_otfs_bin_validity and not base.valid_dd[i, j, q]:
                        continue
                    if j != f and (
                        not base.edge_mask[j, f]
                        or not tables.feasible_comm[f, j]
                    ):
                        continue
                    candidates.append((i, j))
            if not candidates or per_target_budget == 0:
                continue
            singleton_values = sorted(
                (
                    deflection_for_links(
                        cfg, tables, q, [link], weight_mode="deflection",
                        plan=plan, base=base,
                    ),
                    link,
                )
                for link in candidates
            )
            chosen_values: List[float] = []
            local_used = 0
            local_cap = int(cfg.selector.max_local_observations_per_target)
            fusion_cap = int(cfg.selector.max_observations_per_fusion_uav)
            pair_budget = per_target_budget if fusion_cap < 0 else min(per_target_budget, fusion_cap)
            if pair_budget <= 0:
                continue
            for singleton, link in reversed(singleton_values):
                local = int(link[1]) == f
                if local and local_cap >= 0 and local_used >= local_cap:
                    continue
                chosen_values.append(float(singleton))
                local_used += int(local)
                if len(chosen_values) >= pair_budget:
                    break
            if chosen_values:
                values[q, f] = float(sum(chosen_values))

    if np.any(~np.isfinite(np.max(values, axis=1))):
        missing = np.flatnonzero(~np.isfinite(np.max(values, axis=1))).tolist()
        raise ValueError(f"no feasible fusion destination for targets {missing}")

    # Target-wise normalization prevents intrinsically easy targets with very
    # large deflection scales from dominating the assignment.  The bottleneck
    # stage first maximizes the weakest target's retained fraction of its best
    # standalone fusion value; a sum-value assignment only breaks ties inside
    # that optimal bottleneck tier.
    row_max = np.max(values, axis=1)
    relative_values = np.full_like(values, -np.inf)
    positive_rows = row_max > 0.0
    relative_values[positive_rows] = (
        values[positive_rows] / row_max[positive_rows, None]
    )
    for q in np.flatnonzero(~positive_rows):
        relative_values[q, np.isfinite(values[q])] = 1.0
    return ReportingPlan(
        mode="explicit",
        f_q=_solve_bottleneck_capacity_assignment(cfg, relative_values),
    )


def _assign_capacitated_pd_lookahead(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
) -> ReportingPlan:
    """Assign fusion nodes using a detector-level selection lookahead.

    For every fixed ``(target, fusion UAV)`` pair, this routine greedily
    replays the downstream detector-PD marginal under the per-target local,
    fusion-observation, receiver, and an equal-share remote-report allowance.
    It then lexicographically maximizes the minimum and total predicted PD
    across the fixed-utility capacitated assignment.  No communication price
    or fitted scalar weight is introduced.

    The equal-share allowance is ``ceil(K_remote / Q)``.  It is only a
    tractable assignment lookahead; the subsequent global selector still
    enforces the exact shared budgets and may allocate reports unevenly.
    """
    from .fusion import predicted_pd_for_links

    Q, M = cfg.scale.Q, cfg.scale.M
    per_target_cap = max(int(cfg.selector.max_links_per_target), 0)
    fusion_cap = int(cfg.selector.max_observations_per_fusion_uav)
    if fusion_cap >= 0:
        per_target_cap = min(per_target_cap, fusion_cap)
    remote_cap = int(cfg.selector.max_remote_reports)
    remote_share = per_target_cap if remote_cap < 0 else int(np.ceil(remote_cap / max(Q, 1)))
    local_cap = int(cfg.selector.max_local_observations_per_target)
    rx_cap = int(cfg.selector.max_observations_per_receiver)
    shortlist = max(int(cfg.selector.candidate_topk_per_target), per_target_cap, 1)
    values = np.full((Q, M), -np.inf, dtype=float)

    for q in range(Q):
        for f in range(M):
            plan = ReportingPlan(mode="explicit", f_q=np.full(Q, f, dtype=int))
            local_scored: List[tuple[float, Link]] = []
            remote_scored: List[tuple[float, Link]] = []
            for i in range(M):
                for j in range(M):
                    if i == j:
                        continue
                    if cfg.dd.use_otfs_bin_validity and not base.valid_dd[i, j, q]:
                        continue
                    if j != f and (
                        not base.edge_mask[j, f] or not tables.feasible_comm[f, j]
                    ):
                        continue
                    link = (i, j)
                    singleton_pd = predicted_pd_for_links(
                        cfg,
                        tables,
                        q,
                        [link],
                        weight_mode="deflection",
                        plan=plan,
                        base=base,
                    )
                    bucket = local_scored if j == f else remote_scored
                    bucket.append((float(singleton_pd), link))

            # Preserve both evidence types in the shortlist. A global top-K
            # could otherwise remove the local anchor before the replay.
            local_scored.sort(reverse=True)
            remote_scored.sort(reverse=True)
            candidates = [link for _, link in local_scored[:shortlist]]
            candidates.extend(link for _, link in remote_scored[:shortlist])
            if not candidates or per_target_cap == 0:
                continue

            chosen: List[Link] = []
            receiver_counts = np.zeros(M, dtype=int)
            remote_used = 0
            current_pd = 0.0
            while len(chosen) < per_target_cap:
                best_link: Link | None = None
                best_pd = current_pd
                for link in candidates:
                    if link in chosen:
                        continue
                    is_local = int(link[1]) == f
                    if is_local and local_cap >= 0:
                        local_used = sum(int(existing[1]) == f for existing in chosen)
                        if local_used >= local_cap:
                            continue
                    if not is_local and remote_used >= remote_share:
                        continue
                    if rx_cap >= 0 and receiver_counts[link[1]] >= rx_cap:
                        continue
                    pd = predicted_pd_for_links(
                        cfg,
                        tables,
                        q,
                        chosen + [link],
                        weight_mode="deflection",
                        plan=plan,
                        base=base,
                    )
                    if pd > best_pd + EPS:
                        best_pd = float(pd)
                        best_link = link
                if best_link is None:
                    break
                chosen.append(best_link)
                receiver_counts[best_link[1]] += 1
                remote_used += int(best_link[1] != f)
                current_pd = best_pd
            if chosen:
                values[q, f] = current_pd

    if np.any(~np.isfinite(np.max(values, axis=1))):
        missing = np.flatnonzero(~np.isfinite(np.max(values, axis=1))).tolist()
        raise ValueError(f"no feasible lookahead fusion destination for targets {missing}")
    return ReportingPlan(
        mode="explicit",
        f_q=_solve_bottleneck_capacity_assignment(cfg, values),
    )


def _solve_capacity_assignment(cfg: Config, values: np.ndarray) -> np.ndarray:
    """Maximize fixed target--UAV utilities under a per-UAV target cap."""
    from scipy.optimize import linear_sum_assignment

    Q, M = values.shape
    cap = int(cfg.fusion.max_targets_per_uav)
    if cap < 0:
        cap = Q
    if M * cap < Q:
        raise ValueError(
            "fusion target capacity is infeasible: "
            f"M*max_targets_per_uav={M * cap} < Q={Q}"
        )
    if np.any(~np.isfinite(np.max(values, axis=1))):
        missing = np.flatnonzero(~np.isfinite(np.max(values, axis=1))).tolist()
        raise ValueError(f"no feasible fusion destination for targets {missing}")
    slot_uavs = np.repeat(np.arange(M, dtype=int), cap)
    slot_values = values[:, slot_uavs]
    finite = np.isfinite(slot_values)
    penalty = max(1.0, float(np.max(np.abs(slot_values[finite])))) * 1e9
    cost = np.where(finite, -slot_values, penalty)
    rows, cols = linear_sum_assignment(cost)
    if len(rows) != Q or np.any(~finite[rows, cols]):
        raise ValueError("no feasible capacitated target--fusion assignment")
    f_q = np.full(Q, -1, dtype=int)
    f_q[rows] = slot_uavs[cols]
    return f_q


def _solve_bottleneck_capacity_assignment(
    cfg: Config,
    values: np.ndarray,
) -> np.ndarray:
    """Lexicographically maximize minimum then total fixed assignment value."""
    finite_values = np.unique(values[np.isfinite(values)])
    if finite_values.size == 0:
        raise ValueError("no finite target--fusion assignment utilities")
    threshold_star: float | None = None
    for threshold in finite_values[::-1]:
        restricted = np.where(values >= threshold, values, -np.inf)
        try:
            _solve_capacity_assignment(cfg, restricted)
        except ValueError:
            continue
        threshold_star = float(threshold)
        break
    if threshold_star is None:
        raise ValueError("no feasible bottleneck target--fusion assignment")
    restricted = np.where(values >= threshold_star, values, -np.inf)
    return _solve_capacity_assignment(cfg, restricted)


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
        from .fbl import report_latency_s

        if is_local_observation(self.plan, link, q):
            return 0.0
        return report_latency_s(self.cfg, self.rate(link, q))

    def transmitter(self, link: Link, q: int) -> int:
        """The UAV that actually transmits the report (used by ``active_set``)."""
        return int(link[1])
