"""lookahead（自 ``isac_sim/cooperation/reporting.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains, Geometry, LinkTables

from isac_sim.cooperation.reporting.budget import EPS, ReportingPlan
from isac_sim.cooperation.reporting.solvers import _solve_bottleneck_capacity_assignment


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
    from isac_sim.detection.fusion import predicted_pd_for_links

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
