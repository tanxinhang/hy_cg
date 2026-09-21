"""capacitated（自 ``isac_sim/cooperation/reporting.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains, Geometry, LinkTables

from isac_sim.cooperation.reporting.budget import ReportingPlan
from isac_sim.cooperation.reporting.solvers import _solve_bottleneck_capacity_assignment


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
    from isac_sim.detection.fusion import deflection_for_links

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
