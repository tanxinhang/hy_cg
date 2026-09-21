"""assignment（自 ``isac_sim/cooperation/reporting.py`` 拆出）。"""

from __future__ import annotations

import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains, Geometry, LinkTables

from isac_sim.cooperation.reporting.budget import ReportingPlan
from isac_sim.cooperation.reporting.capacitated import _assign_capacitated_value
from isac_sim.cooperation.reporting.lookahead import _assign_capacitated_pd_lookahead
from isac_sim.cooperation.reporting.solvers import _solve_capacity_assignment


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
