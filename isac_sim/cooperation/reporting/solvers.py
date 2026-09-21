"""solvers（自 ``isac_sim/cooperation/reporting.py`` 拆出）。"""

from __future__ import annotations

import numpy as np
from isac_sim.core.config import Config, Link


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
