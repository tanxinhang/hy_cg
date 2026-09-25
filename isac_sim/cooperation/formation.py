"""Deterministic target-coverage formations for controlled experiments."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from isac_sim.sensing.model import Geometry


def minimum_uav_separation(positions) -> float:
    """Return the minimum three-dimensional centre-to-centre distance."""
    points = np.asarray(positions, dtype=float)
    if len(points) < 2:
        return float("inf")
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    distance[np.diag_indices_from(distance)] = np.inf
    return float(np.min(distance))


def project_uav_separation(cfg, positions, origins, max_movement_m=None):
    """Deterministically project planned positions onto the hard safety set."""
    points = np.asarray(positions, dtype=float).copy()
    origins = np.asarray(origins, dtype=float)
    minimum = float(cfg.geometry.min_uav_separation_m)
    budget = None if max_movement_m is None else np.broadcast_to(
        np.asarray(max_movement_m, dtype=float), (len(points),))
    for _ in range(200):
        delta = points[:, None, :] - points[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(distance, np.inf)
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        if distance[i, j] >= minimum - 1e-9:
            return points
        direction = delta[i, j] / max(float(distance[i, j]), 1e-12)
        if distance[i, j] < 1e-12:
            direction = np.array([1.0, 0.0, 0.0])
        shift = 0.5 * (minimum - float(distance[i, j]) + 1e-6) * direction
        points[i] += shift
        points[j] -= shift
        points[:, :2] = np.clip(points[:, :2], 0.0, cfg.geometry.area_xy)
        points[:, 2] = np.clip(points[:, 2], cfg.geometry.h_uav_min,
                               cfg.geometry.h_uav_max)
        if budget is not None:
            motion = points - origins
            norm = np.linalg.norm(motion, axis=1)
            scale = np.minimum(1.0, budget / np.maximum(norm, 1e-30))
            points = origins + motion * scale[:, None]
    raise RuntimeError("formation cannot satisfy the configured UAV separation")


def target_ring_formation(
    cfg,
    geom: Geometry,
    views_per_target: int,
    *,
    horizontal_radius_m: float = 100.0,
    max_movement_m: float | np.ndarray | None = None,
) -> Geometry:
    """Place distinct UAVs around each target for a structural upper bound.

    The construction uses true target positions and is therefore an oracle
    formation experiment, not an online controller.  It answers whether a
    requested number of geometrically distinct views can rescue detection
    before motion budgets and belief error are introduced.
    """
    count = int(views_per_target)
    if count < 0 or count * int(cfg.scale.Q) > int(cfg.scale.M):
        raise ValueError("views_per_target requires Q*count <= M")
    if not np.isfinite(horizontal_radius_m) or horizontal_radius_m <= 0.0:
        raise ValueError("horizontal_radius_m must be positive and finite")
    positions = np.asarray(geom.p_uav, dtype=float).copy()
    anchors = []
    for target in range(int(cfg.scale.Q)):
        for slot in range(count):
            angle = 2.0 * np.pi * slot / max(count, 1) + np.pi * target / 3.0
            xy = geom.p_tgt[target, :2] + horizontal_radius_m * np.array([
                np.cos(angle), np.sin(angle)
            ])
            anchor = np.empty(3, dtype=float)
            anchor[:2] = np.clip(xy, 0.0, float(cfg.geometry.area_xy))
            anchor[2] = np.clip(
                geom.p_tgt[target, 2],
                float(cfg.geometry.h_uav_min),
                float(cfg.geometry.h_uav_max),
            )
            anchors.append(anchor)
    if anchors:
        anchors_array = np.vstack(anchors)
        cost = np.linalg.norm(
            positions[:, None, :] - anchors_array[None, :, :], axis=2
        )
        uav_indices, anchor_indices = linear_sum_assignment(cost)
        positions[uav_indices] = anchors_array[anchor_indices]
    if max_movement_m is not None:
        budget = np.asarray(max_movement_m, dtype=float)
        if budget.ndim == 0:
            budget = np.full(int(cfg.scale.M), float(budget))
        if budget.shape != (int(cfg.scale.M),):
            raise ValueError("max_movement_m must be scalar or one value per UAV")
        if np.any(~np.isfinite(budget)) or np.any(budget < 0.0):
            raise ValueError("max_movement_m must be finite and non-negative")
        displacement = positions - np.asarray(geom.p_uav, dtype=float)
        distance = np.linalg.norm(displacement, axis=1)
        scale = np.minimum(1.0, budget / np.maximum(distance, 1e-30))
        positions = np.asarray(geom.p_uav, dtype=float) + displacement * scale[:, None]
    positions = project_uav_separation(
        cfg, positions, np.asarray(geom.p_uav, dtype=float), max_movement_m)
    return Geometry(
        p_uav=positions,
        v_uav=np.asarray(geom.v_uav).copy(),
        p_tgt=np.asarray(geom.p_tgt).copy(),
        v_tgt=np.asarray(geom.v_tgt).copy(),
    )
