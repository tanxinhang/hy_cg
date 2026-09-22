"""Reproducible, held-out validation utilities for cooperative sensing."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

from isac_sim.cooperation.formation import target_ring_formation
from isac_sim.scenario.belief import BeliefState
from isac_sim.sensing.model import BaseGains, Geometry


def save_frozen_scenario(
    path: str | Path,
    truth: Geometry,
    belief: BeliefState,
    base: BaseGains,
) -> None:
    """Save every scenario quantity needed for a paired algorithm comparison."""
    payload = {
        **{f"truth_{f.name}": np.asarray(getattr(truth, f.name))
           for f in fields(Geometry)},
        "belief_xhat": np.asarray(belief.xhat),
        "belief_P": np.asarray(belief.P),
        **{f"base_{f.name}": np.asarray(getattr(base, f.name))
           for f in fields(BaseGains)},
        "snapshot_version": np.asarray([1], dtype=np.int64),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)


def load_frozen_scenario(
    path: str | Path,
) -> tuple[Geometry, BeliefState, BaseGains]:
    """Load a frozen scenario without drawing any new random variables."""
    with np.load(Path(path), allow_pickle=False) as data:
        version = int(np.asarray(data["snapshot_version"]).ravel()[0])
        if version != 1:
            raise ValueError(f"unsupported scenario snapshot version {version}")
        truth = Geometry(**{
            f.name: np.asarray(data[f"truth_{f.name}"]).copy()
            for f in fields(Geometry)
        })
        belief = BeliefState(
            xhat=np.asarray(data["belief_xhat"]).copy(),
            P=np.asarray(data["belief_P"]).copy(),
        )
        base = BaseGains(**{
            f.name: np.asarray(data[f"base_{f.name}"]).copy()
            for f in fields(BaseGains)
        })
    return truth, belief, base


def belief_driven_target_ring_formation(
    cfg,
    truth: Geometry,
    belief: BeliefState,
    views_per_target: int,
    *,
    horizontal_radius_m: float,
    max_movement_m: float | np.ndarray,
) -> Geometry:
    """Plan UAV motion from belief means, while retaining hidden target truth."""
    if max_movement_m is None:
        raise ValueError("belief-driven formation requires a finite movement budget")
    budget = np.asarray(max_movement_m, dtype=float)
    if np.any(~np.isfinite(budget)):
        raise ValueError("belief-driven formation requires a finite movement budget")
    belief_geometry = belief.as_geometry(truth)
    planned = target_ring_formation(
        cfg,
        belief_geometry,
        views_per_target,
        horizontal_radius_m=horizontal_radius_m,
        max_movement_m=budget,
    )
    return Geometry(
        p_uav=planned.p_uav,
        v_uav=planned.v_uav,
        p_tgt=np.asarray(truth.p_tgt).copy(),
        v_tgt=np.asarray(truth.v_tgt).copy(),
    )


def fixed_train_test_trials(
    master_seed: int,
    n_train: int,
    n_test: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic, non-overlapping trial identifiers."""
    if n_train <= 0 or n_test <= 0:
        raise ValueError("training and test sets must both be non-empty")
    rng = np.random.default_rng(int(master_seed))
    ids = rng.choice(np.arange(1, 2**31 - 1), n_train + n_test, replace=False)
    return np.sort(ids[:n_train]), np.sort(ids[n_train:])


def fit_joint_statistic(
    train_h0: np.ndarray,
    train_h1: np.ndarray,
    *,
    ridge_fraction: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a linear detector from the empirical joint receiver covariance."""
    h0 = np.asarray(train_h0, dtype=float)
    h1 = np.asarray(train_h1, dtype=float)
    if h0.ndim != 2 or h1.shape != h0.shape or h0.shape[0] < 2:
        raise ValueError("train_h0 and train_h1 must be matched N-by-K arrays")
    delta = np.mean(h1, axis=0) - np.mean(h0, axis=0)
    centered0 = h0 - np.mean(h0, axis=0)
    centered1 = h1 - np.mean(h1, axis=0)
    covariance = (centered0.T @ centered0 + centered1.T @ centered1) / (
        2.0 * max(h0.shape[0] - 1, 1)
    )
    scale = max(float(np.trace(covariance)) / h0.shape[1], 1e-12)
    covariance = covariance + float(ridge_fraction) * scale * np.eye(h0.shape[1])
    weights = np.linalg.solve(covariance, delta)
    return weights, covariance


def heldout_empirical_roc(
    weights: np.ndarray,
    calibration_h0: np.ndarray,
    test_h0: np.ndarray,
    test_h1: np.ndarray,
    p_fa: float,
) -> dict:
    """Calibrate on training H0 and evaluate unchanged on held-out samples."""
    if not 0.0 < float(p_fa) < 1.0:
        raise ValueError("p_fa must lie strictly between zero and one")
    w = np.asarray(weights, dtype=float)
    calibration = np.asarray(calibration_h0, dtype=float)
    h0 = np.asarray(test_h0, dtype=float)
    h1 = np.asarray(test_h1, dtype=float)
    if (calibration.ndim != 2 or h0.ndim != 2 or h1.ndim != 2 or
            calibration.shape[1] != w.size or h0.shape[1] != w.size or
            h1.shape[1] != w.size):
        raise ValueError("held-out samples and weights have incompatible shapes")
    calibration_score = calibration @ w
    score0, score1 = h0 @ w, h1 @ w
    threshold = float(np.quantile(
        calibration_score, 1.0 - float(p_fa), method="higher"
    ))
    false_alarm = float(np.mean(score0 > threshold))
    detection = float(np.mean(score1 > threshold))
    n = int(score1.size)
    z = 1.959963984540054
    denominator = 1.0 + z * z / max(n, 1)
    centre = (detection + z * z / (2.0 * max(n, 1))) / denominator
    half = z * np.sqrt(
        detection * (1.0 - detection) / max(n, 1)
        + z * z / (4.0 * max(n, 1) ** 2)
    ) / denominator
    return {
        "threshold": threshold,
        "empirical_pfa": false_alarm,
        "empirical_pd": detection,
        "pd_ci95_wilson": [
            float(max(0.0, centre - half)),
            float(min(1.0, centre + half)),
        ],
        "n_h0": int(score0.size),
        "n_h1": int(score1.size),
    }
