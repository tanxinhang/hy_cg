"""Target-state prior perturbation.

A reviewer asked the obvious but sharp question: ``D_q`` and the DD-domain
feasibility of every link ``(i, j, q)`` depend on the **predicted** target
state ``p_q, v_q`` --- but the prediction itself comes from the tracker that
is fed by the very links we are scheduling.  The paper leaves this implicit.

This module provides the light-weight, OTFS-faithful prior-uncertainty model
used by the ``prior-sweep`` experiment.  It is ported/adapted from
``CodeCg/gate_otfs_collision/experiments.py`` (``predict_tracks``,
``predict_tracks_with_crn_noise``, ``perturb_scene_prediction``) and turned
into pure functions over :class:`isac_sim.model.Geometry`.

Two framings are supported:

* **Robustness** (``pos_sigma_m > 0`` or ``vel_sigma_mps > 0``).  The whole
  simulator consumes the *perturbed* target state, so the result answers
  "how does P_D degrade when the prior is wrong?".  This is the question a
  practitioner will ask and is what the ``prior-sweep`` mode reports.
* **Predict-then-act** (handled by :func:`predicted_geometry`).  Provides the
  predicted state given a tracker output.  Not wired into the simulator
  loop yet, but available for downstream experiments that want to keep the
  detection-side geometry fixed.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import Config
from .model import Geometry


def perturbed_geometry(
    cfg: Config,
    geom: Geometry,
    sigma_pos_m: float,
    sigma_vel_mps: float,
    rng: np.random.Generator,
) -> Geometry:
    """Return a copy of ``geom`` with the target state perturbed by Gaussian noise.

    The UAV state is untouched -- only the targets are perturbed, which is
    where the scheduling-side uncertainty lives.  ``sigma_pos_m = 0`` or
    ``sigma_vel_mps = 0`` produces the unperturbed state for that axis (and
    *does not* consume any RNG draw, which is important for parity).
    """
    if sigma_pos_m <= 0 and sigma_vel_mps <= 0:
        return Geometry(
            p_uav=geom.p_uav.copy(),
            v_uav=geom.v_uav.copy(),
            p_tgt=geom.p_tgt.copy(),
            v_tgt=geom.v_tgt.copy(),
        )

    p_tgt = geom.p_tgt.copy()
    v_tgt = geom.v_tgt.copy()
    if sigma_pos_m > 0:
        # Perturb in 3D but keep the targets at their nominal altitude
        # (``v_tgt`` is mostly 2D anyway since ``cfg.geometry`` builds targets
        # at altitude zero and moves them in-plane).
        eps = rng.normal(0.0, sigma_pos_m, size=p_tgt.shape)
        p_tgt = p_tgt + eps
        p_tgt[:, 2] = geom.p_tgt[:, 2]
    if sigma_vel_mps > 0:
        eps = rng.normal(0.0, sigma_vel_mps, size=v_tgt.shape)
        v_tgt = v_tgt + eps
        v_tgt[:, 2] = 0.0
    return Geometry(
        p_uav=geom.p_uav.copy(),
        v_uav=geom.v_uav.copy(),
        p_tgt=p_tgt,
        v_tgt=v_tgt,
    )


def predicted_geometry(
    cfg: Config,
    geom: Geometry,
    rng: np.random.Generator,
    sigma_pos_m: float | None = None,
    sigma_vel_mps: float | None = None,
) -> Geometry:
    """Constant-velocity KF-style prediction.

    ``p_pred = p + v * dt`` and ``v_pred = v``, with Gaussian perturbation of
    configurable amplitude.  ``dt`` is read from ``cfg.prior.dt_s`` and the
    sigmas default to ``cfg.prior.sigma_pos_m`` and
    ``cfg.prior.sigma_vel_mps``.
    """
    sigma_p = cfg.prior.sigma_pos_m if sigma_pos_m is None else sigma_pos_m
    sigma_v = cfg.prior.sigma_vel_mps if sigma_vel_mps is None else sigma_vel_mps
    dt = cfg.prior.dt_s

    p_pred = geom.p_tgt + geom.v_tgt * dt
    v_pred = geom.v_tgt.copy()

    if sigma_p > 0:
        p_pred = p_pred + rng.normal(0.0, sigma_p, size=p_pred.shape)
        p_pred[:, 2] = geom.p_tgt[:, 2]
    if sigma_v > 0:
        v_pred = v_pred + rng.normal(0.0, sigma_v, size=v_pred.shape)
        v_pred[:, 2] = 0.0

    return Geometry(
        p_uav=geom.p_uav.copy(),
        v_uav=geom.v_uav.copy(),
        p_tgt=p_pred,
        v_tgt=v_pred,
    )


def crn_noise_pair(
    track_pos: np.ndarray,
    track_vel: np.ndarray,
    sigma_pos_m: float,
    sigma_vel_mps: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Common-Random-Number perturbation primitives.

    Returns ``(pos_eps, vel_eps)`` with the given standard deviations.  Two
    methods that consume the same primitives from a shared RNG see exactly
    the same realisation, which is the CRN trick that keeps paired-comparison
    confidence intervals tight.  Matches ``predict_tracks_with_crn_noise`` in
    the sibling ``gate_otfs_collision`` package.
    """
    pos_eps = rng.normal(0.0, 1.0, size=track_pos.shape) if sigma_pos_m > 0 else np.zeros_like(track_pos)
    vel_eps = rng.normal(0.0, 1.0, size=track_vel.shape) if sigma_vel_mps > 0 else np.zeros_like(track_vel)
    return pos_eps, vel_eps