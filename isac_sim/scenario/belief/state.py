"""state（自 ``isac_sim/scenario/belief.py`` 拆出）。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains, Geometry


@dataclass
class BeliefState:
    """Gaussian belief over the target states.

    ``xhat`` is ``(Q, 6)`` (position + velocity per target) and ``P`` is
    ``(Q, 6, 6)``.  The constant-velocity transition ``F`` and process noise
    ``Q_cv`` are provided so a downstream tracker update can close the loop.
    """

    xhat: np.ndarray
    P: np.ndarray

    @classmethod
    def from_truth(cls, cfg: Config, geom: Geometry, rng: np.random.Generator) -> "BeliefState":
        """Build a belief as the truth corrupted by ``cfg.prior.belief_*`` noise.

        This is the minimal *tracker-free* model: the belief is a single noisy
        snapshot of the truth, with an isotropic covariance of the configured
        amplitude.  It is enough to expose the belief-mismatch cost without
        committing to a specific filter.
        """
        Q = cfg.scale.Q
        xhat = np.zeros((Q, 6))
        xhat[:, 0:3] = geom.p_tgt
        xhat[:, 3:6] = geom.v_tgt

        s_pos = max(cfg.prior.belief_sigma_pos_m, 0.0)
        s_vel = max(cfg.prior.belief_sigma_vel_mps, 0.0)
        P = np.zeros((Q, 6, 6))
        # The scenario generator constrains targets to nominal altitude and
        # zero vertical velocity (see the assignments below).  Their stated
        # covariance must encode the same four-dimensional horizontal model;
        # assigning variance to z/v_z would make capture/risk propagation price
        # errors that the belief draw can never realise.
        P[:, 0, 0] = P[:, 1, 1] = s_pos ** 2
        P[:, 3, 3] = P[:, 4, 4] = s_vel ** 2

        if s_pos > 0:
            xhat[:, 0:3] += rng.normal(0.0, s_pos, size=(Q, 3))
            xhat[:, 2] = geom.p_tgt[:, 2]  # keep nominal altitude
        if s_vel > 0:
            xhat[:, 3:6] += rng.normal(0.0, s_vel, size=(Q, 3))
            xhat[:, 5] = 0.0
        return cls(xhat=xhat, P=P)

    def as_geometry(self, geom: Geometry) -> Geometry:
        """A :class:`Geometry` whose targets are the belief means."""
        return Geometry(
            p_uav=geom.p_uav.copy(),
            v_uav=geom.v_uav.copy(),
            p_tgt=self.xhat[:, 0:3].copy(),
            v_tgt=self.xhat[:, 3:6].copy(),
        )


def belief_geometry(cfg: Config, geom_truth: Geometry, rng: np.random.Generator) -> Geometry:
    """The scheduler's view of the scene: target state = belief mean."""
    belief = BeliefState.from_truth(cfg, geom_truth, rng)
    return belief.as_geometry(geom_truth)


def predicted_geometry_from_belief(cfg: Config, geom_truth: Geometry, rng: np.random.Generator) -> Geometry:
    """Constant-velocity one-step prediction from the current belief (utility)."""
    from isac_sim.scenario.prior import predicted_geometry

    return predicted_geometry(cfg, geom_truth, rng)
