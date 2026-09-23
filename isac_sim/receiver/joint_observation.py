"""Joint multi-target H0/H1 observation domain model.

Every target detector in a scene consumes the same receiver observation.  The
global null removes all candidate echoes; it is therefore suitable for a
scene-level ``max_q T_q`` alarm, unlike target-by-target exclusion records.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from isac_sim.receiver import cancellation as cx
from isac_sim.receiver.cancellation import Observation
from isac_sim.sensing.model import radar_hardware_gain


@dataclass(frozen=True)
class JointObservationPair:
    """One receiver/scene under the global null and full-target alternative."""

    scene_id: int
    receiver: int
    h0: Observation
    h1: Observation

    def for_target(self, target: int) -> tuple[Observation, Observation]:
        """Return both shared observations with only the detector index changed."""
        q = int(target)
        ids = self.h1.A_target_ids
        target_count = (
            int(np.max(ids)) + 1 if ids is not None and np.asarray(ids).size else 0
        )
        if q < 0 or q >= target_count:
            raise IndexError(f"target index {q} is outside the joint observation")
        return replace(self.h0, weak_index=q), replace(self.h1, weak_index=q)


def global_null_from_full(full: Observation, rng: np.random.Generator) -> Observation:
    """Create an independent-noise, all-target-absent observation.

    Direct-path state and receiver metadata are retained from ``full``.  Only
    target echoes and their coefficients are removed, so no target-specific H0
    can accidentally be combined into a global alarm.
    """
    noise = (
        rng.normal(size=full.y.size) + 1j * rng.normal(size=full.y.size)
    ) * math.sqrt(float(full.sigma2) / 2.0)
    return replace(
        full,
        y=full.x_direct + noise,
        s_target=np.zeros_like(full.s_target),
        alpha_true=np.zeros_like(full.alpha_true),
    )


def generate_joint_observation_pair(
    cfg,
    truth,
    belief,
    base,
    *,
    scene_id: int,
    receiver: int,
) -> JointObservationPair:
    """Generate one deterministic joint pair for a receiver and scene.

    Seed namespaces are part of the protocol: observation/noise draws and
    direct-path errors are separated, while all target detectors reuse the
    resulting pair through :meth:`JointObservationPair.for_target`.
    """
    scene = int(scene_id)
    rx = int(receiver)
    rng = np.random.default_rng([int(cfg.run.seed), scene, rx, 0, 404])
    direct_rng = np.random.default_rng([int(cfg.run.seed), scene, rx, 405])
    m = int(cfg.scale.M)
    power = np.full(m, float(cfg.radio.P_default) * float(cfg.radio.rho))
    full = cx.build_observation(
        cfg,
        truth,
        belief.as_geometry(truth),
        base,
        rx,
        rng=rng,
        direct_error_rng=direct_rng,
        sense_power=power,
        radiated_power=power,
        processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)),
        active_mask=np.ones(m, dtype=bool),
        include_echo=True,
        weak_index=0,
    )
    return JointObservationPair(scene, rx, global_null_from_full(full, rng), full)


def joint_observation(cfg, truth, belief, base, *, scene: int, receiver: int):
    """Compatibility tuple for experiment drivers migrating to the domain API."""
    pair = generate_joint_observation_pair(
        cfg, truth, belief, base, scene_id=scene, receiver=receiver
    )
    return pair.h0, pair.h1


__all__ = [
    "JointObservationPair",
    "generate_joint_observation_pair",
    "global_null_from_full",
    "joint_observation",
]
