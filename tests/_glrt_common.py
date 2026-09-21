"""Shared fixtures for the TP-UIC GLRT detector test files.

Split out of ``test_cancellation_glrt.py`` (2026-09-21 audit).  Mirrors the
``_optmodel_common.py`` pattern: pytest's prepend import mode puts ``tests/`` on
``sys.path``.  Deliberately separate from ``_tpuic_common.py`` -- the GLRT
fixture pins ``cancellation.max_protected_targets`` and carries the released
15-UAV configuration as well.
"""
from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.sensing.model import build_base_gains, generate_geometry
from isac_sim.scenario.prior import perturbed_geometry


def make_cfg(**overrides) -> Config:
    cfg = apply_preset(Config(), "small-uav-compact-800m")
    base = {
        "geometry.area_xy": 600.0,
        "detect.target_rcs": 0.1,
        "scale.M": 6,
        "scale.Q": 3,
        "run.seed": 2026,
        "run.verbose": False,
        "cancellation.enable": True,
        "cancellation.max_protected_targets": 3,
    }
    base.update(overrides)
    return apply_overrides(cfg, base)


def make_canonical_cfg(**overrides) -> Config:
    """The released scenario: 15 UAVs, 10 targets, 600 m, RCS 0.1.

    Two claims in this file are properties of *that* geometry rather than of the
    detector -- the local patch covering and the off-grid recovery -- so they are
    tested where they live instead of being asserted on a toy configuration
    where three targets do not overlap at all.
    """
    cfg = apply_preset(Config(), "paper-canonical")
    base = {
        "geometry.area_xy": 600.0,
        "detect.target_rcs": 0.1,
        "run.seed": 2026,
        "run.verbose": False,
        "cancellation.enable": True,
        "cancellation.max_protected_targets": 3,
    }
    base.update(overrides)
    return apply_overrides(cfg, base)


def make_trial(cfg: Config, seed: int = 0, belief_error: bool = True, receiver: int = 0):
    """One matched ``(H1, H0)`` pair plus the tested target."""
    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    belief = (
        perturbed_geometry(
            cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
        )
        if belief_error
        else geom
    )
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    probe = cx.build_observation(
        cfg, geom, belief, base, receiver, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0,
    )
    ids = np.asarray(probe.A_target_ids)
    target = int(ids[0]) if ids.size else 0
    obs1, obs0 = cx.build_observation_pair(
        cfg, geom, belief, base, receiver, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0, exclude_target=target, weak_index=target,
    )
    return obs1, obs0, target


def make_arms(cfg: Config, obs):
    arms = cx.cancellation_arms(cfg, obs, weak_target=int(obs.weak_index), threshold=0.0)
    return arms, gl.arm_plans(cfg, obs, arms)
