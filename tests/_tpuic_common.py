"""Shared fixtures for the TP-UIC cancellation test files.

Split out of ``test_cancellation_tp_uic.py`` (2026-09-21 audit) so the three
topical files can share the two builders without duplicating them or importing
from a ``test_*`` module.  Mirrors the existing ``_optmodel_common.py`` pattern:
pytest's prepend import mode puts ``tests/`` on ``sys.path``.
"""
from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.receiver import cancellation as cx
from isac_sim.sensing.model import build_base_gains, generate_geometry


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
    }
    base.update(overrides)
    return apply_overrides(cfg, base)


def make_observation(cfg: Config, seed: int = 0, belief_error: bool = False):
    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    if belief_error:
        from isac_sim.scenario.prior import perturbed_geometry

        belief = perturbed_geometry(
            cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
        )
    else:
        belief = geom
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    obs = cx.build_observation(
        cfg, geom, belief, base, 0, rng=rng, sense_power=sense, radiated_power=sense,
        processing_gain=cfg.waveform.N * cfg.waveform.L, hw_gain=1.0,
    )
    return obs
