"""Shared fixtures for the receiver-context test files.

Split out of ``test_receiver_context.py`` (2026-09-21 audit).  Mirrors the
``_optmodel_common.py`` pattern: pytest's prepend import mode puts ``tests/`` on
``sys.path``.
"""
from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config, apply_overrides, apply_preset
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
    }
    base.update(overrides)
    return apply_overrides(cfg, base)


def make_world(cfg: Config, seed: int = 0):
    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    return geom, base, rng


def make_belief(cfg: Config, geom, rng):
    return perturbed_geometry(
        cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
    )
