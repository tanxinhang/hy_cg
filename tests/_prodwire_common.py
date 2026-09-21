"""Shared fixtures for the production-wiring test files.

Split out of ``test_cancellation_production_wiring.py`` (2026-09-21 audit).
Mirrors the ``_optmodel_common.py`` pattern: pytest's prepend import mode puts
``tests/`` on ``sys.path``.
"""
from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.sensing.model import build_base_gains, generate_geometry


def make_cfg(**overrides) -> Config:
    cfg = apply_preset(Config(), "paper-canonical")
    base = {
        "geometry.area_xy": 600.0,
        "detect.target_rcs": 0.1,
        "run.seed": 2026,
        "run.verbose": False,
    }
    base.update(overrides)
    return apply_overrides(cfg, base)


def make_base(cfg: Config, seed: int = 0):
    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    return geom, build_base_gains(cfg, geom, rng), rng
