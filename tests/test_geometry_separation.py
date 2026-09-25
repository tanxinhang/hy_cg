"""Hard UAV-separation constraints in random geometry generation."""

import numpy as np
import pytest

from isac_sim.core.config import Config, validate_config
from isac_sim.sensing.model import generate_geometry


def _minimum_pair_distance(points):
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    distance[np.diag_indices_from(distance)] = np.inf
    return float(np.min(distance))


def test_generated_uavs_respect_twenty_metre_separation():
    cfg = Config()
    cfg.scale.M = 6
    cfg.scale.Q = 3
    cfg.geometry.area_xy = 400.0
    cfg.geometry.h_uav_min = 800.0
    cfg.geometry.h_uav_max = 1200.0
    cfg.geometry.min_uav_separation_m = 20.0

    for seed in range(100):
        geom = generate_geometry(cfg, np.random.default_rng(seed))
        assert _minimum_pair_distance(geom.p_uav) >= 20.0


def test_invalid_uav_separation_is_rejected():
    cfg = Config()
    cfg.geometry.min_uav_separation_m = -1.0
    with pytest.raises(ValueError, match="min_uav_separation_m"):
        validate_config(cfg)


def test_impossible_uav_separation_fails_explicitly():
    cfg = Config()
    cfg.scale.M = 2
    cfg.scale.Q = 1
    cfg.geometry.area_xy = 0.0
    cfg.geometry.h_uav_min = 100.0
    cfg.geometry.h_uav_max = 100.0
    cfg.geometry.min_uav_separation_m = 20.0

    with pytest.raises(RuntimeError, match="could not place all UAVs"):
        generate_geometry(cfg, np.random.default_rng(7))
