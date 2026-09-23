"""Domain contracts for shared multi-target global-null observations."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from isac_sim.receiver.joint_observation import (
    JointObservationPair,
    global_null_from_full,
)
from _tpuic_common import make_cfg, make_observation


def test_global_null_removes_every_target_but_retains_direct_state():
    cfg = make_cfg()
    full = make_observation(cfg, seed=41)
    null = global_null_from_full(full, np.random.default_rng(99))
    assert np.array_equal(null.x_direct, full.x_direct)
    assert np.count_nonzero(null.s_target) == 0
    assert np.count_nonzero(null.alpha_true) == 0
    assert not np.array_equal(null.y, full.y)
    assert null.y.shape == full.y.shape


def test_target_views_reuse_the_same_physical_observations():
    cfg = make_cfg()
    full = make_observation(cfg, seed=42)
    null = global_null_from_full(full, np.random.default_rng(100))
    pair = JointObservationPair(7, 1, null, full)
    h0a, h1a = pair.for_target(0)
    h0b, h1b = pair.for_target(1)
    assert h0a.y is h0b.y
    assert h1a.y is h1b.y
    assert (h0a.weak_index, h0b.weak_index) == (0, 1)
    assert (h1a.weak_index, h1b.weak_index) == (0, 1)


def test_target_view_rejects_out_of_range_index():
    cfg = make_cfg()
    full = make_observation(cfg, seed=43)
    pair = JointObservationPair(8, 0, replace(full, s_target=0 * full.s_target), full)
    with pytest.raises(IndexError):
        pair.for_target(cfg.scale.Q)
