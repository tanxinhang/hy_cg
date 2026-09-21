"""belief 捕获概率（capture probability）的口径契约。

钉三件事：① 下界是**概率**（落在 [0,1]、随标准差单调下降），不是增益；
② 它只吃 belief 侧的散布，不吃真值几何（§15 belief-only）；
③ 联合 sigma 点在同一目标状态下跨链路共享（自链路恒为零、ij 与 ji 对称）。

这里不钉数字（137/83 这类计数会随 M/Q 与 seed 变），只钉**形状与对称性**。
"""

import numpy as np
import pytest

from isac_sim.scenario.belief import (
    BeliefState,
    belief_capture_probability_lower_bound,
    belief_capture_sigma_points,
)
from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.sensing.model import build_base_gains, generate_geometry


def test_capture_lower_bound_is_belief_only_probability():
    cfg = Config()
    std_l = np.asarray([[[0.0, 1.0, 10.0]]])
    std_k = np.asarray([[[0.0, 2.0, 20.0]]])
    got = belief_capture_probability_lower_bound(cfg, (std_l, std_k))

    assert got.shape == (1, 1, 3)
    assert np.all((0.0 <= got) & (got <= 1.0))
    assert got[0, 0, 0] == pytest.approx(1.0)
    assert got[0, 0, 0] >= got[0, 0, 1] >= got[0, 0, 2]


def test_capture_lower_bound_rejects_invalid_standard_deviation():
    cfg = Config()
    with pytest.raises(ValueError, match="non-negative"):
        belief_capture_probability_lower_bound(
            cfg, (np.zeros((1, 1, 1)), -np.ones((1, 1, 1)))
        )


def test_joint_capture_sigma_points_share_one_target_state_across_links():
    cfg = apply_overrides(Config(), {"scale.M": 4, "scale.Q": 2})
    rng = np.random.default_rng(17)
    geom = generate_geometry(cfg, rng)
    belief = BeliefState.from_truth(cfg, geom, rng)
    got = belief_capture_sigma_points(cfg, belief.as_geometry(geom), belief)

    assert got.shape == (137, 4, 4, 2)
    assert np.all(got[0, np.arange(4)[:, None] != np.arange(4), :])
    for i in range(4):
        assert not np.any(got[:, i, i, :])
        for j in range(i + 1, 4):
            assert np.array_equal(got[:, i, j, :], got[:, j, i, :])


def test_joint_capture_sigma_points_use_exact_near_field_doppler_map():
    """Canonical trial 0 contains a link whose Jacobian misses angular error."""
    cfg = apply_overrides(
        apply_preset(Config(), "paper-canonical"),
        {
            "scale.M": 6,
            "scale.Q": 3,
            "geometry.area_xy": 600.0,
            "run.seed": 2026,
        },
    )
    rng = np.random.default_rng([cfg.run.seed, 0])
    geom = generate_geometry(cfg, rng)
    # Advance the physical-channel stream exactly as the joint experiment does
    # before drawing the tracker belief.
    build_base_gains(cfg, geom, rng)
    belief = BeliefState.from_truth(cfg, geom, rng)
    got = belief_capture_sigma_points(cfg, belief.as_geometry(geom), belief)

    assert got[0, 3, 4, 1]
    assert not got[2, 3, 4, 1]
    assert int(np.sum(got[:, 3, 4, 1])) == 83
