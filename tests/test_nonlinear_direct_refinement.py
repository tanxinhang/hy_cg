"""Pin truth-free nonlinear direct-DD refinement and its input contracts."""

from copy import deepcopy

import numpy as np

from isac_sim.receiver import cancellation as cx
from isac_sim.core.config import apply_overrides
from _tpuic_common import make_cfg, make_observation


def _case():
    cfg = apply_overrides(make_cfg(), {
        "cancellation.direct_estimation_sigma_delay_bins": 0.1,
        "cancellation.direct_estimation_sigma_doppler_bins": 0.1,
    })
    return cfg, make_observation(cfg)


def test_refinement_is_truth_free():
    cfg, obs = _case()
    changed_truth = deepcopy(obs)
    changed_truth.x_direct = 1000.0 * obs.x_direct
    changed_truth.h_true = 1000.0 * obs.h_true
    got = cx.refine_direct_dd(cfg, obs, grid_points=3)
    altered = cx.refine_direct_dd(cfg, changed_truth, grid_points=3)
    np.testing.assert_allclose(got.X, altered.X)
    for left, right in zip(got.direct_est, altered.direct_est):
        assert left.delay_bin == right.delay_bin
        assert left.doppler_bin == right.doppler_bin


def test_refinement_validates_grid_contract():
    cfg, obs = _case()
    with np.testing.assert_raises(ValueError):
        cx.refine_direct_dd(cfg, obs, grid_points=4)


def test_joint_refinement_is_truth_free_and_applies_to_held_out_look():
    cfg, obs = _case()
    reference = deepcopy(obs)
    reference.x_direct = 1000.0 * reference.x_direct
    reference.h_true = 1000.0 * reference.h_true
    left = cx.refine_direct_dd_joint(cfg, [obs], grid_points=3)
    right = cx.refine_direct_dd_joint(cfg, [reference], grid_points=3)
    for a, b in zip(left, right):
        assert a.delay_bin == b.delay_bin
        assert a.doppler_bin == b.doppler_bin
    held_out = cx.apply_direct_dd(cfg, obs, left)
    assert held_out.X.shape == obs.X.shape
    assert held_out.direct_est == left
