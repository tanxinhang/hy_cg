"""Reference-only coefficient MAP contracts."""
from __future__ import annotations

import numpy as np

from isac_sim.receiver import cancellation as cx
from _tpuic_common import make_cfg, make_observation


def test_reference_map_does_not_consume_sensing_observation():
    cfg = make_cfg()
    reference = make_observation(cfg, seed=1)
    sensing = make_observation(cfg, seed=2)
    estimate = cx.fit_reference_map([reference], cfg.cancellation.prior_variance)
    residual_before, _ = cx.apply_reference_map(sensing, estimate)
    sensing.y[:] += 100.0
    residual_after, _ = cx.apply_reference_map(sensing, estimate)
    assert np.allclose(residual_after - residual_before, 100.0)


def test_more_identical_references_contract_posterior_covariance():
    cfg = make_cfg()
    reference = make_observation(cfg, seed=3)
    one = cx.fit_reference_map([reference], cfg.cancellation.prior_variance)
    four = cx.fit_reference_map([reference] * 4, cfg.cancellation.prior_variance)
    assert np.trace(four.covariance).real < np.trace(one.covariance).real
    prior = np.eye(one.covariance.shape[0]) / cfg.cancellation.prior_variance
    information_one = np.linalg.inv(one.covariance) - prior
    information_four = np.linalg.inv(four.covariance) - prior
    assert np.allclose(information_four, 4.0 * information_one)


def test_pilot_estimate_is_independent_of_disjoint_sensing_rows():
    cfg = make_cfg()
    obs = make_observation(cfg, seed=4)
    pilot = np.arange(0, obs.y.size, 2)
    sensing = np.arange(1, obs.y.size, 2)
    before = cx.fit_pilot_map(obs, pilot, cfg.cancellation.prior_variance)
    obs.y[sensing] += 100.0
    after = cx.fit_pilot_map(obs, pilot, cfg.cancellation.prior_variance)
    assert np.array_equal(before.h_hat, after.h_hat)


def test_nested_pilot_rows_contract_posterior_covariance():
    cfg = make_cfg()
    obs = make_observation(cfg, seed=5)
    small = cx.fit_pilot_map(
        obs, np.arange(0, obs.y.size, 4), cfg.cancellation.prior_variance)
    large = cx.fit_pilot_map(
        obs, np.arange(0, obs.y.size, 2), cfg.cancellation.prior_variance)
    assert np.trace(large.covariance).real < np.trace(small.covariance).real
