"""Direct role powers and marginal-PD AO must obey physical power constraints.

These tests pin zero-power role support, legacy-configuration exclusion, weakest-
target search weighting, and exact delivered-PD acceptance semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from isac_sim.cooperation.power_ao import (
    marginal_delivered_pd_power_ao,
    smooth_weakest_pd,
)
from isac_sim.core.config import Config, validate_config
from isac_sim.sensing.model.link_tables.power import power_split


def test_direct_power_vectors_allow_real_sensing_and_reporting_roles():
    cfg = Config()
    cfg.scale.M = 3
    cfg.radio.P_sense_by_uav = (1.0, 0.0, 0.4)
    cfg.radio.P_comm_by_uav = (0.0, 1.0, 0.6)
    split = power_split(cfg)
    np.testing.assert_array_equal(split.P_sense, [1.0, 0.0, 0.4])
    np.testing.assert_array_equal(split.P_comm, [0.0, 1.0, 0.6])
    np.testing.assert_array_equal(split.P, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(split.rho, [1.0, 0.0, 0.4])


def test_direct_and_legacy_power_parameterizations_cannot_be_mixed():
    cfg = Config()
    cfg.radio.P_sense_by_uav = tuple([0.8] * cfg.scale.M)
    cfg.radio.P_comm_by_uav = tuple([0.2] * cfg.scale.M)
    cfg.radio.rho_by_uav = tuple([0.8] * cfg.scale.M)
    with pytest.raises(ValueError, match="cannot be mixed"):
        validate_config(cfg)


def test_smooth_weakest_pd_weights_the_bottleneck():
    base = smooth_weakest_pd(np.array([0.2, 0.9]), 0.02)
    weak_gain = smooth_weakest_pd(np.array([0.3, 0.9]), 0.02) - base
    strong_gain = smooth_weakest_pd(np.array([0.2, 1.0]), 0.02) - base
    assert weak_gain > 1000.0 * strong_gain


def test_marginal_power_ao_accepts_only_true_worst_pd_improvements():
    calls = []

    def replay(sense, comm):
        calls.append((sense.copy(), comm.copy()))
        # Target 0 needs UAV-0 sensing; target 1 needs UAV-1 communication.
        return np.array([
            0.1 + 0.8 * sense[0],
            0.1 + 0.8 * comm[1],
        ])

    result = marginal_delivered_pd_power_ao(
        np.array([0.25, 0.25]),
        np.array([0.25, 0.25]),
        replay,
        fleet_budget_w=1.0,
        peak_power_w=1.0,
        finite_difference_w=0.05,
        initial_trust_w=0.2,
        max_rounds=8,
    )
    assert np.min(result.delivered_pd) > 0.3
    assert np.sum(result.sense + result.comm) <= 1.0 + 1e-12
    assert np.all(result.sense + result.comm <= 1.0 + 1e-12)
    assert all(
        item["new_worst_pd"] > item["old_worst_pd"]
        for item in result.history if item["accepted"]
    )
    np.testing.assert_array_equal(calls[-1][0], result.sense)
    np.testing.assert_array_equal(calls[-1][1], result.comm)
