"""Multi-hypothesis aggregation and calibration contracts for the DD detector."""

import numpy as np
import pytest


def test_logmeanexp_is_bounded_by_mean_and_max():
    values = np.array([1.0, 2.0, 4.0])
    peak = float(np.max(values))
    statistic = peak + float(np.log(np.mean(np.exp(values - peak))))
    assert float(np.mean(values)) <= statistic <= peak


def test_logmeanexp_single_hypothesis_is_identity():
    value = 3.25
    assert value + np.log(np.mean(np.exp(np.array([value]) - value))) == pytest.approx(value)
