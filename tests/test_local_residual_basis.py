"""Contracts for truth-free local bases and covariance-aware coverage geometry."""

import numpy as np
import pytest

from isac_sim.receiver.cancellation import Observation
from isac_sim.receiver.cancellation_glrt.covariance import LowRankCovariance
from isac_sim.receiver.cancellation_glrt.local_residual_basis import (
    projection_coverage,
    receiver_known_local_bases,
    whitened_projection_coverage,
)


def test_projection_coverage_has_expected_extremes_and_scale_invariance():
    basis = np.array([[1.0], [0.0]], dtype=complex)
    assert projection_coverage(np.array([3.0j, 0.0]), basis)[0] == pytest.approx(1.0)
    assert projection_coverage(np.array([0.0, 2.0]), 7.0 * basis)[0] == pytest.approx(0.0)


def test_whitened_coverage_uses_covariance_geometry():
    cov = LowRankCovariance(
        sigma2=1.0,
        W=np.array([[1.0], [0.0]], dtype=complex),
        lam=np.array([3.0]),
        min_ratio=1.0,
    )
    direction = np.array([1.0, 1.0], dtype=complex)
    basis = np.array([[1.0], [0.0]], dtype=complex)
    coverage, rank = whitened_projection_coverage(cov, direction, basis)
    assert coverage == pytest.approx(0.2)
    assert rank == 1


def test_receiver_known_bases_only_assemble_receiver_dictionaries():
    obs = Observation(
        y=np.zeros(3, complex), X=np.eye(3, 1, dtype=complex),
        A=np.eye(3, 2, dtype=complex), x_direct=np.full(3, 99.0),
        s_target=np.full(3, 88.0), h_true=np.zeros(1), alpha_true=np.zeros(2),
        sigma2=1.0, A_target_ids=np.array([0, 1]),
    )
    bases = receiver_known_local_bases(obs, target=1)
    assert list(bases) == ["direct", "direct_tested_target", "direct_all_targets"]
    assert [value.shape[1] for value in bases.values()] == [1, 2, 3]
    assert not any(np.any(value == 99.0) for value in bases.values())
