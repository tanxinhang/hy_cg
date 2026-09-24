"""TP-UIC matrix-information to cooperation reporting contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from isac_sim.cooperation.reporting import (
    ReceiverDetectionQuality,
    select_detection_reports,
)


def _quality() -> ReceiverDetectionQuality:
    # Receiver 1 is deliberately the strongest for target 0.  A geometry-only
    # C2F proxy previously discarded it; detector-consistent fusion must not.
    modes = (
        (np.array([0.10]), np.array([0.05])),
        (np.array([0.70]), np.array([0.15])),
        (np.array([0.20]), np.array([0.60])),
    )
    return ReceiverDetectionQuality(modes, n_looks=8, p_fa=0.05)


def test_quality_is_built_from_tpuic_stochastic_information() -> None:
    items = tuple(tuple(SimpleNamespace(eigenvalues=np.array([value]), n_looks=8)
                        for value in row) for row in ((0.1, 0.2), (0.3, 0.4)))
    quality = ReceiverDetectionQuality.from_stochastic_information(items, p_fa=0.05)
    assert quality.shape == (2, 2)
    assert quality.n_looks == 8
    assert quality.belief_only


def test_detector_quality_selects_the_best_local_fusion_receiver() -> None:
    quality = _quality()
    plan = select_detection_reports(
        quality, np.ones((3, 3)), link_feasible=~np.eye(3, dtype=bool),
        max_reports_per_sender=0,
    )
    np.testing.assert_array_equal(plan.fusion_nodes, [1, 2])
    np.testing.assert_allclose(plan.delivered_pd, plan.local_pd)


def test_reporting_adds_only_detector_improving_feasible_evidence() -> None:
    quality = _quality()
    feasible = ~np.eye(3, dtype=bool)
    plan = select_detection_reports(
        quality, np.full((3, 3), 0.9), link_feasible=feasible,
        max_reports_per_sender=1, max_reports_per_fusion=1,
    )
    assert np.all(plan.delivered_pd >= plan.local_pd)
    assert np.all(plan.sender_load <= 1)
    assert np.all(plan.fusion_load <= 1)
    assert all(feasible[source, destination]
               for source, _target, destination in plan.selected_reports)


def test_quality_rejects_mixed_look_semantics() -> None:
    items = ((SimpleNamespace(eigenvalues=np.array([0.1]), n_looks=8),),
             (SimpleNamespace(eigenvalues=np.array([0.2]), n_looks=16),))
    with pytest.raises(ValueError, match="same look count"):
        ReceiverDetectionQuality.from_stochastic_information(items, p_fa=0.05)
