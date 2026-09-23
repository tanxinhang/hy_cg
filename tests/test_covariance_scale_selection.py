"""Train-role and deterministic-choice contracts for covariance-scale fitting."""

import pytest

from isac_sim.detection.evaluation_split import SplitDataset
from isac_sim.receiver.cancellation_glrt.covariance_scale import (
    select_covariance_scale,
)


def _data(role="train"):
    return SplitDataset(role, tuple({
        "split": role, "scene_id": scene,
        "h0_inflation_by_scale": {0.5: 1.8, 1.0: 1.1, 2.0: 0.7},
    } for scene in range(3)))


def test_scale_selection_uses_train_h0_calibration():
    selected = select_covariance_scale(_data(), (0.5, 1.0, 2.0))
    assert selected.scale == 1.0
    assert selected.median_inflation == pytest.approx(1.1)


def test_scale_selection_rejects_non_train_dataset():
    with pytest.raises(ValueError, match="train split"):
        select_covariance_scale(_data("calibration"), (1.0,))


def test_scale_selection_rejects_missing_candidate_measurement():
    with pytest.raises(ValueError, match="lacks covariance scale"):
        select_covariance_scale(_data(), (3.0,))
