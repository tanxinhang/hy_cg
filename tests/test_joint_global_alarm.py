"""Scene-level global-gate regression tests without an expensive receiver run."""
from __future__ import annotations

import numpy as np

from tools.run_joint_global_alarm import _learn_minimax_weight, _variant


def _row(scene: int, h0: float, h1: float) -> dict:
    return {
        "scene_id": scene,
        "raw_by_hypothesis_receiver_target": [
            [[h0, h0 + 0.2], [h0 + 0.1, h0]],
            [[h1, h1 + 0.2], [h1 + 0.1, h1]],
        ],
    }


def test_global_gate_uses_max_over_targets_and_calibration_only():
    cal = [_row(i, i / 10, i / 10 + 1) for i in range(20)]
    test = [_row(20, 0.0, 3.0)]
    result = _variant(cal, test, field="raw_by_hypothesis_receiver_target",
                      receiver_index=None, alpha=0.05)
    assert result["conformal_order"] == 20
    assert result["marginal_global_pfa_bound"] == 1 / 21
    assert abs(result["threshold"] - 4.0) < 1e-12
    assert result["test_global_false_alarms"] == 0
    assert result["test_per_target_detections"] == [1, 1]
    test[0]["raw_by_hypothesis_receiver_target"][0][0][0] = 1000
    changed = _variant(cal, test, field="raw_by_hypothesis_receiver_target",
                       receiver_index=None, alpha=0.05)
    assert changed["threshold"] == result["threshold"]
    assert changed["test_global_false_alarms"] == 1


def test_global_gate_refuses_nontrivial_alpha_with_too_few_scenes():
    cal = [_row(i, i, i + 1) for i in range(8)]
    result = _variant(cal, [_row(9, 0, 1)],
                      field="raw_by_hypothesis_receiver_target",
                      receiver_index=None, alpha=0.05)
    assert result["threshold"] == float("inf")
    assert result["test_per_target_detections"] == [0, 0]


def test_weight_selection_uses_train_scores_only():
    train = []
    for scene in range(20):
        base = scene % 4
        train.append({"reported_by_hypothesis_receiver_target": [
            [[base, base], [base + 0.1, base + 0.1]],
            [[base + 2, base + 2], [base + 0.1, base + 0.1]],
        ]})
    learned = _learn_minimax_weight(
        train, field="reported_by_hypothesis_receiver_target", quant_step=8 / 7)
    assert learned["selected"]["weights"] == [1.0, 0.0]
    cal = [_row(i, i / 10, i / 10 + 1) for i in range(20)]
    test = [_row(20, 0, 3)]
    first = _variant(cal, test, field="raw_by_hypothesis_receiver_target",
                     receiver_index=None, weights=np.asarray([1.0, 0.0]), alpha=0.05)
    test[0]["raw_by_hypothesis_receiver_target"][0][0][0] = 1e6
    second = _variant(cal, test, field="raw_by_hypothesis_receiver_target",
                      receiver_index=None, weights=np.asarray([1.0, 0.0]), alpha=0.05)
    assert first["threshold"] == second["threshold"]
