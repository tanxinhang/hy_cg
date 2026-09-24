"""Protocol guards for the receiver-target generalization gate."""

from argparse import Namespace

import pytest
import numpy as np

from tools.gate_tpuic_generalization import _evaluate, _pairs
from tools.tpuic_generalization_diagnostics import _heldout_certificate, _score
from _tpuic_common import make_cfg, make_observation


def test_small_runs_cannot_claim_formal_generalization():
    args = Namespace(calibration_scenes=4, test_scenes=4, p_fa=0.05)
    rows = []
    for scene, split in enumerate(["train"] + ["calibration"] * 4 + ["test"] * 4):
        row = {"scene_id": scene, "split": split, "boost_db": 50.0,
               "receiver": 0, "target": 0}
        for name in ("two_cpi", "crossfit_map", "perfect_two_cpi"):
            row[f"{name}_h0"] = float(scene)
            row[f"{name}_h1"] = float(scene + 1)
        rows.append(row)
    _, _, decision = _evaluate(rows, args)
    assert decision == {"eligible": False, "status": "screen_only"}


def test_preregistered_pairs_are_validated_against_selected_axes():
    assert _pairs("0:1,3:2,1:1", (0, 1, 3), (1, 2)) == (
        (0, 1), (3, 2), (1, 1))
    with pytest.raises(ValueError, match="subsets"):
        _pairs("2:1", (0, 1, 3), (1, 2))


def test_heldout_certificate_removes_target_span_and_noise_floor():
    class Obs:
        A = np.eye(4, 1, dtype=complex)
        y = np.zeros(4, dtype=complex)
        sigma2 = 2.0

    class Result:
        residual = np.array([9.0, 2.0, 2.0, 2.0], dtype=complex)

    raw, corrected = _heldout_certificate(Obs(), Result())
    assert raw == pytest.approx(12.0)
    assert corrected == pytest.approx(6.0)


def test_operator_aware_noise_floor_is_physical():
    cfg = make_cfg()
    obs = make_observation(cfg)
    _, _, floor = _score(cfg, obs, target=0)
    assert 0.0 <= floor <= obs.sigma2 * obs.y.size
