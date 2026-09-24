"""Protocol guards for the receiver-target generalization gate."""

from argparse import Namespace

import pytest
import numpy as np

from isac_sim.receiver import cancellation as cx
from tools.gate_tpuic_generalization import _evaluate, _pairs
from tools.tpuic_generalization_diagnostics import _heldout_certificate, _score
from tools.tpuic_residual_decomposition import (
    decompose_residual, noise_energy_spectrum, operator_noise_floor)
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
    _, _, floor, _ = _score(cfg, obs, target=0)
    assert 0.0 <= floor <= obs.sigma2 * obs.y.size


def test_oracle_residual_components_close_under_one_frozen_operator():
    cfg = make_cfg()
    obs = make_observation(cfg)
    results = cx.cancellation_arms(cfg, obs, weak_target=0)
    audit = decompose_residual(cfg, obs, results, "tp_uic_full")
    assert audit["closure_error"] < 1e-10
    assert audit["total"] == pytest.approx(
        audit["direct"] + audit["target"] + audit["noise"] + audit["cross"])
    assert min(audit[key] for key in ("direct", "target", "noise")) >= 0.0


def test_noise_spectrum_reproduces_exact_operator_mean():
    cfg = make_cfg()
    obs = make_observation(cfg)
    results = cx.cancellation_arms(cfg, obs, weak_target=0)
    weights, unit_count = noise_energy_spectrum(
        cfg, obs, results, "tp_uic_full")
    spectral_mean = obs.sigma2 * (unit_count + np.sum(weights))
    assert spectral_mean == pytest.approx(
        operator_noise_floor(cfg, obs, results, "tp_uic_full"), rel=1e-10)
