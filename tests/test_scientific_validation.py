import numpy as np

from isac_sim.cooperation.scientific_validation import (
    belief_driven_target_ring_formation,
    fit_joint_statistic,
    fixed_train_test_trials,
    heldout_empirical_roc,
    load_frozen_scenario,
    save_frozen_scenario,
)
from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.scenario.belief import BeliefState
from isac_sim.sensing.model import build_base_gains, generate_geometry


def _scenario():
    cfg = apply_overrides(apply_preset(Config(), "paper-canonical"), {
        "scale.M": 6, "scale.Q": 3, "geometry.area_xy": 600.0,
        "run.seed": 19, "run.verbose": False,
    })
    rng = np.random.default_rng(19)
    truth = generate_geometry(cfg, rng)
    belief = BeliefState.from_truth(cfg, truth, rng)
    base = build_base_gains(cfg, truth, rng)
    return cfg, truth, belief, base


def test_frozen_scenario_round_trip_is_exact(tmp_path):
    _cfg, truth, belief, base = _scenario()
    path = tmp_path / "scene.npz"
    save_frozen_scenario(path, truth, belief, base)
    got_truth, got_belief, got_base = load_frozen_scenario(path)
    for name in ("p_uav", "v_uav", "p_tgt", "v_tgt"):
        np.testing.assert_array_equal(getattr(got_truth, name), getattr(truth, name))
    np.testing.assert_array_equal(got_belief.xhat, belief.xhat)
    np.testing.assert_array_equal(got_belief.P, belief.P)
    for name in base.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(got_base, name), getattr(base, name))


def test_belief_formation_hides_truth_and_respects_motion_budget():
    cfg, truth, belief, _base = _scenario()
    shifted = BeliefState(belief.xhat.copy(), belief.P.copy())
    shifted.xhat[:, 0] += 75.0
    moved = belief_driven_target_ring_formation(
        cfg, truth, shifted, 2,
        horizontal_radius_m=100.0, max_movement_m=80.0,
    )
    np.testing.assert_array_equal(moved.p_tgt, truth.p_tgt)
    distance = np.linalg.norm(moved.p_uav - truth.p_uav, axis=1)
    assert np.all(distance <= 80.0 + 1e-9)


def test_belief_formation_respects_distinct_per_uav_motion_budgets():
    cfg, truth, belief, _base = _scenario()
    budgets = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    moved = belief_driven_target_ring_formation(
        cfg, truth, belief, 2,
        horizontal_radius_m=100.0, max_movement_m=budgets,
    )
    distance = np.linalg.norm(moved.p_uav - truth.p_uav, axis=1)
    assert np.all(distance <= budgets + 1e-9)
    assert distance[0] == 0.0


def test_train_test_split_is_reproducible_and_disjoint():
    train1, test1 = fixed_train_test_trials(2026, 10, 20)
    train2, test2 = fixed_train_test_trials(2026, 10, 20)
    np.testing.assert_array_equal(train1, train2)
    np.testing.assert_array_equal(test1, test2)
    assert set(train1).isdisjoint(test1)


def test_joint_covariance_and_heldout_roc_detect_common_mode_redundancy():
    rng = np.random.default_rng(71)
    common_train = rng.normal(size=(4000, 1))
    h0_train = common_train + 0.2 * rng.normal(size=(4000, 3))
    h1_train = h0_train + np.array([0.0, 0.35, 0.0])
    weights, covariance = fit_joint_statistic(h0_train, h1_train)
    assert covariance[0, 1] > 0.5 * np.sqrt(covariance[0, 0] * covariance[1, 1])

    common_test = rng.normal(size=(10000, 1))
    h0_test = common_test + 0.2 * rng.normal(size=(10000, 3))
    h1_test = h0_test + np.array([0.0, 0.35, 0.0])
    result = heldout_empirical_roc(
        weights, h0_train, h0_test, h1_test, 0.05
    )
    assert abs(result["empirical_pfa"] - 0.05) < 0.02
    assert result["empirical_pd"] > result["empirical_pfa"]
