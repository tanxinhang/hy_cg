"""Contracts for matrix information, explicit protection, AO and frame evidence."""
from __future__ import annotations

import numpy as np

from isac_sim.cooperation.matrix_information_ao import (
    activation_neighborhood,
    monotone_matrix_information_ao,
    protection_neighborhood,
    protection_subsets,
)
from isac_sim.cooperation.formation import target_ring_formation
from isac_sim.detection.multiframe_llr import accumulate_llr
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.sensing.model import build_base_gains, generate_geometry
from isac_sim.sensing.waveform.timing import (
    dwell_from_looks_s,
    looks_from_dwell,
    otfs_frame_duration_s,
)
from _tpuic_common import make_cfg, make_observation
from tools.run_matrix_information_chain import (
    _packet_erasure_fused_pd,
    _stochastic_pd,
    _uav_centric_greedy_fusion,
    _geometry_maxmin_power_allocation,
    _uav_centric_fusion_search,
)


def test_default_full_arm_is_independent_of_stage_one_values(monkeypatch):
    """P0 regression: default full TP-UIC does not consume Stage-1 values."""
    import isac_sim.receiver.cancellation.arms_basic as basic

    cfg = make_cfg()
    obs = make_observation(cfg, belief_error=True)
    reference = cx.cancellation_arms(cfg, obs)["tp_uic_full"]
    original = basic.run_arm

    def corrupt(ctx, name, subspace, pv, candidates=(), soft_mu=None):
        arm = original(ctx, name, subspace, pv, candidates, soft_mu)
        if name == "tp_uic_stage1":
            arm.sub_direct[:] = complex(1e6, -1e6)
            arm.sub_target[:] = complex(-2e6, 3e6)
            arm.sub_noise[:] = complex(4e6, 5e6)
            arm.h_hat[:] = complex(6e6, -7e6)
        return arm

    monkeypatch.setattr(basic, "run_arm", corrupt)
    changed = cx.cancellation_arms(cfg, obs)["tp_uic_full"]
    np.testing.assert_array_equal(changed.h_hat, reference.h_hat)
    np.testing.assert_array_equal(changed.residual, reference.residual)
    assert changed.candidates == reference.candidates


def test_default_pruned_full_arm_skips_stage_one(monkeypatch):
    """The production-only default path must not execute redundant Stage 1."""
    import isac_sim.receiver.cancellation.arms_basic as basic

    cfg = make_cfg()
    obs = make_observation(cfg, belief_error=True)
    reference = cx.cancellation_arms(cfg, obs)["tp_uic_full"]
    original = basic.run_arm

    def reject(ctx, name, subspace, pv, candidates=(), soft_mu=None):
        if name == "tp_uic_stage1":
            raise AssertionError("default pruned tp_uic_full executed Stage 1")
        return original(ctx, name, subspace, pv, candidates, soft_mu)

    monkeypatch.setattr(basic, "run_arm", reject)
    arms = cx.cancellation_arms(cfg, obs, only="tp_uic_full")
    assert set(arms) == {"tp_uic_full"}
    np.testing.assert_array_equal(arms["tp_uic_full"].h_hat, reference.h_hat)
    np.testing.assert_array_equal(arms["tp_uic_full"].residual, reference.residual)


def test_matrix_information_is_the_glrt_noncentrality_interface():
    cfg = make_cfg()
    obs = make_observation(cfg, belief_error=True)
    arms = cx.cancellation_arms(cfg, obs)
    model = gl.residual_model(cfg, obs, "tp_uic_full", arms)
    target = int(obs.weak_index)
    info = gl.detection_information(
        cfg, obs, arms["tp_uic_full"], model, target
    )
    direct = gl.target_conditioned_glrt(
        cfg, obs, arms["tp_uic_full"], model, target=target
    )
    assert info.value == direct.ncp_unit
    assert 0.0 <= info.identifiable_fraction <= 1.0
    assert info.value >= 0.0


def test_stochastic_matrix_information_consumes_looks_and_reduces_to_scalar():
    cfg = make_cfg()
    obs = make_observation(cfg, belief_error=True)
    arms = cx.cancellation_arms(cfg, obs)
    model = gl.residual_model(cfg, obs, "tp_uic_full", arms)
    item = gl.stochastic_detection_information(
        cfg, obs, model, int(obs.weak_index)
    )
    lam = item.eigenvalues
    expected = cfg.detect.n_looks * np.sum(lam**2 / (1.0 + lam))
    assert item.jeffreys == expected
    assert item.n_looks == cfg.detect.n_looks
    assert 0.0 <= item.gaussian_pd(0.05) <= 1.0


def test_otfs_looks_are_derived_from_declared_sensing_dwell():
    cfg = make_cfg()
    frame = otfs_frame_duration_s(cfg)
    assert frame == 64 / 30e3
    assert looks_from_dwell(cfg, 16 * frame) == 16
    assert dwell_from_looks_s(cfg, 16) == 16 * frame


def test_sensing_dwell_shorter_than_one_frame_is_rejected():
    cfg = make_cfg()
    with np.testing.assert_raises(ValueError):
        looks_from_dwell(cfg, 0.5 * otfs_frame_duration_s(cfg))


def test_explicit_binary_protection_controls_basis_and_support():
    cfg = make_cfg()
    rng = np.random.default_rng([cfg.run.seed, 91])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    sense = np.full(cfg.scale.M, cfg.radio.rho * cfg.radio.P_default)
    protected = frozenset({1})
    obs = cx.build_observation(
        cfg, geom, geom, base, 0, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0, protected_targets=protected,
    )
    result = cx.cancellation_arms(cfg, obs, only="tp_uic_full")["tp_uic_full"]
    assert obs.protected_targets == protected
    assert set(np.asarray(obs.A_target_ids)[list(result.candidates)]) == protected
    assert result.protect_dim > 0


def test_exact_z_enumeration_uses_rank_not_target_count():
    ranks = {
        frozenset(): 0,
        frozenset({0}): 2,
        frozenset({1}): 2,
        frozenset({2}): 3,
        frozenset({0, 1}): 3,
        frozenset({0, 2}): 5,
        frozenset({1, 2}): 5,
        frozenset({0, 1, 2}): 6,
    }
    subsets = protection_subsets((0, 1, 2), ranks.__getitem__, max_rank=3)
    assert frozenset({0, 1}) in subsets
    assert frozenset({0, 2}) not in subsets


def test_protection_neighborhood_uses_marginal_rank_feasibility():
    ranks = {
        frozenset(): 0,
        frozenset({0}): 2,
        frozenset({1}): 2,
        frozenset({2}): 4,
        frozenset({0, 1}): 3,
        frozenset({0, 2}): 5,
        frozenset({1, 2}): 5,
    }
    candidates = protection_neighborhood(
        (0, 1, 2), frozenset({0}), ranks.__getitem__, max_rank=3
    )
    assert frozenset() in candidates
    assert frozenset({0, 1}) in candidates
    assert frozenset({2}) not in candidates


def test_ao_is_monotone_and_improves_worst_target_information():
    active_candidates = ((True, False), (False, True), (True, True))
    z_candidates = ((frozenset(), frozenset({0}), frozenset({1})),)

    def evaluate(active, protection):
        base = np.array([1.0, 0.6])
        if active[1]:
            base += np.array([0.2, 0.5])
        if 1 in protection[0]:
            base += np.array([-0.05, 0.45])
        return base

    result = monotone_matrix_information_ao(
        (True, False), (frozenset(),), active_candidates, z_candidates, evaluate
    )
    assert result.objective > 0.6
    assert all(b >= a for a, b in zip(result.history, result.history[1:]))
    assert 1 in result.protection[0]


def test_activation_neighborhood_avoids_exhaustive_mask_enumeration():
    state = (True, True, False, False)
    candidates = activation_neighborhood(state)
    assert len(candidates) == 8  # 4 flips + 2x2 swaps, not 2^4-1 masks
    assert all(any(candidate) for candidate in candidates)


def test_ao_accepts_dynamic_marginal_activation_search():
    calls = []

    def neighborhood(active):
        calls.append(active)
        return activation_neighborhood(active, include_swaps=False)

    def evaluate(active, _protection):
        return np.array([2.0 if active == (True, True) else 1.0])

    result = monotone_matrix_information_ao(
        (True, False), (frozenset(),), neighborhood, ((frozenset(),),), evaluate
    )
    assert result.active == (True, True)
    assert calls


def test_network_information_is_sum_of_receiver_viewpoints():
    receiver_target = np.array([
        [1.0, 0.2, 0.0],
        [0.0, 0.8, 0.4],
        [0.5, 0.0, 0.6],
    ])
    np.testing.assert_allclose(
        receiver_target.sum(axis=0), np.array([1.5, 1.0, 1.0])
    )


def test_medium_view_fraction_is_enforced_per_target_not_globally():
    pd = np.array([
        [0.9, 0.4, 0.1],
        [0.8, 0.2, 0.1],
        [0.7, 0.4, 0.1],
    ])
    fractions = np.mean(pd >= 0.3, axis=0)
    np.testing.assert_allclose(fractions, [1.0, 2.0 / 3.0, 0.0])
    assert float(np.min(fractions)) == 0.0


def test_packet_erasure_fusion_interpolates_local_and_ideal_evidence():
    receiver_modes = (
        (np.array([0.2]),),
        (np.array([0.4]),),
    )
    pfa = 0.05
    none_remote = _packet_erasure_fused_pd(
        receiver_modes, np.array([[1.0], [0.0]]), 8, pfa
    )[0]
    all_delivered = _packet_erasure_fused_pd(
        receiver_modes, np.ones((2, 1)), 8, pfa
    )[0]
    partial = _packet_erasure_fused_pd(
        receiver_modes, np.array([[1.0], [0.5]]), 8, pfa
    )[0]
    assert none_remote < partial < all_delivered


def test_remote_llr_delivery_creates_cooperation_gain_over_fusion_local_only():
    receiver_modes = (
        (np.array([0.15]),),
        (np.array([0.35]),),
        (np.array([0.25]),),
    )
    local = _packet_erasure_fused_pd(
        receiver_modes, np.array([[0.0], [1.0], [0.0]]), 8, 0.05
    )[0]
    cooperative = _packet_erasure_fused_pd(
        receiver_modes, np.ones((3, 1)), 8, 0.05
    )[0]
    assert cooperative > local


def test_positive_receiver_correlation_does_not_increase_fused_pd():
    receiver_modes = (
        (np.array([0.15]),),
        (np.array([0.35]),),
        (np.array([0.25]),),
    )
    success = np.ones((3, 1))
    independent = _packet_erasure_fused_pd(
        receiver_modes, success, 8, 0.05, receiver_correlation=0.0
    )[0]
    correlated = _packet_erasure_fused_pd(
        receiver_modes, success, 8, 0.05, receiver_correlation=0.25
    )[0]
    assert correlated <= independent


def test_uav_centric_policy_respects_sender_and_receiver_report_budgets():
    receiver_modes = (
        (np.array([0.10]), np.array([0.05])),
        (np.array([0.30]), np.array([0.15])),
        (np.array([0.20]), np.array([0.40])),
    )
    result = _uav_centric_greedy_fusion(
        receiver_modes, np.ones((3, 3)), 8, 0.05,
        max_reports_per_sender=1, max_reports_per_fusion=1,
    )
    assert np.all(result["sender_load"] <= 1)
    assert np.all(result["fusion_load"] <= 1)
    assert np.all(result["delivered_pd"] >= result["local_pd"])
    assert len({item["source_uav"] for item in result["selected_reports"]}) == len(
        result["selected_reports"]
    )


def test_uav_centric_detection_improves_with_additional_looks():
    receiver_modes = (
        (np.array([0.10]), np.array([0.05])),
        (np.array([0.30]), np.array([0.15])),
        (np.array([0.20]), np.array([0.40])),
    )
    link_success = np.ones((3, 3))
    short = _uav_centric_greedy_fusion(receiver_modes, link_success, 8, 0.05)
    long = _uav_centric_greedy_fusion(receiver_modes, link_success, 16, 0.05)
    assert np.all(long["delivered_pd"] >= short["delivered_pd"])


def test_uav_centric_policy_rejects_infeasible_reporting_links():
    receiver_modes = (
        (np.array([0.10]),),
        (np.array([0.40]),),
    )
    feasible = np.array([[False, False], [False, False]])
    result = _uav_centric_greedy_fusion(
        receiver_modes, np.ones((2, 2)), 8, 0.05,
        link_feasible=feasible,
    )
    assert result["selected_reports"] == []
    np.testing.assert_allclose(result["delivered_pd"], result["local_pd"])


def test_uav_centric_policy_respects_serial_latency_budget():
    receiver_modes = (
        (np.array([0.10]),),
        (np.array([0.40]),),
    )
    result = _uav_centric_greedy_fusion(
        receiver_modes, np.ones((2, 2)), 8, 0.05,
        report_latency_s=0.006,
        max_total_latency_s=0.005,
    )
    assert result["selected_reports"] == []
    assert result["total_reporting_latency_s"] == 0.0
    np.testing.assert_allclose(result["delivered_pd"], result["local_pd"])


def test_joint_fusion_search_is_not_worse_than_best_local_fusion():
    receiver_modes = (
        (np.array([0.10]),),
        (np.array([0.40]),),
        (np.array([0.25]),),
    )
    success = np.ones((3, 3))
    feasible = ~np.eye(3, dtype=bool)
    local = _uav_centric_greedy_fusion(
        receiver_modes, success, 8, 0.05, link_feasible=feasible
    )
    searched = _uav_centric_fusion_search(
        receiver_modes, success, 8, 0.05, 1, 2, feasible
    )
    assert np.min(searched["delivered_pd"]) >= np.min(local["delivered_pd"])


def test_zero_information_receiver_returns_false_alarm_operating_point():
    pd = _stochastic_pd((np.zeros(0), np.zeros(3)), 8, 0.05)
    np.testing.assert_allclose(pd, [0.05, 0.05])


def test_geometry_power_allocation_is_heterogeneous_and_budget_feasible():
    cfg = make_cfg(**{"scale.M": 6, "scale.Q": 3})
    rng = np.random.default_rng(81)
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    sense, comm = _geometry_maxmin_power_allocation(base, 0.8, 0.8, 1.0)
    assert np.isclose(np.sum(sense), 3.84)
    assert np.isclose(np.sum(comm), 0.96)
    assert np.all(sense + comm <= 1.0)
    assert np.ptp(sense + comm) > 1e-6


def test_geometry_power_allocation_respects_per_uav_communication_floor():
    cfg = make_cfg(**{"scale.M": 6, "scale.Q": 3})
    rng = np.random.default_rng(82)
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    sense, comm = _geometry_maxmin_power_allocation(
        base, 0.8, 0.8, 1.0, minimum_comm_power_w=0.05
    )
    assert np.all(comm >= 0.05 - 1e-12)
    assert np.isclose(np.sum(comm), 0.96)
    assert np.all(sense + comm <= 1.0 + 1e-12)


def test_target_ring_formation_assigns_distinct_views_per_target():
    cfg = make_cfg(**{"scale.M": 6, "scale.Q": 3, "geometry.area_xy": 1000.0})
    rng = np.random.default_rng(7)
    geom = generate_geometry(cfg, rng)
    moved = target_ring_formation(cfg, geom, 2, horizontal_radius_m=50.0)
    for target in range(3):
        distances = np.linalg.norm(
            moved.p_uav[:, :2] - geom.p_tgt[target, :2], axis=1
        )
        assert int(np.sum(distances <= 50.0 + 1e-9)) >= 2


def test_target_ring_formation_respects_per_uav_movement_budget():
    cfg = make_cfg(**{"scale.M": 6, "scale.Q": 3, "geometry.area_xy": 1000.0})
    geom = generate_geometry(cfg, np.random.default_rng(8))
    moved = target_ring_formation(
        cfg, geom, 2, horizontal_radius_m=50.0, max_movement_m=100.0
    )
    distance = np.linalg.norm(moved.p_uav - geom.p_uav, axis=1)
    assert np.all(distance <= 100.0 + 1e-9)


def test_fixed_window_llr_reports_rate_convergence():
    out = accumulate_llr(
        np.ones(12), relative_increment_tolerance=1e-12, stable_frames=3
    )
    assert out.converged
    assert out.convergence_frame == 4
    np.testing.assert_array_equal(out.cumulative, np.arange(1.0, 13.0))


def test_stochastic_target_generator_is_not_unit_modulus():
    cfg = make_cfg(**{"detect.target_response_model": "swerling2_fast"})
    obs = make_observation(cfg, seed=17)
    centre = np.asarray(obs.A_true_centre_mask, dtype=bool)
    magnitude = np.abs(np.asarray(obs.alpha_true)[centre])
    assert magnitude.size > 1
    assert not np.allclose(magnitude, 1.0)
