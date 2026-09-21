"""Contracts for matrix information, explicit protection, AO and frame evidence."""
from __future__ import annotations

import numpy as np

from isac_sim.cooperation.matrix_information_ao import (
    monotone_matrix_information_ao,
    protection_subsets,
)
from isac_sim.detection.multiframe_llr import accumulate_llr
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.sensing.model import build_base_gains, generate_geometry
from _tpuic_common import make_cfg, make_observation


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


def test_fixed_window_llr_reports_rate_convergence():
    out = accumulate_llr(
        np.ones(12), relative_increment_tolerance=1e-12, stable_frames=3
    )
    assert out.converged
    assert out.convergence_frame == 4
    np.testing.assert_array_equal(out.cumulative, np.arange(1.0, 13.0))
