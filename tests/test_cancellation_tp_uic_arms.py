"""TP-UIC cancellation arms: protection invariant, residual accounting, prediction.

Split out of ``test_cancellation_tp_uic.py`` (2026-09-21 audit, 773 -> 3 files).
Function bodies are moved verbatim; the two builders now live in
``_tpuic_common.py``.  Covered by ``tests/_test_inventory.py``.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from isac_sim.receiver import cancellation as cx
from isac_sim.core.config import apply_overrides
from isac_sim.sensing.model import generate_geometry, build_base_gains, noise_power
from _tpuic_common import make_cfg, make_observation


# --------------------------------------------------------------------------
# Module M2: the protection invariant
# --------------------------------------------------------------------------
def test_protected_component_is_untouched_by_stage_one():
    """Eq. (2c): ``P r1 = P y`` for the *stage-1* protected arms.

    This is the property that separates "the canceller left some interference"
    from "the canceller ate the target": the subspace the receiver declared as
    target-carrying is passed through bit-for-bit.

    Stage 2 is deliberately excluded.  ``tp_uic_full`` models the targets
    explicitly and then releases that protection -- that is its entire purpose
    -- so requiring the invariant there would forbid the arm from working.  The
    second half of the test pins the release instead of ignoring it.
    """
    cfg = make_cfg()
    obs = make_observation(cfg, belief_error=True)
    assert obs.basis_belief.shape[1] > 0, "protection basis should not be empty"
    U = obs.basis_belief
    project = lambda v: U @ (U.conj().T @ v)
    arms = cx.cancellation_arms(cfg, obs)
    expected = project(obs.y)
    scale = max(float(np.linalg.norm(expected)), 1e-30)
    for name in ("protected_ls", "tp_uic_stage1"):
        err = np.linalg.norm(project(arms[name].residual) - expected) / scale
        assert err < 1e-8, f"{name}: ||P r - P y|| / ||P y|| = {err:.2e}"

    stage1 = np.linalg.norm(project(arms["tp_uic_stage1"].residual))
    full = np.linalg.norm(project(arms["tp_uic_full"].residual))
    assert full < stage1, "the joint stage must release energy from the protected subspace"


def test_unprotected_arms_do_not_protect():
    cfg = make_cfg()
    obs = make_observation(cfg, belief_error=True)
    arms = cx.cancellation_arms(cfg, obs)
    for name in ("no_ic", "plain_ls", "ridge_ls", "perfect_channel"):
        assert arms[name].protect_dim == 0
    assert arms["protected_ls"].protect_dim > 0


# --------------------------------------------------------------------------
# Module M3/M4: eq. (3) must hold, and must reduce to the known limit
# --------------------------------------------------------------------------
def test_unprotected_ls_prediction_is_sigma2_times_rank():
    """Without a prior, the estimation residual is exactly ``sigma^2 * rank(X)``.

    ``tr(C_h G) = tr(sigma^2 G^+ G) = sigma^2 * rank(G)``.  The elementwise form
    ``sum_i C_h[i,i] G[i,i]`` does *not* give this whenever the Gram is
    non-diagonal, and on this scenario it is not -- which is how the wrong
    formula was caught.
    """
    # ``0.0`` (not ``None``): the config field is typed ``float | None`` and the
    # override coercion only accepts numbers, which is itself a small reminder
    # that "no prior" has to mean a number here.
    cfg = apply_overrides(make_cfg(), {"cancellation.prior_variance": 0.0})
    obs = make_observation(cfg)
    empty = cx.orthonormalise(np.zeros((obs.y.size, 0)))
    retained, estimate = cx.residual_accounting(obs.X, empty, obs.sigma2, None)
    rank = int(np.linalg.matrix_rank(obs.X, tol=1e-12 * np.linalg.norm(obs.X)))
    assert retained == 0.0
    assert estimate == pytest.approx(obs.sigma2 * rank, rel=1e-6)


def test_full_arm_prediction_uses_actual_xh_subtraction_operator():
    """The joint arm subtracts X h, so uncertainty propagates through X."""
    cfg = make_cfg()
    obs = make_observation(cfg)
    full = cx.cancellation_arms(cfg, obs, threshold=0.0)["tp_uic_full"]
    A_cand = obs.A[:, list(full.candidates)]
    pv = cfg.cancellation.prior_variance
    cov_h = cx.joint_interference_covariance(
        obs.X, A_cand, obs.sigma2, pv, pv or 1.0
    )
    expected = float(np.real(np.trace(cov_h @ (obs.X.conj().T @ obs.X))))

    assert full.candidates
    assert full.i_res_retained == pytest.approx(0.0, abs=1e-18)
    assert full.i_res_pred_estimate == pytest.approx(expected, rel=1e-10)


def test_residual_moments_match_exported_quadratic_descriptors():
    cfg = make_cfg()
    obs = make_observation(cfg)
    result = cx.cancellation_arms(cfg, obs)["tp_uic_full"]
    gram = np.asarray(result.residual_direct_gram)
    eigenvalues = np.asarray(result.residual_noise_eigenvalues)
    mean = float(np.real(np.trace(gram)) + np.sum(eigenvalues))
    structural_var = float(
        np.sum(np.abs(gram) ** 2)
        - np.sum(np.abs(np.diag(gram)) ** 2)
    )
    variance = structural_var + float(np.sum(eigenvalues ** 2))
    assert result.i_res_moment_mean == pytest.approx(mean, rel=1e-10)
    assert result.i_res_pred_var == pytest.approx(variance, rel=1e-10)


def test_prior_residual_quantile_is_deterministic_and_ordered():
    cfg = make_cfg()
    obs = make_observation(cfg)
    result = cx.cancellation_arms(cfg, obs)["tp_uic_full"]
    q50a = cx.residual_power_prior_quantile(result, 0.50)
    q50b = cx.residual_power_prior_quantile(result, 0.50)
    q80 = cx.residual_power_prior_quantile(result, 0.80)
    assert q50a == q50b
    assert 0.0 <= q50a <= q80


def test_soft_tpuic_zero_weight_is_exactly_ridge_ls():
    cfg = make_cfg(**{"cancellation.soft_protection_mu": 0.0})
    obs = make_observation(cfg)
    arms = cx.cancellation_arms(cfg, obs)
    assert np.allclose(arms["soft_tpuic"].h_hat, arms["ridge_ls"].h_hat)
    assert np.allclose(arms["soft_tpuic"].residual, arms["ridge_ls"].residual)


def test_soft_weight_monotonically_reduces_target_subspace_removal():
    cfg = make_cfg()
    obs = make_observation(cfg)
    belief = cx.Subspace(obs.basis_belief, obs.basis_belief.shape[1])
    removed = []
    for mu in (0.0, 0.1, 1.0, 10.0, 100.0):
        h, _cov = cx.soft_protected_map(
            obs.s_target, obs.X, belief, obs.sigma2,
            cfg.cancellation.prior_variance, mu,
        )
        projected = belief.project(obs.X @ h)
        removed.append(float(np.vdot(projected, projected).real))
    assert all(b <= a * (1.0 + 1e-10) + 1e-24 for a, b in zip(removed, removed[1:]))
    assert removed[-1] < removed[0]


def test_adaptive_soft_is_belief_pareto_safe_or_exact_hard_fallback():
    cfg = make_cfg(**{
        "cancellation.adaptive_soft_enable": True,
        "cancellation.adaptive_soft_mu_grid": (0.0, 0.1, 1.0, 10.0, 100.0),
    })
    obs = make_observation(cfg)
    arms = cx.cancellation_arms(cfg, obs)
    hard = arms["tp_uic_full"]
    adaptive = arms["adaptive_soft_tpuic"]

    assert adaptive.i_res_moment_mean <= hard.i_res_moment_mean * (1.0 + 1e-10)
    assert adaptive.eta_survive_risk_q + 1e-12 >= hard.eta_survive_risk_q
    assert cx.residual_power_prior_quantile(
        adaptive, cfg.cancellation.adaptive_soft_residual_quantile
    ) <= cx.residual_power_prior_quantile(
        hard, cfg.cancellation.adaptive_soft_residual_quantile
    ) * (1.0 + 1e-10)
    if adaptive.soft_mu is None:
        assert np.array_equal(adaptive.residual, hard.residual)
        assert np.array_equal(adaptive.h_hat, hard.h_hat)
    else:
        assert adaptive.soft_mu in cfg.cancellation.adaptive_soft_mu_grid


def test_adaptive_soft_pruned_measurement_keeps_hard_reference_dependency():
    """生产测量只装 adaptive 臂时仍必须保留 hard 参考并返回同一结果。

    背景：联合闭环 pilot 通过 ``measure_prune_arms`` 请求单臂时曾因 adaptive
    没有被装配而直接 KeyError。这里钉装配依赖，不覆盖端到端检测增益。
    """
    cfg = make_cfg(**{"cancellation.adaptive_soft_enable": True})
    obs = make_observation(cfg)
    full = cx.cancellation_arms(cfg, obs)["adaptive_soft_tpuic"]
    pruned = cx.cancellation_arms(
        cfg, obs, only="adaptive_soft_tpuic"
    )["adaptive_soft_tpuic"]

    assert pruned.soft_mu == full.soft_mu
    np.testing.assert_array_equal(pruned.residual, full.residual)
    np.testing.assert_array_equal(pruned.h_hat, full.h_hat)


def test_adaptive_soft_arm_is_absent_by_default():
    cfg = make_cfg()
    obs = make_observation(cfg)
    assert "adaptive_soft_tpuic" not in cx.cancellation_arms(cfg, obs)


def test_targeted_hard_tpuic_spends_rank_only_on_tested_target():
    cfg = make_cfg()
    obs = make_observation(cfg)
    target = int(obs.weak_index)
    arms = cx.cancellation_arms(cfg, obs, weak_target=target)
    targeted = arms["targeted_tpuic_stage1"]
    union = arms["tp_uic_stage1"]
    ids = np.asarray(obs.A_target_ids)
    target_block = cx.orthonormalise(obs.A[:, ids == target])

    assert 0 < targeted.protect_dim <= union.protect_dim
    assert set(ids[list(arms["targeted_tpuic_full"].candidates)]) == {target}
    projected_before = target_block.project(obs.y)
    projected_after = target_block.project(targeted.residual)
    assert np.allclose(projected_after, projected_before, rtol=1e-9, atol=1e-14)


def test_prediction_tracks_measurement_for_plain_ls():
    """Measured estimation residual vs eq. (3), on one trial."""
    cfg = make_cfg()
    obs = make_observation(cfg)
    arms = cx.cancellation_arms(cfg, obs)
    r = arms["plain_ls"]
    # Plain LS removes the direct component essentially exactly (the model is
    # linear in X), so the residual is the estimation term.
    assert r.i_res_structural < 1e-3 * r.i_res_estimate
    assert r.i_res == pytest.approx(r.i_res_pred, rel=0.25)


def test_reference_budget_scales_the_estimation_term():
    """``n_cpi`` enters the prediction as a genuine budget, not decoration."""
    cfg = make_cfg()
    i_sense = np.array([1e-9, 4e-9])
    n_illum = np.array([14.0, 14.0])
    total1, retained1 = cx.predict_cancellation(cfg, i_sense, n_illum)
    cfg2 = apply_overrides(cfg, {"cancellation.n_cpi": 16})
    total2, retained2 = cx.predict_cancellation(cfg2, i_sense, n_illum)
    # 16 CPIs of coherent reference integration buy 12.0 dB on the estimation
    # term and nothing on the retention term.  The total barely moves (0.2 dB)
    # because retention dominates -- which is the finding, not an inconvenience.
    estimate1, estimate2 = total1 - retained1, total2 - retained2
    assert 10.0 * math.log10(estimate1[0] / estimate2[0]) == pytest.approx(12.04, abs=0.1)
    assert retained1 == pytest.approx(retained2)
    assert 10.0 * math.log10(total1[0] / total2[0]) < 0.5


# --------------------------------------------------------------------------
# The arms that are definitions, not estimators
# --------------------------------------------------------------------------
def test_the_definition_arms_are_the_only_two_bounds():
    """下界与 oracle 两条"定义臂"必须仍是唯二不估计的臂。

    ``fixed_kappa`` 曾在这里：它把模型的常数读回样本域，好让那个常数与算法
    同单位比较。常数连同 ``interference.direct_cancellation_db`` 一起被删了
    （没有接收机实现支撑却撑着分母），所以这条臂也不该再存在 —— 若它复活，
    说明有人又把一个未被实现的对消假设塞回了比较表。
    """
    cfg = make_cfg()
    obs = make_observation(cfg)
    arms = cx.cancellation_arms(cfg, obs)
    assert "fixed_kappa" not in arms
    assert arms["perfect_channel"].i_res == pytest.approx(0.0, abs=1e-30)
    assert arms["no_ic"].kappa_db == pytest.approx(0.0, abs=1e-12)
    assert arms["no_ic"].i_res == pytest.approx(arms["no_ic"].i_in)


def test_direct_field_matches_the_link_table_field():
    """The bridge: ``||X h||^2`` is the model's ``I_sense_field``.

    The two are not identical -- distinct illuminators' DD kernels are not
    exactly orthogonal -- so the test pins the *size* of the cross term rather
    than pretending it is zero.  This is the identification that lets a measured
    residual be fed back as an effective ``kappa_dc``.
    """
    cfg = make_cfg()
    obs = make_observation(cfg)
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    rng = np.random.default_rng([cfg.run.seed, 0])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    model_field = float((sense @ base.direct_gain)[0])
    measured = float(np.vdot(obs.x_direct, obs.x_direct).real)
    assert measured == pytest.approx(model_field, rel=0.15)


# --------------------------------------------------------------------------
# Accounting guards: a ratio must not be clamped by a fixed epsilon
# --------------------------------------------------------------------------
def test_echo_survival_is_a_pure_ratio_below_the_eps_guard():
    """``eta_survive`` must be 1.0 for an arm that removes nothing, at any scale.

    The old denominator was ``max(s_energy, EPS)`` with ``EPS = 1e-12``, and a
    real echo on this scenario is smaller than that (measured 1.21e-13 W on one
    single-target trial), so ``no_ic`` -- whose survival is 1.0 by definition --
    reported 0.1209.  The same trap is documented on ``_positive_sigma``; it
    survived here only because the old echo generator wrote three unit-modulus
    scatterers per target and inflated ``s_energy`` above the guard.
    """
    cfg = make_cfg()
    obs = make_observation(cfg)
    tiny = replace(obs, s_target=obs.s_target * 1e-8)
    assert float(np.vdot(tiny.s_target, tiny.s_target).real) < cx.EPS
    arms = cx.cancellation_arms(cfg, tiny)
    assert arms["no_ic"].eta_survive == pytest.approx(1.0)
    assert arms["perfect_channel"].eta_survive == pytest.approx(1.0)
    assert 0.0 <= arms["no_ic"].eta_protect <= 1.0
