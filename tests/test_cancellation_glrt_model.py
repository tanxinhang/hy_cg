"""TP-UIC GLRT: the chi-square helper, the affine-arm claim and the covariance claim.

Split out of ``test_cancellation_glrt.py`` (2026-09-21 audit, 512 -> 2 files).
Function bodies are moved verbatim; the four builders live in
``_glrt_common.py``.  Covered by ``tests/_test_inventory.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from _glrt_common import make_arms, make_cfg, make_trial


# --------------------------------------------------------------------------
# The chi-square helper: the library must not need scipy to set a threshold
# --------------------------------------------------------------------------
def test_chi2_cdf_is_monotone_and_normalised():
    for half in (1, 2, 3, 8, 14):
        xs = np.linspace(0.0, 40.0 * half, 60)
        cdf = [gl._chi2_cdf_even(x, half) for x in xs]
        assert all(b >= a - 1e-15 for a, b in zip(cdf, cdf[1:]))
        assert cdf[0] == 0.0
        assert gl._chi2_cdf_even(1e4 * half, half) > 1.0 - 1e-12


def test_chi2_quantile_matches_scipy():
    """Cross-check the dependency-free quantile against the reference."""
    stats = pytest.importorskip("scipy.stats")
    for half in (1, 2, 3, 6, 14, 40):
        for p in (0.01, 0.05, 0.5, 0.9, 0.99):
            got = gl._chi2_quantile_even(p, half)
            want = float(stats.chi2.ppf(p, 2 * half))
            assert got == pytest.approx(want, rel=1e-6, abs=1e-9)


def test_glrt_threshold_refuses_an_odd_dof():
    with pytest.raises(ValueError, match="even"):
        gl.glrt_threshold(0.05, 7)


def test_threshold_achieves_the_requested_false_alarm_rate():
    """The dof convention: ``T`` is ``(1/2) chi^2`` with ``2 * rank`` dof.

    This is the claim that makes the CFAR level meaningful at all, and it is
    cheap to check on white noise instead of hoping it holds on the scenario.
    """
    rng = np.random.default_rng(0)
    K, trials, p_fa = 256, 8000, 0.05
    B = rng.normal(size=(K, 3)) + 1j * rng.normal(size=(K, 3))
    rank = gl._numerical_rank(B)
    threshold = gl.glrt_threshold(p_fa, 2 * rank)
    noise = (rng.normal(size=(K, trials)) + 1j * rng.normal(size=(K, trials))) / math.sqrt(2.0)
    proj = B @ np.linalg.lstsq(B, noise, rcond=None)[0]
    stat = np.real(np.sum(np.abs(proj) ** 2, axis=0))
    rate = float(np.mean(stat > threshold))
    assert rate == pytest.approx(p_fa, abs=3.0 * math.sqrt(p_fa * (1 - p_fa) / trials))
    # the mean of the statistic is the complex dimension -- the dof convention
    assert float(np.mean(stat)) == pytest.approx(rank, rel=0.05)


# --------------------------------------------------------------------------
# Claim 1: the arms are affine maps
# --------------------------------------------------------------------------
def test_every_arm_is_reproduced_by_its_low_rank_form():
    """``remove(y) + const`` must equal what the arm actually removes.

    If V1's arm table is ever edited without this file following, the probe
    stops reproducing the arm and every whitened statistic silently describes a
    canceller that does not exist.
    """
    cfg = make_cfg()
    obs, _, target = make_trial(cfg)
    arms, plans = make_arms(cfg, obs)
    for name in gl.ARM_ORDER:
        model = gl.residual_model(cfg, obs, name, arms, plans=plans)
        got = model.remove(obs.y) + model.const
        want = obs.y - arms[name].residual
        scale = max(float(np.linalg.norm(want)), 1e-300)
        assert float(np.linalg.norm(got - want)) / scale < 1e-6, name
        assert model.probe_error < 1e-6, name


def test_adaptive_soft_arm_is_reproduced_by_its_low_rank_form():
    cfg = make_cfg(**{
        "cancellation.adaptive_soft_enable": True,
        "cancellation.adaptive_soft_risk_slack": 1.0,
    })
    obs, _, target = make_trial(cfg)
    arms = cx.cancellation_arms(cfg, obs, weak_target=target)
    adaptive = arms["adaptive_soft_tpuic"]
    assert adaptive.soft_mu is not None
    plans = gl.arm_plans(cfg, obs, arms)
    model = gl.residual_model(
        cfg, obs, "adaptive_soft_tpuic", arms, plans=plans
    )
    got = model.remove(obs.y) + model.const
    want = obs.y - adaptive.residual
    scale = max(float(np.linalg.norm(want)), 1e-300)
    assert float(np.linalg.norm(got - want)) / scale < 1e-6
    assert model.probe_error < 1e-6


def test_null_covariance_threshold_reduces_to_standard_for_same_model():
    cfg = make_cfg()
    obs, _, target = make_trial(cfg)
    arms, plans = make_arms(cfg, obs)
    model = gl.residual_model(cfg, obs, "tp_uic_full", arms, plans=plans)
    standard = gl.target_conditioned_glrt(
        cfg, obs, arms["tp_uic_full"], model, target=target
    )
    calibrated = gl.target_conditioned_glrt(
        cfg, obs, arms["tp_uic_full"], model, target=target,
        null_cov_model=model,
    )
    assert calibrated.threshold == pytest.approx(standard.threshold, rel=1e-10)


def test_oracle_arms_do_not_depend_on_the_observation():
    """``no_ic`` and ``perfect_channel`` remove a constant, so ``F = 0``.

    Their transfer on the echo is therefore the identity: an oracle that
    subtracts a known vector does not attenuate a target.

    ``fixed_kappa`` used to sit in this loop; it went away with
    ``interference.direct_cancellation_db``, so there is no longer a third
    non-estimator arm.
    """
    cfg = make_cfg()
    obs, _, _ = make_trial(cfg)
    arms, _ = make_arms(cfg, obs)
    for name in ("no_ic", "perfect_channel"):
        model = gl.residual_model(cfg, obs, name, arms)
        assert model.rank == 0
        A = obs.A[:, :4]
        assert np.allclose(model.signal_transfer(A), A)


def test_residual_model_needs_the_recorded_result():
    cfg = make_cfg()
    obs, _, _ = make_trial(cfg)
    with pytest.raises(KeyError):
        gl.residual_model(cfg, obs, "plain_ls", None)


# --------------------------------------------------------------------------
# Claim 2: C_res is identity plus low rank, and is whitened exactly
# --------------------------------------------------------------------------
def test_whitener_satisfies_the_defining_identity():
    """``(M u)^H C (M v) = u^H v`` for ``M = C^{-1/2}``."""
    cfg = make_cfg()
    obs, _, _ = make_trial(cfg)
    arms, plans = make_arms(cfg, obs)
    rng = np.random.default_rng(1)
    for name in ("no_ic", "plain_ls", "tp_uic_stage1", "perfect_channel"):
        model = gl.residual_model(cfg, obs, name, arms, plans=plans)
        cov = model.cov
        for _ in range(3):
            u = rng.normal(size=obs.y.size) + 1j * rng.normal(size=obs.y.size)
            v = rng.normal(size=obs.y.size) + 1j * rng.normal(size=obs.y.size)
            lhs = np.vdot(cov.whiten(u), cov.apply(cov.whiten(v)))
            assert lhs == pytest.approx(np.vdot(u, v), rel=1e-8, abs=1e-12), name


def test_whitener_inverse_round_trips():
    cfg = make_cfg()
    obs, _, _ = make_trial(cfg)
    arms, _ = make_arms(cfg, obs)
    rng = np.random.default_rng(2)
    model = gl.residual_model(cfg, obs, "tp_uic_full", arms)
    v = rng.normal(size=obs.y.size) + 1j * rng.normal(size=obs.y.size)
    assert np.allclose(model.cov.inverse(model.cov.apply(v)), v, rtol=1e-8, atol=1e-14)


def test_residual_covariance_matches_its_own_generative_model():
    """Monte-Carlo the story the covariance claims, not the simulation's.

    ``C_res`` describes a *reference-based* canceller: the direct field is scaled
    by ``T_d``, the noise floor is ``sigma^2 I``, and the coefficient error is an
    independent ``C_h / n_cpi``.  Sampling exactly that and comparing Rayleigh
    quotients is what validates the low-rank eigen representation; comparing
    against the self-fit simulation would (correctly) disagree, because that is a
    different canceller.
    """
    cfg = make_cfg()
    obs, _, _ = make_trial(cfg)
    arms, plans = make_arms(cfg, obs)
    rng = np.random.default_rng(3)
    n_samples = 1500
    for name in ("plain_ls", "tp_uic_stage1", "tp_uic_full"):
        model = gl.residual_model(cfg, obs, name, arms, plans=plans)
        plan = plans[name]
        L = gl._psd_sqrt(gl._estimator_covariance(cfg, obs, plan)) / math.sqrt(
            float(cfg.cancellation.n_cpi)
        )
        subtraction_dictionary = (
            obs.X if plan.candidates else plan.subspace.complement_matrix(obs.X)
        )
        X = obs.X
        d = X.shape[1]
        res = np.zeros((obs.y.size, n_samples), dtype=complex)
        for j in range(n_samples):
            h = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=d))
            e = (rng.normal(size=d) + 1j * rng.normal(size=d)) / math.sqrt(2.0)
            n = (rng.normal(size=obs.y.size) + 1j * rng.normal(size=obs.y.size)) * math.sqrt(
                obs.sigma2 / 2.0
            )
            res[:, j] = (
                model.signal_transfer(X @ h) + n
                - subtraction_dictionary @ (L @ e)
            )
        for _ in range(3):
            v = rng.normal(size=obs.y.size) + 1j * rng.normal(size=obs.y.size)
            v = v / np.linalg.norm(v)
            empirical = float(np.mean(np.abs(v.conj() @ res) ** 2))
            claimed = float(np.real(np.vdot(v, model.cov.apply(v))))
            assert empirical == pytest.approx(claimed, rel=0.15), (name, empirical, claimed)


def test_reference_covariance_is_conservative_against_a_self_fit():
    """The stated floor is ``sigma^2 I``, which can only over-state a self-fit.

    A canceller that fits on the observation it cleans also removes the noise in
    the fitted subspace; stating ``sigma^2 I`` instead leaves that noise in the
    model, so the detector is conservative by ``sigma^2 tr(F F^H)`` plus the
    coefficient-error term.  The direction of the error is pinned here so nobody
    later "fixes" the calibration in the flattering direction.
    """
    cfg = make_cfg()
    obs, _, _ = make_trial(cfg)
    arms, plans = make_arms(cfg, obs)
    for name in ("plain_ls", "protected_ls"):
        model = gl.residual_model(cfg, obs, name, arms, plans=plans)
        s = model.small
        # tr((I-F)(I-F)^H) for the *self-fit* noise floor
        self_fit = obs.sigma2 * (
            obs.y.size - 2.0 * np.real(np.trace(s)) + np.real(np.trace(s.conj().T @ s))
        )
        assert model.cov.trace > self_fit - 1e-12, name
