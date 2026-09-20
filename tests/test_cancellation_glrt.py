"""Invariants of the TP-UIC V1.1 detector (``isac_sim.cancellation_glrt``).

The V1.1 detector is built on four claims, and each of them is a property that
fails quietly if it fails at all:

1. every arm is an affine map, so the residual covariance can be written down;
2. ``C_res`` is identity plus low rank, so it can be whitened exactly;
3. ``T_q`` is calibrated, so a 5 % threshold really is a 5 % threshold;
4. ``rho`` measures masking and nothing else.

The tests below pin those four and the two limits of ``rho``.  They deliberately
do **not** pin any measured number: a scenario number belongs in the experiment
and the document, where it can carry its caveats, not in a test that would then
fail for the wrong reason when the geometry changes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from isac_sim import cancellation as cx
from isac_sim import cancellation_glrt as gl
from isac_sim.config import Config, apply_overrides, apply_preset
from isac_sim.model import build_base_gains, generate_geometry
from isac_sim.prior import perturbed_geometry


def make_cfg(**overrides) -> Config:
    cfg = apply_preset(Config(), "small-uav-compact-800m")
    base = {
        "geometry.area_xy": 600.0,
        "detect.target_rcs": 0.1,
        "scale.M": 6,
        "scale.Q": 3,
        "run.seed": 2026,
        "run.verbose": False,
        "cancellation.enable": True,
        "cancellation.max_protected_targets": 3,
    }
    base.update(overrides)
    return apply_overrides(cfg, base)


def make_canonical_cfg(**overrides) -> Config:
    """The released scenario: 15 UAVs, 10 targets, 600 m, RCS 0.1.

    Two claims in this file are properties of *that* geometry rather than of the
    detector -- the local patch covering and the off-grid recovery -- so they are
    tested where they live instead of being asserted on a toy configuration
    where three targets do not overlap at all.
    """
    cfg = apply_preset(Config(), "paper-canonical")
    base = {
        "geometry.area_xy": 600.0,
        "detect.target_rcs": 0.1,
        "run.seed": 2026,
        "run.verbose": False,
        "cancellation.enable": True,
        "cancellation.max_protected_targets": 3,
    }
    base.update(overrides)
    return apply_overrides(cfg, base)


def make_trial(cfg: Config, seed: int = 0, belief_error: bool = True, receiver: int = 0):
    """One matched ``(H1, H0)`` pair plus the tested target."""
    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    belief = (
        perturbed_geometry(
            cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
        )
        if belief_error
        else geom
    )
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    probe = cx.build_observation(
        cfg, geom, belief, base, receiver, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0,
    )
    ids = np.asarray(probe.A_target_ids)
    target = int(ids[0]) if ids.size else 0
    obs1, obs0 = cx.build_observation_pair(
        cfg, geom, belief, base, receiver, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0, exclude_target=target, weak_index=target,
    )
    return obs1, obs0, target


def make_arms(cfg: Config, obs):
    arms = cx.cancellation_arms(cfg, obs, weak_target=int(obs.weak_index), threshold=0.0)
    return arms, gl.arm_plans(cfg, obs, arms)


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
    """``fixed_kappa`` and ``perfect_channel`` remove a constant, so ``F = 0``.

    Their transfer on the echo is therefore the identity: an oracle that
    subtracts a known vector does not attenuate a target.
    """
    cfg = make_cfg()
    obs, _, _ = make_trial(cfg)
    arms, _ = make_arms(cfg, obs)
    for name in ("no_ic", "fixed_kappa", "perfect_channel"):
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


# --------------------------------------------------------------------------
# Claim 3: rho is masking and nothing else
# --------------------------------------------------------------------------
def _nuisance_block(cfg, obs, target):
    """Exactly the nuisance columns the GLRT will use: centre, other targets."""
    A, ids = obs.A, np.asarray(obs.A_target_ids)
    mask = gl._centre_mask(cfg, A.shape[1])
    return A[:, mask & (ids != int(target))]


def test_rho_is_one_when_there_is_no_nuisance():
    cfg = make_cfg()
    obs, _, target = make_trial(cfg)
    arms, plans = make_arms(cfg, obs)
    model = gl.residual_model(cfg, obs, "no_ic", arms, plans=plans)
    got = gl.target_conditioned_glrt(
        cfg, obs, arms["no_ic"], model, target=target, nuisance_targets=[]
    )
    assert got.n_templates > 0
    assert np.allclose(got.rho, 1.0, atol=1e-9)
    assert got.rho_weighted == pytest.approx(1.0, abs=1e-9)


def test_rho_is_zero_for_a_template_inside_the_nuisance():
    """The lower limit: an echo that is a combination of the others is invisible."""
    cfg = make_cfg()
    obs, _, target = make_trial(cfg)
    arms, plans = make_arms(cfg, obs)
    model = gl.residual_model(cfg, obs, "no_ic", arms, plans=plans)
    A_neg = _nuisance_block(cfg, obs, target)
    coef = np.arange(1, A_neg.shape[1] + 1, dtype=float)[:, None]
    fake = A_neg @ (coef * (1.0 + 0j))
    got = gl.target_conditioned_glrt(
        cfg, obs, arms["no_ic"], model, target=target, template_override=fake
    )
    assert got.rho_weighted < 1e-9
    # ... and the same template is perfectly identifiable once the nuisance is gone
    free = gl.target_conditioned_glrt(
        cfg, obs, arms["no_ic"], model, target=target, template_override=fake,
        nuisance_targets=[],
    )
    assert free.rho_weighted == pytest.approx(1.0, abs=1e-9)


def test_glrt_is_invariant_to_the_nuisance_amplitudes():
    """``T_q`` must not move when the other targets' echoes change.

    That is the whole content of projecting the nuisance out, and it is exact
    rather than statistical: what the projector removes is the nuisance itself.
    """
    from dataclasses import replace

    cfg = make_cfg()
    obs, _, target = make_trial(cfg)
    arms, plans = make_arms(cfg, obs)
    model = gl.residual_model(cfg, obs, "no_ic", arms, plans=plans)
    base = gl.target_conditioned_glrt(cfg, obs, arms["no_ic"], model, target=target)
    rng = np.random.default_rng(4)
    A_neg = _nuisance_block(cfg, obs, target)
    delta = rng.normal(size=A_neg.shape[1]) + 1j * rng.normal(size=A_neg.shape[1])
    shifted = replace(obs, y=obs.y + A_neg @ delta)
    got = gl.target_conditioned_glrt(cfg, shifted, arms["no_ic"], model, target=target)
    assert got.statistic == pytest.approx(base.statistic, rel=1e-8)
    assert np.allclose(got.rho, base.rho)


def test_masking_grows_with_the_number_of_nuisance_targets():
    cfg = make_cfg()
    obs, _, target = make_trial(cfg)
    arms, plans = make_arms(cfg, obs)
    model = gl.residual_model(cfg, obs, "no_ic", arms, plans=plans)
    curve = gl.masking_curve(cfg, obs, model, arms["no_ic"], target=target, counts=(0, 1, 2))
    assert curve.rho_weighted[0] == pytest.approx(1.0, abs=1e-9)
    assert all(
        b <= a + 1e-12 for a, b in zip(curve.rho_weighted, curve.rho_weighted[1:])
    ), curve.rho_weighted


def test_nuisance_manifold_is_lifted_into_spatial_observation_space():
    cfg = make_cfg(**{"aperture.enable": True, "aperture.m_rx": 4})
    obs, _, target = make_trial(cfg)
    extra = gl._manifold_columns(cfg, obs, target, order=1)
    sources = [s for s in obs.targets_belief if int(s.target) != int(target)]
    assert extra.shape[0] == obs.y.size
    assert extra.shape[1] == 4 * len(sources)  # centre DD + 2 DD tangents + angle


def test_belief_error_covariance_is_lifted_into_spatial_observation_space():
    cfg = make_cfg(**{
        "aperture.enable": True,
        "aperture.m_rx": 4,
        "cancellation.belief_error_in_cres": True,
    })
    obs, _, _ = make_trial(cfg)
    factor = gl._belief_error_factor(cfg, obs)
    assert factor.shape[0] == obs.y.size
    arms = cx.cancellation_arms(cfg, obs)
    model = gl.residual_model(cfg, obs, "tp_uic_full", arms)
    assert model.cov.W.shape[0] == obs.y.size


def test_moving_the_template_off_the_grid_restores_identifiability():
    """The audit's decisive test: masking is local, so a far template escapes.

    Run on the released 600 m / 10-target geometry, because that is where the
    co-location exists at all.  If this ever fails the conclusion flips: the
    masking would be intrinsic to the kernel family rather than a property of
    where the scatterers sit, and "more physical resolution" would stop being
    the remedy.
    """
    cfg = make_canonical_cfg()
    obs, _, target = make_trial(cfg, receiver=1)
    arms, plans = make_arms(cfg, obs)
    model = gl.residual_model(cfg, obs, "no_ic", arms, plans=plans)
    audit = gl.identifiability_audit(
        cfg, obs, model, arms["no_ic"], target=target,
        deltas=((5.0, 5.0), (-5.0, -5.0)),
    )
    on_grid = float(np.median(audit.rho_on_grid))
    far = float(np.median(audit.off_grid[(5.0, 5.0)]))
    assert on_grid < 0.1, "the scenario no longer co-locates the targets"
    assert far > 10.0 * max(on_grid, 1e-12), (on_grid, far)


def test_the_same_template_moves_with_the_canceller():
    """Masking must be a geometric fact, not an artefact of the canceller.

    ``rho`` is computed on the *whitened* templates, so a scenario in which the
    answer depended on the arm would be a scenario where the whitening, not the
    geometry, decided identifiability.  The released geometry is not one of
    those, and that is what lets the paper attribute the failure to
    co-location instead of to TP-UIC.
    """
    cfg = make_canonical_cfg()
    obs, _, target = make_trial(cfg, receiver=1)
    arms, plans = make_arms(cfg, obs)
    values = []
    for name in ("no_ic", "plain_ls", "tp_uic_full", "perfect_channel"):
        model = gl.residual_model(cfg, obs, name, arms, plans=plans)
        got = gl.target_conditioned_glrt(cfg, obs, arms[name], model, target=target)
        values.append(got.rho_weighted)
    assert max(values) - min(values) < 0.15, values


# --------------------------------------------------------------------------
# Claim 4: the single-target arm really has one target
# --------------------------------------------------------------------------
def test_restrict_to_target_leaves_exactly_one_echo():
    cfg = make_cfg()
    obs, _, target = make_trial(cfg)
    single = gl.restrict_to_target(cfg, obs, target)
    ids = np.asarray(single.A_true_target_ids)
    assert set(ids.tolist()) == {int(target)}
    assert {int(s.target) for s in single.targets} == {int(target)}
    assert {int(s.target) for s in single.targets_belief} == {int(target)}
    assert set(np.asarray(single.A_target_ids).tolist()) == {int(target)}
    # the noise draw is preserved, so the two observations differ only in the echo
    noise_obs = obs.y - obs.x_direct - obs.s_target
    noise_single = single.y - single.x_direct - single.s_target
    assert np.allclose(noise_obs, noise_single)
    assert float(np.vdot(single.s_target, single.s_target).real) < float(
        np.vdot(obs.s_target, obs.s_target).real
    )


def test_single_target_arm_has_no_nuisance_and_detects():
    """The mechanism validation must be a *detection*, or it validates nothing.

    Run with a strong echo (RCS 50) on purpose: the assertion here is that the
    pipeline detects a target that is unambiguously there, i.e. that removing the
    multi-target question leaves a working detector.  Whether it still detects at
    RCS 0.1 is an experimental result, not an invariant, and it is reported as
    one.
    """
    cfg = make_cfg(**{"detect.target_rcs": 50.0})
    obs, _, target = make_trial(cfg)
    single = gl.restrict_to_target(cfg, obs, target)
    arms, plans = make_arms(cfg, single)
    for name in ("no_ic", "tp_uic_full", "perfect_channel"):
        model = gl.residual_model(cfg, single, name, arms, plans=plans)
        got = gl.target_conditioned_glrt(cfg, single, arms[name], model, target=target)
        assert got.n_nuisance_columns == 0
        assert got.rho_weighted == pytest.approx(1.0, abs=1e-9)
        assert got.detected, name
