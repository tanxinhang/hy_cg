"""Invariants of the TP-UIC V1 canceller (``isac_sim.cancellation``).

Every assertion here is a property the first experiment relies on.  They are
cheap (a handful of DD kernels) and they are the ones that would otherwise fail
silently and inflate a reported cancellation depth.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from isac_sim import cancellation as cx
from isac_sim import cancellation_glrt as gl
from isac_sim.config import Config, apply_overrides, apply_preset
from isac_sim.model import build_base_gains, generate_geometry, noise_power


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
    }
    base.update(overrides)
    return apply_overrides(cfg, base)


def make_observation(cfg: Config, seed: int = 0, belief_error: bool = False):
    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    if belief_error:
        from isac_sim.prior import perturbed_geometry

        belief = perturbed_geometry(
            cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
        )
    else:
        belief = geom
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    obs = cx.build_observation(
        cfg, geom, belief, base, 0, rng=rng, sense_power=sense, radiated_power=sense,
        processing_gain=cfg.waveform.N * cfg.waveform.L, hw_gain=1.0,
    )
    return obs


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
def test_fixed_kappa_arm_reproduces_the_constant():
    """The frozen model's constant, read back in the sample domain."""
    cfg = make_cfg()
    cfg = apply_overrides(cfg, {"interference.direct_cancellation_db": 40.0})
    obs = make_observation(cfg)
    arms = cx.cancellation_arms(cfg, obs)
    assert arms["fixed_kappa"].kappa_db == pytest.approx(40.0, abs=1e-6)
    assert arms["fixed_kappa"].calibration_error_db == pytest.approx(0.0, abs=1e-9)
    assert arms["perfect_channel"].i_res == pytest.approx(0.0, abs=1e-30)
    assert arms["no_ic"].kappa_db == pytest.approx(0.0, abs=1e-12)
    assert arms["no_ic"].i_res == pytest.approx(arms["no_ic"].i_in)


# --------------------------------------------------------------------------
# Module M2: the protection is a statement about a neighbourhood
# --------------------------------------------------------------------------
def test_on_grid_target_is_fully_covered_by_its_tangent_basis():
    """``a(theta)`` lies in ``span{a, da/dk, da/dl}`` by construction."""
    cfg = make_cfg()
    a = cx.kernel_vector(cfg, 2.37, 1.44)
    J = cx.tangent_columns(cfg, 2.37, 1.44, order=1, step=0.05)
    U = cx.orthonormalise(J)
    assert U.rank == 3
    residual = a - U.project(a)
    assert np.linalg.norm(residual) / np.linalg.norm(a) < 1e-10


def test_protection_covers_a_neighbouring_offset_to_first_order():
    """A target a fraction of a bin away is still mostly protected.

    The residual grows quadratically, which is the error term the method note
    quotes: ``||M a_true|| = O(||dtheta||^2)``.
    """
    cfg = make_cfg()
    J = cx.tangent_columns(cfg, 2.37, 1.44, order=1, step=0.05)
    U = cx.orthonormalise(J)
    previous, ratios = None, []
    for delta in (0.005, 0.01, 0.02, 0.04):
        a = cx.kernel_vector(cfg, 2.37 + delta, 1.44)
        leak = float(np.linalg.norm(a - U.project(a)) / np.linalg.norm(a))
        ratios.append(leak)
        if previous is not None:
            assert leak > previous, "leakage must grow with the offset"
        previous = leak
    assert ratios[-1] < 0.10, "a 0.04-bin offset should stay well inside the protection"


# --------------------------------------------------------------------------
# Numerical traps that have already bitten once
# --------------------------------------------------------------------------
def test_orthonormalise_rank_ignores_column_scaling():
    """Rank must not depend on ``1/step``: the finite-difference trap.

    The DD tangent columns are difference directions whose norms differ from the
    centre column's, and that gap grows as ``step`` shrinks.  With the default
    ``step=0.05`` it is only ~3.6x, but a rank decision made *relative to the
    largest Gram eigenvalue* lets the gap decide the answer: on one real 420-
    column block the old rule returned 219 / 186 / 148 for tolerances
    1e-12 / 1e-9 / 1e-6, while the normalise-then-absolute-threshold rule returns
    a stable 169.  A rank that moves with the tolerance cannot be an input to a
    protection-cost calculation, so scale invariance is the invariant worth
    pinning -- which is why the scales below are deliberately extreme rather
    than whatever the default ``step`` happens to produce.
    """
    cfg = make_cfg()
    block = cx.tangent_columns(cfg, 2.37, 1.44, order=1, step=0.05)
    scaled = block @ np.diag([1.0, 1000.0, 0.001])
    assert cx.orthonormalise(block).rank == cx.orthonormalise(scaled).rank


def test_positive_sigma_refuses_a_clamped_variance():
    """``EPS = 1e-12`` is larger than the noise power: it must not be a divisor."""
    assert noise_power(make_cfg()) < 1e-12, (
        "if this ever fails the EPS guard would be harmless; while it holds, "
        "clamping sigma2 to EPS silently inflates every prediction by 26x"
    )
    cx._positive_sigma(3.83e-14)
    with pytest.raises(ValueError):
        cx._positive_sigma(0.0)


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
# The H1/H0 pair: the detection metric is only as good as this pairing
# --------------------------------------------------------------------------
def _pair(cfg: Config, seed: int = 0, share_noise: bool = False):
    """Build a matched ``(H1, H0)`` pair and report the tested target.

    The tested target is resolved with a throw-away generator so the returned
    pair still consumes a clean ``rng`` stream; it must be a target that really
    has columns, otherwise ``build_observation_pair`` refuses.
    """
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    gain = cfg.waveform.N * cfg.waveform.L

    peek = np.random.default_rng([cfg.run.seed, seed])
    geom_peek = generate_geometry(cfg, peek)
    base_peek = build_base_gains(cfg, geom_peek, peek)
    probe = cx.build_observation(
        cfg, geom_peek, geom_peek, base_peek, 0, rng=peek, sense_power=sense,
        radiated_power=sense, processing_gain=gain, hw_gain=1.0,
    )
    target = int(np.asarray(probe.A_target_ids)[0])

    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    obs1, obs0 = cx.build_observation_pair(
        cfg, geom, geom, base, 0, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=gain, hw_gain=1.0,
        exclude_target=target, share_noise=share_noise,
    )
    return obs1, obs0, target


def test_observation_pair_shares_the_direct_field():
    """``x_direct``, ``X``, ``A`` and ``h_true`` are the *same* in H1 and H0.

    This is the whole point: without it the two hypotheses differ because the
    illumination was redrawn, and a canceller is then scored on luck.
    """
    cfg = make_cfg()
    obs1, obs0, _ = _pair(cfg)
    assert np.array_equal(obs1.x_direct, obs0.x_direct)
    assert np.array_equal(obs1.X, obs0.X)
    assert np.array_equal(obs1.A, obs0.A)
    assert np.array_equal(obs1.h_true, obs0.h_true)
    # ``x_direct`` equal but the observations different -- the echo/noise did
    # change, otherwise the pair would be a no-op.
    assert not np.array_equal(obs1.y, obs0.y)


def test_h0_is_h1_minus_exactly_the_tested_echo():
    """With ``share_noise=True`` the difference is algebraically exact."""
    cfg = make_cfg()
    obs1, obs0, target = _pair(cfg, share_noise=True)
    A_true = cx.target_dictionary(cfg, obs1.targets)
    drop = np.asarray(obs1.A_true_target_ids) == target
    s_weak = A_true[:, drop] @ obs1.alpha_true[drop]
    # The echo must be non-trivial or the identity below would be vacuous.
    assert float(np.vdot(s_weak, s_weak).real) > 0.0
    diff = (obs1.y - obs1.x_direct) - (obs0.y - obs0.x_direct)
    assert np.allclose(diff, s_weak, rtol=1e-10, atol=0.0)


def test_independent_builds_are_not_a_matched_pair():
    """Pins the defect ``build_observation_pair`` exists to prevent.

    Two successive ``build_observation`` calls consume different rng draws for
    the direct phases, so the direct field moves.  If a future refactor makes
    this test fail, the pair builder has become redundant -- do not simply
    delete the assertion without re-checking the detection metric.
    """
    cfg = make_cfg()
    rng = np.random.default_rng([cfg.run.seed, 0])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    kw = dict(rng=rng, sense_power=sense, radiated_power=sense,
              processing_gain=cfg.waveform.N * cfg.waveform.L, hw_gain=1.0)
    first = cx.build_observation(cfg, geom, geom, base, 0, **kw)
    second = cx.build_observation(cfg, geom, geom, base, 0, **kw)
    assert not np.allclose(first.x_direct, second.x_direct)


def test_observation_pair_refuses_a_target_with_no_columns():
    cfg = make_cfg()
    rng = np.random.default_rng([cfg.run.seed, 0])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    with pytest.raises(ValueError, match="no column"):
        cx.build_observation_pair(
            cfg, geom, geom, base, 0, rng=rng, sense_power=sense,
            radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
            hw_gain=1.0, exclude_target=10 ** 6,
        )


def test_the_paired_metric_can_see_the_echo():
    """The oracle arm removes exactly the direct field, so the H1/H0 ratio is known.

    ``perfect_channel`` subtracts ``x_direct`` and nothing else, and the pair
    builder gives both hypotheses the same ``x_direct``, with ``s_h0`` differing
    from ``s_h1`` by exactly the tested echo.  With ``share_noise=True`` the two
    observations differ by that echo alone, so

        T(H1) / T(H0) = ||U^H (s + n)||^2 / ||U^H (s_h0 + n)||^2

    with every symbol computable from the observation.  The test asserts that
    identity (which pins the arm's action) and then that the ratio exceeds 1 --
    a direction check, not a calibrated detection figure.

    The ratio is *not* asserted against a dB threshold any more.  It used to
    read 2.85 dB here and the old threshold was 3.0 dB, tuned to an echo
    generator that wrote a third of its energy onto tangent columns no detector
    modelled; a threshold that only a broken generator clears is not a test.
    """
    cfg = make_cfg()
    obs1, obs0, target = _pair(cfg, share_noise=True)
    A = obs1.A
    U = cx.orthonormalise(A[:, np.asarray(obs1.A_target_ids) == target])
    n = obs1.y - obs1.x_direct - obs1.s_target

    def energy(v: np.ndarray) -> float:
        p = U.U.conj().T @ v
        return float(np.vdot(p, p).real) / U.rank

    arm1 = cx.cancellation_arms(cfg, obs1, weak_target=target)["perfect_channel"]
    arm0 = cx.cancellation_arms(cfg, obs0, weak_target=target)["perfect_channel"]
    assert arm0.matched_stat > 0.0
    expected = energy(obs1.s_target + n) / energy(obs0.s_target + n)
    got = arm1.matched_stat / arm0.matched_stat
    assert got == pytest.approx(expected, rel=1e-8), (
        "the oracle arm is not subtracting exactly the direct field"
    )
    assert got > 1.5, "the paired metric can barely see the echo (ratio %.2f)" % got


# --------------------------------------------------------------------------
# The generation contract: what the echo generator is allowed to write
# --------------------------------------------------------------------------
def test_generated_echo_puts_its_truth_on_the_centre_columns_only():
    """``alpha_true`` is non-zero on the centre column of each block, and nowhere else.

    The tangent columns are the receiver's *model* of where the echo might have
    moved, not extra scatterers.  Giving them true coefficients changes the
    physics being simulated -- three unit-modulus echoes per target instead of
    one -- and, because the detector's nuisance block keeps centre columns, it
    also puts signal into a subspace the null hypothesis declares empty.  That
    is what broke the CFAR level (see the two oracle tests below); this test
    pins the generator, which is the cheaper of the two guards.
    """
    cfg = make_cfg()
    obs = make_observation(cfg)
    n_basis = 1 + 2 * int(cfg.cancellation.tangent_order)
    alpha = np.asarray(obs.alpha_true)
    assert alpha.size % n_basis == 0, "columns per source must be 1 + 2*order"
    centre = np.zeros(alpha.size, dtype=bool)
    centre[::n_basis] = True
    assert np.all(alpha[~centre] == 0.0), (
        "%d tangent columns carry a true coefficient" % int(np.count_nonzero(alpha[~centre]))
    )
    assert np.allclose(np.abs(alpha[centre]), 1.0)

    # And the structural consequence: the true echo lies in the span of the
    # centre columns of the true dictionary, so a centre-only nuisance block can
    # remove it exactly.
    A_true = cx.target_dictionary(cfg, obs.targets)
    U = cx.orthonormalise(A_true[:, ::n_basis])
    leak = obs.s_target - U.project(obs.s_target)
    assert (
        float(np.linalg.norm(leak)) / max(float(np.linalg.norm(obs.s_target)), 1e-300)
        < 1e-10
    )


def test_direct_path_truth_is_centre_only_too():
    """The rule is not echo-specific, and the direct field already obeyed it."""
    cfg = make_cfg()
    obs = make_observation(cfg)
    n_basis = 1 + 2 * int(cfg.cancellation.interference_tangent_order)
    h = np.asarray(obs.h_true)
    centre = np.zeros(h.size, dtype=bool)
    centre[::n_basis] = True
    assert np.all(h[~centre] == 0.0)


def _oracle_observations(cfg: Config, seed: int, alpha: np.ndarray | None = None):
    """A matched ``(H1, H0)`` pair with a perfect tracker and a chosen truth.

    ``alpha=None`` uses the generator's own coefficients.  Passing ``alpha``
    lets a test write a *different* echo into the same direct field, which is how
    the two oracle tests below separate "the detector is miscalibrated" from
    "the generator wrote something the detector does not model".
    """
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    gain = cfg.waveform.N * cfg.waveform.L

    peek = np.random.default_rng([cfg.run.seed, seed])
    geom_peek = generate_geometry(cfg, peek)
    base_peek = build_base_gains(cfg, geom_peek, peek)
    probe = cx.build_observation(
        cfg, geom_peek, geom_peek, base_peek, 0, rng=peek, sense_power=sense,
        radiated_power=sense, processing_gain=gain, hw_gain=1.0,
    )
    target = int(np.asarray(probe.A_target_ids)[0])

    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    obs1, obs0 = cx.build_observation_pair(
        cfg, geom, geom, base, 0, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=gain, hw_gain=1.0,
        exclude_target=target,
    )
    if alpha is None:
        return obs1, obs0, target
    A_true = cx.target_dictionary(cfg, obs1.targets)
    ids = np.asarray(obs1.A_true_target_ids)
    noise = obs1.y - obs1.x_direct - obs1.s_target
    s1 = A_true @ alpha
    drop = ids == int(target)
    s0 = s1 - A_true[:, drop] @ alpha[drop]
    return (
        replace(obs1, y=obs1.x_direct + s1 + noise, s_target=s1, alpha_true=alpha.copy()),
        replace(obs1, y=obs1.x_direct + s0 + noise, s_target=s0, alpha_true=np.where(drop, 0.0, alpha)),
        target,
    )


def _oracle_statistic_ratio(cfg: Config, seed: int, alpha: np.ndarray | None = None):
    """``T_H0 / E[T_H0]`` for perfect_channel + truth.  Should be 1 when calibrated."""
    obs1, obs0, target = _oracle_observations(cfg, seed, alpha)
    arms = cx.cancellation_arms(cfg, obs0, weak_target=target)
    model = gl.residual_model(
        cfg, obs0, "perfect_channel", arms, plans=gl.arm_plans(cfg, obs0, arms)
    )
    got = gl.target_conditioned_glrt(
        cfg, obs0, arms["perfect_channel"], model, target=target, p_fa=0.05,
        dictionary="truth",
    )
    assert got.dof_real > 0
    return float(got.statistic / (got.dof_real / 2.0))


def test_oracle_cfar_level_is_met_when_the_echo_is_the_generated_one():
    """``perfect_channel`` + ``truth`` on H0: the analytic level must be a CFAR level.

    Nothing is left to blame here.  The direct field is removed exactly, the
    other targets' echoes are stated exactly and projected out exactly, there is
    no belief error and no estimation error, so ``T_H0`` must be a
    ``(1/2) chi^2_dof`` draw and the ratio below must be 1.  This is the
    closure the V1.1 headline could not run: it reported ``P_FA = 0.40 .. 0.94``
    and had no way to tell a mis-stated ``C_res`` from a mis-stated signal model.
    """
    cfg = make_cfg()
    ratios = [_oracle_statistic_ratio(cfg, s) for s in range(10)]
    med = float(np.median(ratios))
    assert 0.5 < med < 2.0, (
        "H0 statistic sits at %.2f x its stated mean -- the stated C_res is not "
        "the covariance of the residual this arm leaves" % med
    )


def test_a_tangent_contaminated_echo_breaks_the_oracle_cfar_level():
    """The defect that was fixed, kept as a measurement.

    Same arm, same dictionary, same noise stream -- the only difference is that
    the generator writes unit-modulus coefficients onto the *tangent* columns as
    well.  The nuisance projection is centre-only, so those components survive
    into H0 and the statistic leaves its stated distribution.  If this test ever
    stops failing to reject, the generator and the detector have been changed
    together in a way that hides which of them is wrong.
    """
    cfg = make_cfg()
    rng = np.random.default_rng([cfg.run.seed, 10 ** 6])
    obs1, _, _ = _oracle_observations(cfg, 0)
    contaminated = np.exp(
        1j * rng.uniform(0.0, 2.0 * np.pi, size=np.asarray(obs1.alpha_true).size)
    )
    bad = float(np.median([_oracle_statistic_ratio(cfg, s, contaminated) for s in range(6)]))
    good = float(np.median([_oracle_statistic_ratio(cfg, s) for s in range(6)]))
    assert bad > 1.5 * good, (
        "contaminated %.2f vs centre-only %.2f: the tangent truth no longer "
        "moves the H0 statistic, so this test can no longer detect the defect "
        "it exists for" % (bad, good)
    )


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


# --------------------------------------------------------------------------
# Stage 2: a declared support rule, not a test whose level is an assumption
# --------------------------------------------------------------------------
def test_joint_support_is_the_protection_set_by_default():
    """The default support is declared, and it is not the gate's output.

    ``threshold = 0`` is the extreme case: every target clears it, so the gate's
    output is "all targets".  The default policy must still model only the
    protected ones -- the test would be vacuous with a positive threshold, which
    is exactly why it uses the degenerate one.
    """
    cfg = make_cfg(**{"cancellation.max_protected_targets": 1})
    obs = make_observation(cfg, belief_error=True)
    shielded = cx.protected_target_ids(cfg, obs.targets_belief or obs.targets)
    default = cx.cancellation_arms(cfg, obs, threshold=0.0)["tp_uic_full"]
    ablation = cx.cancellation_arms(
        cfg, obs, threshold=0.0, candidate_policy="statistic"
    )["tp_uic_full"]
    assert shielded, "the protection budget must select something"
    assert set(default.supported_targets) == set(shielded)
    assert set(ablation.supported_targets) == set(ablation.gate_targets)
    assert len(default.candidates) < len(ablation.candidates)
    with pytest.raises(ValueError):
        cx.cancellation_arms(cfg, obs, candidate_policy="whatever")


def test_protected_target_ids_agrees_with_the_protection_basis():
    """The extracted helper still describes the basis that is actually built."""
    cfg = make_cfg(**{"cancellation.max_protected_targets": 2})
    obs = make_observation(cfg, belief_error=True)
    sources = obs.targets_belief or obs.targets
    ids = cx.protected_target_ids(cfg, sources)
    subset = [s for s in sources if int(s.target) in ids]
    built = cx._protection_basis(cfg, subset, int(obs.y.size))
    assert np.array_equal(built, obs.basis_belief)
