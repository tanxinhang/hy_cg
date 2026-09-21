"""TP-UIC H1/H0 pairing, echo-generation contract and oracle CFAR closure.

Split out of ``test_cancellation_tp_uic.py`` (2026-09-21 audit).  Function bodies
are moved verbatim; the builders live in ``_tpuic_common.py``.  Covered by
``tests/_test_inventory.py``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from isac_sim.core.config import Config
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.sensing.model import build_base_gains, generate_geometry
from _tpuic_common import make_cfg, make_observation


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
