"""Receiver measurement: the two bridges, retention shapes and explicit budgets.

Split out of ``test_receiver_context.py`` (2026-09-21 audit).  Function bodies are
moved verbatim; the builders live in ``_receiverctx_common.py``.  Covered by
``tests/_test_inventory.py``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from isac_sim.receiver import cancellation as cx
from isac_sim.core.config import apply_overrides
from _receiverctx_common import make_belief, make_cfg, make_world


# --------------------------------------------------------------------------
# The full receiver product: both bridges, one call
# --------------------------------------------------------------------------
def test_the_measurement_reports_both_bridges():
    """Denominator (``fraction``) *and* numerator (echo survival), together.

    A scheduler wired to the denominator alone is a "power-equivalent" bridge:
    it prices the interference the canceller leaves but credits the target with
    energy the canceller may have taken.  Both must come from the same run.
    """
    cfg = make_cfg()
    geom, base, rng = make_world(cfg)
    belief = make_belief(cfg, geom, rng)
    ctx = cx.ReceiverContext.from_trial(cfg, geom, belief, base)
    m = cx.measure_receiver_context(ctx, rng=np.random.default_rng([cfg.run.seed, 31]))

    m_rx = int(cfg.scale.M)
    assert m.fraction.shape == (m_rx,)
    assert m.eta_survive.shape == (m_rx,)
    assert m.eta_survive_q.shape == (m_rx,)
    assert np.all(m.fraction > 0.0) and np.all(m.fraction <= 1.0)
    assert np.all(m.eta_survive > 0.0)
    assert np.all(m.eta_protect >= 0.0)
    # ``eta_survive`` is a *ratio of energies on a block* and can exceed 1: the
    # joint stage leaves behind whatever projects onto the target's block, so a
    # strict ``<= 1`` here would be a false invariant (measured 1.001).  The
    # bridge clamps it; the diagnosis keeps it.
    assert np.all(m.as_retention() <= 1.0)
    assert np.all(m.as_retention() > 0.0)
    # ... and the clamp must be lossless below 1.
    below = m.eta_survive < 1.0
    if np.any(below):
        assert np.allclose(m.as_retention()[below], m.eta_survive[below])
    # The residual is not free: the estimator pays for it in noise enhancement.
    assert np.all(np.isfinite(m.noise_enhance_db))
    assert np.all(m.i_res_estimate >= 0.0)
    # The analytic residual certificate is exported alongside, but remains
    # distinct from the realized residual used by ``as_fraction``.
    assert m.i_res_pred.shape == (m_rx,)
    assert m.i_in_pred.shape == (m_rx,)
    assert m.predicted_fraction.shape == (m_rx,)
    assert m.predicted_kappa_db.shape == (m_rx,)
    assert m.model_fraction.shape == (m_rx,)
    assert np.all(m.i_res_pred >= 0.0)
    positive = m.i_in_pred > 0.0
    assert np.allclose(
        m.predicted_fraction[positive],
        m.i_res_pred[positive] / m.i_in_pred[positive],
    )
    assert np.allclose(
        m.model_fraction[positive],
        m.i_res[positive] / m.i_in_pred[positive],
    )
    np.testing.assert_array_equal(m.as_residual_power(), m.i_res)
    np.testing.assert_array_equal(
        m.as_predicted_residual_power(), m.i_res_pred
    )
    assert np.allclose(
        m.predicted_kappa_db,
        -10.0 * np.log10(np.clip(m.predicted_fraction, cx.EPS, None)),
    )


def test_the_retention_bridge_broadcasts_to_the_target_axis():
    """``as_retention`` must produce the shape the numerator consumes."""
    cfg = make_cfg()
    geom, base, rng = make_world(cfg)
    ctx = cx.ReceiverContext.from_trial(cfg, geom, geom, base)
    m = cx.measure_receiver_context(ctx, rng=np.random.default_rng([cfg.run.seed, 41]))

    per_rx = m.as_retention()
    assert per_rx.shape == (int(cfg.scale.M),)
    per_target = m.as_retention(per_target=int(cfg.scale.Q))
    assert per_target.shape == (int(cfg.scale.M), int(cfg.scale.Q))
    assert np.allclose(per_target, np.repeat(per_rx[:, None], int(cfg.scale.Q), axis=1))


def test_predicted_fraction_is_invariant_to_receiver_noise_and_random_phase():
    """An analytic planning certificate must not be another receiver draw.

    The realised denominator changes with the unknown direct-path phases, but
    both terms of the predicted ratio are conditional on the same geometry and
    therefore remain identical across observation RNG seeds.
    """
    cfg = make_cfg()
    geom, base, rng = make_world(cfg)
    belief = make_belief(cfg, geom, rng)
    ctx = cx.ReceiverContext.from_trial(cfg, geom, belief, base)
    first = cx.measure_receiver_context(
        ctx, rng=np.random.default_rng([cfg.run.seed, 51])
    )
    second = cx.measure_receiver_context(
        ctx, rng=np.random.default_rng([cfg.run.seed, 52])
    )

    assert np.array_equal(first.predicted_fraction, second.predicted_fraction)
    assert np.array_equal(first.i_in_pred, second.i_in_pred)
    assert not np.array_equal(first.i_in, second.i_in)


def test_target_conditioned_retention_is_measured_not_broadcast():
    """The joint scheduler consumes one survival factor per receiver/target.

    Repeating target zero or a whole-field ratio across Q is a scalar proxy, not
    a target-conditioned receiver.  The opt-in measurement must expose the real
    matrix and the bridge must return that matrix unchanged below the clamp.
    """
    cfg = make_cfg()
    geom, base, rng = make_world(cfg)
    belief = make_belief(cfg, geom, rng)
    ctx = cx.ReceiverContext.from_trial(cfg, geom, belief, base)
    m = cx.measure_receiver_context(
        ctx, rng=np.random.default_rng([cfg.run.seed, 42]), all_targets=True
    )

    eta = m.as_retention(source="per_target", per_target=int(cfg.scale.Q))
    assert eta.shape == (int(cfg.scale.M), int(cfg.scale.Q))
    assert np.all(np.isfinite(eta))
    assert np.all((eta >= 0.0) & (eta <= 1.0))
    assert not np.allclose(eta, np.repeat(eta[:, :1], int(cfg.scale.Q), axis=1))
    measured_fraction = m.as_target_fraction()
    model_fraction = m.as_model_fraction(per_target=True)
    residual_power = m.as_residual_power(per_target=True)
    predicted_power = m.as_predicted_residual_power(per_target=True)
    assert measured_fraction.shape == eta.shape
    assert model_fraction.shape == eta.shape
    assert np.all(np.isfinite(measured_fraction))
    assert np.all(np.isfinite(model_fraction))
    np.testing.assert_allclose(
        residual_power,
        model_fraction * m.i_in_pred[:, None],
    )
    np.testing.assert_allclose(
        predicted_power,
        m.predicted_fraction_by_target * m.i_in_pred[:, None],
    )
    predicted = m.as_predicted_retention()
    assert predicted.shape == eta.shape
    assert np.all(np.isfinite(predicted))
    assert np.all((predicted >= 0.0) & (predicted <= 1.0))
    risk = m.as_predicted_retention(risk=True)
    assert risk.shape == eta.shape
    assert np.all(np.isfinite(risk))
    assert np.all((risk >= 0.0) & (risk <= predicted + 1e-12))


def test_predicted_target_retention_is_truth_echo_independent():
    """The planning survival certificate may depend on belief, not truth phase."""
    cfg = make_cfg()
    geom, base, rng = make_world(cfg)
    belief = make_belief(cfg, geom, rng)
    ctx = cx.ReceiverContext.from_trial(cfg, geom, belief, base)
    first = cx.measure_receiver_context(
        ctx, rng=np.random.default_rng([cfg.run.seed, 61]), all_targets=True
    )
    second = cx.measure_receiver_context(
        ctx, rng=np.random.default_rng([cfg.run.seed, 62]), all_targets=True
    )

    assert np.array_equal(
        first.predicted_retention_by_target,
        second.predicted_retention_by_target,
    )
    assert np.array_equal(
        first.risk_retention_by_target,
        second.risk_retention_by_target,
    )
    assert not np.array_equal(
        first.eta_survive_by_target,
        second.eta_survive_by_target,
    )


def test_covariance_protection_expands_the_belief_subspace():
    base_cfg = make_cfg()
    robust_cfg = apply_overrides(
        base_cfg, {"cancellation.covariance_protection": True}
    )
    geom, base, rng = make_world(base_cfg)
    belief = make_belief(base_cfg, geom, rng)
    sense = np.full(int(base_cfg.scale.M), base_cfg.radio.rho * base_cfg.radio.P_default)

    def result(cfg):
        obs = cx.build_observation(
            cfg, geom, belief, base, 0,
            rng=np.random.default_rng([cfg.run.seed, 71]),
            sense_power=sense,
            radiated_power=sense,
            processing_gain=float(cfg.waveform.N * cfg.waveform.L),
            hw_gain=1.0,
            include_echo=True,
        )
        return obs, cx.cancellation_arms(cfg, obs)["tp_uic_full"]

    nominal_obs, nominal = result(base_cfg)
    robust_obs, robust = result(robust_cfg)
    assert base_cfg.cancellation.covariance_protection is False
    assert robust.protect_dim >= nominal.protect_dim
    assert robust.protect_dim > nominal.protect_dim
    assert robust_obs.A.shape[1] > nominal_obs.A.shape[1]
    assert robust_obs.A_true_target_ids.size == nominal_obs.A_true_target_ids.size
    assert robust_obs.A_target_ids.size == robust_obs.A.shape[1]
    assert robust_obs.A_centre_mask.shape == (robust_obs.A.shape[1],)
    assert robust_obs.A_true_centre_mask.shape == (robust_obs.A_true_target_ids.size,)
    assert int(np.sum(robust_obs.A_centre_mask)) == len(robust_obs.targets_belief)
    assert int(np.sum(robust_obs.A_true_centre_mask)) == len(robust_obs.targets)


def test_truth_and_belief_target_gain_views_are_separate():
    """Changing the belief gain must move A without changing the true echo.

    This pins the no-oracle-amplitude contract: the receiver may use a scheduler
    gain view for its dictionary while the generated echo still uses truth.
    """
    cfg = make_cfg()
    geom, base, _rng = make_world(cfg)
    belief_base = replace(base, target_gain=0.25 * base.target_gain)
    sense = np.full(int(cfg.scale.M), cfg.radio.rho * cfg.radio.P_default)
    kwargs = dict(
        sense_power=sense,
        radiated_power=sense,
        processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0,
    )
    obs_shared = cx.build_observation(
        cfg, geom, geom, base, 0,
        rng=np.random.default_rng([cfg.run.seed, 43]), **kwargs,
    )
    obs_split = cx.build_observation(
        cfg, geom, geom, base, 0, base_belief=belief_base,
        rng=np.random.default_rng([cfg.run.seed, 43]), **kwargs,
    )
    assert np.array_equal(obs_shared.s_target, obs_split.s_target)
    assert np.linalg.norm(obs_split.A) == pytest.approx(
        0.5 * np.linalg.norm(obs_shared.A), rel=1e-12
    )


def test_the_context_states_power_and_gain_explicitly():
    """Nothing about the receiver's budget may be re-derived from ``Config``.

    The production chain runs ``rho_by_uav``, active sensing power scaling, a
    configurable processing gain, ``G_hw`` and a coordinated transmit mask.  A
    measurement that defaults all of them describes a receiver the trial is not
    running -- which is exactly how a selector and a receiver end up in two
    worlds.
    """
    cfg = make_cfg()
    geom, base, rng = make_world(cfg)
    m_rx = int(cfg.scale.M)
    sense = np.full(m_rx, 0.5 * cfg.radio.rho * cfg.radio.P_default)
    radiated = 2.0 * sense

    ctx = cx.ReceiverContext.from_trial(
        cfg, geom, geom, base, sense_power=sense, radiated_power=radiated,
        processing_gain=7.0, hw_gain=3.0,
        active_mask=np.ones(m_rx, dtype=bool),
    )
    assert np.array_equal(ctx.sense_power, sense)
    assert np.array_equal(ctx.radiated_power, radiated)
    assert ctx.processing_gain == pytest.approx(7.0)
    assert ctx.hw_gain == pytest.approx(3.0)

    default = cx.ReceiverContext.from_trial(cfg, geom, geom, base)
    assert default.processing_gain == pytest.approx(cfg.waveform.N * cfg.waveform.L)
    assert np.allclose(default.sense_power, cfg.radio.rho * cfg.radio.P_default)
