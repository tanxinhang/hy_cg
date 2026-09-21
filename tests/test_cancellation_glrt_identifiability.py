"""TP-UIC GLRT: the masking (``rho``) claim and the single-target arm claim.

Split out of ``test_cancellation_glrt.py`` (2026-09-21 audit).  Function bodies
are moved verbatim; the builders live in ``_glrt_common.py``.  Covered by
``tests/_test_inventory.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from _glrt_common import make_arms, make_canonical_cfg, make_cfg, make_trial


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
