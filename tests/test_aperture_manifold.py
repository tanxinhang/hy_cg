"""P1-2, second half: the observation space becomes ``DD (x) array``.

The lift is a change of the *observation space*, so the invariants are about what
must not move and what must:

* ``m_rx = 1`` (or the gate closed) must be bit-exact -- this is the release;
* with more elements the dimension multiplies and ``rho`` must rise (that is the
  whole point, measured by the angular probe as 0.0004 -> 0.74/0.94/0.98);
* the array must not silently change the link budget -- the templates stay
  unit-norm, so a matched filter sees the same SNR, and only the *coherence*
  between templates moves;
* the Kronecker identity must hold, or the array half is silently ignored and
  every number degrades to the DD-only one -- the failure that looks exactly like
  a negative result.
"""
from __future__ import annotations

import numpy as np
import pytest

from isac_sim import aperture as apx
from isac_sim import cancellation as cx
from isac_sim.config import Config, apply_preset
from isac_sim.model import build_base_gains, generate_geometry
from isac_sim.prior import perturbed_geometry

TRIAL = 11
RECEIVER = 7


def build(m_rx: int | None):
    cfg = apply_preset(Config(), "paper-canonical")
    cfg.geometry.area_xy = 600.0
    cfg.detect.target_rcs = 0.1
    cfg.cancellation.enable = True
    if m_rx is not None and int(m_rx) > 1:
        cfg.aperture.enable = True
        cfg.aperture.m_rx = int(m_rx)
    rng = np.random.default_rng([cfg.run.seed, TRIAL])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    belief = perturbed_geometry(
        cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
    )
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    obs1, obs0 = cx.build_observation_pair(
        cfg, geom, belief, base, RECEIVER, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0, exclude_target=0, weak_index=0,
    )
    return cfg, obs1, obs0


def test_the_lift_is_inert_at_one_element():
    """The release: gate closed and ``m_rx = 1`` are the same observation."""
    _, a1, a0 = build(None)
    _, b1, b0 = build(1)
    for x, y in ((a1, b1), (a0, b0)):
        assert x.y.shape == y.y.shape == (4096,)
        assert np.array_equal(x.y, y.y)
        assert np.array_equal(x.X, y.X)
        assert np.array_equal(x.A, y.A)
        assert np.array_equal(x.A_true if hasattr(x, "A_true") else x.y,
                              y.A_true if hasattr(y, "A_true") else y.y)


def test_more_elements_multiplies_the_observation_dimension():
    """``dim = K * m``: the lifted space is the tensor product, nothing else."""
    for m in (1, 4, 8):
        _, o1, _ = build(m)
        expected = 4096 * (m if m > 1 else 1)
        assert o1.y.size == expected
        assert o1.X.shape[0] == expected
        assert o1.A.shape[0] == expected


def test_the_steering_is_unit_norm_and_induces_the_documented_inner_product():
    """``<a(u1), a(u2)> = A(u1 - u2)`` -- the Kronecker identity's array half.

    If this drifts, every lifted inner product silently becomes the DD-only one
    and the whole aperture degrades to no aperture, which reads as a negative
    result rather than as a bug.
    """
    for m in (4, 8, 16):
        a1 = cx.steering_vector(m, 0.3)
        assert np.vdot(a1, a1).real == pytest.approx(1.0, abs=1e-12)
        for du in (0.05, 0.2, 0.5):
            a2 = cx.steering_vector(m, 0.3 + du)
            got = abs(np.vdot(a1, a2))
            assert got == pytest.approx(apx.array_factor(m, du), abs=1e-12)


def test_lifting_preserves_the_template_norm():
    """No budget dressed as a mechanism: a matched filter sees the same SNR.

    The array's collection gain is a link-budget quantity; if the lifted columns
    came out louder, every detection improvement would be ambiguous between
    "more selective" and "simply more power".
    """
    cfg, o1, _ = build(1)
    cfg4, o4, _ = build(4)
    # one column of the direct dictionary, before and after the lift
    n1 = float(np.linalg.norm(o1.X[:, 0]))
    n4 = float(np.linalg.norm(o4.X[:, 0]))
    assert n4 == pytest.approx(n1, rel=1e-12)


def test_the_array_raises_the_escape_fraction():
    """The point of the exercise, measured inside the observation model."""
    from isac_sim import cancellation_glrt as gl

    def rho(cfg, obs):
        arms = cx.cancellation_arms(cfg, obs, weak_target=0,
                                    candidate_policy="protected_only")
        model = gl.residual_model(cfg, obs, "perfect_channel", arms,
                                  plans=gl.arm_plans(cfg, obs, arms))
        got = gl.target_conditioned_glrt(cfg, obs, arms["perfect_channel"], model,
                                         target=0, p_fa=0.05, dictionary="belief")
        return float(got.rho_weighted)

    vals = {m: rho(*build(m)[:2]) for m in (1, 4, 8)}
    assert vals[4] > vals[1] + 0.1, "the array must resolve co-bin targets"
    assert vals[8] > vals[4]
    assert vals[8] > 0.5
