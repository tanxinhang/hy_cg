"""Receiver measurement: belief is stated, never guessed.

Split out of ``test_receiver_context.py`` (2026-09-21 audit, 473 -> 2 files).
Function bodies are moved verbatim; the builders live in
``_receiverctx_common.py``.  Covered by ``tests/_test_inventory.py``.

``measure_residual_fraction`` used to call ``build_observation(cfg, geom, geom,
...)`` -- the truth geometry served as the receiver's own belief.  The measured
depth was therefore the *perfect-tracker* depth: an upper bound on what TP-UIC
can do, while the config simultaneously declares
``prior.belief_sigma_pos_m``/``belief_sigma_vel_mps`` and the scheduler runs on
a perturbed geometry.  Two halves of one trial, two different receivers.
"""
from __future__ import annotations

import numpy as np
import pytest

from isac_sim.receiver import cancellation as cx
from isac_sim.scenario.belief import BeliefState, belief_dd_std_bins
from _receiverctx_common import make_belief, make_cfg, make_world


def test_receiver_link_dd_uncertainty_matches_scheduler_propagation():
    """Receiver and scheduler must propagate the same belief covariance."""
    cfg = make_cfg()
    geom, _base, _rng = make_world(cfg)
    belief = BeliefState.from_truth(
        cfg, geom, np.random.default_rng([cfg.run.seed, 991])
    )
    assert np.all(belief.P[:, 2, 2] == 0.0)
    assert np.all(belief.P[:, 5, 5] == 0.0)
    geom_belief = belief.as_geometry(geom)
    std_l, std_k = belief_dd_std_bins(cfg, geom_belief, belief)

    for i, j, q in ((0, 1, 0), (2, 4, 1), (5, 3, 2)):
        got_k, got_l = cx.target_link_std_bins(cfg, geom_belief, i, j, q)
        assert got_k == pytest.approx(std_k[i, j, q], rel=1e-12)
        assert got_l == pytest.approx(std_l[i, j, q], rel=1e-12)


def test_receiver_exports_belief_side_bearing_uncertainty():
    cfg = make_cfg(**{"aperture.enable": True, "aperture.m_rx": 4})
    geom, _base, rng = make_world(cfg)
    belief = make_belief(cfg, geom, rng)
    sigma_u = cx.target_bearing_std(cfg, belief, receiver=1, q=0)
    assert np.isfinite(sigma_u)
    assert sigma_u > 0.0


# --------------------------------------------------------------------------
# Backwards compatibility: the old call must still be the oracle world
# --------------------------------------------------------------------------
def test_omitting_geom_belief_is_the_oracle_belief_world():
    """The legacy call keeps its numbers, and is *labelled* as what it is.

    Bit-exactness of the released measurement matters (the 37.14 dB figure was
    produced this way), but the label matters more: it is the only thing that
    stops an upper bound being quoted as a system depth.
    """
    cfg = make_cfg()
    geom, base, rng = make_world(cfg)

    frac_omitted, kappa_omitted = cx.measure_residual_fraction(
        cfg, geom, base, rng=np.random.default_rng([cfg.run.seed, 11])
    )
    frac_explicit, kappa_explicit = cx.measure_residual_fraction(
        cfg, geom, base, geom_belief=geom, rng=np.random.default_rng([cfg.run.seed, 11])
    )
    assert np.array_equal(frac_omitted, frac_explicit)
    assert np.array_equal(kappa_omitted, kappa_explicit)

    ctx = cx.ReceiverContext.from_trial(cfg, geom, geom, base)
    assert ctx.belief_is_truth is True


def test_geom_belief_is_used_and_not_reconstructed():
    """The context must carry the caller's geometry object itself.

    A helper that "helpfully" regenerated a belief would silently break the one
    property this refactor exists for: the scheduler and the receiver must see
    the same belief, from the same RNG draw, in the same trial.
    """
    cfg = make_cfg()
    geom, base, rng = make_world(cfg)
    belief = make_belief(cfg, geom, rng)

    ctx = cx.ReceiverContext.from_trial(cfg, geom, belief, base)
    assert ctx.geom_belief is belief
    assert ctx.geom_true is geom
    assert ctx.belief_is_truth is False


def test_actual_belief_is_a_different_receiver_from_the_oracle_one():
    """Negative control: giving the receiver a belief must change the answer.

    If the belief geometry were ignored (the pre-refactor bug), the two calls
    below would agree and every "measured kappa" would be an oracle bound
    wearing a system label.
    """
    cfg = make_cfg()
    geom, base, rng = make_world(cfg)
    belief = make_belief(cfg, geom, rng)

    oracle_ctx = cx.ReceiverContext.from_trial(cfg, geom, geom, base)
    actual_ctx = cx.ReceiverContext.from_trial(cfg, geom, belief, base)

    oracle = cx.measure_receiver_context(
        oracle_ctx, rng=np.random.default_rng([cfg.run.seed, 21])
    )
    actual = cx.measure_receiver_context(
        actual_ctx, rng=np.random.default_rng([cfg.run.seed, 21])
    )
    assert oracle.belief_is_truth is True
    assert actual.belief_is_truth is False
    assert not np.allclose(oracle.fraction, actual.fraction), (
        "the belief geometry is being ignored -- the measurement is still oracle"
    )


def test_the_belief_error_is_paid_on_the_echo_side_not_on_kappa():
    """Where the declared tracker error actually lands -- and where it does not.

    Measured on the 600 m / RCS 0.1 scenario (M=6, Q=3, ``tp_uic_full``):

    ==================  ==============  ==============
    quantity            oracle belief   actual belief
    ==================  ==============  ==============
    kappa (median)      35.89 dB        35.94 dB
    eta_protect         1.000           0.264
    eta_survive         0.978           0.918
    ==================  ==============  ==============

    Two conclusions, and the second one is the surprising one:

    * The cancellation **depth is nearly belief-insensitive** -- the direct path
      never depends on the belief, only the target dictionary and the protection
      basis do, so ``kappa`` moves by ~0.06 dB.  Quoting ``kappa`` is therefore
      *not* where an oracle-belief measurement flatters the system.
    * The belief error is paid on the **echo side**: protection coverage of the
      true echo collapses by ~4x, because the receiver protects where it thinks
      the target is.  Echo survival still only loses ~6 % because stage 1 is
      conservative (it refrains from subtracting, and a wrong refrain costs less
      than a wrong subtraction) -- which is exactly why the whole-field survival
      figure must never be read as "the protection worked".

    The two assertions below are the ones worth keeping: coverage must fall, and
    depth must not move much.  Neither is a numerical target -- both are
    statements about *where* the model puts the cost.
    """
    cfg = make_cfg()
    geom, base, rng = make_world(cfg)
    belief = make_belief(cfg, geom, rng)

    oracle = cx.measure_receiver_context(
        cx.ReceiverContext.from_trial(cfg, geom, geom, base),
        rng=np.random.default_rng([cfg.run.seed, 21]),
    )
    actual = cx.measure_receiver_context(
        cx.ReceiverContext.from_trial(cfg, geom, belief, base),
        rng=np.random.default_rng([cfg.run.seed, 21]),
    )
    # Coverage collapses: the receiver protects the wrong subspace.
    assert float(np.median(oracle.eta_protect)) > 0.9
    assert float(np.median(actual.eta_protect)) < 0.5 * float(
        np.median(oracle.eta_protect)
    )
    # Depth barely moves: the direct dictionary does not depend on the belief.
    assert abs(
        float(np.median(actual.kappa_db)) - float(np.median(oracle.kappa_db))
    ) < 1.0
    # ... and the echo damage stays bounded, which is what makes the arm usable.
    assert 0.0 < float(np.median(actual.eta_survive)) <= 1.0
