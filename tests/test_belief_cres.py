"""Invariants of the belief-error term in ``C_res``.

The term exists to answer one question: is the detector's false-alarm rate above
its nominal level because the *templates are misplaced* (the tracker's belief
differs from the target state), and does charging that misalignment to ``C_res``
calibrate it?  Three things can quietly destroy the answer:

* charging the **coefficient prior** instead (a different uncertainty -- about
  the amplitudes being fitted, not about where the template sits);
* charging it while the detector builds templates from the **truth**, where there
  is no misalignment at all;
* leaving it **on by default**, which would move every released number.

The validity limit is pinned numerically rather than asserted, because that is
the finding: at the declared belief error the offset is about one whole
resolution cell, where a first-order expansion is no longer a small correction.
"""
from __future__ import annotations

import numpy as np
import pytest

from isac_sim import cancellation as cx
from isac_sim import cancellation_glrt as gl
from isac_sim.config import Config, apply_preset


def scene(trial: int = 3, receiver: int = 7):
    cfg = apply_preset(Config(), "paper-canonical")
    cfg.geometry.area_xy = 600.0
    cfg.detect.target_rcs = 0.1
    cfg.cancellation.enable = True
    rng = np.random.default_rng([cfg.run.seed, trial])
    from isac_sim.model import build_base_gains, generate_geometry
    from isac_sim.prior import perturbed_geometry
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    belief = perturbed_geometry(
        cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
    )
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    obs1, obs0 = cx.build_observation_pair(
        cfg, geom, belief, base, receiver, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0, exclude_target=0, weak_index=0,
    )
    return cfg, obs1, obs0


def _whitened(model, v: np.ndarray) -> np.ndarray:
    """``C^{-1/2} v`` -- the only thing the detector ever asks of ``C_res``."""
    c = model.cov
    if c.rank == 0:
        return v / np.sqrt(c.sigma2)
    coeff = (1.0 / np.sqrt(1.0 + c.lam / c.sigma2)) - 1.0
    return (v + c.W @ (coeff * (c.W.conj().T @ v))) / np.sqrt(c.sigma2)


def test_the_term_is_off_by_default_and_the_released_model_is_untouched():
    """The gate: with the flag closed, ``C_res`` is exactly what it was.

    Compared through ``C^{-1/2}`` applied to a fixed probe, because that is the
    only operation the detector performs -- two covariances that differ nowhere
    the detector looks are the same model for every number it reports.
    """
    cfg, obs1, _ = scene()
    assert cfg.cancellation.belief_error_in_cres is False
    arms = cx.cancellation_arms(cfg, obs1, weak_target=0,
                                candidate_policy="protected_only")
    plans = gl.arm_plans(cfg, obs1, arms)
    probe = np.exp(1j * np.linspace(0.0, 3.0, obs1.y.size))

    off_a = _whitened(gl.residual_model(cfg, obs1, "tp_uic_stage1", arms,
                                        plans=plans), probe)
    cfg.cancellation.belief_error_in_cres = True
    on = _whitened(gl.residual_model(cfg, obs1, "tp_uic_stage1", arms,
                                     plans=plans), probe)
    cfg.cancellation.belief_error_in_cres = False
    off_b = _whitened(gl.residual_model(cfg, obs1, "tp_uic_stage1", arms,
                                        plans=plans), probe)

    assert np.array_equal(off_a, off_b), "flag closed must restore the released model"
    assert not np.allclose(off_a, on), "flag open must change something, or the " \
        "experiment cannot measure it"
    assert np.isfinite(on).all()


def test_it_is_not_charged_when_the_dictionary_is_the_truth():
    """A truth-side template cannot be misplaced, so the term must vanish."""
    cfg, obs1, _ = scene()
    cfg.cancellation.belief_error_in_cres = True
    arms = cx.cancellation_arms(cfg, obs1, weak_target=0,
                                candidate_policy="protected_only")
    belief_model = gl.residual_model(cfg, obs1, "tp_uic_stage1", arms,
                                     plans=gl.arm_plans(cfg, obs1, arms),
                                     dictionary="belief")
    truth_model = gl.residual_model(cfg, obs1, "tp_uic_stage1", arms,
                                    plans=gl.arm_plans(cfg, obs1, arms),
                                    dictionary="truth")
    # ``ResidualModel.rank`` is the rank of the *removal* operator, not of the
    # covariance -- comparing it here would compare the canceller, not the term.
    assert truth_model.cov.rank < belief_model.cov.rank
    assert truth_model.cov.trace < belief_model.cov.trace


def test_the_tested_targets_own_leakage_is_not_charged():
    """``C_res`` is the covariance *under H0*, where that target is absent.

    Charging its leakage would inflate the covariance exactly where the
    false-alarm level is set -- the term would then appear to calibrate by
    weakening the test, which is the failure mode worth pinning.
    """
    cfg, obs1, _ = scene()
    sources = obs1.targets_belief or obs1.targets
    tested = int(obs1.weak_index)
    factor = gl._belief_error_factor(cfg, obs1)
    assert factor.shape[1] > 0
    # 2 columns per *other* target's source; none for the tested one
    expected = 2 * sum(1 for s in sources if int(s.target) != tested)
    assert factor.shape[1] == expected


def test_the_belief_error_is_measured_in_resolution_cells():
    """The finding, pinned: the declared error is about one whole DD cell.

    150 m against a 78.1 m delay cell and a 11.9 m/s velocity cell is where the
    first-order (tangent) expansion stops being a correction.  This test is the
    place that number is recorded, so a change of scenario that silently moves
    the detector in or out of the linear regime has to be acknowledged here.
    """
    cfg, _, _ = scene()
    sigma_l, sigma_k = gl._belief_bin_sigmas(cfg)
    assert sigma_l == pytest.approx(0.96, abs=0.02)
    assert sigma_k == pytest.approx(1.26, abs=0.02)
    assert sigma_l > 0.3, "inside the linear regime the term is a small correction"
