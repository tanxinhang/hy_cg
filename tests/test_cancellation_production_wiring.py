"""The production wiring of a receiver-side canceller into the link tables.

``model.compute_link_tables`` multiplies the aggregated direct field by a
constant ``kappa_dc = 10^(-interference.direct_cancellation_db/10)``.  The
argument added on 2026-09-19 lets a *measured* per-receiver fraction take that
constant's place, which is the difference between "the chain assumes 40 dB" and
"the chain uses the algorithm".

The invariants below are the ones that keep that substitution honest.  They are
deliberately about *equivalence and non-silence* rather than about a numerical
target: the value of the produced ``rinr`` belongs to the run's record, while
"the frozen path did not move", "passing the constant back reproduces the frozen
path exactly", and "the reuse fast path cannot hand back a table built under a
different receiver model" are properties that must hold forever.
"""

from __future__ import annotations

import numpy as np
import pytest

from isac_sim.config import Config, apply_overrides, apply_preset
from isac_sim.model import (
    build_base_gains,
    compute_link_tables,
    generate_geometry,
    radar_hardware_gain,
)


def make_cfg(**overrides) -> Config:
    cfg = apply_preset(Config(), "paper-canonical")
    base = {
        "geometry.area_xy": 600.0,
        "detect.target_rcs": 0.1,
        "run.seed": 2026,
        "run.verbose": False,
    }
    base.update(overrides)
    return apply_overrides(cfg, base)


def make_base(cfg: Config, seed: int = 0):
    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    return geom, build_base_gains(cfg, geom, rng), rng


def test_the_constant_passed_as_an_array_reproduces_the_frozen_path():
    """``frac[j] = 10^(-kappa/10)`` must be bit-for-bit the scalar constant.

    This is the equivalence that makes the substitution a *substitution*: the
    measured fraction is on the same scale as the constant, so the frozen path
    must be exactly the special case ``frac = const``.  If the two ever differ,
    a comparison between "assumed receiver" and "measured receiver" is comparing
    an arithmetic difference on top of the thing being measured.
    """
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    kappa = float(cfg.interference.direct_cancellation_db)
    frac = np.full(cfg.scale.M, 10.0 ** (-kappa / 10.0))

    frozen = compute_link_tables(cfg, base)
    via_array = compute_link_tables(cfg, base, residual_fraction_by_receiver=frac)
    assert np.array_equal(frozen.rinr, via_array.rinr)
    assert np.array_equal(frozen.gamma_sense, via_array.gamma_sense)


def test_a_worse_fraction_raises_the_residual_ratio():
    """Direction check: a larger surviving fraction must mean more residual."""
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    m = cfg.scale.M
    frozen = compute_link_tables(cfg, base)

    better = compute_link_tables(cfg, base, residual_fraction_by_receiver=np.full(m, 1e-6))
    worse = compute_link_tables(cfg, base, residual_fraction_by_receiver=np.full(m, 1e-2))
    assert np.all(worse.rinr >= frozen.rinr)
    assert np.all(better.rinr <= frozen.rinr)
    # ... and it must actually move, otherwise the argument is being dropped.
    assert np.any(worse.rinr > frozen.rinr)
    assert np.any(better.rinr < frozen.rinr)


def test_reuse_cannot_hand_back_a_table_built_under_another_receiver_model():
    """The reuse fast path copies the sensing block -- it must be disabled here.

    Without the guard this test fails *silently in production*: the caller passes
    a measured fraction, ``reuse_from`` supplies a sensing block built with the
    frozen constant, and the returned table reports the constant's numbers while
    the caller believes it is reading the algorithm's.  Nothing raises and every
    number looks plausible.
    """
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    m = cfg.scale.M
    frac = np.full(m, 1e-3)

    reused = compute_link_tables(cfg, base, reuse_from=compute_link_tables(cfg, base),
                                 residual_fraction_by_receiver=frac)
    fresh = compute_link_tables(cfg, base, residual_fraction_by_receiver=frac)
    assert np.array_equal(reused.rinr, fresh.rinr)
    assert np.array_equal(reused.gamma_sense, fresh.gamma_sense)
    # And the guard is load-bearing: the reused table differs from the frozen one.
    assert not np.array_equal(reused.rinr, compute_link_tables(cfg, base).rinr)


def test_measure_mode_is_refused_with_a_pointer_to_the_real_interface():
    """``mode="measure"`` cannot be honoured here, so it must not be ignored.

    The estimator needs the trial geometry, which this function is not given.  A
    silent fallback to the constant would be the worst outcome: the field would
    read "measure" while every number came from the assumption.
    """
    cfg = make_cfg(**{"cancellation.mode": "measure"})
    _, base, _ = make_base(cfg)
    with pytest.raises(ValueError, match="measure_residual_fraction"):
        compute_link_tables(cfg, base)
    # Supplying the measured array is the supported route and must be accepted.
    compute_link_tables(
        cfg, base,
        residual_fraction_by_receiver=np.full(cfg.scale.M, 1e-4),
    )


def test_a_misspelt_mode_raises_instead_of_using_the_constant():
    """A typo must not be read as "off".

    Silent fallback is the most expensive failure mode here: the run would look
    configured-for-the-algorithm while every number came from the frozen
    assumption.  Same rule as the rest of the config surface.
    """
    cfg = make_cfg(**{"cancellation.mode": "mesure"})
    _, base, _ = make_base(cfg)
    with pytest.raises(ValueError, match="Unknown cancellation.mode"):
        compute_link_tables(cfg, base)


def test_measured_fraction_agrees_with_the_prediction_by_receiver_shape():
    """``measure_residual_fraction`` must return one fraction per receiver in (0, 1].

    It is the array the production call consumes, so its contract is the same as
    the substitution's: same length, same scale as ``10^(-kappa/10)``, and a
    reported dB figure consistent with the fraction it came from.
    """
    from isac_sim import cancellation as cx

    cfg = make_cfg(**{"cancellation.enable": True})
    geom, base, _ = make_base(cfg)
    rng = np.random.default_rng([cfg.run.seed, 10 ** 6])
    fraction, kappa = cx.measure_residual_fraction(cfg, geom, base, rng=rng)
    m = cfg.scale.M
    assert fraction.shape == (m,)
    assert kappa.shape == (m,)
    assert np.all(fraction > 0.0) and np.all(fraction <= 1.0)
    finite = fraction < 1.0
    assert np.allclose(
        kappa[finite], -10.0 * np.log10(fraction[finite]), rtol=1e-9, atol=1e-9
    )
    # The estimator must be a receiver, not a pass-through: on this scenario it
    # removes tens of dB of the aggregated direct field.
    assert float(np.median(kappa)) > 10.0
    # ... and the released hardware gain is 0 dB, which is the budget the
    # estimator is working against (G_hw = 1).
    assert radar_hardware_gain(cfg) == pytest.approx(1.0)
