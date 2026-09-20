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


def test_target_conditioned_fraction_matches_independent_target_tables():
    """Each target slice must use its own denominator without moving raw SINR."""
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    m, q = cfg.scale.M, cfg.scale.Q
    fractions = np.empty((m, q), dtype=float)
    for target in range(q):
        fractions[:, target] = 10.0 ** (-float(25 + 5 * target) / 10.0)

    joint = compute_link_tables(
        cfg, base, residual_fraction_by_receiver=fractions
    )
    for target in range(q):
        separate = compute_link_tables(
            cfg, base, residual_fraction_by_receiver=fractions[:, target]
        )
        np.testing.assert_array_equal(
            joint.gamma_sense[:, :, target],
            separate.gamma_sense[:, :, target],
        )
        np.testing.assert_array_equal(
            joint.raw_gamma_sense[:, :, target],
            separate.raw_gamma_sense[:, :, target],
        )
    # Pair-level diagnostics conservatively report the largest residual slice.
    worst = compute_link_tables(
        cfg, base, residual_fraction_by_receiver=np.max(fractions, axis=1)
    )
    np.testing.assert_array_equal(joint.rinr, worst.rinr)


def test_target_conditioned_fraction_validates_shape():
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    with pytest.raises(ValueError, match="residual_fraction_by_receiver must have shape"):
        compute_link_tables(
            cfg, base,
            residual_fraction_by_receiver=np.ones((cfg.scale.M, 2, 2)),
        )


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


def test_all_ones_retention_reproduces_the_frozen_numerator():
    """``eta = 1`` must be bit-for-bit the released numerator.

    The substitution has to be a substitution on *both* halves of the fraction,
    or the "algorithm" arm carries an arithmetic difference that has nothing to
    do with the algorithm.
    """
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    m, q = cfg.scale.M, cfg.scale.Q
    frozen = compute_link_tables(cfg, base)

    for shape in ((m,), (m, q)):
        via = compute_link_tables(
            cfg, base, target_retention_by_receiver=np.ones(shape)
        )
        assert np.array_equal(frozen.gamma_sense, via.gamma_sense)
        assert np.array_equal(frozen.raw_gamma_sense, via.raw_gamma_sense)
        assert np.array_equal(frozen.rinr, via.rinr)


def test_retention_moves_the_numerator_and_only_the_numerator():
    """The echo-survival factor belongs in ``gamma``, not in ``raw_gamma``.

    ``raw_gamma_sense`` is the raw-SINR baseline: the link before any receiver
    processing.  If the canceller's survival factor leaked into it, the
    "before" and "after" baselines would collapse into one number and every
    statement about what the receiver costs would become unfalsifiable.
    """
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    m = cfg.scale.M
    frozen = compute_link_tables(cfg, base)
    attenuated = compute_link_tables(
        cfg, base, target_retention_by_receiver=np.full(m, 0.5)
    )
    assert np.all(attenuated.gamma_sense <= frozen.gamma_sense)
    assert np.any(attenuated.gamma_sense < frozen.gamma_sense)
    assert np.array_equal(attenuated.raw_gamma_sense, frozen.raw_gamma_sense)
    # ... and the denominator must be untouched by a numerator argument.
    assert np.array_equal(attenuated.rinr, frozen.rinr)


def test_a_retention_above_one_is_refused():
    """A survival factor > 1 is an artefact, not gain -- refuse rather than clamp.

    The measured ratio can exceed 1 (1.001 on the 600 m scenario) because the
    joint stage leaves behind whatever else projects onto a target's block.
    Multiplying a detection statistic by that would manufacture detection
    probability out of a bookkeeping artefact.
    """
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    m = cfg.scale.M
    with pytest.raises(ValueError, match="must not exceed 1.0"):
        compute_link_tables(
            cfg, base, target_retention_by_receiver=np.full(m, 1.001)
        )
    with pytest.raises(ValueError, match="target_retention_by_receiver must have shape"):
        compute_link_tables(
            cfg, base, target_retention_by_receiver=np.ones((m, 2, 2))
        )


def test_the_numerator_bridge_also_disables_the_reuse_fast_path():
    """Same guard as the denominator's, and for the same reason.

    The fast path copies a sensing block built under whatever receiver model the
    source table used.  A caller passing a survival factor and getting back a
    table built without it would see the released numbers while believing it
    reads the algorithm's.
    """
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    m = cfg.scale.M
    eta = np.full(m, 0.9)
    reused = compute_link_tables(
        cfg, base, reuse_from=compute_link_tables(cfg, base),
        target_retention_by_receiver=eta,
    )
    fresh = compute_link_tables(cfg, base, target_retention_by_receiver=eta)
    assert np.array_equal(reused.gamma_sense, fresh.gamma_sense)
    assert not np.array_equal(reused.gamma_sense, compute_link_tables(cfg, base).gamma_sense)


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
