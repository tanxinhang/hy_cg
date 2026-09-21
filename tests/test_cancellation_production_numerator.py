"""Production wiring, numerator half: echo retention, mode validation, measurement.

Split out of ``test_cancellation_production_wiring.py`` (2026-09-21 audit).
Function bodies are moved verbatim; the builders live in ``_prodwire_common.py``.
Covered by ``tests/_test_inventory.py``.

Since 2026-09-20 the assumed constant is gone.
``interference.direct_cancellation_db`` was deleted because it asserted a fixed
40 dB with no receiver implementation behind it while propping up the entire
SINR denominator -- at 0 dB the detection probability collapses to the
false-alarm rate, so every gain booked on it was bookkeeping.  The default is
therefore **no cancellation** (``kappa_dc = 1``), and the only way to obtain
cancellation is to inject a measured residual.
"""
from __future__ import annotations

import numpy as np
import pytest

from isac_sim.sensing.model import compute_link_tables, radar_hardware_gain
from _prodwire_common import make_base, make_cfg


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
    with pytest.raises(ValueError, match="as_residual_power"):
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
    from isac_sim.receiver import cancellation as cx

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
