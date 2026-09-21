"""Production wiring, denominator half: the measured residual fraction / power.

Split out of ``test_cancellation_production_wiring.py`` (2026-09-21 audit,
390 -> 2 files).  Function bodies are moved verbatim; the builders live in
``_prodwire_common.py``.  Covered by ``tests/_test_inventory.py``.

``model.compute_link_tables`` multiplies the aggregated direct field by a
surviving fraction ``kappa_dc``.  The argument added on 2026-09-19 lets a
*measured* per-receiver fraction drive that factor, which is the difference
between "the chain assumes a cancellation depth" and "the chain uses the
algorithm".
"""
from __future__ import annotations

import numpy as np
import pytest

from isac_sim.sensing.model import compute_link_tables
from _prodwire_common import make_base, make_cfg


def test_no_cancellation_is_the_default_and_the_array_special_case():
    """``frac = 1`` (0 dB) must be bit-for-bit the default path.

    This is the equivalence that makes the substitution a *substitution*: the
    measured fraction lives on the same scale as the model's own factor, so the
    default path must be exactly the special case ``frac = 1``.  If the two ever
    differ, a comparison between "no receiver" and "measured receiver" is
    comparing an arithmetic difference on top of the thing being measured.
    """
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    frozen = compute_link_tables(cfg, base)
    via_array = compute_link_tables(
        cfg, base, residual_fraction_by_receiver=np.ones(cfg.scale.M)
    )
    assert np.array_equal(frozen.rinr, via_array.rinr)
    assert np.array_equal(frozen.gamma_sense, via_array.gamma_sense)


def test_the_default_path_really_carries_no_cancellation():
    """删除常数后的反证：默认路径绝不能被悄悄衰减。

    这条是"没有对消"这个声明的可证伪形式 —— 它断言默认分母等于全额直连场，
    因此任何注入的对消都必须把它压下去。若哪天有人把某个常数放回默认路径，
    这条会先于任何性能数字翻面。
    """
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    frozen = compute_link_tables(cfg, base)
    cancelled = compute_link_tables(
        cfg, base, residual_fraction_by_receiver=np.full(cfg.scale.M, 1e-4)
    )
    assert np.all(cancelled.rinr <= frozen.rinr)
    assert np.any(cancelled.rinr < frozen.rinr)


def test_a_worse_fraction_raises_the_residual_ratio():
    """Direction check: a larger surviving fraction must mean more residual.

    The baseline is an explicit reference fraction, not the default path --
    the default *is* the worst case (``frac = 1``) now that no cancellation is
    assumed, so comparing against it could only ever pass trivially.
    """
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    m = cfg.scale.M
    reference = compute_link_tables(
        cfg, base, residual_fraction_by_receiver=np.full(m, 1e-3)
    )

    better = compute_link_tables(cfg, base, residual_fraction_by_receiver=np.full(m, 1e-6))
    worse = compute_link_tables(cfg, base, residual_fraction_by_receiver=np.full(m, 1e-2))
    assert np.all(worse.rinr >= reference.rinr)
    assert np.all(better.rinr <= reference.rinr)
    # ... and it must actually move, otherwise the argument is being dropped.
    assert np.any(worse.rinr > reference.rinr)
    assert np.any(better.rinr < reference.rinr)


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


def test_absolute_residual_power_matches_the_model_aligned_fraction():
    """The canonical power bridge must equal the legacy aligned bridge.

    This is the system-level denominator invariant: if ``fraction`` is defined
    against the exact field used by the link model, multiplying it by that
    field and passing the resulting absolute power must be bit-equivalent.
    """
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    # Reproduce the orthogonal shared-spectrum field used by the model.
    rho = float(cfg.radio.rho)
    input_power = (
        np.full(cfg.scale.M, rho * cfg.radio.P_default) @ base.direct_gain
    )
    fractions = np.linspace(1e-5, 5e-4, cfg.scale.M)
    via_fraction = compute_link_tables(
        cfg, base, residual_fraction_by_receiver=fractions
    )
    via_power = compute_link_tables(
        cfg,
        base,
        residual_power_by_receiver_target=fractions * input_power,
    )
    np.testing.assert_array_equal(via_fraction.rinr, via_power.rinr)
    np.testing.assert_array_equal(
        via_fraction.gamma_sense, via_power.gamma_sense
    )


def test_target_conditioned_absolute_power_uses_each_target_slice():
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    m, q = cfg.scale.M, cfg.scale.Q
    power = np.arange(1, m * q + 1, dtype=float).reshape(m, q) * 1e-12
    joint = compute_link_tables(
        cfg, base, residual_power_by_receiver_target=power
    )
    for target in range(q):
        separate = compute_link_tables(
            cfg, base, residual_power_by_receiver_target=power[:, target]
        )
        np.testing.assert_array_equal(
            joint.gamma_sense[:, :, target],
            separate.gamma_sense[:, :, target],
        )


def test_residual_power_bridge_validates_contract_and_is_exclusive():
    cfg = make_cfg()
    _, base, _ = make_base(cfg)
    m = cfg.scale.M
    with pytest.raises(ValueError, match="either residual_fraction"):
        compute_link_tables(
            cfg,
            base,
            residual_fraction_by_receiver=np.ones(m),
            residual_power_by_receiver_target=np.ones(m),
        )
    with pytest.raises(ValueError, match="residual_power_by_receiver_target"):
        compute_link_tables(
            cfg, base, residual_power_by_receiver_target=np.ones((m, 2, 2))
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        compute_link_tables(
            cfg, base, residual_power_by_receiver_target=-np.ones(m)
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
