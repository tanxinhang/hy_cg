"""P1-2: the receive array enters the production collision penalty.

Three things must hold, and each has its own failure mode:

* **the gate** -- with ``aperture.enable`` closed every released number is
  bit-exact, and even with the gate open ``m_rx = 1`` must reproduce the
  integer count, because that is what the released model is;
* **direction** -- more aperture may only ever *remove* masking, never add it,
  and it must not touch the echo power.  A "gain" that appears here is a bug:
  the array's collection gain is a link-budget quantity and belongs with
  ``radar_net_gain_db``, not here;
* **no threshold** -- the soft count is continuous in the angular separation, so
  there is nothing to tune after seeing the result.
"""
from __future__ import annotations

import numpy as np
import pytest

from isac_sim import aperture as ap
from isac_sim.config import Config, apply_preset
from isac_sim.model import build_base_gains, compute_link_tables, generate_geometry

TRIAL = 5


def scene(m_rx: int | None = None):
    """One fixed geometry.  ``m_rx=None`` leaves the gate closed (the release)."""
    cfg = apply_preset(Config(), "paper-canonical")
    cfg.geometry.area_xy = 600.0
    cfg.detect.target_rcs = 0.1
    if m_rx is not None:
        cfg.aperture.enable = True
        cfg.aperture.m_rx = int(m_rx)
    rng = np.random.default_rng([cfg.run.seed, TRIAL])
    geom = generate_geometry(cfg, rng)
    return cfg, geom, build_base_gains(cfg, geom, rng)


def test_the_array_model_is_unit_modulus():
    """Aperture buys selectivity, not loudness -- pinned at the source.

    ``rho`` is a coherence: 0 head-on, ~1 beyond a beamwidth, and it must never
    exceed 1 or go negative (the latter is what the Gram-domain defect produced,
    and it read as "no escape at all").
    """
    assert ap.array_escape(8, 0.0) == pytest.approx(0.0, abs=1e-15)
    assert ap.array_escape(1, 0.7) == 0.0          # one element resolves nothing
    assert 0.0 <= ap.array_escape(8, 0.3) <= 1.0
    assert ap.array_escape(8, ap.beamwidth_u(8)) == pytest.approx(0.5, abs=0.02)
    assert ap.array_escape(8, 0.05) < ap.array_escape(8, 0.2)


def test_the_gate_is_closed_by_default():
    cfg, _, _ = scene()
    assert cfg.aperture.enable is False
    assert cfg.aperture.m_rx == 1
    assert cfg.aperture.axis == 0


def test_one_element_is_exactly_the_released_dd_only_model():
    """``|A|^2 == 1`` for a single element, so the soft sum *is* the integer count.

    Compared through the whole link table, not just the count: the count only
    matters because it divides the sensing signal, and a count that matches
    while ``gamma_sense`` moves would be a worse bug than either alone.
    """
    cfg_off, _, base_off = scene()
    cfg_on, _, base_on = scene(m_rx=1)
    assert np.array_equal(base_off.dd_collision_count, base_on.dd_collision_count)
    assert np.allclose(base_on.dd_collision_count,
                       np.round(base_on.dd_collision_count), atol=1e-12)
    assert np.array_equal(compute_link_tables(cfg_off, base_off).gamma_sense,
                          compute_link_tables(cfg_on, base_on).gamma_sense)


def test_more_aperture_only_ever_removes_masking():
    """Direction, pinned: the count falls with ``m_rx`` and never rises."""
    counts = {m: scene(m_rx=m)[2].dd_collision_count.copy() for m in (1, 4, 8, 16)}
    for m in (4, 8, 16):
        assert np.all(counts[m] <= counts[1] + 1e-12), "m must not add masking"
    assert np.all(counts[8] <= counts[4] + 1e-12)
    assert np.all(counts[16] <= counts[8] + 1e-12)
    # and it must actually do something on this scene, or the wiring is inert
    assert counts[8].max() > 1.0 + 1e-12
    assert float(np.mean(counts[16])) < float(np.mean(counts[1]))


def test_the_aperture_does_not_inflate_the_echo_power():
    """The trap the whole design exists to avoid.

    Turning the array on may change how much co-bin interference masks the echo,
    but it must not change the echo itself: ``target_gain`` is the propagation
    and RCS law, and a collection gain appearing there would be a budget dressed
    as a mechanism -- indistinguishable from raising ``radar_net_gain_db``.
    """
    _, _, base_off = scene()
    _, _, base_on = scene(m_rx=16)
    assert np.array_equal(base_off.target_gain, base_on.target_gain)
    assert np.array_equal(base_off.dd_frac_loss, base_on.dd_frac_loss)
    assert np.array_equal(base_off.direct_gain, base_on.direct_gain)


def test_the_soft_count_is_continuous_in_separation():
    """No threshold to tune: masking decays smoothly with angular separation."""
    m = 8
    values = [ap.masking_fraction(m, du) for du in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)]
    assert values[0] == pytest.approx(1.0)
    assert all(v <= 1.0 + 1e-12 for v in values)
    # strictly decreasing well inside the main lobe
    assert all(b <= a for a, b in zip(values[:5], values[1:5]))
    assert values[4] < values[1]
