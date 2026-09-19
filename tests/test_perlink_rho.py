"""Invariants for the P1-3 admission measurement (per-link ``rho``).

The measurement answers "does a ``rho``-aware selector have anything to choose".
Its two helper statistics are therefore part of the claim, and each one is a
place where a silent defect would read as a conclusion:

* ``_headroom`` mixes up *which* k links are compared -- if it ever returned the
  same subset twice it would report zero headroom, i.e. "P1-3 is pointless";
* ``_spearman`` decides (B); a sign or tie-handling error there would decide
  whether ``rho`` counts as new information;
* the receiver-level spread is a **between-receiver** quantity -- computed inside
  one receiver's loop it is a single-element group and returns NaN quietly, which
  is exactly how the first run reported it.
"""
from __future__ import annotations

import numpy as np
import pytest

from isac_sim.config import Config, apply_preset
from tools import probe_perlink_rho as pp


def scene_cfg() -> Config:
    """The *same* scene the per-link probe and the angular probe measure.

    A bare ``Config()`` is a different waveform and power budget; using it here
    is how a comparison between "per-link median 0.30" and "scene median 0.94"
    can look like a finding when it is only a different scenario.
    """
    cfg = apply_preset(Config(), "paper-canonical")
    cfg.geometry.area_xy = 600.0
    cfg.detect.target_rcs = 0.1
    return cfg


def one_scene():
    cfg = scene_cfg()
    rng = np.random.default_rng([cfg.run.seed, 10 ** 6])
    geom = pp.generate_geometry(cfg, rng)
    base = pp.build_base_gains(cfg, geom, rng)
    belief = pp.perturbed_geometry(
        cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
    )
    return cfg, geom, base, belief


def test_headroom_compares_different_subsets():
    """``top - median-ranked`` must be zero only when there is nothing to choose.

    Both ends are means over ``k`` links, so with ``k = 3`` the top end is the
    mean of the three best (0.875 on this profile), not the single maximum --
    an earlier version of this test asserted 1.0 and failed against correct code.
    """
    flat = [0.5] * 9
    top, mid = pp._headroom(flat, 3)
    assert top == pytest.approx(0.5)
    assert mid == pytest.approx(0.5)
    assert top - mid == pytest.approx(0.0), "a flat profile must give zero headroom"

    spread = list(np.linspace(0.0, 1.0, 9))   # descending: 1.0, 0.875, 0.75, ...
    top, mid = pp._headroom(spread, 3)
    assert top == pytest.approx(0.875)         # mean of the three best
    assert mid == pytest.approx(0.5)           # three around the median rank
    assert top - mid > 0.3


def test_headroom_is_bounded_by_the_profile():
    """k=1 is the maximum; k >= n is *no selection*, so it returns the mean.

    The second half is the one worth pinning: clamping ``k`` to the number of
    links means "take everything", whose mean is the average, not the best --
    i.e. the headroom readout has to stay zero there, or an over-large budget
    would look like a selection gain.
    """
    v = [0.1, 0.9, 0.4]
    assert pp._headroom(v, 1)[0] == pytest.approx(0.9)
    assert pp._headroom(v, 99)[0] == pytest.approx(float(np.mean(v)))
    assert pp._headroom(v, 99)[1] == pytest.approx(float(np.mean(v)))


def test_spearman_has_the_right_sign_and_range():
    x = list(range(20))
    assert pp._spearman(x, x) == pytest.approx(1.0)
    assert pp._spearman(x, x[::-1]) == pytest.approx(-1.0)
    rng = np.random.default_rng(7)
    y = list(rng.normal(size=20))
    assert abs(pp._spearman(x, y)) <= 1.0
    # rank-based, so any monotone transform of one side leaves it unchanged
    assert pp._spearman(x, [v ** 3 for v in y]) == pytest.approx(pp._spearman(x, y))
    # too few points is "not measured", not "no correlation"
    assert not np.isfinite(pp._spearman([0.1, 0.2], [0.3, 0.4]))


def test_the_dd_only_baseline_is_degenerate_for_the_typical_link():
    """``M = 1`` must leave the *typical* link with no escape.

    Asserted on the **median**, not the maximum: a few targets are isolated in
    DD and keep a large ``rho`` even without an array (measured p90 = 0.32 at
    M = 1), so a max-based assertion would fail on a correct implementation.
    That heavy tail is also why the scene-level figure must be quoted as a
    median -- the power-weighted mean at M = 1 is already 0.23.
    """
    cfg, geom, base, belief = one_scene()
    rows = pp.per_link_rho(cfg, belief, base, 0, 1, pp.PRIMARY_AXIS)
    assert rows, "the scene must produce templates"
    med = float(np.median([r["rho"] for r in rows]))
    assert med < 1e-2, "DD-only median escape should be ~0; got %.4g" % med


def test_more_aperture_gives_more_escape_on_the_same_scene():
    """Median per-link ``rho`` must increase with ``M`` (no aperture => no escape)."""
    cfg, geom, base, belief = one_scene()
    med = {}
    for m in (1, 4, 16):
        rows = pp.per_link_rho(cfg, belief, base, 0, m, pp.PRIMARY_AXIS)
        med[m] = float(np.median([r["rho"] for r in rows]))
    assert med[16] >= med[4] >= med[1]


def test_the_receiver_level_spread_is_a_between_receiver_quantity():
    """Regression for the NaN: aggregating must happen across receivers."""
    cfg, geom, base, belief = one_scene()
    tables = pp.compute_link_tables(cfg, base)
    pooled = []
    for j in range(3):
        pooled.extend(pp.per_link_rho(cfg, belief, base, j, 8, pp.PRIMARY_AXIS))
    rows = pp.target_rows(0, 8, pp.PRIMARY_AXIS, pooled, tables.gamma_sense)
    assert rows
    for r in rows:
        assert r["n_receivers"] >= 3, "links were not pooled over receivers"
        assert np.isfinite(r["spread_receiver_level"]), (
            "receiver-level spread is NaN -- the pool is a single receiver")


def test_the_pre_registered_constants_do_not_contradict_each_other():
    """The previous probe shipped a rule whose PASS trigger equalled its FAIL
    threshold; guard against repeating that rather than re-checking by eye."""
    assert pp.SPREAD_PASS > pp.SPREAD_FAIL, "PASS and FAIL thresholds must differ"
    assert 0.0 < pp.RHO_S_MAX < 1.0
    assert set(pp.CANDIDATE_M) <= set(pp.M_LIST)
    assert pp.PRIMARY_AXIS in pp.AXES
