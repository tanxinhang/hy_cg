"""Pin the waveform mapping before anyone sweeps ``N``/``L`` as "resolution".

These are not physics assertions -- they are *bookkeeping* assertions about what
the model currently does, so that a decision to change the mapping has to be
taken deliberately instead of being discovered from a sweep that nobody can
explain.  The finding they record is in ``ANGULAR_IDENTIFIABILITY_PROBE.md`` §七:

the delay axis is quantised on ``L * delta_f`` while the noise bandwidth -- and
therefore the physical delay resolution ``1 / B`` -- is ``N * delta_f``.  At the
default ``N == L == 64`` the two readings of "the bandwidth" are equal, which is
why the model is self-consistent out of the box and stops being so the moment
the two are swept independently.
"""
from __future__ import annotations

import numpy as np
import pytest

from isac_sim.sensing import model as md
from isac_sim.core.config import Config

C = 3e8


def cfg_of(n: int, l: int, area: float = 600.0) -> Config:
    cfg = Config()
    cfg.waveform.N = n
    cfg.waveform.L = l
    cfg.geometry.area_xy = area
    return cfg


def test_the_default_grid_is_the_self_consistent_one():
    """The released work point is exactly the degenerate case that hides this.

    ``N == L`` is what makes ``1 / (L * delta_f)`` and ``1 / B`` the same number.
    Nothing in the config enforces it, so the next person to sweep either one
    silently leaves this regime -- which is the whole reason these tests exist.
    """
    cfg = Config()
    assert cfg.waveform.N == cfg.waveform.L


def test_the_noise_bandwidth_is_set_by_N_and_not_by_L():
    """``bandwidth = N * delta_f``, so ``L`` is free of noise consequences.

    If this ever changes (e.g. someone "fixes" the mapping by moving the delay
    axis onto ``N``) the resolution sweep changes meaning, and this test is the
    place that change has to be made visible.
    """
    base = md.bandwidth(cfg_of(64, 64))
    assert md.bandwidth(cfg_of(64, 256)) == pytest.approx(base)
    assert md.bandwidth(cfg_of(256, 64)) == pytest.approx(4.0 * base)
    assert md.noise_power(cfg_of(256, 64)) == pytest.approx(
        4.0 * md.noise_power(cfg_of(64, 64))
    )


def test_the_delay_axis_is_quantised_by_L_and_the_doppler_axis_by_N():
    """Measured on a real geometry, not derived from the docstring.

    ``delay_bin = round(tau L delta_f)`` and ``doppler_bin = round(nu N T)``.
    Doubling ``L`` doubles every delay index; doubling ``N`` doubles every
    Doppler index.  The two axes must not be cross-wired.
    """
    geom = md.generate_geometry(cfg_of(64, 64), np.random.default_rng([1, 7]))

    def bins(n: int, l: int):
        base = md.build_base_gains(cfg_of(n, l), geom, np.random.default_rng([4242, 0]))
        return np.asarray(base.delay_bin, dtype=float), np.asarray(base.doppler_bin, dtype=float)

    d64, k64 = bins(64, 64)
    d128, _ = bins(64, 128)
    _, k128 = bins(128, 64)
    # atol 1: round() can land on either side of a .5 boundary
    assert np.allclose(d128, 2.0 * d64, atol=1.0)
    assert np.allclose(k128, 2.0 * k64, atol=1.0)


def test_the_two_readings_of_the_delay_cell_agree_only_when_N_equals_L():
    """The finding itself: ``1/(L delta_f)`` vs ``1/B`` diverge as ``N / L``.

    This is what makes an ``N``/``L`` sweep uninterpretable until the mapping is
    adjudicated -- a sweep that only moves ``L`` refines the *grid* while the
    physical ``1/B`` resolution (and the noise floor) does not move at all.
    """
    for n, l in ((64, 64), (128, 128)):
        cfg = cfg_of(n, l)
        assert C / (2.0 * l * cfg.waveform.delta_f) == pytest.approx(
            C / (2.0 * md.bandwidth(cfg))
        ), "delay cell should agree at N == L"

    cfg = cfg_of(64, 256)
    grid_cell = C / (2.0 * 256 * cfg.waveform.delta_f)
    band_cell = C / (2.0 * md.bandwidth(cfg))
    assert grid_cell == pytest.approx(band_cell / 4.0)
    assert band_cell > grid_cell, "the grid must not claim more than 1/B resolves"


def test_processing_gain_is_the_time_bandwidth_product():
    """``G_proc = N * L`` is the one quantity that is unambiguously physical.

    Both dimensions legitimately earn integration gain, which is why raising
    ``L`` looks like a free 3 dB: it is real, but it is *integration*, not
    resolution -- and the collision penalty below is where the two get confused.
    """
    for n, l in ((64, 64), (64, 256), (256, 64)):
        cfg = cfg_of(n, l)
        assert md.bandwidth(cfg) * (l * cfg.waveform.T) == pytest.approx(float(n * l))


def test_the_collision_penalty_tracks_the_rounding_grid_not_the_bandwidth():
    """Decision-grade: more delay bins at fixed ``B`` reduces collisions anyway.

    ``dd_collision_count`` counts targets sharing one *rounded* bin, so refining
    the grid alone separates them on paper.  Measured on one fixed scene at the
    600 m work point: 11.9% of valid links collide at 64x64, 2.3% at 64x256 --
    with ``B`` and the noise floor untouched -- while 256x64, which quadruples
    ``B`` and the noise, only reaches 2.7%.

    The thresholds are loose because they are direction checks on one scene; the
    point is the *ordering*, not the exact percentage.
    """
    geom = md.generate_geometry(cfg_of(64, 64), np.random.default_rng([12345, 777]))

    def colliding(n: int, l: int) -> float:
        cfg = cfg_of(n, l)
        assert cfg.dd.enable_dd_collision_penalty, "penalty must be on for this to mean anything"
        base = md.build_base_gains(cfg, geom, np.random.default_rng([12345, 0]))
        cc = np.asarray(base.dd_collision_count, dtype=float)
        valid = np.asarray(base.valid_dd, dtype=bool)
        m = cc.shape[0]
        sel = (~np.eye(m, dtype=bool)[:, :, None]) & valid
        return float((cc[sel] > 1.0).mean())

    at_64 = colliding(64, 64)
    more_delay_bins = colliding(64, 256)
    more_bandwidth = colliding(256, 64)

    assert more_delay_bins < at_64, "refining the delay grid must separate targets"
    assert more_bandwidth < at_64
    assert more_delay_bins <= more_bandwidth, (
        "re-quantisation must not buy less separation than a 4x bandwidth increase: "
        "%.4f vs %.4f" % (more_delay_bins, more_bandwidth)
    )
