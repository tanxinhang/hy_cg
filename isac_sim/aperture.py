"""The receive-array model: one implementation, used by the probe and by the model.

Everything here is **unit-modulus**: the steering vectors are normalised, so a
larger aperture buys *spatial selectivity* and nothing else.  That is the whole
point of keeping it in its own module.

The collection gain of an ``m``-element array (``10 log10 m`` on the desired
echo) is a **link-budget** quantity of exactly the same kind as
``radio.radar_net_gain_db``: it makes everything louder, signal and interference
alike, and it is therefore *not* modelled here.  The angular probe's entire
verdict rests on separating the two -- reporting a detection gain from an
aperture without saying which of the two produced it repeats the ``G_hw = 15 dB``
mistake of reading a budget as a mechanism.  If the platform is credited with a
collection gain, it belongs in the link budget and must be declared as such.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "beamwidth_u",
    "array_factor",
    "array_escape",
    "bearings",
    "masking_fraction",
]


def beamwidth_u(m_rx: int) -> float:
    """3 dB beamwidth of a half-wavelength ULA, in ``u = sin(phi)`` units.

    ``0.886 / m`` is the two-sided half-power width of the ``|sin(m x)/(m sin x)|
    `` main lobe.  Every separation in this project is reported in these units
    rather than in degrees, because the whole angular result is a function of
    ``m * delta_u`` alone: in degrees the same geometry would read differently for
    every aperture, and "how many beamwidths apart" would move with the hardware
    choice it is supposed to inform.
    """
    return 0.886 / float(m_rx)


def array_factor(m_rx: int, delta_u: float) -> float:
    """``|<a(u), a(u + du)>|`` for unit-norm steering, spacing ``lambda/2``."""
    if m_rx <= 1:
        return 1.0
    x = math.pi * delta_u / 2.0
    if x % math.pi == 0.0:
        return 1.0
    return abs(math.sin(m_rx * x) / (m_rx * math.sin(x)))


def array_escape(m_rx: int, delta_u: float) -> float:
    """``rho = 1 - |A|^2``: the fraction of one steering vector that escapes the
    span of the other -- the array-only analogue of the DD escape fraction.

    ``rho`` is a *coherence*, not a gain: it is 0 for two targets seen in the same
    direction and tends to 1 once they are more than a beamwidth apart.  It says
    nothing about how loud either echo is.
    """
    if m_rx <= 1:
        return 0.0
    a = array_factor(m_rx, delta_u)
    return 1.0 - a * a


def bearings(geom, receiver: int, q_count: int, axis: int) -> np.ndarray:
    """``(Q,)`` direction cosines of the targets, seen from ``receiver``.

    Projected onto one body axis (``0`` = x, ``1`` = y) because a one-dimensional
    array resolves along one axis only.  A 1-D array also has a front/back
    ambiguity (``|A|`` returns to 1 at ``delta_u = 2``), so this is a declared
    hardware assumption rather than a free choice: the axis is fixed in
    configuration and never chosen after the numbers are seen.
    """
    us = np.zeros(q_count, dtype=float)
    p_rx = np.asarray(geom.p_uav[receiver], dtype=float)
    for q in range(q_count):
        v = np.asarray(geom.p_tgt[q], dtype=float)[:3] - p_rx[:3]
        n = float(np.linalg.norm(v))
        us[q] = float(v[axis] / n) if n > 0.0 else 0.0
    return us


def masking_fraction(m_rx: int, delta_u: float) -> float:
    """How much of one echo the other masks: ``1 - rho = |A|^2``.

    This is the quantity the production collision penalty uses.  With one element
    it is exactly 1.0 (the DD-only model: any co-bin target masks completely), and
    it shrinks as the array separates the pair -- which is what makes the
    collision penalty array-aware without introducing a hard threshold.
    """
    a = array_factor(m_rx, delta_u)
    return a * a
