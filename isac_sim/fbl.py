"""Finite-blocklength (FBL) reporting reliability.

Why this module exists
----------------------
The legacy model defines the reporting reliability as

    chi = gamma / (gamma + gamma_req),

which is a smooth monotone map from SINR into ``[0, 1)`` and nothing more.
Three reviewers pushed back on it because it cannot be read as a *packet
success probability*: the payload size plays no role, the blocklength plays no
role, and the number it produces has no probabilistic interpretation.

A soft sensing report is a short control packet, so the natural model is the
normal-approximation finite-blocklength error probability

    eps(gamma) ~= Q( ( C(gamma) - k/n ) / sqrt( V(gamma)/n ) ),
    C(gamma)   = log2(1 + gamma),
    V(gamma)   = (1 - (1 + gamma)^-2) * (log2 e)^2,

with payload ``k`` bits carried in ``n`` channel uses.  Then

    chi = 1 - eps

is literally the probability that the report arrives intact.  This single
change makes reliability, payload and latency one coupled system instead of
three independently hand-tuned knobs:

* ``k`` (bits) enters the error exponent -- a bigger report is less reliable;
* ``n`` (channel uses) enters both the error exponent and the latency
  ``T = n / B`` -- a longer block is more reliable but slower;
* the latency no longer depends on the instantaneous link rate, which is
  what makes the scheduling problem well posed.

Reference behaviour
-------------------
``reliability_model="heuristic"`` reproduces the legacy
``gamma / (gamma + gamma_req)`` value bit-for-bit, so every frozen result
stays valid.  Only ``"fbl"`` switches to the new model.
"""

from __future__ import annotations

import math

import numpy as np

from .config import Config

LN2 = math.log(2.0)
LOG2E = 1.0 / LN2


# ==========================================================================
# Core FBL quantities
# ==========================================================================
def channel_dispersion(gamma: np.ndarray | float) -> np.ndarray | float:
    """Channel dispersion ``V(gamma) = (1 - (1+gamma)^-2) (log2 e)^2``."""
    g = np.asarray(gamma, dtype=float)
    v = (1.0 - 1.0 / (1.0 + np.maximum(g, 0.0)) ** 2) * LOG2E ** 2
    return float(v) if np.ndim(gamma) == 0 else v


def fbl_error_prob(
    gamma: np.ndarray | float,
    n_block: int,
    k_bits: float,
) -> np.ndarray | float:
    r"""Normal-approximation packet error probability.

        eps = Q( (C(gamma) - k/n) / sqrt(V(gamma)/n) )

    ``eps`` is clipped to ``[0, 1]``.  When ``k/n`` exceeds the Shannon
    capacity the argument of ``Q`` is negative and ``eps > 0.5``, i.e. the
    packet is more likely than not to fail -- which is the physically correct
    behaviour for a rate request the channel cannot support.
    """
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    n = float(n_block)
    C = np.log2(1.0 + g)
    V = channel_dispersion(g)
    denom = np.sqrt(np.maximum(V, 1e-30) / n)
    z = (C - k_bits / n) / denom
    eps = _qfunc(z)
    return float(eps) if np.ndim(gamma) == 0 else eps


def fbl_success_prob(
    gamma: np.ndarray | float,
    n_block: int,
    k_bits: float,
) -> np.ndarray | float:
    """Packet success probability ``chi = 1 - eps``."""
    return 1.0 - fbl_error_prob(gamma, n_block, k_bits)


def blocklength_latency_s(cfg: Config) -> float:
    """Latency of one FBL report block: ``n`` channel uses at bandwidth ``B``."""
    B = cfg.waveform.N * cfg.waveform.delta_f
    return float(cfg.comm.n_block) / B


def min_blocklength_for_target(cfg: Config, eps_target: float, gamma: float) -> float:
    """Smallest blocklength that reaches ``eps_target`` at the given SINR.

    Closed form from inverting the normal approximation.  Returns ``inf`` when
    the requested error floor is unreachable (i.e. ``k/n`` stays above capacity
    for every ``n``), which is the strict FBL counterpart of the legacy
    ``rate >= R_min`` feasibility test.
    """
    k = float(cfg.comm.K_candidates) * cfg.comm.b_d
    C = math.log2(1.0 + max(gamma, 0.0))
    V = float(channel_dispersion(gamma))
    qinv = _qinv(eps_target)
    if C <= 0.0 or V <= 0.0:
        return math.inf
    # k/n + q * sqrt(V/n) = C  ->  let u = 1/sqrt(n):
    #     k u^2 + q sqrt(V) u - C = 0
    a = k
    b = qinv * math.sqrt(V)
    c = -C
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return math.inf
    u = (-b + math.sqrt(disc)) / (2.0 * a)
    if u <= 0.0:
        return math.inf
    return 1.0 / (u * u)


# ==========================================================================
# Dispatch: legacy vs FBL
# ==========================================================================
def chi_from_gamma(
    cfg: Config,
    gamma: np.ndarray | float,
    gamma_req: float,
    k_bits: float | None = None,
) -> np.ndarray | float:
    """Reliability of a reporting link, under whichever model is configured."""
    if cfg.comm.reliability_model.lower() == "fbl":
        k = packet_bits(cfg) if k_bits is None else float(k_bits)
        return fbl_success_prob(gamma, cfg.comm.n_block, k)
    # Legacy heuristic (unchanged arithmetic -- parity critical).
    g = np.asarray(gamma, dtype=float)
    out = g / (g + gamma_req + 1e-12)
    return float(out) if np.ndim(gamma) == 0 else out


def packet_bits(cfg: Config) -> float:
    """Payload of one soft-information report, in bits."""
    return float(max(cfg.comm.K_candidates, 0) * cfg.comm.b_d)


def report_latency_s(cfg: Config, rate_bps: float) -> float:
    """Latency of one report under the configured latency bookkeeping."""
    if cfg.comm.latency_model.lower() == "blocklength":
        return blocklength_latency_s(cfg)
    return packet_bits(cfg) / max(float(rate_bps), 1e-12)


# ==========================================================================
# Numeric helpers (local, so this module has no scipy dependency)
# ==========================================================================
def _qfunc(x: np.ndarray | float) -> np.ndarray | float:
    """Gaussian Q-function, accurate and fast enough for O(M^2) tables."""
    arr = np.asarray(x, dtype=float)
    # erfc is not a numpy ufunc; a small cached vectorisation keeps the table
    # build at ~microseconds for the M=15 regime used throughout.
    flat = arr.ravel()
    out = np.array([0.5 * math.erfc(v / math.sqrt(2.0)) for v in flat], dtype=float)
    out = out.reshape(arr.shape)
    return float(out) if np.ndim(x) == 0 else out


def _qinv(p: float) -> float:
    """Inverse Q-function (Acklam-style rational approximation + refinement)."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    # Initial guess via the Beasley-Springer-Moro approximation on the
    # standard normal quantile, then two Newton steps on erfc.
    x = _norm_ppf(1.0 - p)
    for _ in range(3):
        f = 0.5 * math.erfc(x / math.sqrt(2.0)) - p
        fp = -math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        step = f / fp
        x -= step
        if abs(step) < 1e-14:
            break
    return x


def _norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
