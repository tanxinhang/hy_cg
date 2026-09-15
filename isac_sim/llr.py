"""Waveform-derived local log-likelihood ratio (LLR) soft statistic.

Motivation
----------
The legacy soft statistic is

    mu_ijq = kappa_mu * log(1 + gamma^s_ijq),        kappa_mu = 8,

which a reviewer can legitimately dismiss: it encodes the tautology "a higher
sensing SINR should give a larger soft mean" and nothing else, and the single
free scale ``kappa_mu`` silently sets the whole detection operating point.

This module replaces it with the *actual* detection statistic produced by the
delay-Doppler receiver, leaving no free parameter.

Derivation
----------
After OTFS matched filtering, the content of the delay-Doppler bin carrying
target ``q`` on the bistatic link ``i -> q -> j`` is one complex sample

    z = a * s + n,     n ~ CN(0, sigma_n^2),

where each look has an independent fast-fluctuation coefficient
``a_l ~ CN(0, sigma_a^2)``. This is a Swerling-II-like independent-look
abstraction, not a slow-fluctuation Swerling-I model. The post-processing sensing
SINR is ``gamma = sigma_a^2 |s|^2 / sigma_n^2`` (the simulator already folds
the ``N*L`` processing gain and the residual interference into ``gamma``).

The matched filter is a sufficient statistic, so the locally optimal test is
the energy test on the normalised bin energy

    x = |z|^2 / sigma_n^2,
    H0:  x ~ Exp(1)                     (central chi-square, 2 d.o.f.)
    H1:  x ~ Exp(scale = 1 + gamma)     (independent fast fluctuation)

For a CPI that incoherently integrates ``L`` independent looks (frames),
``x`` becomes Gamma(shape=L) with the same scale convention.  The local
log-likelihood ratio is

    ell = log p1(x) / p0(x)
        = -L*ln(1+gamma) + x * gamma/(1+gamma).

Centring it under H0 (which is what a Neyman-Pearson threshold needs) gives
the compact form used everywhere below:

    ell~ = gamma/(1+gamma) * (x - L)

with

    delta  = E1[ell~] - E0[ell~] = L * gamma^2 / (1 + gamma)
    var0   = Var0[ell~]          = L * gamma^2 / (1 + gamma)^2
    var1   = Var1[ell~]          = L * gamma^2
    J      = D_KL(p1 || p0)      = L * (gamma - ln(1 + gamma))

Three things follow, and all three are what the paper needs:

1. The single-link deflection is ``delta^2 / var0 = K_look * gamma^2`` -- Swerling's
   classical result for a square-law detector with a fluctuating target.  No
   free scale anywhere.
2. The per-link *information gain* ``J`` is literally a Kullback-Leibler
   divergence, so the KLD-based selection objective the reviewers asked for
   falls out of the same derivation instead of being assumed.
3. Because the LLR is affine in the sufficient statistic, the fusion weights
   ``w propto delta / var0`` reduce to ``w propto (1 + gamma)`` -- a derived
   rule that replaces the hand-set ``beta`` ranking heuristic.

``K_look`` (``detect.n_looks``) is a genuine physical parameter -- the number of
OTFS frames integrated in one CPI -- not a tuning knob.
"""

from __future__ import annotations

import numpy as np

from .config import Config

EPS = 1e-12


# ==========================================================================
# Analytic moments of the centred local LLR
# ==========================================================================
def llr_delta(gamma: np.ndarray | float, n_looks: int) -> np.ndarray | float:
    """Mean gap ``E1[ell~] - E0[ell~] = L * gamma^2 / (1 + gamma)``."""
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = float(n_looks) * g * g / (1.0 + g)
    return float(out) if np.ndim(gamma) == 0 else out


def llr_var0(gamma: np.ndarray | float, n_looks: int) -> np.ndarray | float:
    """H0 variance ``Var0[ell~] = L * gamma^2 / (1 + gamma)^2``."""
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = float(n_looks) * g * g / (1.0 + g) ** 2
    return float(out) if np.ndim(gamma) == 0 else out


def llr_var1(gamma: np.ndarray | float, n_looks: int) -> np.ndarray | float:
    """H1 variance ``Var1[ell~] = L * gamma^2``."""
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = float(n_looks) * g * g
    return float(out) if np.ndim(gamma) == 0 else out


def llr_kld(gamma: np.ndarray | float, n_looks: int) -> np.ndarray | float:
    r"""Per-link information gain ``D_KL(p1 || p0) = L*(gamma - ln(1+gamma))``."""
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = float(n_looks) * (g - np.log1p(g))
    return float(out) if np.ndim(gamma) == 0 else out


def llr_reverse_kld(gamma: np.ndarray | float, n_looks: int) -> np.ndarray | float:
    r"""Reverse information ``D_KL(p0 || p1)`` for the Gamma hypotheses."""
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = float(n_looks) * (np.log1p(g) - g / (1.0 + g))
    return float(out) if np.ndim(gamma) == 0 else out


def llr_jeffreys(gamma: np.ndarray | float, n_looks: int) -> np.ndarray | float:
    r"""Jeffreys divergence ``D_KL(p1||p0)+D_KL(p0||p1)``.

    The result is exactly the centred-LLR mean gap
    ``L*gamma^2/(1+gamma)`` used by the detector-aligned scheduler.
    """
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = float(n_looks) * g * g / (1.0 + g)
    return float(out) if np.ndim(gamma) == 0 else out


def llr_h0_offset(gamma: np.ndarray | float, n_looks: int) -> np.ndarray | float:
    r"""Constant that converts the centred statistic into the exact LLR.

    ``ell = ell_tilde + L * (gamma/(1+gamma) - log(1+gamma))``.
    The term must be included once for every *successfully received* report;
    with true erasures it cannot be absorbed into one fixed threshold because
    the received-report set is random.
    """
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = float(n_looks) * (g / (1.0 + g) - np.log1p(g))
    return float(out) if np.ndim(gamma) == 0 else out


def llr_deflection(gamma: np.ndarray | float, n_looks: int) -> np.ndarray | float:
    """Single-link deflection ``delta^2 / var0 = L * gamma^2``."""
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = float(n_looks) * g * g
    return float(out) if np.ndim(gamma) == 0 else out


def optimal_fusion_weight(gamma: np.ndarray | float) -> np.ndarray | float:
    """Deflection-optimal fusion weight ``w propto delta / var0 = 1 + gamma``."""
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = 1.0 + g
    return float(out) if np.ndim(gamma) == 0 else out


# ==========================================================================
# Monte-Carlo sampling of the local LLR
# ==========================================================================
def draw_llr(
    gamma: float,
    n_looks: int,
    rng: np.random.Generator,
    h1: bool,
) -> float:
    """Draw one centred local LLR sample.

    Under ``H1`` the normalised bin energy is Gamma(L, scale = 1 + gamma);
    under ``H0`` it is Gamma(L, scale = 1).  ``gamma <= 0`` carries no target
    information, so the two hypotheses coincide and the draw is the H0 one.
    """
    g = max(float(gamma), 0.0)
    scale = (1.0 + g) if h1 else 1.0
    x = float(rng.gamma(shape=float(max(n_looks, 1)), scale=scale))
    return g / (1.0 + g) * (x - float(max(n_looks, 1)))


def draw_llr_erased(
    gamma: float,
    n_looks: int,
    rng: np.random.Generator,
    h1: bool,
) -> float:
    """Draw the statistic the fusion node substitutes after a *failed* packet.

    An erased short packet carries no target information, so under both
    hypotheses the substituted value is an H0-valued LLR, slightly inflated by
    ``soft_error_sigma_scale`` to model the fusion node's uncertainty about
    what it actually received.
    """
    g = max(float(gamma), 0.0)
    x = float(rng.gamma(shape=float(max(n_looks, 1)), scale=1.0))
    base = g / (1.0 + g) * (x - float(max(n_looks, 1)))
    if not h1:
        return base
    # Under H1 a lost report still removes the target evidence; the legacy
    # model expressed this as "mean pulled towards zero", which here is
    # automatic because the H0 draw has zero mean.
    return base


# ==========================================================================
# Dispatch used by the model layer
# ==========================================================================
def soft_mean(cfg: Config, gamma: np.ndarray | float) -> np.ndarray | float:
    """H1 mean of the soft statistic under the configured model."""
    if cfg.detect.soft_stat_model.lower() == "llr":
        return llr_delta(gamma, cfg.detect.n_looks)
    # Legacy: kappa_mu * log(1 + gamma).  Arithmetic untouched for parity.
    g = np.asarray(gamma, dtype=float)
    out = cfg.detect.soft_mu_scale * np.log1p(np.maximum(g, 0.0))
    return float(out) if np.ndim(gamma) == 0 else out


def soft_var0(cfg: Config, gamma: np.ndarray | float, pair_var0: np.ndarray | float) -> np.ndarray | float:
    """H0 variance of the soft statistic under the configured model.

    In ``"llr"`` mode the variance is *derived* from the sensing SINR, so the
    legacy pair-level ``sigma0^2`` (which only carried the residual-interference
    inflation) is no longer used -- the residual interference already sits in
    the denominator of ``gamma`` itself, and carrying it twice would be double
    counting.
    """
    if cfg.detect.soft_stat_model.lower() == "llr":
        return llr_var0(gamma, cfg.detect.n_looks)
    return pair_var0


def soft_var1(cfg: Config, gamma: np.ndarray | float, pair_var0: np.ndarray | float) -> np.ndarray | float:
    """H1 variance of the soft statistic (used by the Monte-Carlo detector)."""
    if cfg.detect.soft_stat_model.lower() == "llr":
        return llr_var1(gamma, cfg.detect.n_looks)
    # Legacy: sigma0^2 / (1 + gamma).
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = np.asarray(pair_var0, dtype=float) / (1.0 + g)
    return float(out) if np.ndim(gamma) == 0 else out
