"""divergences（自 ``isac_sim/detection/llr.py`` 拆出）。"""

from __future__ import annotations

import numpy as np


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


def optimal_fusion_weight(gamma: np.ndarray | float) -> np.ndarray | float:
    """Deflection-optimal fusion weight ``w propto delta / var0 = 1 + gamma``."""
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = 1.0 + g
    return float(out) if np.ndim(gamma) == 0 else out
