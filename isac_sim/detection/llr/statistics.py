"""statistics（自 ``isac_sim/detection/llr.py`` 拆出）。"""

from __future__ import annotations

import numpy as np


EPS = 1e-12


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


def llr_deflection(gamma: np.ndarray | float, n_looks: int) -> np.ndarray | float:
    """Single-link deflection ``delta^2 / var0 = L * gamma^2``."""
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = float(n_looks) * g * g
    return float(out) if np.ndim(gamma) == 0 else out
