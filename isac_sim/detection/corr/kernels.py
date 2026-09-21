"""kernels（自 ``isac_sim/detection/corr.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np
from isac_sim.core.config import Config, Link


EPS = 1e-12


def _rhos(cfg: Config) -> Tuple[float, float, float, float]:
    """Return validated ``(rho_tx, rho_rx, rho_target, rho_dd)`` values.

    Configuration validation owns clipping policy: silently changing recorded
    experiment parameters here would make the saved manifest disagree with the
    covariance actually used by the simulator.
    """
    c = cfg.corr
    raw = [float(c.rho_tx), float(c.rho_rx), float(c.rho_target), float(c.rho_dd)]
    if not all(np.isfinite(value) and value >= 0.0 for value in raw):
        raise ValueError("correlation coefficients must be finite and non-negative")
    if sum(raw) > 1.0 + EPS:
        raise ValueError("correlation coefficients must sum to at most one")
    return raw[0], raw[1], raw[2], raw[3]


def dd_laplace_kernel(
    delay_bin: Sequence[int],
    doppler_bin: Sequence[int],
    kappa: float = 1.0,
) -> np.ndarray:
    """PSD kernel on the delay-Doppler grid: product of two Laplace kernels."""
    l = np.asarray(delay_bin, dtype=float)
    k = np.asarray(doppler_bin, dtype=float)
    dl = np.abs(l[:, None] - l[None, :])
    dk = np.abs(k[:, None] - k[None, :])
    return np.exp(-dl / max(kappa, EPS)) * np.exp(-dk / max(kappa, EPS))
