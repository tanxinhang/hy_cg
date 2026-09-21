"""weights（自 ``isac_sim/detection/corr.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains

from isac_sim.detection.corr.kernels import EPS
from isac_sim.detection.corr.matrices import covariance_matrix


def solve_psd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` for a PSD ``A``, with a ridge / pinv fallback."""
    n = A.shape[0]
    try:
        return np.linalg.solve(A + 1e-10 * np.eye(n), b)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return np.linalg.pinv(A) @ b


def correlation_aware_weights(
    cfg: Config,
    links: Sequence[Link],
    delta: np.ndarray,
    sigma: np.ndarray,
    base: BaseGains | None = None,
    q: int | None = None,
) -> np.ndarray:
    r"""Deflection-optimal weights ``w propto Sigma^{-1} delta``, normalised.

    Falls back to the independent-observation rule ``w propto delta/sigma^2``
    when the correlation model is switched off or the system is 1-dimensional.
    """
    delta = np.asarray(delta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(links)
    if n == 0:
        return np.zeros(0, dtype=float)
    if (not cfg.corr.enable) or n == 1:
        w = np.maximum(delta, 0.0) / (sigma ** 2 + EPS)
        total = float(np.sum(w))
        return w / total if total > EPS else np.full(n, 1.0 / n)

    Sigma = covariance_matrix(cfg, links, sigma, base=base, q=q)
    w = solve_psd(Sigma, delta)
    # Signed weights are part of the unconstrained deflection optimum.  The
    # Monte-Carlo fusion is linear and does not require a convex combination;
    # clipping negative entries would invalidate w propto Sigma^{-1} delta and
    # the corresponding closed-form deflection claim.
    scale = float(np.sum(np.abs(w)))
    if scale <= EPS:
        return np.full(n, 1.0 / n)
    return w / scale


def correlated_deflection(
    cfg: Config,
    links: Sequence[Link],
    delta: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    base: BaseGains | None = None,
    q: int | None = None,
) -> float:
    """Fused deflection ``(w^T delta)^2 / (w^T Sigma w)``."""
    delta = np.asarray(delta, dtype=float)
    w = np.asarray(weights, dtype=float)
    num = float(w @ delta)
    if num <= 0.0:
        return 0.0
    if cfg.corr.enable and len(links) > 1:
        Sigma = covariance_matrix(cfg, links, sigma, base=base, q=q)
        den = float(w @ Sigma @ w)
    else:
        den = float(np.sum((w ** 2) * (np.asarray(sigma, dtype=float) ** 2)))
    return float(num * num / (den + EPS))


def redundancy_index(
    cfg: Config,
    links: Sequence[Link],
    delta: np.ndarray,
    sigma: np.ndarray,
    base: BaseGains | None = None,
    q: int | None = None,
) -> float:
    """How much of the naive (independence-assuming) deflection is spurious.

    ``1 - D_corr / D_ind`` in ``[0, 1)``: ``0`` means the links are genuinely
    independent, values approaching ``1`` mean the naive fusion was counting
    the same information several times.  Useful as a diagnostic and as an
    ablation axis.
    """
    if len(links) == 0:
        return 0.0
    if not cfg.corr.enable or len(links) == 1:
        return 0.0
    w = correlation_aware_weights(cfg, links, delta, sigma, base=base, q=q)
    d_corr = correlated_deflection(cfg, links, delta, sigma, w, base=base, q=q)
    s = np.asarray(sigma, dtype=float)
    d_ind = float(np.sum((np.asarray(delta, dtype=float) ** 2) / (s ** 2 + EPS)))
    if d_ind <= EPS:
        return 0.0
    return float(max(0.0, 1.0 - d_corr / d_ind))
