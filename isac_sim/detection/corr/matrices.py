"""matrices（自 ``isac_sim/detection/corr.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains

from isac_sim.detection.corr.kernels import _rhos, dd_laplace_kernel


def correlation_matrix(
    cfg: Config,
    links: Sequence[Link],
    delay_bin: Sequence[int] | None = None,
    doppler_bin: Sequence[int] | None = None,
) -> np.ndarray:
    """Unit-diagonal correlation matrix of the local statistics in ``links``."""
    n = len(links)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    rho_tx, rho_rx, rho_tgt, rho_dd = _rhos(cfg)

    tx = np.array([link[0] for link in links])
    rx = np.array([link[1] for link in links])
    B_tx = (tx[:, None] == tx[None, :]).astype(float)
    B_rx = (rx[:, None] == rx[None, :]).astype(float)
    J = np.ones((n, n), dtype=float)

    R = (1.0 - rho_tx - rho_rx - rho_tgt - rho_dd) * np.eye(n)
    R = R + rho_tx * B_tx + rho_rx * B_rx + rho_tgt * J
    if rho_dd > 0.0 and delay_bin is not None and doppler_bin is not None:
        R = R + rho_dd * dd_laplace_kernel(delay_bin, doppler_bin)
    # Enforce an exact unit diagonal (the construction already gives one up to
    # floating-point noise; this keeps the idiosyncratic share well defined).
    np.fill_diagonal(R, 1.0)
    return 0.5 * (R + R.T)


def link_dd_bins(
    cfg: Config,
    base: BaseGains | None,
    q: int,
    links: Sequence[Link],
) -> Tuple[List[int], List[int]]:
    """Delay / Doppler bin indices of ``links`` for target ``q``."""
    if base is None:
        return [0] * len(links), [0] * len(links)
    return (
        [int(base.delay_bin[i, j, q]) for i, j in links],
        [int(base.doppler_bin[i, j, q]) for i, j in links],
    )


def covariance_matrix(
    cfg: Config,
    links: Sequence[Link],
    sigma: np.ndarray,
    base: BaseGains | None = None,
    q: int | None = None,
) -> np.ndarray:
    """Full covariance ``Sigma = S R S`` of the stacked local statistics."""
    R = correlation_matrix(
        cfg,
        links,
        *(link_dd_bins(cfg, base, q if q is not None else 0, links) if base is not None else (None, None))
    )
    S = np.asarray(sigma, dtype=float)
    return R * np.outer(S, S)
