"""bounds（自 ``isac_sim/sensing/fbl.py`` 拆出）。"""

from __future__ import annotations

import math
import numpy as np
from isac_sim.core.config import Config

from isac_sim.sensing.fbl.dispersion import channel_dispersion, packet_bits
from isac_sim.sensing.fbl.gaussian import _qfunc, _qinv


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
