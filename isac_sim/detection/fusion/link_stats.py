"""link_stats（自 ``isac_sim/detection/fusion.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.soft_channel import (
    local_moments,
    received_full_llr_moments,
    received_h0_third_central,
    received_moments,
)


def effective_h1_mean_for_link(
    cfg: Config, tables, link: Link, q: int, plan: "object | None" = None
) -> float:
    """Communication-error-calibrated H1 mean of the soft statistic.

    The soft statistic ``s_{ijq}`` is produced at the *receiving* UAV ``j``
    and reported to the destination (the sensing initiator ``i`` in the legacy
    architecture, the fusion UAV ``f_q`` in the explicit one), so the reporting
    reliability is the ``j -> dest`` communication quality.
    """
    if (not cfg.detect.enable_comm_error_pollution) or (not cfg.selector.use_comm_error_calibration):
        return local_moments(cfg, tables, link, q).gap
    return received_moments(cfg, tables, link, q, plan).gap


def _base_std_for_link(cfg: Config, tables, link: Link, q: int) -> float:
    """H0 standard deviation of the soft statistic (before comm-error mix).

    In the Gaussian model this is the pair-level ``sigma0[i, j]``; in the LLR
    model it is ``sqrt(var0_q[i, j, q])`` (SINR-derived).  The two coincide
    numerically in the Gaussian case because ``var0_q`` is filled with the
    exact ``sigma0^2`` broadcast.
    """
    i, j = link
    if cfg.detect.soft_stat_model.lower() == "llr":
        return float(np.sqrt(max(tables.var0_q[i, j, q], 0.0)))
    return float(tables.sigma0[i, j])


def h0_variance_for_link(
    cfg: Config, tables, link: Link, q: int | None = None, plan: "object | None" = None
) -> float:
    """H0 variance of the soft statistic, inflated by failed packets."""
    qq = q if q is not None else 0
    return received_moments(cfg, tables, link, qq, plan).v0


def deflection_variance_for_link(
    cfg: Config, tables, link: Link, q: int, plan: "object | None" = None
) -> float:
    """H0 variance used by the ordinary deflection ``(m1-m0)^2/v0``."""
    if (not cfg.detect.enable_comm_error_pollution) or (not cfg.selector.use_comm_error_calibration):
        return local_moments(cfg, tables, link, q).v0
    return received_moments(cfg, tables, link, q, plan).v0


def _per_link_arrays(
    cfg: Config, tables, q: int, links: List[Link], plan: "object | None"
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.array(
        [effective_h1_mean_for_link(cfg, tables, link, q, plan) for link in links],
        dtype=float,
    )
    sigma = np.array(
        [np.sqrt(max(deflection_variance_for_link(cfg, tables, link, q, plan), 0.0)) for link in links],
        dtype=float,
    )
    return delta, sigma
