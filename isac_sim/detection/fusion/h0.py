"""h0（自 ``isac_sim/detection/fusion.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import (
    EPS,
    BaseGains,
    d_pd_d_D,
    pd_from_deflection,
    qfunc,
    threshold_from_pfa,
)
from isac_sim.sensing.soft_channel import (
    local_moments,
    received_full_llr_moments,
    received_h0_third_central,
    received_moments,
)

from isac_sim.detection.fusion.link_stats import h0_variance_for_link


def fused_h0_variance(
    cfg: Config,
    tables,
    q: int,
    links: List[Link],
    weights: Dict[Link, float],
    plan: "object | None" = None,
    base: BaseGains | None = None,
) -> float:
    """H0 variance of the fused statistic, honouring observation correlation.

    Used as the detection-threshold denominator.  Without the correlation model
    this reduces to ``sum_k w_k^2 * sigma_k^2`` (the legacy threshold); with it
    the denominator is ``w^T Sigma_H0 w``.
    """
    if cfg.corr.enable and base is not None and len(links) > 1:
        from isac_sim.detection.corr import covariance_matrix

        sigma = np.array(
            [np.sqrt(max(h0_variance_for_link(cfg, tables, link, q, plan), 0.0)) for link in links],
            dtype=float,
        )
        Sigma = covariance_matrix(cfg, links, sigma, base=base, q=q)
        w = np.array([weights[link] for link in links], dtype=float)
        return float(w @ Sigma @ w)

    return float(sum((w ** 2) * h0_variance_for_link(cfg, tables, link, q, plan) for link, w in weights.items()))


def fused_h0_skewness(
    cfg: Config,
    tables,
    q: int,
    links: List[Link],
    weights: Dict[Link, float],
    plan: "object | None" = None,
    base: BaseGains | None = None,
) -> float:
    """Independent-link H0 skewness for Cornish--Fisher threshold calibration."""
    # The correlated sampler is moment-matched Gaussian only when at least two
    # observations are sampled jointly.  A singleton still uses the exact
    # finite-look LLR marginal and therefore retains its Gamma skewness.
    if (cfg.corr.enable and len(links) > 1) or not links:
        return 0.0
    var0 = fused_h0_variance(cfg, tables, q, links, weights, plan, base)
    if var0 <= EPS:
        return 0.0
    mu3 = sum(
        (w ** 3) * received_h0_third_central(cfg, tables, link, q, plan)
        for link, w in weights.items()
    )
    return float(mu3 / (var0 ** 1.5))
