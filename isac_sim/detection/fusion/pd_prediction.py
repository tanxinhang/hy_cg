"""pd_prediction（自 ``isac_sim/detection/fusion/weights.py`` 拆出）。"""

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
from isac_sim.detection.fusion.h0 import fused_h0_skewness, fused_h0_variance
from isac_sim.detection.fusion.threshold import calibrated_fused_threshold

from isac_sim.detection.fusion.weights import compute_weights


def predicted_pd_for_links(
    cfg: Config,
    tables,
    q: int,
    links: List[Link],
    weight_mode: str = "deflection",
    plan: "object | None" = None,
    base: BaseGains | None = None,
) -> float:
    """Moment-matched detection probability for the implemented detector.

    Unlike :func:`pd_from_deflection`, this prediction retains the unequal H0
    and H1 variances of the finite-look LLR, the post-report packet mixture,
    and the Cornish--Fisher H0 threshold correction.  It remains an analytic
    scheduler surrogate; Monte-Carlo detection is still the release metric.
    """
    if not links:
        return 0.0

    weights = compute_weights(cfg, tables, q, links, mode=weight_mode, plan=plan, base=base)
    moments = [
        received_full_llr_moments(cfg, tables, link, q, plan)
        if weight_mode == "exact_llr_sum"
        else received_moments(cfg, tables, link, q, plan)
        for link in links
    ]
    mean1 = float(sum(weights[link] * mom.m1 for link, mom in zip(links, moments)))
    if weight_mode == "exact_llr_sum":
        var0 = float(sum((weights[link] ** 2) * mom.v0 for link, mom in zip(links, moments)))
    else:
        var0 = fused_h0_variance(cfg, tables, q, links, weights, plan=plan, base=base)

    if cfg.corr.enable and base is not None and len(links) > 1:
        from isac_sim.detection.corr import covariance_matrix

        sigma1 = np.array([np.sqrt(max(mom.v1, 0.0)) for mom in moments], dtype=float)
        Sigma1 = covariance_matrix(cfg, links, sigma1, base=base, q=q)
        w = np.array([weights[link] for link in links], dtype=float)
        var1 = float(w @ Sigma1 @ w)
    else:
        var1 = float(sum(
            (weights[link] ** 2) * mom.v1 for link, mom in zip(links, moments)
        ))

    if (
        weight_mode == "exact_llr_sum"
        or cfg.detect.exact_gaussian_replacement_threshold
    ):
        threshold = calibrated_fused_threshold(
            cfg, tables, q, links, weights, plan=plan, base=base,
            statistic_mode=(
                "exact_llr_sum" if weight_mode == "exact_llr_sum" else weight_mode
            ),
        )
    else:
        skew0 = fused_h0_skewness(cfg, tables, q, links, weights, plan=plan, base=base)
        z0 = threshold_from_pfa(cfg)
        z_cf = z0 + (skew0 / 6.0) * (z0 * z0 - 1.0)
        threshold = z_cf * np.sqrt(max(var0, EPS))
    return float(qfunc((threshold - mean1) / np.sqrt(max(var1, EPS))))
