"""Soft-information fusion: weights, deflection and the target priority alpha_q.

The deflection of a fused statistic is

    D_q = (sum_k w_k * mu_eff_k)^2 / (sum_k w_k^2 * sigma_k^2)

with weights ``w_k`` proportional to ``mu_eff_k / sigma_k^2``.  ``mu_eff`` and
``sigma_k`` fold in the communication-error calibration, which is what the
``w/o comm. calib.`` ablation switches off.

Two upgrades are layered on top, both optional and both default-off so the
frozen results stay bit-exact:

1. **Local-LLR soft statistic** (``detect.soft_stat_model = "llr"``): ``mu`` is
   the centred LLR mean ``L*gamma^2/(1+gamma)`` and the H0 variance is the
   derived ``L*gamma^2/(1+gamma)^2`` (via ``tables.var0_q``).  No free scale.
2. **Correlation-aware fusion** (``corr.enable``): the fused deflection becomes
   ``delta^T Sigma^{-1} delta`` with the weights ``w propto Sigma^{-1} delta``,
   so correlated / redundant UAV observations are automatically down-weighted.

The reporting destination (fusion UAV ``f_q``) is resolved through the
``plan`` argument; ``plan=None`` keeps the legacy ``j -> i`` direction.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import Config, Link
from .model import (
    EPS,
    BaseGains,
    d_pd_d_D,
    pd_from_deflection,
    qfunc,
    threshold_from_pfa,
)
from .reporting import report_dest
from .soft_channel import local_moments, received_h0_third_central, received_moments


# ==========================================================================
# Per-link effective first / second order statistics
# ==========================================================================
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


# ==========================================================================
# Fusion weights and deflection
# ==========================================================================
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


def compute_weights(
    cfg: Config,
    tables,
    q: int,
    links: List[Link],
    mode: str = "deflection",
    plan: "object | None" = None,
    base: BaseGains | None = None,
) -> Dict[Link, float]:
    if not links:
        return {}

    if mode == "equal":
        return {link: 1.0 / len(links) for link in links}

    if mode == "deflection" and cfg.corr.enable and base is not None and len(links) > 1:
        from .corr import correlation_aware_weights

        delta, sigma = _per_link_arrays(cfg, tables, q, links, plan)
        w = correlation_aware_weights(cfg, links, delta, sigma, base=base, q=q)
        return {link: float(w[k]) for k, link in enumerate(links)}

    vals: List[float] = []
    for link in links:
        if mode == "deflection":
            mu_eff = effective_h1_mean_for_link(cfg, tables, link, q, plan)
            vals.append(max(mu_eff, 0.0) / (deflection_variance_for_link(cfg, tables, link, q, plan) + EPS))
        else:
            raise ValueError(mode)

    total = float(np.sum(vals))
    if total <= EPS:
        return {link: 1.0 / len(links) for link in links}
    return {link: v / total for link, v in zip(links, vals)}


def fusion_weight_mode_for_method(method: str) -> str:
    """Self-consistent fusion weights for each method."""
    return "equal" if method == "raw_sense_sinr" else "deflection"


def deflection_for_links(
    cfg: Config,
    tables,
    q: int,
    links: List[Link],
    weight_mode: str = "deflection",
    plan: "object | None" = None,
    base: BaseGains | None = None,
) -> float:
    if not links:
        return 0.0

    weights = compute_weights(cfg, tables, q, links, mode=weight_mode, plan=plan, base=base)

    if weight_mode == "deflection" and cfg.corr.enable and base is not None and len(links) > 1:
        from .corr import correlated_deflection, correlation_aware_weights

        delta, sigma = _per_link_arrays(cfg, tables, q, links, plan)
        w = np.array([weights[link] for link in links], dtype=float)
        return correlated_deflection(cfg, links, delta, sigma, w, base=base, q=q)

    mean_gap = 0.0
    var0 = 0.0
    for link, w in weights.items():
        mean_gap += w * effective_h1_mean_for_link(cfg, tables, link, q, plan)
        var0 += (w ** 2) * deflection_variance_for_link(cfg, tables, link, q, plan)

    return float((max(mean_gap, 0.0) ** 2) / (var0 + EPS))


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
    moments = [received_moments(cfg, tables, link, q, plan) for link in links]
    mean1 = float(sum(weights[link] * mom.m1 for link, mom in zip(links, moments)))
    var0 = fused_h0_variance(cfg, tables, q, links, weights, plan=plan, base=base)

    if cfg.corr.enable and base is not None and len(links) > 1:
        from .corr import covariance_matrix

        sigma1 = np.array([np.sqrt(max(mom.v1, 0.0)) for mom in moments], dtype=float)
        Sigma1 = covariance_matrix(cfg, links, sigma1, base=base, q=q)
        w = np.array([weights[link] for link in links], dtype=float)
        var1 = float(w @ Sigma1 @ w)
    else:
        var1 = float(sum(
            (weights[link] ** 2) * mom.v1 for link, mom in zip(links, moments)
        ))

    skew0 = fused_h0_skewness(cfg, tables, q, links, weights, plan=plan, base=base)
    z0 = threshold_from_pfa(cfg)
    z_cf = z0 + (skew0 / 6.0) * (z0 * z0 - 1.0)
    threshold = z_cf * np.sqrt(max(var0, EPS))
    return float(qfunc((threshold - mean1) / np.sqrt(max(var1, EPS))))


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
        from .corr import covariance_matrix

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
    if cfg.corr.enable or not links:
        return 0.0
    var0 = fused_h0_variance(cfg, tables, q, links, weights, plan, base)
    if var0 <= EPS:
        return 0.0
    mu3 = sum(
        (w ** 3) * received_h0_third_central(cfg, tables, link, q, plan)
        for link, w in weights.items()
    )
    return float(mu3 / (var0 ** 1.5))


# ==========================================================================
# Target priority
# ==========================================================================
def selection_utility_from_pd(
    cfg: Config, D_fuse: np.ndarray, predicted_pd: np.ndarray
) -> float:
    """Evaluate the canonical fair potential for supplied target-level PDs."""
    s = cfg.selector
    D = np.asarray(D_fuse, dtype=float)
    if not s.use_target_priority:
        return float(np.sum(D))

    pd = np.asarray(predicted_pd, dtype=float)
    pd_required = max(float(cfg.detect.pd_required), EPS)
    # Detection utility is intentionally saturated at the declared operating
    # requirement. This prevents zero-communication local observations from
    # being accumulated merely for numerically tiny gains after every target
    # already meets the design point.
    pd_effective = np.minimum(pd, pd_required)
    if s.use_softmin_alpha:
        tau = max(float(s.softmin_tau), EPS)
        z = -pd_effective / tau
        zmax = float(np.max(z)) if z.size else 0.0
        sensing = -float(cfg.scale.Q) * tau * (
            zmax + np.log(max(float(np.sum(np.exp(z - zmax))), EPS))
        )
    else:
        sensing = float(np.sum(pd_effective))

    deficit = np.maximum(pd_required - pd, 0.0)
    penalty = (
        float(s.mu_deficit) * float(np.sum(deficit * deficit))
        / (2.0 * pd_required)
    )
    return sensing - penalty


def selection_utility(cfg: Config, D_fuse: np.ndarray) -> float:
    r"""Fair sensing utility whose gradient is the unclipped ``alpha_q``.

    With soft-min target prioritisation enabled,

    ``U(D) = -Q*tau*log sum_q exp(-P_D,q/tau)
             - mu/(2*P_D_req) * sum_q [P_D_req-P_D,q]_+^2``.

    The first term emphasizes weak targets and the second penalizes detection
    deficits.  Disabling soft-min replaces its first term by ``sum_q P_D,q``;
    disabling target priority altogether yields ``sum_q D_q``.
    """
    D = np.asarray(D_fuse, dtype=float)
    pd = np.asarray(pd_from_deflection(cfg, D), dtype=float)
    return selection_utility_from_pd(cfg, D, pd)


def target_alpha(cfg: Config, D_fuse: np.ndarray) -> np.ndarray:
    r"""Marginal value of improving each target.

    ``alpha_q = dU/dD_q`` for :func:`selection_utility` before numerical
    clipping.  With
    ``use_softmin_alpha`` the derivative term is weighted by a soft-min over the
    predicted ``P_D`` so that weak targets receive larger priority.
    """
    s = cfg.selector
    D = np.asarray(D_fuse, dtype=float)
    if not s.use_target_priority:
        return np.ones_like(D, dtype=float)

    pd = pd_from_deflection(cfg, D)
    dpd = d_pd_d_D(cfg, D)
    below_requirement = pd < float(cfg.detect.pd_required)

    if s.use_softmin_alpha:
        tau = max(s.softmin_tau, EPS)
        logits = -(pd - np.min(pd)) / tau
        logits = logits - np.max(logits)  # stable softmax; max logit becomes 0
        w = np.exp(logits)
        w = w / max(float(np.sum(w)), EPS)
        marginal = w * dpd * cfg.scale.Q * below_requirement
    else:
        marginal = dpd * below_requirement

    pd_required = max(float(cfg.detect.pd_required), EPS)
    deficit = np.maximum(pd_required - pd, 0.0) / pd_required
    alpha = marginal + s.mu_deficit * deficit * dpd
    return np.clip(alpha, s.alpha_floor, s.alpha_cap)
