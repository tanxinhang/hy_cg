"""Soft-information fusion: weights, deflection and the target priority alpha_q.

The deflection of a fused statistic is

    D_q = (sum_k w_k * mu_eff_k)^2 / (sum_k w_k^2 * sigma_k^2)

with weights ``w_k`` proportional to ``mu_eff_k / sigma_k^2``.  ``mu_eff`` and
``sigma_k`` fold in the communication-error calibration, which is what the
``w/o comm. calib.`` ablation switches off.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import Config, Link
from .model import EPS, d_pd_d_D, pd_from_deflection


# ==========================================================================
# Per-link effective first / second order statistics
# ==========================================================================
def effective_h1_mean_for_link(cfg: Config, tables, link: Link, q: int) -> float:
    """Communication-error-calibrated H1 mean of the soft statistic.

    The soft statistic ``s_{ijq}`` is produced at the *receiving* UAV ``j``.
    It is then reported back to the *transmitting* UAV ``i`` (the sensing
    initiator that fuses the information for target ``q``), so the reporting
    reliability is the ``j -> i`` communication quality ``chi_comm[j, i]``
    (NOT ``chi_comm[i, j]``).
    """
    i, j = link
    mu = float(tables.mu_soft[i, j, q])

    if (not cfg.detect.enable_comm_error_pollution) or (not cfg.selector.use_comm_error_calibration):
        return mu

    # chi_comm is already clipped to [0, 1] when the tables are built.
    chi = float(tables.chi_comm[j, i])
    model = cfg.detect.comm_error_model
    if model == "erasure":
        return chi * mu
    if model == "flip":
        return (chi - (1.0 - chi) * cfg.detect.soft_error_flip_scale) * mu
    if model == "biased":
        return (chi + (1.0 - chi) * cfg.detect.soft_error_bias_scale) * mu
    raise ValueError(model)


def h0_variance_for_link(cfg: Config, tables, link: Link) -> float:
    """H0 variance of the soft statistic, inflated by failed packets.

    Uses the ``j -> i`` reporting reliability (see
    :func:`effective_h1_mean_for_link`).
    """
    i, j = link
    sigma0 = float(tables.sigma0[i, j])

    if not cfg.detect.enable_comm_error_pollution:
        return sigma0 ** 2

    # chi_comm is already clipped to [0, 1] when the tables are built.
    chi = float(tables.chi_comm[j, i])
    sigma_err = cfg.detect.soft_error_sigma_scale * sigma0
    return chi * sigma0 ** 2 + (1.0 - chi) * sigma_err ** 2


def deflection_variance_for_link(cfg: Config, tables, link: Link, q: int) -> float:
    """Full effective variance of the soft statistic used by the deflection.

    Models the communication error as a Bernoulli drop-out mixture
    ``s_eff = B * s + (1 - B) * e`` with ``B ~ Bern(chi)``.  The total
    variance is, by the law of total variance,

        Var(s_eff) = E[Var(s_eff | B)] + Var(E[s_eff | B])
                   = chi*sigma^2 + (1-chi)*sigma_err^2   (within-group)
                   + chi*(1-chi)*mu^2                    (between-group)

    The first term is :func:`h0_variance_for_link`; the second (between-group)
    term ``chi*(1-chi)*mu^2`` was previously dropped and is added here.

    ``q`` is required because the between-group term depends on the H1 mean
    ``mu_soft[i, j, q]``.

    Detection still uses :func:`h0_variance_for_link` (the standard
    detection-threshold denominator).  For the ``w/o comm. calib.`` ablation
    this function intentionally ignores the communication-error inflation
    while the Monte-Carlo detector remains polluted.
    """
    i, j = link
    sigma0 = float(tables.sigma0[i, j])
    if (not cfg.detect.enable_comm_error_pollution) or (not cfg.selector.use_comm_error_calibration):
        return sigma0 ** 2

    var_within = h0_variance_for_link(cfg, tables, link)
    # chi_comm is already clipped to [0, 1] when the tables are built.
    chi = float(tables.chi_comm[j, i])
    mu = float(tables.mu_soft[i, j, q])
    var_between = chi * (1.0 - chi) * mu ** 2
    return var_within + var_between


# ==========================================================================
# Fusion weights and deflection
# ==========================================================================
def compute_weights(cfg: Config, tables, q: int, links: List[Link], mode: str = "beta") -> Dict[Link, float]:
    if not links:
        return {}

    if mode == "equal":
        return {link: 1.0 / len(links) for link in links}

    vals: List[float] = []
    for link in links:
        i, j = link
        if mode == "deflection":
            mu_eff = effective_h1_mean_for_link(cfg, tables, link, q)
            vals.append(max(mu_eff, 0.0) / (deflection_variance_for_link(cfg, tables, link, q) + EPS))
        else:
            vals.append(max(float(tables.beta[i, j, q]), 0.0))

    total = float(np.sum(vals))
    if total <= EPS:
        return {link: 1.0 / len(links) for link in links}
    return {link: v / total for link, v in zip(links, vals)}


def fusion_weight_mode_for_method(method: str) -> str:
    """Self-consistent fusion weights for each method.

    ``raw_sense_sinr`` is a pure sensing-only baseline and keeps equal weights.
    Every other method uses deflection weights proportional to
    ``mu_eff / sigma0^2``, so that the beta-based *selection* utility is never
    mixed into the *effective-H1 deflection* estimate.
    """
    return "equal" if method == "raw_sense_sinr" else "deflection"


def deflection_for_links(cfg: Config, tables, q: int, links: List[Link], weight_mode: str = "deflection") -> float:
    if not links:
        return 0.0

    weights = compute_weights(cfg, tables, q, links, mode=weight_mode)
    mean_gap = 0.0
    var0 = 0.0
    for link, w in weights.items():
        mean_gap += w * effective_h1_mean_for_link(cfg, tables, link, q)
        var0 += (w ** 2) * deflection_variance_for_link(cfg, tables, link, q)

    return float((max(mean_gap, 0.0) ** 2) / (var0 + EPS))


# ==========================================================================
# Target priority
# ==========================================================================
def target_alpha(cfg: Config, D_fuse: np.ndarray) -> np.ndarray:
    r"""Marginal value of improving each target.

    ``alpha_q = dR/dD_q + mu * normalized_deficit``.  With
    ``use_softmin_alpha`` the derivative term is weighted by a soft-min over the
    predicted ``P_D`` so that weak targets receive larger priority.
    """
    s = cfg.selector
    D = np.asarray(D_fuse, dtype=float)
    if not s.use_target_priority:
        return np.ones_like(D, dtype=float)

    pd = pd_from_deflection(cfg, D)
    dpd = d_pd_d_D(cfg, D)

    if s.use_softmin_alpha:
        tau = max(s.softmin_tau, EPS)
        logits = -(pd - np.min(pd)) / tau
        logits = logits - np.max(logits)  # stable softmax; max logit becomes 0
        w = np.exp(logits)
        w = w / max(float(np.sum(w)), EPS)
        # w sums to one; multiplying by Q keeps the marginal scale comparable to
        # a sum-P_D objective instead of shrinking alpha by roughly 1/Q.
        marginal = w * dpd * cfg.scale.Q
    else:
        marginal = dpd

    deficit = np.maximum(cfg.detect.D_min - D, 0.0) / max(cfg.detect.D_min, EPS)
    alpha = marginal + s.mu_deficit * deficit
    return np.clip(alpha, s.alpha_floor, s.alpha_cap)
