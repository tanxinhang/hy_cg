"""utility（自 ``isac_sim/detection/fusion.py`` 拆出）。"""

from __future__ import annotations

import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.detection.fusion.leximin import leximin_gradient, leximin_potential
from isac_sim.detection.fusion.maxmin import maxmin_support, worst_case_pd
from isac_sim.sensing.model import (
    EPS,
    BaseGains,
    d_pd_d_D,
    pd_from_deflection,
    qfunc,
    threshold_from_pfa,
)


def selection_utility_from_pd(
    cfg: Config, D_fuse: np.ndarray, predicted_pd: np.ndarray
) -> float:
    """Evaluate the canonical fair potential for supplied target-level PDs.

    ``selector.maxmin_objective`` switches the potential to the proposal's
    epigraph form ``Phi = min_q P_D,q`` (see
    :mod:`isac_sim.detection.fusion.maxmin`); the default soft-min/sum form is
    untouched, so the released numbers stay bit-identical.
    """
    s = cfg.selector
    D = np.asarray(D_fuse, dtype=float)
    if not s.use_target_priority:
        return float(np.sum(D))
    pd = np.asarray(predicted_pd, dtype=float)
    if s.maxmin_objective:
        # 贪心用的势是 leximin（字典序细化）；纯 min 的边际恒为 0 会死锁，
        # 详见 isac_sim.detection.fusion.leximin 的模块说明。
        return leximin_potential(cfg, pd, s.leximin_rho)
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

    if s.maxmin_objective:
        # leximin 的梯度：第 k 弱的目标拿到 rho^k（并列均分）。
        return np.clip(
            leximin_gradient(cfg, pd, s.leximin_rho) * dpd, s.alpha_floor, s.alpha_cap
        )

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
