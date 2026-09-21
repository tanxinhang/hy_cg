"""capture（自 ``isac_sim/scenario/belief.py`` 拆出）。"""

from __future__ import annotations

import math
from typing import Dict, List
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains, Geometry

from isac_sim.scenario.belief.state import BeliefState
from isac_sim.scenario.belief.truth import truth_captured_links


def belief_dd_std_bins(
    cfg: Config,
    geom_belief: Geometry,
    belief: BeliefState,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate ``P`` to bistatic delay/Doppler standard deviations in bins.

    A first-order Jacobian of the bistatic measurement model is used for each
    ``(i,j,q)``.  This makes the DD search support depend on the tracker
    covariance, rather than comparing two rounded bins for exact equality.
    """
    M, Q = cfg.scale.M, cfg.scale.Q
    std_l = np.zeros((M, M, Q), dtype=float)
    std_k = np.zeros((M, M, Q), dtype=float)
    delay_scale = cfg.waveform.L * cfg.waveform.delta_f
    doppler_scale = cfg.waveform.N * cfg.waveform.T
    lam = cfg.waveform.c / cfg.waveform.fc
    eye3 = np.eye(3)

    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            for q in range(Q):
                p = geom_belief.p_tgt[q]
                v = geom_belief.v_tgt[q]
                ri = p - geom_belief.p_uav[i]
                rj = p - geom_belief.p_uav[j]
                di = max(float(np.linalg.norm(ri)), 1.0)
                dj = max(float(np.linalg.norm(rj)), 1.0)
                ui, uj = ri / di, rj / dj

                g_tau = np.zeros(6, dtype=float)
                g_tau[:3] = (ui + uj) / cfg.waveform.c

                vi = v - geom_belief.v_uav[i]
                vj = v - geom_belief.v_uav[j]
                g_nu = np.zeros(6, dtype=float)
                g_nu[:3] = (
                    ((eye3 - np.outer(ui, ui)) @ vi) / di
                    + ((eye3 - np.outer(uj, uj)) @ vj) / dj
                ) / lam
                g_nu[3:] = (ui + uj) / lam

                Pq = belief.P[q]
                var_tau = max(float(g_tau @ Pq @ g_tau), 0.0)
                var_nu = max(float(g_nu @ Pq @ g_nu), 0.0)
                std_l[i, j, q] = delay_scale * np.sqrt(var_tau)
                std_k[i, j, q] = doppler_scale * np.sqrt(var_nu)
    return std_l, std_k


def belief_capture_probability_lower_bound(
    cfg: Config,
    dd_std_bins: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Belief-only lower bound on DD-window capture probability.

    Each marginal DD error is Gaussian under the linearised belief model.  We
    compute its probability of falling in the same gate used by
    :func:`truth_captured_links`, then combine delay and Doppler with the
    Bonferroni bound ``max(0, p_delay + p_doppler - 1)``.  Unlike multiplying
    the two marginals, this remains a valid lower bound without assuming their
    errors are independent.

    The result is an ``(M,M,Q)`` array and depends only on information visible
    to the scheduler; it never inspects the truth state.
    """
    std_l = np.asarray(dd_std_bins[0], dtype=float)
    std_k = np.asarray(dd_std_bins[1], dtype=float)
    if std_l.shape != std_k.shape or std_l.ndim != 3:
        raise ValueError("DD standard-deviation arrays must share shape (M,M,Q)")
    if np.any(std_l < 0.0) or np.any(std_k < 0.0):
        raise ValueError("DD standard deviations must be non-negative")

    def marginal(std: np.ndarray) -> np.ndarray:
        width = 0.5 + float(cfg.prior.search_gate_sigma) * std
        z = np.full(std.shape, np.inf, dtype=float)
        np.divide(width, np.sqrt(2.0) * std, out=z, where=std > 0.0)
        erf = np.vectorize(math.erf, otypes=[float])
        return erf(z)

    p_l = marginal(std_l)
    p_k = marginal(std_k)
    return np.clip(p_l + p_k - 1.0, 0.0, 1.0)


def belief_capture_rate(
    cfg: Config,
    base_truth: BaseGains,
    base_belief: BaseGains,
    selected: Dict[int, List[Link]],
    dd_std_bins: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Fraction of selected links whose belief window captures the true bin."""
    total = 0
    hit = 0
    for q, links in selected.items():
        for (i, j) in links:
            total += 1
            captured = truth_captured_links(
                cfg, base_truth, base_belief, {q: [(i, j)]}, dd_std_bins
            ).get(q, [])
            if captured:
                hit += 1
    return hit / max(total, 1)
