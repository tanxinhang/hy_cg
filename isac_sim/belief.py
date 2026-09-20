"""Truth vs belief: the predict -> schedule -> sense -> update loop.

The reviewers' third recurring objection is that the scheduler *already knows*
the target position, velocity and RCS when it computes delay, Doppler, bin and
sensing gain -- yet detection has not happened yet.  The old ``prior-sweep``
only *perturbed the world*: it wrote the perturbed state back into the geometry
so the scheduler and the detector shared the same (wrong) state.  That answers
"how does P_D degrade if the tracker is wrong", not the reviewers' actual
question.

This module splits the two states:

* **truth** ``x_{q,t} = (p_{q,t}, v_{q,t})`` drives echo generation, the true
  delay-Doppler, the true sensing gain and the detector.
* **belief** ``b_{q,t} = N(xhat_{q,t}, P_{q,t})`` drives link selection, the DD
  search window and the resource budget.

The scheduler may only *see* the belief; the physical world runs on the truth.
A selected link only delivers target evidence when the belief-guided search
window actually captures the true delay-Doppler bin -- which is exactly the
cost of a mismatched prior, and exactly the effect a reviewer expects to see
when ``xhat != x``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, List

import numpy as np

from .config import Config, Link
from .model import BaseGains, Geometry
from .prior import perturbed_geometry


@dataclass
class BeliefState:
    """Gaussian belief over the target states.

    ``xhat`` is ``(Q, 6)`` (position + velocity per target) and ``P`` is
    ``(Q, 6, 6)``.  The constant-velocity transition ``F`` and process noise
    ``Q_cv`` are provided so a downstream tracker update can close the loop.
    """

    xhat: np.ndarray
    P: np.ndarray

    @classmethod
    def from_truth(cls, cfg: Config, geom: Geometry, rng: np.random.Generator) -> "BeliefState":
        """Build a belief as the truth corrupted by ``cfg.prior.belief_*`` noise.

        This is the minimal *tracker-free* model: the belief is a single noisy
        snapshot of the truth, with an isotropic covariance of the configured
        amplitude.  It is enough to expose the belief-mismatch cost without
        committing to a specific filter.
        """
        Q = cfg.scale.Q
        xhat = np.zeros((Q, 6))
        xhat[:, 0:3] = geom.p_tgt
        xhat[:, 3:6] = geom.v_tgt

        s_pos = max(cfg.prior.belief_sigma_pos_m, 0.0)
        s_vel = max(cfg.prior.belief_sigma_vel_mps, 0.0)
        P = np.zeros((Q, 6, 6))
        # The scenario generator constrains targets to nominal altitude and
        # zero vertical velocity (see the assignments below).  Their stated
        # covariance must encode the same four-dimensional horizontal model;
        # assigning variance to z/v_z would make capture/risk propagation price
        # errors that the belief draw can never realise.
        P[:, 0, 0] = P[:, 1, 1] = s_pos ** 2
        P[:, 3, 3] = P[:, 4, 4] = s_vel ** 2

        if s_pos > 0:
            xhat[:, 0:3] += rng.normal(0.0, s_pos, size=(Q, 3))
            xhat[:, 2] = geom.p_tgt[:, 2]  # keep nominal altitude
        if s_vel > 0:
            xhat[:, 3:6] += rng.normal(0.0, s_vel, size=(Q, 3))
            xhat[:, 5] = 0.0
        return cls(xhat=xhat, P=P)

    def as_geometry(self, geom: Geometry) -> Geometry:
        """A :class:`Geometry` whose targets are the belief means."""
        return Geometry(
            p_uav=geom.p_uav.copy(),
            v_uav=geom.v_uav.copy(),
            p_tgt=self.xhat[:, 0:3].copy(),
            v_tgt=self.xhat[:, 3:6].copy(),
        )


def belief_geometry(cfg: Config, geom_truth: Geometry, rng: np.random.Generator) -> Geometry:
    """The scheduler's view of the scene: target state = belief mean."""
    belief = BeliefState.from_truth(cfg, geom_truth, rng)
    return belief.as_geometry(geom_truth)


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


def belief_capture_sigma_points(
    cfg: Config,
    geom_belief: Geometry,
    belief: BeliefState,
) -> np.ndarray:
    """Joint DD-gate capture over correlated horizontal-state sigma points.

    The output has shape ``(137,M,M,Q)``: the belief centre, the eight axis
    points, and 64 deterministic antipodal direction pairs on the 95%
    four-dimensional Gaussian boundary in ``x,y,vx,vy``.  Each point is passed
    through the exact nonlinear bistatic delay/Doppler map; only the gate width
    uses the usual covariance Jacobian.  A state perturbation is shared by
    every bistatic link of the target, so this certificate retains the
    dependence that marginal per-link probabilities discard.  It is a
    discrete boundary-coverage diagnostic, not 137 independent probabilities
    or a proof of continuous ellipsoid coverage.
    """
    m, q_count = int(cfg.scale.M), int(cfg.scale.Q)
    radius95_4d = 3.080215745168048
    axes = (0, 1, 3, 4)
    rng = np.random.default_rng(0x43415054)
    random_dirs = rng.normal(size=(64, 4))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    directions = np.concatenate([
        np.zeros((1, 4), dtype=float),
        np.eye(4), -np.eye(4),
        random_dirs, -random_dirs,
    ], axis=0)
    out = np.ones((directions.shape[0], m, m, q_count), dtype=bool)
    delay_scale = float(cfg.waveform.L * cfg.waveform.delta_f)
    doppler_scale = float(cfg.waveform.N * cfg.waveform.T)
    lam = float(cfg.waveform.c / cfg.waveform.fc)
    eye3 = np.eye(3)
    for q in range(q_count):
        points = np.zeros((directions.shape[0], 6), dtype=float)
        scales = np.asarray([
            math.sqrt(max(float(belief.P[q, dim, dim]), 0.0))
            for dim in axes
        ])
        points[:, list(axes)] = radius95_4d * directions * scales[None, :]
        p_points = geom_belief.p_tgt[q][None, :] + points[:, :3]
        v_points = geom_belief.v_tgt[q][None, :] + points[:, 3:]
        for i in range(m):
            for j in range(m):
                if i == j:
                    out[:, i, j, q] = False
                    continue
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
                p_q = belief.P[q]
                std_l = delay_scale * math.sqrt(
                    max(float(g_tau @ p_q @ g_tau), 0.0)
                )
                std_k = doppler_scale * math.sqrt(
                    max(float(g_nu @ p_q @ g_nu), 0.0)
                )
                gate_l = 0.5 + float(cfg.prior.search_gate_sigma) * std_l
                gate_k = 0.5 + float(cfg.prior.search_gate_sigma) * std_k
                # Do not linearise the certificate itself.  At the canonical
                # 150 m position uncertainty, a target can move through a
                # near-field angular sector where the Doppler Jacobian at the
                # belief mean severely understates the actual mismatch.
                ri_points = p_points - geom_belief.p_uav[i][None, :]
                rj_points = p_points - geom_belief.p_uav[j][None, :]
                di_points = np.maximum(
                    np.linalg.norm(ri_points, axis=1), 1.0
                )
                dj_points = np.maximum(
                    np.linalg.norm(rj_points, axis=1), 1.0
                )
                ui_points = ri_points / di_points[:, None]
                uj_points = rj_points / dj_points[:, None]
                tau0 = (di + dj) / cfg.waveform.c
                tau_points = (di_points + dj_points) / cfg.waveform.c
                nu0 = (
                    float(np.dot(vi, ui)) + float(np.dot(vj, uj))
                ) / lam
                nu_points = (
                    np.sum(
                        (v_points - geom_belief.v_uav[i][None, :])
                        * ui_points,
                        axis=1,
                    )
                    + np.sum(
                        (v_points - geom_belief.v_uav[j][None, :])
                        * uj_points,
                        axis=1,
                    )
                ) / lam
                dl = np.abs(tau_points - tau0) * delay_scale
                dk = np.abs(nu_points - nu0) * doppler_scale
                out[:, i, j, q] = (dl <= gate_l) & (dk <= gate_k)
    return out


def truth_captured_links(
    cfg: Config,
    base_truth: BaseGains,
    base_belief: BaseGains,
    selected: Dict[int, List[Link]],
    dd_std_bins: tuple[np.ndarray, np.ndarray] | None = None,
) -> Dict[int, List[Link]]:
    """Restrict ``selected`` to links whose belief-guided search captures the truth.

    A selected sensing link yields target evidence when the true continuous DD
    coordinate falls inside the covariance-derived search gate around the
    predicted coordinate.  Half a bin accounts for quantisation; the remaining
    width is ``prior.search_gate_sigma`` times the propagated standard deviation.
    """
    out: Dict[int, List[Link]] = {}
    for q, links in selected.items():
        kept: List[Link] = []
        for (i, j) in links:
            if cfg.dd.use_otfs_bin_validity and not base_truth.valid_dd[i, j, q]:
                continue
            if dd_std_bins is None:
                std_l = std_k = 0.0
            else:
                std_l = float(dd_std_bins[0][i, j, q])
                std_k = float(dd_std_bins[1][i, j, q])
            dl = abs(base_truth.tau[i, j, q] - base_belief.tau[i, j, q]) \
                * cfg.waveform.L * cfg.waveform.delta_f
            dk = abs(base_truth.doppler[i, j, q] - base_belief.doppler[i, j, q]) \
                * cfg.waveform.N * cfg.waveform.T
            if dl > 0.5 + cfg.prior.search_gate_sigma * std_l:
                continue
            if dk > 0.5 + cfg.prior.search_gate_sigma * std_k:
                continue
            kept.append((i, j))
        out[q] = kept
    return out


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


def geometry_robust_base(cfg: Config, base: BaseGains) -> BaseGains:
    r"""Return a conservative path-gain view over a position confidence ball.

    The belief error is horizontal isotropic Gaussian with standard deviation
    ``sigma``.  Its radius has Rayleigh CDF, hence the configured confidence
    mass corresponds to

    ``r = sigma * sqrt(-2 log(1-confidence))``.

    For every target position within that ball, the triangle inequality gives
    ``d_true <= d_hat + r``.  Since the bistatic target gain is proportional to
    ``d_i^-2 d_j^-2``, multiplying the predicted gain by

    ``[d_i/(d_i+r)]^2 [d_j/(d_j+r)]^2``

    is a valid lower bound on the distance-dependent component.  RCS, DD
    leakage, correlation and communication uncertainty are intentionally not
    claimed by this bound.
    """
    sigma = max(float(cfg.prior.belief_sigma_pos_m), 0.0)
    if not cfg.prior.belief_mode or sigma <= 0.0:
        return base
    confidence = float(cfg.prior.robust_position_confidence)
    radius = sigma * np.sqrt(-2.0 * np.log1p(-confidence))
    distances = np.maximum(np.asarray(base.d_uav_tgt, dtype=float), 1.0)
    one_way = (distances / (distances + radius)) ** 2
    bistatic_factor = one_way[:, None, :] * one_way[None, :, :]
    return replace(base, target_gain=base.target_gain * bistatic_factor)


def predicted_geometry_from_belief(cfg: Config, geom_truth: Geometry, rng: np.random.Generator) -> Geometry:
    """Constant-velocity one-step prediction from the current belief (utility)."""
    from .prior import predicted_geometry

    return predicted_geometry(cfg, geom_truth, rng)
