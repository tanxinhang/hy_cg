"""sigma（自 ``isac_sim/scenario/belief.py`` 拆出）。"""

from __future__ import annotations

import math
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains, Geometry

from isac_sim.scenario.belief.state import BeliefState


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
