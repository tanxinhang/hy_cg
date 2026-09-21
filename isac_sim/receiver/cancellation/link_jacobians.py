"""cancellation 积木：目标链路雅可比与标准差。"""

from __future__ import annotations

from isac_sim.core.config import Config
import math
import numpy as np
from typing import Tuple

from isac_sim.receiver.cancellation.link_offsets import wavelength

def target_link_jacobians(
    cfg: Config, geom_belief, i: int, j: int, q: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Doppler/delay-bin Jacobians with respect to ``(p,v)``.

    The ordering is the six-state convention ``(x,y,z,vx,vy,vz)``.
    """
    p = np.asarray(geom_belief.p_tgt[q], dtype=float)
    v = np.asarray(geom_belief.v_tgt[q], dtype=float)
    ri = p - np.asarray(geom_belief.p_uav[i], dtype=float)
    rj = p - np.asarray(geom_belief.p_uav[j], dtype=float)
    di = max(float(np.linalg.norm(ri)), 1.0)
    dj = max(float(np.linalg.norm(rj)), 1.0)
    ui, uj = ri / di, rj / dj
    vi = v - np.asarray(geom_belief.v_uav[i], dtype=float)
    vj = v - np.asarray(geom_belief.v_uav[j], dtype=float)

    g_tau = np.zeros(6, dtype=float)
    g_tau[:3] = (ui + uj) / float(cfg.waveform.c)
    eye3 = np.eye(3)
    g_nu = np.zeros(6, dtype=float)
    g_nu[:3] = (
        ((eye3 - np.outer(ui, ui)) @ vi) / di
        + ((eye3 - np.outer(uj, uj)) @ vj) / dj
    ) / wavelength(cfg)
    g_nu[3:] = (ui + uj) / wavelength(cfg)

    g_l = g_tau * float(cfg.waveform.L * cfg.waveform.delta_f)
    g_k = g_nu * float(cfg.waveform.N * cfg.waveform.T)
    return np.asarray(g_k, dtype=float), np.asarray(g_l, dtype=float)


def target_link_std_bins(
    cfg: Config, geom_belief, i: int, j: int, q: int
) -> Tuple[float, float]:
    """Propagate the configured horizontal target covariance to one DD link."""
    g_k, g_l = target_link_jacobians(cfg, geom_belief, i, j, q)
    spos = max(float(cfg.prior.belief_sigma_pos_m), 0.0)
    svel = max(float(cfg.prior.belief_sigma_vel_mps), 0.0)
    diag_p = np.asarray([
        spos * spos, spos * spos, 0.0,
        svel * svel, svel * svel, 0.0,
    ])
    std_l = math.sqrt(max(float(np.sum(g_l * g_l * diag_p)), 0.0))
    std_k = math.sqrt(max(float(np.sum(g_k * g_k * diag_p)), 0.0))
    return float(std_k), float(std_l)


def target_bearing_std(
    cfg: Config, geom_belief, receiver: int, q: int
) -> float:
    """First-order standard deviation of the receive direction cosine.

    For ``u = r_axis / ||r||``, ``grad_r u = (e_axis-u*rhat)/||r||``.
    The configured isotropic position covariance is propagated through that
    gradient.  This is belief-only and therefore available before sensing.
    """
    if not cfg.aperture.enable or int(cfg.aperture.m_rx) <= 1:
        return 0.0
    r = (
        np.asarray(geom_belief.p_tgt[q], dtype=float)
        - np.asarray(geom_belief.p_uav[receiver], dtype=float)
    )
    distance = max(float(np.linalg.norm(r)), 1.0)
    r_hat = r / distance
    axis = int(cfg.aperture.axis)
    e_axis = np.zeros(3, dtype=float)
    e_axis[axis] = 1.0
    u = float(r_hat[axis])
    gradient = (e_axis - u * r_hat) / distance
    sigma_pos = max(float(cfg.prior.belief_sigma_pos_m), 0.0)
    return float(sigma_pos * np.linalg.norm(gradient[:2]))


def target_bearing_jacobian(
    cfg: Config, geom_belief, receiver: int, q: int
) -> np.ndarray:
    """Six-state Jacobian of receive direction cosine ``u``."""
    out = np.zeros(6, dtype=float)
    if not cfg.aperture.enable or int(cfg.aperture.m_rx) <= 1:
        return out
    r = (
        np.asarray(geom_belief.p_tgt[q], dtype=float)
        - np.asarray(geom_belief.p_uav[receiver], dtype=float)
    )
    distance = max(float(np.linalg.norm(r)), 1.0)
    r_hat = r / distance
    axis = int(cfg.aperture.axis)
    e_axis = np.zeros(3, dtype=float)
    e_axis[axis] = 1.0
    out[:3] = (e_axis - float(r_hat[axis]) * r_hat) / distance
    return out
