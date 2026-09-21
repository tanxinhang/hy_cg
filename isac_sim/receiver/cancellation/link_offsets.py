"""cancellation 积木：物理量到延迟-多普勒偏移的换算。"""

from __future__ import annotations

from isac_sim.core.config import Config
import numpy as np
from typing import Tuple

def wavelength(cfg: Config) -> float:
    return cfg.waveform.c / cfg.waveform.fc


def dd_offset_from_physical(cfg: Config, tau_s: float, nu_hz: float) -> Tuple[float, float]:
    """Physical ``(tau, nu)`` -> fractional ``(doppler_bin, delay_bin)``.

    Identical to the conversion used when ``model.build_base_gains`` fills
    ``doppler_bin`` / ``delay_bin`` for the target echoes, so the sample-level
    observation and the link tables agree bin for bin.
    """
    w = cfg.waveform
    return nu_hz * w.N * w.T, tau_s * w.L * w.delta_f


def direct_link_offset(
    cfg: Config, geom, i: int, j: int
) -> Tuple[float, float]:
    """Fractional DD offset of the UAV ``i`` -> UAV ``j`` direct path.

    Delay is the geometric range over ``c``; Doppler is the *bistatic* radial
    rate ``(v_i - v_j) . u_ij`` over the wavelength.  The target-echo convention
    in ``build_base_gains`` is the same one with the target substituted, so a
    target that happens to sit on the baseline is indistinguishable from a
    direct path -- which is exactly the near-far problem this module is about.
    """
    w = cfg.waveform
    vec = np.asarray(geom.p_uav[j], dtype=float) - np.asarray(geom.p_uav[i], dtype=float)
    dist = max(float(np.linalg.norm(vec)), 1.0)
    unit = vec / dist
    tau = dist / w.c
    nu = float(np.dot(np.asarray(geom.v_uav[i]) - np.asarray(geom.v_uav[j]), unit)) / wavelength(cfg)
    return dd_offset_from_physical(cfg, tau, nu)


def target_link_offset(cfg: Config, geom, i: int, j: int, q: int) -> Tuple[float, float]:
    """Fractional DD offset of the bistatic ``i -> q -> j`` echo path."""
    w = cfg.waveform
    p_i, p_j, p_q = geom.p_uav[i], geom.p_uav[j], geom.p_tgt[q]
    v_i, v_j, v_q = geom.v_uav[i], geom.v_uav[j], geom.v_tgt[q]
    vec_iq, vec_jq = p_q - p_i, p_q - p_j
    d_iq = max(float(np.linalg.norm(vec_iq)), 1.0)
    d_jq = max(float(np.linalg.norm(vec_jq)), 1.0)
    tau = (d_iq + d_jq) / w.c
    rr_tx = float(np.dot(v_q - v_i, vec_iq / d_iq))
    rr_rx = float(np.dot(v_q - v_j, vec_jq / d_jq))
    return dd_offset_from_physical(cfg, tau, (rr_tx + rr_rx) / wavelength(cfg))
