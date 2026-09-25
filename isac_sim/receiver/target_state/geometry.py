"""积木：把一个共享的水平偏移落到信念几何与目标源上。"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from isac_sim.core.config import Config
from isac_sim.receiver.cancellation.link_jacobians import (
    target_bearing_jacobian,
    target_bearing_std,
    target_link_jacobians,
    target_link_std_bins,
)
from isac_sim.receiver.cancellation.link_offsets import target_link_offset
from isac_sim.receiver.cancellation.steering import _u_of

__all__ = ["shift_belief_geometry", "shifted_target_sources"]


def shift_belief_geometry(geom, target: int, delta_xy_m):
    """只把第 ``target`` 个目标的**信念**水平位置平移 ``delta``。

    UAV 位置/速度、其它目标、目标高度与速度一字不改。返回新对象，输入不被
    就地修改。
    """
    delta = np.asarray(delta_xy_m, dtype=float).reshape(2)
    p_tgt = np.array(geom.p_tgt, dtype=float, copy=True)
    p_tgt[int(target), 0] += float(delta[0])
    p_tgt[int(target), 1] += float(delta[1])
    return replace(geom, p_tgt=p_tgt)


def shifted_target_sources(cfg: Config, geom, receiver: int, target: int,
                           sources: Sequence) -> list:
    """按平移后的信念几何重建**该目标**的源，其它目标的源原样返回。

    除了 DD 偏移本身，链路标准差与状态雅可比也一并重算，否则偏移后的字典会
    带着"不确定性还挂在旧位置"的旧数字。源**功率不动**：本仓的信念增益表与
    真值增益表是同一份（``base_belief=None``），去改它就会引入真值泄漏。
    """
    m_rx = int(cfg.aperture.m_rx) if cfg.aperture.enable else 1
    u_vec = None
    if m_rx > 1:
        p_rx = np.asarray(geom.p_uav[receiver], dtype=float)[:3]
        u_vec = _u_of(np.asarray(geom.p_tgt, dtype=float), p_rx,
                      int(cfg.scale.Q), int(cfg.aperture.axis))
    sigma_u = float(target_bearing_std(cfg, geom, receiver, target))
    jac_u = target_bearing_jacobian(cfg, geom, receiver, target)
    out = []
    for src in sources:
        if int(src.target) != int(target):
            out.append(src)
            continue
        i, q = int(src.uav), int(src.target)
        k, l = target_link_offset(cfg, geom, i, receiver, q)
        sigma_k, sigma_l = target_link_std_bins(cfg, geom, i, receiver, q)
        jac_k, jac_l = target_link_jacobians(cfg, geom, i, receiver, q)
        out.append(replace(
            src,
            doppler_bin=float(k),
            delay_bin=float(l),
            u=(float(u_vec[q]) if u_vec is not None else 0.0),
            sigma_doppler_bin=float(sigma_k),
            sigma_delay_bin=float(sigma_l),
            sigma_bearing_u=sigma_u,
            jacobian_doppler_state=jac_k,
            jacobian_delay_state=jac_l,
            jacobian_bearing_state=jac_u,
        ))
    return out
