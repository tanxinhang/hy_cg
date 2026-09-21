"""信念误差：把"跟踪器报的位置/速度误差"换成模板失配的协方差因子。"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from isac_sim.receiver.cancellation import (
    EPS,
    Observation,
    lift_dictionary,
    steering_derivative,
    tangent_columns,
)


def _belief_bin_sigmas(cfg) -> Tuple[float, float]:
    """``(sigma_delay_bins, sigma_doppler_bins)``：把声明的信念误差换成 bin 单位。

    接收机的跟踪器报的是位置误差与速度误差，而检测器需要用"模板索引的单位"
    来计它们。延迟换算：单向路程误差 ``sigma_p / c`` 除以延迟格间距
    ``1 / (L * delta_f)``；多普勒换算：双站径向速率误差 ``2 sigma_v / lambda``
    除以多普勒格间距 ``1 / (N * T)``。两者取的是观测几何下的**最大**误差，所以
    这一项是保守的，不是调出来的。
    """
    w = cfg.waveform
    lam = float(w.c) / float(w.fc)
    sigma_delay_bins = (float(cfg.prior.belief_sigma_pos_m) / float(w.c)
                        * float(w.L) * float(w.delta_f))
    sigma_doppler_bins = (2.0 * float(cfg.prior.belief_sigma_vel_mps) / lam
                          * float(w.N) * float(w.T))
    return float(sigma_delay_bins), float(sigma_doppler_bins)


def _belief_error_factor(cfg, obs: Observation) -> np.ndarray:
    """``B``，满足 ``B B^H = C_belief``：模板失配的协方差。

    每个被相信的回波最多贡献三列，分别描述延迟、多普勒与方位失配。DD 导数由
    源的导向向量提升，方位导数是 ``a_DD (x) da_array/du``，它们的外积构成一阶
    失配协方差。``P`` 是被相信的回波功率，所以接收机几乎看不见的目标只贡献
    几乎为零的失配。

    **被测目标自己的源被排除。** ``C_res`` 是 **H0 下**残余的协方差，而 H0 下
    该目标的回波并不存在，把它自己的泄漏算进去恰好会在设定虚警电平的地方把
    协方差抬高。代价是公开且已声明的：H1 下被测回波自身的失配没有被计费，
    于是这一项换来的是**标定过的** ``P_FA``，价格是略微乐观的 ``P_D``。

    用的是**被相信的**源及其被相信的 bin —— 接收机不可能知道真值，按真值计费
    的协方差只是"伪装成标定的 oracle"。
    """
    sources = obs.targets_belief if obs.targets_belief is not None else obs.targets
    n_bins = int(obs.y.size)
    if not sources:
        return np.zeros((n_bins, 0), dtype=complex)
    sigma_l, sigma_k = _belief_bin_sigmas(cfg)
    if sigma_l <= 0.0 and sigma_k <= 0.0:
        return np.zeros((n_bins, 0), dtype=complex)
    step = float(cfg.cancellation.tangent_step_bins)
    tested = int(getattr(obs, "weak_index", 0))
    cols: List[np.ndarray] = []
    for src in sources:
        if int(src.target) == tested:
            continue
        block = tangent_columns(cfg, float(src.doppler_bin), float(src.delay_bin),
                                1, step)
        if block.shape[1] < 3:
            continue
        centre, d_delay, d_doppler = block[:, 0], block[:, 1], block[:, 2]
        norm = float(np.sqrt(max(np.vdot(centre, centre).real, EPS)))
        amp = math.sqrt(max(float(src.power), 0.0)) / norm
        sigma_l_src = (
            sigma_l if src.sigma_delay_bin is None
            else max(float(src.sigma_delay_bin), 0.0)
        )
        sigma_k_src = (
            sigma_k if src.sigma_doppler_bin is None
            else max(float(src.sigma_doppler_bin), 0.0)
        )
        sigma_u_src = max(float(src.sigma_bearing_u or 0.0), 0.0)
        u_src = float(getattr(src, "u", 0.0))
        if sigma_l_src > 0.0:
            cols.append(lift_dictionary(
                cfg, ((amp * sigma_l_src) * d_delay)[:, None], [u_src]
            )[:, 0])
        if sigma_k_src > 0.0:
            cols.append(lift_dictionary(
                cfg, ((amp * sigma_k_src) * d_doppler)[:, None], [u_src]
            )[:, 0])
        m_rx = int(cfg.aperture.m_rx) if cfg.aperture.enable else 1
        if sigma_u_src > 0.0 and m_rx > 1:
            cols.append(
                (amp * sigma_u_src)
                * np.kron(centre, steering_derivative(m_rx, u_src))
            )
    if not cols:
        return np.zeros((n_bins, 0), dtype=complex)
    return np.stack(cols, axis=1)
