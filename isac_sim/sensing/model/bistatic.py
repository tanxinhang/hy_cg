"""逐 (i,j,q) 的双站几何量。

⚠️ 本函数是随机数契约的**第二段**：Swerling-I 的共享 RCS（每目标一次）
先抽，再逐链路抽 RCS。次序不可交换。
"""
from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model.containers import BaseGains, BistaticFields, Geometry
from isac_sim.sensing.model.mathkit import wavelength


def _bistatic_fields(
    cfg: Config,
    geom: Geometry,
    rng: np.random.Generator,
    channel: BaseGains | None,
    rcs_view: str,
) -> BistaticFields:
    """填 per-(i,j,q) 的时延、多普勒、DD 格、捕获损耗与目标增益。

    ``rcs_view="mean"`` 时目标增益用配置的均值 RCS —— 调度器在感知之前
    不应看到本 CPI 的 RCS 实现。
    """
    M, Q = cfg.scale.M, cfg.scale.Q
    d = cfg.detect
    w = cfg.waveform
    lam = wavelength(cfg)

    diff_uav_tgt = geom.p_uav[:, None, :] - geom.p_tgt[None, :, :]
    d_uav_tgt = np.linalg.norm(diff_uav_tgt, axis=-1)

    tau = np.zeros((M, M, Q))
    doppler = np.zeros((M, M, Q))
    delay_bin = np.zeros((M, M, Q), dtype=int)
    doppler_bin = np.zeros((M, M, Q), dtype=int)
    valid_dd = np.zeros((M, M, Q), dtype=bool)
    dd_frac_loss = np.ones((M, M, Q), dtype=float)
    delay_frac_full = np.zeros((M, M, Q), dtype=float)
    doppler_frac_full = np.zeros((M, M, Q), dtype=float)
    l_float_grid = np.zeros((M, M, Q), dtype=float)
    k_float_grid = np.zeros((M, M, Q), dtype=float)
    target_gain = np.zeros((M, M, Q))
    geom_factor = np.zeros((M, M, Q))
    aspect_azimuth = np.zeros((M, M, Q))
    rcs_fluct = np.zeros((M, M, Q))

    # Swerling-I 目标：每个 (目标, CPI) 抽一次指数分布 RCS，被所有观测该目标的
    # 双站对共享。只在 ``detect.rcs_model == "swerling1"`` 时抽，好让默认的
    # 独立同分布路径保持精确的随机流。
    rcs_shared: np.ndarray | None = None
    if channel is None and d.rcs_model == "swerling1":
        rcs_shared = rng.exponential(scale=d.target_rcs, size=Q)
    elif channel is None and d.rcs_model == "mean":
        rcs_shared = np.full(Q, d.target_rcs, dtype=float)

    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            for q in range(Q):
                p_i, p_j, p_q = geom.p_uav[i], geom.p_uav[j], geom.p_tgt[q]
                v_i, v_j, v_q = geom.v_uav[i], geom.v_uav[j], geom.v_tgt[q]

                vec_iq = p_q - p_i
                vec_jq = p_q - p_j
                d_iq = max(float(np.linalg.norm(vec_iq)), 1.0)
                d_jq = max(float(np.linalg.norm(vec_jq)), 1.0)
                u_iq = vec_iq / d_iq
                u_jq = vec_jq / d_jq

                tau_ijq = (d_iq + d_jq) / w.c
                rr_tx = float(np.dot(v_q - v_i, u_iq))
                rr_rx = float(np.dot(v_q - v_j, u_jq))
                nu_ijq = (rr_tx + rr_rx) / lam

                tau[i, j, q] = tau_ijq
                doppler[i, j, q] = nu_ijq

                l_float = tau_ijq * w.L * w.delta_f
                k_float = nu_ijq * w.N * w.T
                l_bin = int(np.round(l_float))
                k_bin = int(np.round(k_float))
                delay_bin[i, j, q] = l_bin
                doppler_bin[i, j, q] = k_bin
                valid_dd[i, j, q] = (0 <= l_bin < w.L) and (-(w.N // 2) <= k_bin < w.N // 2)

                delay_frac = abs(l_float - l_bin)
                doppler_frac = abs(k_float - k_bin)
                dd_frac_loss[i, j, q] = float((np.sinc(delay_frac) ** 2) * (np.sinc(doppler_frac) ** 2))
                # 带符号的分数偏移（eta_local_sinc / Dirichlet 需要）。
                delay_frac_full[i, j, q] = float(l_float - l_bin)
                doppler_frac_full[i, j, q] = float(k_float - k_bin)
                l_float_grid[i, j, q] = float(l_float)
                k_float_grid[i, j, q] = float(k_float)

                a = (p_i - p_q) / d_iq
                b = (p_j - p_q) / d_jq
                sin_angle = float(np.linalg.norm(np.cross(a, b)))
                geom_factor[i, j, q] = 0.1 + 0.9 * min(max(sin_angle, 0.0), 1.0)
                bisector_xy = a[:2] + b[:2]
                if float(np.linalg.norm(bisector_xy)) <= 1e-12:
                    bisector_xy = a[:2]
                aspect_azimuth[i, j, q] = float(
                    np.arctan2(bisector_xy[1], bisector_xy[0])
                )

                if rcs_view == "mean":
                    rcs_fluct[i, j, q] = float(d.target_rcs)
                    if d.rcs_aspect_enable:
                        rcs_fluct[i, j, q] *= 0.2 + 0.8 * min(max(sin_angle, 0.0), 1.0)
                elif channel is not None:
                    rcs_fluct[i, j, q] = float(channel.rcs_fluct[i, j, q])
                elif rcs_shared is not None:
                    rcs_fluct[i, j, q] = float(rcs_shared[q])
                    if d.rcs_aspect_enable:
                        # 双站观测角因子：前向/后向散射看到完整 RCS，
                        # 接近镜面方向则更少。
                        rcs_fluct[i, j, q] *= 0.2 + 0.8 * min(max(sin_angle, 0.0), 1.0)
                else:
                    rcs_fluct[i, j, q] = rng.exponential(scale=d.target_rcs)
                target_gain[i, j, q] = lam ** 2 * rcs_fluct[i, j, q] / ((4.0 * np.pi) ** 3 * d_iq ** 2 * d_jq ** 2)

    return BistaticFields(
        d_uav_tgt=d_uav_tgt,
        tau=tau,
        doppler=doppler,
        delay_bin=delay_bin,
        doppler_bin=doppler_bin,
        valid_dd=valid_dd,
        dd_frac_loss=dd_frac_loss,
        delay_frac_full=delay_frac_full,
        doppler_frac_full=doppler_frac_full,
        l_float_grid=l_float_grid,
        k_float_grid=k_float_grid,
        target_gain=target_gain,
        geom_factor=geom_factor,
        aspect_azimuth=aspect_azimuth,
        rcs_fluct=rcs_fluct,
    )
