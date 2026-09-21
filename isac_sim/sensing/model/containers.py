"""层间数据容器与由表派生的反向标定：几何、基础增益、链路量表。"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from isac_sim.core.config import Config
from isac_sim.detection.llr import soft_mean as _soft_mean, soft_var0 as _soft_var0


# ==========================================================================
# 数据容器
# ==========================================================================
@dataclass
class Geometry:
    """一次 trial 的部署几何与运动学（位置与速度）。"""

    p_uav: np.ndarray
    v_uav: np.ndarray
    p_tgt: np.ndarray
    v_tgt: np.ndarray

@dataclass
class BaseGains:
    """与几何相关的随机量，不依赖任何算法。"""

    edge_mask: np.ndarray
    direct_gain: np.ndarray
    d_uu: np.ndarray
    d_uav_tgt: np.ndarray
    tau: np.ndarray
    doppler: np.ndarray
    delay_bin: np.ndarray
    doppler_bin: np.ndarray
    valid_dd: np.ndarray
    dd_frac_loss: np.ndarray
    dd_collision_count: np.ndarray
    target_gain: np.ndarray
    geom_factor: np.ndarray
    # 双站视线角平分线在目标水平系里的方位角。
    # 只被 opt-in 的主动证据"观测角场景"模型使用。
    aspect_azimuth: np.ndarray
    # 逐 (i,j,q) 的 RCS 实现值。存下来是为了让第二套几何（例如调度器的
    # *belief*）能复用完全相同的物理信道 —— UAV 间衰落与目标 RCS 都是物理量，
    # 不能因为跟踪器的 belief 与真值不同就重抽。
    rcs_fluct: np.ndarray
    # C2F DD 细化数组（论文式 coarse/fine_dd_gain）。``cfg.refine`` 关闭时两个
    # 字段都等于 ``dd_frac_loss``，从而原有的感知 SINR 计算逐位不变。
    eta_loc: np.ndarray
    eta_fine: np.ndarray
    # 以坐标轴为单位的（带符号）DD 分数中心，供 eta^loc 的窗口求和用。
    delay_frac: np.ndarray
    doppler_frac: np.ndarray

@dataclass
class LinkTables:
    """由 :class:`BaseGains` 与功率划分导出的逐链路量。"""

    gamma_comm: np.ndarray
    rate: np.ndarray
    chi_comm: np.ndarray
    feasible_comm: np.ndarray
    raw_gamma_sense: np.ndarray
    gamma_sense: np.ndarray
    rinr: np.ndarray
    mu_soft: np.ndarray
    sigma0: np.ndarray
    # 逐 (i,j,q) 的软统计量 H0 方差。旧的高斯模型里它等于 ``sigma0[i,j]^2``
    # 在 q 上广播；LLR 模型里则由感知 SINR **反推**
    # （``L*gamma^2/(1+gamma)^2``）。
    var0_q: np.ndarray

def rescale_sensing_tables_for_rcs(
    cfg: Config,
    tables: LinkTables,
    factor: float | np.ndarray,
) -> LinkTables:
    """Return a non-mutating RCS-counterfactual sensing table.

    在几何、功率、干扰与波形损伤都固定的前提下，RCS 与期望回波功率成正比，
    因而与感知 SINR 成正比。``factor`` 可以是标量，也可以是逐目标的正乘子。
    通信字段不变。软统计量的矩**由缩放后的 SINR 重算**，而不是直接按比例缩放，
    这样才能保住非线性的 LLR 模型。
    """
    factors = np.asarray(factor, dtype=float)
    if factors.ndim == 0:
        factors = np.full(cfg.scale.Q, float(factors), dtype=float)
    if factors.shape != (cfg.scale.Q,):
        raise ValueError(
            f"RCS factor must be scalar or shape ({cfg.scale.Q},), got {factors.shape}"
        )
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("RCS factors must be finite and strictly positive")

    scale = factors.reshape(1, 1, cfg.scale.Q)
    raw_gamma = np.asarray(tables.raw_gamma_sense, dtype=float) * scale
    gamma = np.asarray(tables.gamma_sense, dtype=float) * scale
    mu_soft = np.asarray(_soft_mean(cfg, gamma), dtype=float)
    pair_var0 = np.broadcast_to(
        np.asarray(tables.sigma0, dtype=float)[:, :, None] ** 2,
        gamma.shape,
    )
    var0_q = np.asarray(_soft_var0(cfg, gamma, pair_var0), dtype=float)
    return LinkTables(
        gamma_comm=tables.gamma_comm,
        rate=tables.rate,
        chi_comm=tables.chi_comm,
        feasible_comm=tables.feasible_comm,
        raw_gamma_sense=raw_gamma,
        gamma_sense=gamma,
        rinr=tables.rinr,
        mu_soft=mu_soft,
        sigma0=tables.sigma0,
        var0_q=var0_q,
    )

# ==========================================================================
# 通信与感知 SINR 表
# ==========================================================================
def packet_bits_for_target(cfg: Config, q: int) -> float:
    """每个候选包承载的载荷比特数（当前与目标无关）。"""
    return float(max(cfg.comm.K_candidates, 0) * cfg.comm.b_d)


@dataclass
class BistaticFields:
    """逐 (i,j,q) 的双站量，由 :mod:`...bistatic` 填好一次后传给下游。"""

    d_uav_tgt: np.ndarray
    tau: np.ndarray
    doppler: np.ndarray
    delay_bin: np.ndarray
    doppler_bin: np.ndarray
    valid_dd: np.ndarray
    dd_frac_loss: np.ndarray
    delay_frac_full: np.ndarray
    doppler_frac_full: np.ndarray
    l_float_grid: np.ndarray
    k_float_grid: np.ndarray
    target_gain: np.ndarray
    geom_factor: np.ndarray
    aspect_azimuth: np.ndarray
    rcs_fluct: np.ndarray
