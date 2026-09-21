"""逐 (i,j) 对级量：残余干扰、RINR、sigma0 与有效感知功率。

这些量只依赖链路 (i,j)，与目标 q 无关，所以在 q 循环外算一次。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model.containers import BaseGains
from isac_sim.sensing.model.link_tables.fields import InterferenceFields
from isac_sim.sensing.model.link_tables.power import PowerSplit
from isac_sim.sensing.model.link_tables.residual import ResidualModel


@dataclass
class PairTerms:
    """一条链路 (i,j) 的对级量。"""

    #: 残余干扰 / 噪声地板
    rinr: float
    #: 对级 H0 标准差
    sigma0: float
    #: 残余总功率（逐目标时由调用方按 q 覆盖）
    residual_total: float
    #: 自干扰残余（与 i/j/q 无关的那部分）
    residual_self: float
    #: 多机感知泄漏残余
    residual_multi: float
    #: 逐目标的残余直连功率 (Q,)；None = 用标量
    residual_direct_by_target: np.ndarray | None
    #: 有效感知功率（含功率模型与协同门控）
    effective_sensing_power: float


def pair_terms(
    cfg: Config,
    base: BaseGains,
    power: PowerSplit,
    fields: InterferenceFields,
    residual: ResidualModel,
    chi_comm: np.ndarray,
    active_tx_mask: np.ndarray | None,
    M: int,
    i: int,
    j: int,
) -> PairTerms:
    """算链路 (i,j) 的对级量。"""
    r, d = cfg.radio, cfg.detect
    P, P_sense, P_comm = power.P, power.P_sense, power.P_comm
    n0, eps_den = power.n0, power.eps_den

    residual_self = r.residual_self_factor * P[j]
    residual_direct_by_target = None
    if fields.shared:
        # 干扰集、增益、辐射功率都与 j 处的通信接收机相同 —— 只有接收机的
        # 抑制能力不同。注意照射机 i **不**被排除：它的直连路径正是双站感知
        # 接收机必须对消的那一项近远效应。
        if residual.residual_power_vec is not None:
            if residual.residual_power_vec.ndim == 2:
                residual_direct_by_target = residual.residual_power_vec[j, :]
                residual_direct = float(np.max(residual_direct_by_target))
            else:
                residual_direct = float(residual.residual_power_vec[j])
        elif residual.kappa_dc_vec is not None and residual.kappa_dc_vec.ndim == 2:
            residual_direct_by_target = residual.kappa_dc_vec[j, :] * float(fields.I_sense_field[j])
            # 对级 RINR/sigma0 编码不了 q。这里报**最保守的最差目标**，而下面的
            # gamma_sense/var0_q 用的是精确的目标条件分母。
            residual_direct = float(np.max(residual_direct_by_target))
        else:
            residual_direct = (
                residual.kappa_dc
                if residual.kappa_dc_vec is None
                else residual.kappa_dc_vec[j]
            ) * float(fields.I_sense_field[j])
        residual_multi = 0.0
    else:
        residual_direct = 0.0
        residual_multi = 0.0
        # 直连对消误差与多机感知泄漏共用同一批干扰源，一次循环里一起累加。
        for k in range(M):
            if k == i or k == j:
                continue
            residual_direct += r.residual_direct_factor * P[k] * base.direct_gain[k, j]
            residual_multi += r.residual_multi_uav_factor * P_sense[k] * base.direct_gain[k, j]

    residual_total = residual_self + residual_direct + residual_multi
    rinr = residual_total / (n0 + eps_den)
    sigma0 = d.soft_sigma0 * math.sqrt(1.0 + r.rinr_sigma_factor * rinr)

    if r.isac_power_model == "sensing_only":
        effective_sensing_power = P_sense[i]
    elif r.isac_power_model == "joint_waveform":
        effective_sensing_power = P_sense[i] + P_comm[i]
    elif r.isac_power_model == "reliable_comm_assisted":
        effective_sensing_power = P_sense[i] + chi_comm[i, j] * P_comm[i]
    else:
        raise ValueError(f"Unknown isac_power_model={r.isac_power_model!r}")

    if fields.gate_echo and not active_tx_mask[i]:
        # 被静默的照射机：本 CPI 没有辐射任何波形，也就没东西可收。``signal``
        # 与 ``raw_signal`` 都由这个因子构成，所以一次赋值同时覆盖门控观测
        # 与它的原始 SINR 基线。
        effective_sensing_power = 0.0

    return PairTerms(
        rinr=rinr,
        sigma0=sigma0,
        residual_total=residual_total,
        residual_self=residual_self,
        residual_multi=residual_multi,
        residual_direct_by_target=residual_direct_by_target,
        effective_sensing_power=effective_sensing_power,
    )
