"""通信侧：SINR、速率、可靠度与上报链路的可行性。

注意**上报方向是反的**：软统计量 ``s_{ijq}`` 在接收端 j 产生、要报回发射端 i，
所以上报可行性看的是 ``rate[j, i] / chi_comm[j, i]``，**不是** ``rate[i, j]``。
"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.fbl import chi_from_gamma as _chi_from_gamma
from isac_sim.sensing.model.containers import BaseGains
from isac_sim.sensing.model.link_tables.fields import InterferenceFields
from isac_sim.sensing.model.link_tables.power import PowerSplit


def comm_tables(
    cfg: Config,
    base: BaseGains,
    power: PowerSplit,
    fields: InterferenceFields,
    active_tx_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(gamma_comm, rate, chi_comm, feasible_comm)``。"""
    M = cfg.scale.M
    c = cfg.comm
    P_sense, P_comm = power.P_sense, power.P_comm
    n0, eps_den = power.n0, power.eps_den

    gamma_comm = np.zeros((M, M))
    rate = np.zeros((M, M))
    chi_comm = np.zeros((M, M))
    feasible_comm = np.zeros((M, M), dtype=bool)

    for i in range(M):
        for j in range(M):
            if i == j or not base.edge_mask[i, j]:
                continue

            signal = P_comm[i] * base.direct_gain[i, j]
            if fields.shared:
                # 与 j 处感知接收机同一个场，只减去 i 自己的贡献 —— 在这条链路上
                # i 是想要的信源，不是干扰源。被减掉的那些项必须与当初加进场里的
                # 功率一致，否则一个"不发"的节点会减掉它从未做出的贡献，干扰就
                # 可能变成负数。
                interf = (
                    fields.I_pay_field[j]
                    - fields.P_rad_pay[i] * base.direct_gain[i, j]
                    + c.comm_leakage_from_sensing
                    * (fields.I_leak_field[j] - fields.P_leak[i] * base.direct_gain[i, j])
                )
            else:
                interf = 0.0
                for k in range(M):
                    if k == i or k == j:
                        continue
                    # "active_set" 模型：UAV 只有在真的在发（即它是某条被选中的
                    # 上报链路的发送端）时才算干扰。静默的 UAV 不贡献任何东西。
                    if active_tx_mask is not None and not active_tx_mask[k]:
                        continue
                    interf += (P_comm[k] + c.comm_leakage_from_sensing * P_sense[k]) * base.direct_gain[k, j]

            direct_leakage = c.comm_direct_leakage_factor * P_sense[i] * base.direct_gain[i, j]
            gamma = signal / (n0 + interf + direct_leakage + eps_den)
            gamma_comm[i, j] = gamma
            rate[i, j] = power.B * np.log2(1.0 + gamma)
            # 按配置的可靠度模型：旧的启发式 ``gamma/(gamma+gamma_req)``，或
            # 有限块长成功概率 ``1 - Q(...)``。经 fbl.py 分派，好让两者在
            # ``reliability_model="heuristic"`` 下逐位兼容。
            chi_comm[i, j] = _chi_from_gamma(cfg, gamma, power.gamma_req)

    # ``chi_comm = gamma / (gamma + gamma_req)`` 按构造就落在 [0, 1)。
    # 在这里一次性向量化裁剪（O(M^2)），好让会被调用上百万次的融合热路径
    # 永远不用为一次标量 np.clip 付出代价。
    np.clip(chi_comm, 0.0, 1.0, out=chi_comm)

    # 上报链路的可行性。软统计量 s_{ijq} 在接收端 j 产生、报回发射端 i，
    # 所以上报方向是 j -> i，可行性看 rate[j, i] / chi_comm[j, i]（不是 [i, j]）。
    for i in range(M):
        for j in range(M):
            if i == j or not base.edge_mask[i, j]:
                continue
            if c.enforce_chi_min:
                feasible_comm[i, j] = (rate[j, i] >= c.R_min) and (chi_comm[j, i] >= c.chi_min)
            else:
                feasible_comm[i, j] = (rate[j, i] >= c.R_min)

    return gamma_comm, rate, chi_comm, feasible_comm
