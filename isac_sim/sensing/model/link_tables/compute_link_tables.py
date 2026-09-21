"""逐链路通信 / 感知 SINR 表的装配入口。

本函数只做编排：入参解析 → 功率划分 → 干扰场 → 对消模型 → 通信块 → 感知块。
每一段的实现都在同目录的独立模块里，可以单独拿来用（例如只想算通信块、
或只想算残余量）。
"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model.containers import BaseGains, LinkTables
from isac_sim.sensing.model.link_tables.comm import comm_tables
from isac_sim.sensing.model.link_tables.fields import interference_fields
from isac_sim.sensing.model.link_tables.inputs import resolve_inputs
from isac_sim.sensing.model.link_tables.power import power_split
from isac_sim.sensing.model.link_tables.residual import resolve_residual
from isac_sim.sensing.model.link_tables.sensing import sensing_tables
from isac_sim.sensing.model.link_tables.trial_certificate import resolve_injection


def compute_link_tables(
    cfg: Config,
    base: BaseGains,
    dd_gain: np.ndarray | None = None,
    active_tx_mask: np.ndarray | None = None,
    reuse_from: LinkTables | None = None,
    sensing_power_scale_by_uav: np.ndarray | None = None,
    residual_fraction_by_receiver: np.ndarray | None = None,
    residual_power_by_receiver_target: np.ndarray | None = None,
    target_retention_by_receiver: np.ndarray | None = None,
) -> LinkTables:
    """逐链路的通信 / 感知 SINR 量。

    ``dd_gain`` 是可选的 ``(M, M, Q)`` DD 增益覆盖。省略（默认）时用原始的粗
    分辨率 ``dd_frac_loss``，从而试验输出逐位不变。C2F 选择器会传
    ``base.eta_fine``，得到细阶段用的细粒度感知 SINR 表。

    ``active_tx_mask`` 是可选的 ``(M,)`` 布尔数组，标记当前并发发射的 UAV。
    当 ``cfg.comm.interference_model == "active_set"`` 时只有这些 UAV 计入通信
    干扰项，所以选中的链路集越小看到的干扰越少；``None``（默认，也是
    ``"full_concurrent"`` 模型）保持原先"所有 UAV 都贡献"的最坏情况。
    在 ``interference_model="orthogonal"`` 下上报载荷干扰为零，但持续辐射的
    感知波形泄漏仍留在分母里。

    ``reuse_from`` 是 ``active_set`` 模型的性能快速路径。在默认的
    ``sensing_only``（或 ``joint_waveform``）功率模型下，感知侧的量
    —— ``gamma_sense``、``mu_soft``、``sigma0``、``rinr``、``raw_gamma_sense``
    —— **不**依赖通信干扰，于是可以从选择阶段建好的表里原样拷贝，只需重算
    O(M^2) 的通信块。这样每次复评都不必再跑 O(M^2·Q) 的感知循环（成本远高于
    其它部分）。``reliable_comm_assisted`` 下该路径被忽略：那里的感知功率本身
    就依赖 ``chi_comm``。

    ``sensing_power_scale_by_uav`` 在发射端、在形成想要的信号与泄漏场之前施加
    主动模式功率。因此抬高某架 UAV 的感知功率会同时增强它自己的回波、并增加
    其它接收机遇见的干扰。默认全 1 向量与已发布的被动模型完全一致。

    ``residual_fraction_by_receiver`` 把**没有对消**（默认的 ``kappa_dc = 1``）
    换成逐接收机 ``(M,)`` 或 ``(M,Q)`` 的**实测**残留比例；``target_retention_by_receiver``
    是同一替换的**另一半** —— 对消器既留干扰（分母 ``I_res``）也衰减回波
    （分子 ``eta_surv``），只给分母等于让回路半闭合。两者 ``None``（默认）时
    逐位不变，也就是每个冻结基线的分子与分母。

    ``residual_power_by_receiver_target`` 是首选接口：**绝对**残余功率，形状
    ``(M,)`` 或 ``(M,Q)``，与比例接口互斥（比例接口为冻结基线保留）。三者的
    优先级与口径见 :mod:`residual`；生产接线见 :mod:`trial_certificate`。
    """
    dd_used = base.dd_frac_loss if dd_gain is None else dd_gain

    # 生产接线：两参数均省略时从 trial 上下文回填（显式传参优先；关时逐位不变）。
    residual_fraction_by_receiver, target_retention_by_receiver = resolve_injection(
        residual_fraction_by_receiver, target_retention_by_receiver
    )

    inputs = resolve_inputs(
        cfg,
        dd_gain=dd_gain,
        active_tx_mask=active_tx_mask,
        reuse_from=reuse_from,
        sensing_power_scale_by_uav=sensing_power_scale_by_uav,
        residual_fraction_by_receiver=residual_fraction_by_receiver,
        residual_power_by_receiver_target=residual_power_by_receiver_target,
        target_retention_by_receiver=target_retention_by_receiver,
    )
    power = power_split(cfg, sensing_power_scale_by_uav)
    fields = interference_fields(cfg, base, power, active_tx_mask)
    residual = resolve_residual(
        cfg,
        base,
        fields,
        power,
        residual_fraction_by_receiver=residual_fraction_by_receiver,
        residual_power_by_receiver_target=residual_power_by_receiver_target,
    )
    gamma_comm, rate, chi_comm, feasible_comm = comm_tables(
        cfg, base, power, fields, active_tx_mask
    )

    # ---- 快速路径 --------------------------------------------------------
    # 只有通信块依赖当前发射集，所以（大得多的）感知块可以复用。
    # 这正是 "active_set" 模型能在 Monte-Carlo 循环里跑得起的原因。
    if inputs.can_reuse_sensing:
        return LinkTables(
            gamma_comm=gamma_comm,
            rate=rate,
            chi_comm=chi_comm,
            feasible_comm=feasible_comm,
            raw_gamma_sense=reuse_from.raw_gamma_sense,
            gamma_sense=reuse_from.gamma_sense,
            rinr=reuse_from.rinr,
            mu_soft=reuse_from.mu_soft,
            sigma0=reuse_from.sigma0,
            var0_q=reuse_from.var0_q,
        )

    sens = sensing_tables(
        cfg, base, power, fields, residual, inputs, chi_comm, active_tx_mask, dd_used
    )
    return LinkTables(
        gamma_comm=gamma_comm,
        rate=rate,
        chi_comm=chi_comm,
        feasible_comm=feasible_comm,
        raw_gamma_sense=sens.raw_gamma_sense,
        gamma_sense=sens.gamma_sense,
        rinr=sens.rinr,
        mu_soft=sens.mu_soft,
        sigma0=sens.sigma0,
        var0_q=sens.var0_q,
    )
