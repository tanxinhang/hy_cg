"""干扰场：通信接收机与感知接收机共用同一个并发发射集。

在 ``interference.coupling="shared_spectrum"`` 下，节点 j 的通信接收机与感知
接收机由**同一批并发发射机、同一组直连增益、同一组发射功率**构成。下面三个
场向量只依赖接收节点 j、不依赖照射机 i，所以每建一张表只算一次。

``direct_gain`` 的对角线为零，因此 ``P @ G`` 自动是对 k != j 求和。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model.containers import BaseGains
from isac_sim.sensing.model.link_tables.power import PowerSplit


@dataclass
class InterferenceFields:
    """节点级干扰场与其配套的辐射功率。"""

    #: 是否走共享频谱口径（false = legacy，不做任何耦合）
    shared: bool
    #: 感知观测期间各节点的辐射功率 (M,)
    P_rad_sense: np.ndarray | None
    #: 上报期间各节点的辐射功率 (M,)
    P_rad_pay: np.ndarray | None
    #: 持续辐射的感知波形泄漏功率 (M,)
    P_leak: np.ndarray | None
    #: 是否把 active_tx_mask 同时门控到**观测**路径上
    gate_echo: bool
    #: j 处收到的直连场 (M,)
    I_sense_field: np.ndarray | None
    #: j 处收到的上报载荷场 (M,)
    I_pay_field: np.ndarray | None
    #: j 处收到的感知波形泄漏场 (M,)
    I_leak_field: np.ndarray | None


def interference_fields(
    cfg: Config,
    base: BaseGains,
    power: PowerSplit,
    active_tx_mask: np.ndarray | None,
) -> InterferenceFields:
    """构造三个干扰场与协同门控。"""
    c = cfg.comm
    P_sense, P_comm = power.P_sense, power.P_comm

    shared = cfg.interference.coupling == "shared_spectrum"
    if not shared and cfg.interference.coupling != "legacy":
        raise ValueError(
            f"Unknown interference.coupling={cfg.interference.coupling!r}; "
            f"expected 'legacy' or 'shared_spectrum'"
        )
    if not shared:
        # legacy 口径：不做耦合，三个场不存在，对消常数也不参与。
        return InterferenceFields(
            shared=False, P_rad_sense=None, P_rad_pay=None, P_leak=None,
            gate_echo=False, I_sense_field=None, I_pay_field=None, I_leak_field=None,
        )

    ic = cfg.interference
    # 每架 UAV 持续辐射它的感知分量。载荷功率跟随实际上报 MAC：全并发模式下
    # 所有节点都发，active-set 模式下发的是被选中的上报者，而会议版的正交阶段
    # 在感知观测期间不发载荷。
    if c.interference_model == "orthogonal":
        P_rad_sense = P_sense
        P_rad_pay = np.zeros_like(P_comm)
    elif active_tx_mask is not None:
        P_rad_sense = P_sense + P_comm * active_tx_mask
        P_rad_pay = P_comm * active_tx_mask
    else:
        P_rad_sense = P_sense + P_comm
        P_rad_pay = P_comm
    P_leak = P_sense                                  # 一直在辐射

    # ---- 协同门控 --------------------------------------------------------
    # 被静默的节点**什么都不辐射**：既不发它的感知波形，也不向其它接收机泄漏。
    # 门控后的功率必须复现当前口径下的**未门控**表达式 —— 早先的实现一律用
    # ``(P_sense + P_comm) * mask``，那在正交口径下会把辐射功率从 ``P_sense``
    # （= rho*P）悄悄换成 ``P_sense + P_comm``（= P）：干扰凭空多 25%，也就是
    # 开启协同反而先把基线弄差。修好之后，全 True 的掩码门控是恒等操作。
    # 只在门控开关打开时生效，所以冻结的默认路径逐位不变。
    #
    # ``gate_echo`` 把下面这块镜像到**观测**路径上：被静默的节点什么都不辐射，
    # 它的回波自然也不该被收到。没有这一条时，门控删掉了被静默照射机的干扰
    # 却保留了它的观测，于是"掩码漏掉了调度仍在使用的照射机"会被记上它根本
    # 拿不到的检测量（修之前实测：全部节点静默时，2250 个感知 SINR 里仍有
    # 2100 个为正）。
    gate_echo = bool(ic.sense_gate_by_active_tx and active_tx_mask is not None)
    if gate_echo:
        if c.interference_model == "orthogonal":
            P_rad_sense = P_sense * active_tx_mask
            P_rad_pay = np.zeros_like(P_comm)
        else:
            P_rad_sense = (P_sense + P_comm) * active_tx_mask
            P_rad_pay = P_comm * active_tx_mask
        P_leak = P_sense * active_tx_mask

    I_sense_field = P_rad_sense @ base.direct_gain      # j 处的直连场 (M,)
    I_pay_field = P_rad_pay @ base.direct_gain          # j 处的上报载荷场 (M,)
    I_leak_field = P_leak @ base.direct_gain            # j 处的感知波形泄漏场 (M,)
    return InterferenceFields(
        shared=True,
        P_rad_sense=P_rad_sense,
        P_rad_pay=P_rad_pay,
        P_leak=P_leak,
        gate_echo=gate_echo,
        I_sense_field=I_sense_field,
        I_pay_field=I_pay_field,
        I_leak_field=I_leak_field,
    )
