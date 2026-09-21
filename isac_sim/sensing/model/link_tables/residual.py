"""接收端对消深度的四种来源（按优先级）。

``kappa_dc`` 是"聚合直连场经对消后存活的比例"。它**故意是比例而不是 dB**，
这样接收机模型与出厂常数就是同一种对象。

四个来源，按优先级：

* ``residual_power_by_receiver_target`` —— 首选的**绝对**残余功率，
  与任何输入场的归一化无关。
* ``residual_fraction_by_receiver`` —— 旧的逐接收机 / 逐(接收机,目标)比例，
  由 :func:`isac_sim.receiver.cancellation.measure_residual_fraction` 或标定表
  产生。为冻结基线与诊断保留。
* ``cancellation.mode == "predict"`` —— 解析桥
  :func:`isac_sim.receiver.cancellation.predict_cancellation`，即维度比例留存
  加参考预算的估计项。它是**乐观**的（600 m 场景实测差 5 倍），存在的意义是
  把链路端到端闭合起来，不是用来报结果。
* 否则就是**没有对消**：``kappa_dc = 1.0``，直连场全额存活。

最后一项是删除 ``interference.direct_cancellation_db`` 之后唯一的兜底，必须把
它的含义说准：它**不是**"默认对消 40 dB"，而是"这条链路上没有任何接收机实现
提供对消"。那个常数之所以被删，是因为它撑着 SINR 的分母却没有任何实现支撑
（取 0 dB 会让 P_D 掉到 P_FA），挂在它上面的增益全是记账。因此这里宁可让
数字难看，也不给一个不存在的接收机发工钱。

要拿到对消，唯一办法是把 TP-UIC 的**实测**残余喂进来（生产接线见
:mod:`isac_sim.sensing.model.link_tables.trial_certificate`）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model.containers import BaseGains
from isac_sim.sensing.model.link_tables.fields import InterferenceFields
from isac_sim.sensing.model.link_tables.power import PowerSplit


@dataclass
class ResidualModel:
    """对消模型解析结果。"""

    #: 标量比例（仅当没有逐机向量时使用）
    kappa_dc: float
    #: 逐接收机或逐(接收机,目标)比例；None = 用标量
    kappa_dc_vec: np.ndarray | None
    #: 绝对残余功率，形状 (M,) 或 (M,Q)；None = 用比例
    residual_power_vec: np.ndarray | None


def resolve_residual(
    cfg: Config,
    base: BaseGains,
    fields: InterferenceFields,
    power: PowerSplit,
    *,
    residual_fraction_by_receiver: np.ndarray | None,
    residual_power_by_receiver_target: np.ndarray | None,
) -> ResidualModel:
    """按优先级解析出对消模型。"""
    M, Q = cfg.scale.M, cfg.scale.Q

    if not fields.shared:
        # legacy 口径：对消常数不参与，残余直连在多径分支里另行累加。
        return ResidualModel(kappa_dc=0.0, kappa_dc_vec=None, residual_power_vec=None)

    kappa_dc_vec = None
    residual_power_vec = None
    if residual_power_by_receiver_target is not None:
        residual_power_vec = np.asarray(residual_power_by_receiver_target, dtype=float)
        if residual_power_vec.shape not in ((M,), (M, Q)):
            raise ValueError(
                "residual_power_by_receiver_target must have shape (%d,) "
                "or (%d, %d), got %s"
                % (M, M, Q, np.shape(residual_power_by_receiver_target))
            )
        if not np.all(np.isfinite(residual_power_vec)) or np.any(residual_power_vec < 0.0):
            raise ValueError(
                "residual_power_by_receiver_target must be finite and non-negative"
            )
        kappa_dc_vec = None
        kappa_dc = 0.0
    elif residual_fraction_by_receiver is not None:
        kappa_dc_vec = np.asarray(residual_fraction_by_receiver, dtype=float)
        if kappa_dc_vec.shape not in ((M,), (M, Q)):
            raise ValueError(
                "residual_fraction_by_receiver must have shape (%d,) or "
                "(%d, %d), got %s"
                % (M, M, Q, np.shape(residual_fraction_by_receiver))
            )
        if not np.all(np.isfinite(kappa_dc_vec)) or np.any(kappa_dc_vec < 0.0):
            raise ValueError(
                "residual_fraction_by_receiver must be finite and non-negative"
            )
        kappa_dc = 0.0
    elif cfg.cancellation.mode == "predict":
        from isac_sim.receiver.cancellation import predict_cancellation

        # 究竟有多少照射机真的照到了接收机 j：``direct_gain`` 的对角线与
        # 非连通项都是零，所以数"严格为正的乘积"与模型当初求 ``I_sense_field``
        # 时的计数是同一个。
        n_illum = np.asarray(
            [
                np.count_nonzero((power.P_rad_sense * base.direct_gain[:, j]) > 0.0)
                for j in range(M)
            ],
            dtype=float,
        )
        kappa_dc_vec = predict_cancellation(cfg, fields.I_sense_field, n_illum)[0]
        kappa_dc = 0.0
    else:
        # 没有任何接收机提供对消 ⇒ 直连场全额存活。见模块 docstring。
        kappa_dc = 1.0
    return ResidualModel(
        kappa_dc=kappa_dc, kappa_dc_vec=kappa_dc_vec, residual_power_vec=residual_power_vec
    )
