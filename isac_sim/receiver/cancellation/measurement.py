"""cancellation 积木：接收机测量容器。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from isac_sim.receiver.cancellation.measurement_views_a import _MeasurementViewsA
from isac_sim.receiver.cancellation.measurement_views_b import _MeasurementViewsB
from isac_sim.receiver.cancellation.measurement_views_b import _MeasurementViewsB

@dataclass(eq=False)
class ReceiverMeasurement(_MeasurementViewsA, _MeasurementViewsB):
    """TP-UIC 真正交付的三样东西，逐接收机。

    ``fraction`` / ``kappa_db`` —— **分母**桥：``I_res / I_in``，冻结的
    ``kappa_dc`` 的直插替代品。

    ``eta_survive`` / ``eta_survive_q`` —— **分子**桥：对消器留在观测里的回波
    能量占比。整场那个数是标量 SINR 能消费的；``_q`` 那个是被测目标自己的存活率，
    也是唯一能回答"TP-UIC 是否伤害弱目标"的量。两个都报是故意的：它们不一样，
    而那个差值就是"报出来的存活率里有多少是从未受检目标那里借来的"。

    ``statistic`` / ``dof_real`` / ``residual_cov_diag`` —— 一个 GLRT 会去消费的
    东西（而不是标量 SINR）。保留在这里，是为了即使生产模型仍在消费标量，接线
    也能被检查。
    """

    fraction: np.ndarray  # (M,) I_res / I_in
    kappa_db: np.ndarray  # (M,) 10 log10(I_in / I_res)
    eta_survive: np.ndarray  # (M,) 整场回波存活率
    eta_survive_q: np.ndarray  # (M,) 被测目标的回波存活率
    eta_protect: np.ndarray  # (M,) 保护对**真值**回波的覆盖度
    i_res: np.ndarray  # (M,) 绝对残余直达功率
    i_in: np.ndarray  # (M,) 绝对输入直达功率
    i_res_estimate: np.ndarray  # (M,) ||f(n)||^2 那部分，单独报出
    i_res_pred: np.ndarray  # (M,) 解析的 E[残余直达功率 | 拟合模型]
    i_res_moment_mean: np.ndarray  # (M,) 显式映射的均值，供矩审计用
    i_res_pred_var: np.ndarray  # (M,) 解析的残差功率方差
    i_in_pred: np.ndarray  # (M,) 解析的 E[输入直达功率 | 几何]
    predicted_fraction: np.ndarray  # (M,) i_res_pred / i_in
    predicted_kappa_db: np.ndarray  # (M,) -10 log10(predicted_fraction)
    model_fraction: np.ndarray  # (M,) 实现的 i_res / 与模型对齐的 E[i_in]
    noise_enhance_db: np.ndarray  # (M,)
    selected_soft_mu: np.ndarray | None = None  # (M,)，NaN 表示硬回退
    # 目标条件化的分母证书。对"并集保护"臂，各列是相同的；目标条件化的臂则
    # 真正随 q 变化。
    fraction_by_target: np.ndarray | None = None  # (M, Q)
    predicted_fraction_by_target: np.ndarray | None = None  # (M, Q)
    model_fraction_by_target: np.ndarray | None = None  # (M, Q)
    i_res_by_target: np.ndarray | None = None  # (M, Q) 绝对实现功率
    i_res_pred_by_target: np.ndarray | None = None  # (M, Q) 仅信念的均值
    moment_mean_by_target: np.ndarray | None = None  # (M, Q)
    moment_var_by_target: np.ndarray | None = None  # (M, Q)
    prior_quantile_fraction: np.ndarray | None = None  # (M,)
    prior_quantile_fraction_by_target: np.ndarray | None = None  # (M,Q)
    selected_soft_mu_by_target: np.ndarray | None = None  # (M,Q)
    # 真正的目标条件化存活率。为了兼容早期的"单弱目标"测量它是可选的，但一个
    # 接收机/调度器的联合实验需要它。
    eta_survive_by_target: np.ndarray | None = None  # (M, Q)
    predicted_retention_by_target: np.ndarray | None = None  # (M, Q)，仅信念
    risk_retention_by_target: np.ndarray | None = None  # (M, Q)，信念 95% 轴
    arm: str = "tp_uic_full"
    belief_is_truth: bool = False
