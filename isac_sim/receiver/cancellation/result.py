"""cancellation 积木：单臂对消结果容器。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from typing import Tuple

@dataclass
class CancellationResult:
    """下游调度器需要从接收机拿到的一切。

    这里的核算是一个**分量转移**，不是总量之差。每条臂都是观测上的一个线性
    映射 ``f``，于是观测残差可以精确分解::

        y - f(y) = [x - f(x)] + [s - f(s)] - f(n)

    三个方括号项分别是：存活下来的干扰、存活下来的回波、以及估计器从噪声里
    拉进观测的那部分。单算一个 ``||x - f(y)||^2`` 会把它们混在一起，而且是朝
    **美化保护**的方向混 —— 这正是下面这些字段必须分开的原因。
    """

    name: str
    residual: np.ndarray  # (K,) 复向量，清洗后的观测 y - f(y)
    h_hat: np.ndarray  # (d,) 作用在 y 上的干扰系数估计
    c_h_diag: np.ndarray  # (d,) h_hat 的后验方差
    i_in: float  # ||x_direct||^2，输入直达功率
    i_res: float  # i_res_structural + i_res_estimate，留下的干扰
    i_res_structural: float  # ||x_direct - f(x_direct)||^2，f 拒绝减掉的部分
    i_res_estimate: float  # ||f(n)||^2，从噪声里引入的估计误差
    i_res_pred: float  # 式 (3) 的预测值
    i_res_moment_mean: float  # 同一残差均值，但由显式映射推出
    i_res_pred_var: float  # structural + 拟合噪声 的残差功率方差
    i_in_pred: float  # 物理随机相位模型下 E[||Xh||^2]
    i_res_retained: float  # 式 (3) 第一项：保护所付的代价
    i_res_pred_estimate: float  # 式 (3) 第二项：估计不确定性
    eta_protect: float  # ||P s||^2 / ||s||^2 —— 保护覆盖度（设计量）
    eta_survive: float  # ||s - f(s)||^2 / ||s||^2 —— 回波存活率（实现量）
    noise_enhance_db: float  # 10 log10(||f(n)||^2 / ||n||^2)
    protect_dim: int  # 保护子空间的秩
    n_coefficients: int  # 拟合的复干扰系数个数
    residual_direct_gram: np.ndarray | None = None
    residual_noise_eigenvalues: np.ndarray | None = None
    matched_stat: float = 0.0  # 信念偏移处弱目标的统计量
    candidates: Tuple[int, ...] = ()
    # stage-2 支撑集的出处，是**记录**下来的而不是推断出来的。
    # ``gate_targets`` 是 stage-1 残差上的能量检验本会选中的目标；
    # ``supported_targets`` 是联合拟合真正建模了的目标。两者都报出来，才能让
    # **声明式**支撑规则与**统计式**支撑规则的差别出现在结果文件里，而不只是
    # 停留在 docstring 中 —— 也正是这项测量显示：旧门是在"保护自己放进去的
    # 那部分能量"上触发的。
    gate_targets: Tuple[int, ...] = ()
    supported_targets: Tuple[int, ...] = ()
    # **被测目标自己**的回波存活率 ``||s_q - f(s_q)||^2 / ||s_q||^2``。
    #
    # 上面的 ``eta_survive`` 是整场平均，**不能**替代这一个：它会随联合阶段
    # 恰好建模了多少**其它**目标而移动，于是即使保护预算没动，改一下 stage-2
    # 支撑集也会让它变。在 600 m 场景、用修正后的回波生成器实测，
    # ``tp_uic_full`` 读 0.705 而 ``plain_ls`` 读 0.700 —— 这 0.5 个点的差别对
    # 弱目标什么都没说明，因为联合支撑里只有三个受保护目标，而平均是在十条
    # 回波上做的。Q1（“TP-UIC 是否更少伤害弱目标”）问的是 ``s_q``，所以直接报。
    # 观测中无真值可隔离时为 ``0.0``。
    eta_survive_q: float = 0.0
    # 信念侧对被测目标的期望存活率。它由目标字典的物理中心列与该臂的线性
    # 削减算子构成；与 ``eta_survive_q`` 不同，它不使用任何真值回波。
    eta_survive_pred_q: float = 1.0
    eta_survive_risk_q: float = 1.0
    s_q_energy: float = 0.0  # ||s_q||^2，便于调用方给比值加权
    soft_mu: float | None = None  # 被选中的软惩罚乘子（若适用）

    @property
    def kappa_db(self) -> float:
        """实测对消深度，即有效的 ``kappa_dc``。

        只在**干扰**残差上定义，而把它拆开正是全部要点：
        ``model.compute_link_tables`` 把噪声地板与残余直达场放进同一个分母，
        但作为**分开的两项**，所以常数 ``kappa_dc`` 从来不是描述噪声的。
        因此 ``I_res = ||x - f(x)||^2 + ||f(n)||^2``；取哪一项由
        ``cancellation.residual_accounting`` 决定（默认两项都记）。

        ⚠️ 2026-09-21 实测推翻了这里原先的陈述「两者正交，所以这个和是精确的」：
        实测正交性 **0.356**（不正交）；且 ``||f(n)||^2`` 只占 ``||n||^2`` 的
        **0.12%**（残差保留 99.93%）⇒ 它是**被减掉的噪声**，不是残余干扰，
        却占 ``I_res`` 的 **99.8%**（estimation/structural = 506）。
        于是 ``structural`` 口径读的是**对消能力**（实测 65 dB > 需求
        52.25 dB），而默认口径读的是含噪声增强的合计（38 dB）。
        ⚠️ ``structural`` 口径须配 ``direct_estimation_sigma_* > 0``：
        delta=0 时字典完备把结构残差压成 0，那个 68 dB 是**上界**，不是可达值。
        """
        if self.i_res <= 0.0 or self.i_in <= 0.0:
            return float("inf")
        return 10.0 * math.log10(self.i_in / self.i_res)

    @property
    def kappa_pred_db(self) -> float:
        if self.i_res_pred <= 0.0 or self.i_in <= 0.0:
            return float("inf")
        return 10.0 * math.log10(self.i_in / self.i_res_pred)

    @property
    def calibration_error_db(self) -> float:
        """|实测 - 预测| 的深度差，即残差协方差自检。"""
        if self.i_res <= 0.0 or self.i_res_pred <= 0.0:
            return float("nan")
        return abs(10.0 * math.log10(self.i_res_pred / self.i_res))
