"""软融合与链路相关性模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fusion:
    """一个目标的软统计量在哪里汇合。

    感知链路是 ``i -> q -> j``：UAV ``i`` 照射，UAV ``j`` 收到回波并**产生**
    软统计量。随后这个统计量必须被报送到某处才能融合。

    * ``mode="tx"``（旧口径）：统计量报回感知发起者 ``i``。这让上报方向变成
      ``j -> i``，并悄悄把"感知对"当成"上报对"复用 —— 正是审稿人反对的那个
      建模捷径。
    * ``mode="explicit"``：每个目标 ``q`` 被显式指定一个融合 UAV ``f_q``，
      所有统计量按 ``j -> f_q`` 上报。感知对与上报关系完全解耦，上报的可靠度 /
      速率 / 时延都是 ``(j, f_q)`` 的。

    ``mode="explicit"`` 时由 ``rule`` 选融合 UAV：
    ``"max_in_rate"``（来自候选接收机的总入速率最大）、``"max_min_rate"``
    （最大最小公平）、``"nearest_target"``（离预测目标位置最近）或
    ``"nearest_target_capacitated"``（在与所提外层问题相同的逐 UAV 目标容量下
    做最小距离分配）。``"nearest_centroid"`` 作为 ``"nearest_target"`` 的向后
    兼容别名保留。``"capacitated_value"`` 在"逐 UAV 目标数硬容量"约束下最大化
    一个检测质量代理量。容量是系统预算，不是标量化权重。
    """

    mode: str = "tx"
    rule: str = "max_in_rate"
    # 一个融合 UAV 最多分到多少个目标。``-1`` 表示不限，并保持所有已发布的
    # V1 配置。
    max_targets_per_uav: int = -1
    # 同构的逐 UAV CPU 模型。非负速率会激活预算 C_f = F_f * T_proc；
    # 取负值则让已发布配置保持不限。
    cpu_rate_cycles_per_s: float = -1.0
    processing_window_s: float = 0.01
    cpu_fixed_cycles: float = 0.0
    cpu_per_observation_cycles: float = 1.0
    cpu_cubic_cycles: float = 0.0


@dataclass
class Corr:
    """融合软统计量的观测相关性模型。

    旧的复合偏转假设 ``a != b`` 时 ``Cov(s_a, s_b) = 0``，而多 UAV 双站网络
    显然违反这一点：各对共享发射端、接收端、目标的 RCS 起伏，还可能共享相邻的
    时延-多普勒支撑。打开相关模型后，融合用

        D_q = delta^T Sigma^{-1} delta,      w* propto Sigma^{-1} delta

    其中 ``Sigma = S R S``，``R`` 由**因子模型**给出，即非对角为
    ``R = rho_tx * 1{i = i'} + rho_rx * 1{j = j'} + rho_target
    + rho_dd * 1{相邻 DD 支撑}``，对角为 1。
    因为是因子模型，只要各 ``rho`` 非负且和不超过 1，结果按构造就是半正定的 ——
    剩下的份额属于特质（独立）部分。
    """

    enable: bool = False
    rho_tx: float = 0.15
    rho_rx: float = 0.15
    rho_target: float = 0.10
    rho_dd: float = 0.10
