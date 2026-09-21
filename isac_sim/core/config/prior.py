"""目标先验与调度器可见的 belief 口径。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prior:
    """目标状态预测的不确定性模型。

    当 ``sigma_pos_m > 0`` 或 ``sigma_vel_mps > 0`` 时，``prior-sweep`` 实验会把
    每个 MC trial 的目标状态按给定额度的独立高斯抽样扰动。默认值让两个 sigma
    都为零，从而模拟器使用**真实的**目标状态，与原始行为一致。

    ``dt_s`` 是请求常速预测时 :func:`isac_sim.scenario.prior.predicted_geometry`
    用的前视时间。
    """

    sigma_pos_m: float = 0.0
    sigma_vel_mps: float = 0.0
    dt_s: float = 0.1
    # --- 真值 vs belief ---------------------------------------------------
    # ``belief_mode=False``（旧口径）：把**被扰动的**目标状态写回几何，于是调度器
    # 与物理世界共用一个状态。它回答的是"跟踪器错了 P_D 会掉多少"，但那**不是**
    # 审稿人的问题 —— 他们问的是"一个只能**看到** belief 的调度器，面对真目标
    # 表现如何"。
    #
    # ``belief_mode=True``：同时携带两个状态，
    #     真值  x_{q,t}        -> 回波生成、DD、感知增益、检测器
    #     belief b_q = N(xhat, P) -> 链路选择、DD 搜索窗、预算
    # 这才是诚实的 ``predict -> schedule -> sense -> update`` 回路。
    belief_mode: bool = False
    # 用来构造 (xhat, P) 的 **belief 误差**标准差。belief 是"被这些 sigma 污染过的
    # 真值"，所以 sigma = 0 退化为完美跟踪器。
    belief_sigma_pos_m: float = 150.0
    belief_sigma_vel_mps: float = 15.0
    # 可选的鲁棒 detector-PD 选择器所用的水平高斯位置误差球的置信质量。
    # 对二维各向同性 belief，r = sigma * sqrt(-2 log(1-confidence))。
    robust_position_confidence: float = 0.50
    # 调度器（链路选择 / 融合节点分配 / 预算）是否只看**鲁棒化**后的增益视野：
    # 把 ``geometry_robust_base``（belief 位置误差球上的双基增益下界）喂给它。
    # 因为 ``target_gain ∝ 1/(d_i^2 d_j^2)`` 是凸的，点估计对"目标可能在别处"
    # 系统性乐观 —— 这个门控让调度器按悲观值定价。
    #
    # 默认 ``False`` ⇒ 已发布数字逐位不变（接线点只在 belief 分支的
    # ``base_belief`` 上，检测侧 ``base_truth`` 不受影响，由
    # ``tests/test_direction2_robust_gate.py`` 钉住）。它是"接线已有实现"，
    # 不是新算法；打开它会改调度决策，因此属于 B 类变更，需显式批准。
    robust_geometry_for_scheduler: bool = False
    # 由预测协方差导出的椭球 DD 搜索门：当某条被选链路时延与多普勒的残差落在
    # 这么多倍标准差之内（再加半个量化格）时，认为它捕获到了真值。
    search_gate_sigma: float = 3.0
    # 调度器对目标 RCS 知道什么。``"realized"`` 是旧的类 oracle 路径；
    # ``"mean"`` 用 E[sigma_q]，避免本 CPI 的 RCS 实现值泄漏进感知前的调度决策。
    scheduler_rcs: str = "realized"
    # 目标类 RCS 不确定区间的下端点，表示为 ``detect.target_rcs`` 的一个比例。
    # RCS 鲁棒束方法按这个端点给每一列定价。它是认识论意义上的设计界，
    # 既不是本 CPI 的 RCS 观测，也不是可调的目标函数权重。
    rcs_lower_factor: float = 0.5
