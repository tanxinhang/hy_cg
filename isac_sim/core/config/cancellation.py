"""接收端干扰消除（TP-UIC）配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Cancellation:
    """接收端直连干扰消除（TP-UIC V1）。

    ``interference.direct_cancellation_db`` 是一个**常数**：它断言感知接收机
    从聚合直连场里消掉固定的 40 dB，不论几何、参考预算或目标状态如何。
    拿生产链路表去量就会发现，那个常数是一个**需求**（见
    ``KAPPA_DERIVATION.md``），而不是对某个接收机的描述。

    本段用可执行估计器的输出替换该常数：保目标、含不确定性的干扰消除
    （TP-UIC）。接收机用**已知的**协同波形重建每个在发照射机的直连贡献，
    但只在"与局部目标流形正交"的子空间里拟合，使干扰估计无法吞掉弱目标。
    剩下的干扰随后被分成两部分：

      * 因为投影到目标子空间而被**故意保留**的干扰（保护的代价），
      * 由后验协方差描述的估计不确定性残余。

    因此实际达到的对消深度是一个**输出**：同一套算法在一套几何上读到 40 dB、
    在另一套上读到 20 dB，并同时给出自己的目标存活比。

    ``enable`` 默认 ``False``：门关着时每个已发布数字都逐位不变，与
    ``coordination.enable`` 完全一样。本段除
    :mod:`isac_sim.receiver.cancellation` 之外无人读取。
    """

    enable: bool = False
    # ``model.compute_link_tables`` 如何形成残余直连场（也就是"接收机消掉了
    # 多少直连"那一项）：
    #
    # ``"off"``     **没有对消**：直连场全额存活（``kappa_dc = 1``）。默认值。
    #               这不是"默认 40 dB" —— 那个常数已被删除，因为它撑着 SINR 的
    #               分母却没有任何接收机实现支撑它。要拿到对消必须提供实测残余。
    # ``"predict"`` 解析桥 :func:`cancellation.predict_cancellation`，逐接收机：
    #               维度比例留存 + 参考预算的估计项。它**偏乐观**
    #               （600 m 场景实测差 5 倍），存在的意义是把链路端到端闭合，
    #               不是用来报结果。
    # ``"measure"`` 会被 ``compute_link_tables`` 拒绝。实测残余需要本 trial 的
    #               几何量，而那个函数拿不到；用
    #               ``cancellation.measure_residual_fraction(cfg, geom, base)``
    #               建出数组，再以 ``residual_fraction_by_receiver`` 传给
    #               ``compute_link_tables``。调度器被允许消费的是**那个实测
    #               数组**，不是本字段。
    mode: str = "off"
    # --- 参考预算 ---------------------------------------------------------
    # 用于估计直连信道时相干积累的 CPI 数。⚠️ 实测（2026-09-21）：对消臂
    # ``tp_uic_full`` 的**实测**深度对本键逐位不敏感 —— 它目前只进
    # ``cancellation_glrt`` 的 C_res 与解析桥 ``predict``。别拿它当深度杠杆。
    n_cpi: int = 1
    # --- 干扰字典 ---------------------------------------------------------
    # 每个在发照射机的分数 DD 切向列数。默认 0 = 假设直连时延/多普勒由共享位置算出。
    # 加切向列是在建模残余失配，代价是可测的深度损失 —— 当鲁棒性轴用，不要当基线。
    interference_tangent_order: int = 0
    # --- 目标保护 ---------------------------------------------------------
    protect_targets: bool = True
    # 一个接收机保护多少个目标。
    #
    # 这是承重的设计选择，第一次 TP-UIC 实验是硬碰硬发现的：保护**每一个**被
    # 相信的回波，等于取 140 个切空间（15 UAV × 10 目标）的并，它在 600 m 场景
    # 上的秩是 4096 个格里的 195 —— 这个子空间被证明**完整包含**了 14 维直连
    # 子空间（每个照射机的核都有 ≥99.9% 落在里面），于是第一级对消塌到零深度。
    # 保护一个目标要花掉干扰学习空间，代价在被保护目标数上是凸的。
    #
    # 默认保护本接收机上"回波/总场比"最低的 ``max_protected_targets`` 个回波 ——
    # 恰好就是本方法要救的弱目标。设为 ``0`` 表示保护全部（记录那次塌陷的消融）。
    max_protected_targets: int = 3
    # 0 = 只保护预测的 DD 中心；1 = 同时保护一阶导数 d/df_delay 与 d/df_doppler，
    # 即目标流形的切空间。1 才是有意义的取值：目标不会正好落在预测格上，
    # 只保护中心窗口等于没有对"响应离开中心有多快"作任何陈述。
    tangent_order: int = 1
    # 切向列的有限差分步长，以 DD 格为单位。
    tangent_step_bins: float = 0.05
    # 实验性的 belief 协方差保护。启用后，硬保护基除中心切空间外，还包含确定性
    # 的 95% 边缘 DD/方位 sigma 点。默认关闭，好让已发布的接收机逐位不变。
    covariance_protection: bool = False
    # --- 估计器 -----------------------------------------------------------
    # 复直连系数的信道先验方差（单位功率莱斯 LOS 分量）。``None`` 关闭先验，
    # 退化为普通（带保护的）最小二乘。
    prior_variance: float | None = 1.0
    # 软目标保护权重。零就是完全不加保护的岭回归 LS；越大则越重地惩罚"重建出的
    # 直连干扰落在被相信的目标子空间里"。实验臂 ``soft_tpuic`` 在 n_cpi=1 下用它。
    soft_protection_mu: float = 1.0
    # 只用 belief 的自适应软保护。每个乘子都要与硬 TP-UIC 臂对照预测残余与风险
    # 存活率做筛选。默认关闭，因为每个网格点都要再解一次。
    adaptive_soft_enable: bool = False
    adaptive_soft_mu_grid: Tuple[float, ...] = (
        0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0,
    )
    adaptive_soft_risk_slack: float = 0.0
    adaptive_soft_residual_quantile: float = 0.80
    # --- 检测器标定 -------------------------------------------------------
    # 把 **belief 误差**记到残余协方差 ``C_res`` 上。
    #
    # 每个回波模板都在**被相信的**目标状态上构造，而回波来自真实状态，所以
    # 每个模板一阶地偏了 ``delta_l * da/dl + delta_k * da/dk``，其中
    # ``delta_l`` / ``delta_k`` 是用 DD 格表示的位置与速度误差。于是回波会穿过
    # 模板漏进残余 —— 这正是把实测 ``P_FA`` 抬到标称值以上的原因。
    #
    # 这**不是**上面的系数先验：先验描述的是接收机对所**拟合**干扰幅度的不确定
    # 性；本项描述的则是"模板**无法**被精确放置"这一几何事实。把两者混为一谈，
    # 会让检测器恰好按正在被测量的那个量失准。
    #
    # 默认关闭，好让每个已发布数字保持逐位不变；这个开关存在的意义是让收益
    # （标定好的 ``P_FA``）与代价（略微膨胀的 ``C_res``，因而损失一点 ``P_D``）
    # 都能被量出来，而不是被断言。
    belief_error_in_cres: bool = False
    # --- 第二阶段 ---------------------------------------------------------
    joint_refine: bool = True
    # 形成检测统计量时用的局部 DD 搜索半宽。
    search_half_width: int = 1
    # --- 记账 -------------------------------------------------------------
    hw_ceiling_db: float = 60.0
    # --- 生产接线（实验性） -----------------------------------------------
    # 把 TP-UIC 的**实测**证书接进生产链路：每个 trial 测量一次接收机，把
    # (残余比例, 回波存活率) 广播给该 trial 内所有 ``compute_link_tables``
    # 调用点，取代"没有对消"的默认（``kappa_dc = 1``）。
    #
    # 默认关闭 ⇒ 每个已发布数字逐位不变。必须默认关闭有两层理由：
    #   * 语义层：接上后分母不再由假设给出，发布基线整体下移 —— 实测对消约
    #     37 dB < 常数 40 dB，且分子从此要记回波存活率（半闭环不再高估）。
    #   * 算力层：一次测量约 40 s/trial，比不测量的 trial 贵两个数量级。
    production_wire: bool = False
    # 用哪个估计臂测量；``tp_uic_full`` = 保目标的最小二乘（方法本体）。
    production_wire_arm: str = "tp_uic_full"
    # 分子桥口径：``"q"`` = 被测目标自己的存活率（逐目标 GLRT 的自然口径）；
    # ``"field"`` = 整场口径，更保守但随支撑集大小漂移。
    production_wire_retention: str = "q"
    # 这是工程提速，不是算法改动 —— 剪掉的是没有被读取的量，被测臂的数值必须
    # 逐位不变（由 tests/test_tpuic_production_wire.py 钉住）。
    # 若哪天实测不再逐位不变，这个键必须立刻改回 False 并当作 bug 处理。
    measure_prune_arms: bool = True
    # --- 直连参数估计误差（实验性） --------------------------------------
    # 干扰字典每条直连路径的 DD 偏移相对真值的估计误差标准差（DD 格）。默认 0.0 = 完美估计，
    # 即此前只在注释里声称、却让 span(X) 完整包含直连场并把结构残差压成 0 的假设；打开后字典用带
    # 误差的 bin、直连场仍用真 bin ⇒ kappa 由数据决定（逐位不变由 tests/test_direct_estimation_error.py 钉住）。
    direct_estimation_sigma_delay_bins: float = 0.0
    direct_estimation_sigma_doppler_bins: float = 0.0
    # --- 残余干扰记账口径 -------------------------------------------------
    # ``measured``（默认）：i_res = ||x-f(x)||^2 + ||f(n)||^2，逐位不变。⚠️ 第二项是
    # **被减掉的**噪声（占 ||n||^2 的 0.12%），却被记成残余干扰并占 i_res 的 99.8%。
    # ``structural``：只算 ||x-f(x)||^2；⚠️ 须配 delta>0，delta=0 给出的只是上界。
    residual_accounting: str = "measured"
