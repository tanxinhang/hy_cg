"""链路选择规则的参数与 DD 细化开关。"""

from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------
# 算法
# --------------------------------------------------------------------------
@dataclass
class Refine:
    """DD 格的由粗到细（C2F）细化（论文表 I：W、kappa_dd、eta_min、L_short）。
    默认关闭，好让现有数值行为不变。

    ``window_kernel`` 选择局部窗口能量怎么算：

    * ``"dirichlet"`` —— 周期性 Dirichlet 泄漏（默认）。这是忠于物理的选择，
      也是同源的 ``gate_otfs_collision`` 包用的那一个。
    * ``"sinc"``      —— 解析 sinc 窗；当消融用。

    ``apply_to_all`` 关掉候选短名单筛选，让每条可行链路都被细化。这就是论文里
    用来量化 ``41.1%`` 细格求值节省的**全量局部细化**对照。
    """

    enable: bool = False
    half_width: int = 1
    kappa_dd: float = 0.45
    eta_min: float = 0.25
    shortlist_size: int = 25
    window_kernel: str = "dirichlet"
    apply_to_all: bool = False
    # --- 细化模型 ---------------------------------------------------------
    # "interp":  eta^f = min{1, max[eta_min, eta^c + kappa_dd (eta^loc - eta^c)]}。
    #            两个手工设定的参数（kappa_dd、eta_min）没有任何波形解释 ——
    #            正是审稿人指出的启发式。为可比性保留为默认。
    # "window":  细估计器把分数时延-多普勒偏移解出来，因而**恢复了**局部窗口
    #            能量，于是 eta^f = eta^loc。没有自由参数；C2F 这才成为真正的
    #            计算加速机制（便宜的主格粗筛 → 基于波形的局部细化），
    #            而不是"把所选链路的感知增益抬高"的手段。
    mode: str = "interp"

@dataclass
class Selector:
    """以检测器边际为准的观测选择及其行为开关。

    已发布的 V1 用标量上报价格。V1.1 后继版关掉该价格，改由下面的硬资源预算
    来定义可行性。

    四个 ``use_*`` 开关**不是**调节旋钮：每一个恰好关掉一个机制，而且只会被
    消融变体翻转。它们放在这里（而不是物理分组里），因为它们改的是算法，
    不是场景。
    """

    # 边际价值价格：score = alpha_q * dD - lambda_c * delay_ms。
    lambda_c: float = 0.005
    mu_deficit: float = 1.0
    use_softmin_alpha: bool = True
    softmin_tau: float = 0.10
    alpha_floor: float = 0.0
    alpha_cap: float = 10.0
    # ``first_order`` 复现历史上的 alpha_q * DeltaD 规则。
    # ``exact_utility`` 求的是真实的公平效用增量，是论文主口径；
    # 它的贪心与穷举 oracle 共用同一个目标函数。
    # ``detector_pd`` 求上报后的 H0/H1 矩与实现里那个 CF 校正门限，
    # 通信价格口径不变。
    score_mode: str = "first_order"
    # 历史上的 D_min 提前退出为复现旧结果而保留。主口径里它是关的，因为它可能
    # 在"所述效用仍有正的可行边际增益"的时候就停下来。
    stop_at_D_min: bool = True
    # 消融开关。
    use_target_priority: bool = True
    use_delay_price: bool = True
    use_comm_error_calibration: bool = True
    # --- 最差目标（max-min）目标 -------------------------------------------
    # 提案 (P0)/(P1) 的 epigraph 形式：把公平势换成
    #     U = Phi = min_q min(P_D,q, pd_required)
    # 而不是 softmin 平滑代理。实测 softmin（tau=0.10）会让
    # U([.70,.70,.70]) < U([.99,.99,.65]) —— 最弱目标更差的方案反而胜出，
    # 而 max-min 下 0.70 > 0.65，最弱目标优先。
    #
    # 与 ``use_softmin_alpha`` / ``mu_deficit`` 的二次 deficit 罚互斥：本开关
    # 打开时二者都不参与（软最小值不是 min，调温度永远调不成 min）。
    # 受 ``use_target_priority`` 管辖：显式关掉目标优先级时 max-min 也不生效。
    #
    # **默认关闭** ⇒ 冻结的发布路径逐位不变；它是一个新假设，不是调节旋钮。
    maxmin_objective: bool = False
    # leximin 势的几何权重 ``rho``：``U = sum_k rho^k P_D,(k)``（升序）。
    # ``rho=0`` 是精确的 max-min，但那样贪心会死锁（min 的边际恒为 0，
    # 实测选中链路数 12.4 -> 0）。``rho`` 越小越"只盯最弱目标"：
    # 最弱目标权重 1、次弱 rho、再次 rho^2。它**不是** softmin 的温度 ——
    # 这里的排序是**硬**的（按秩给权），不是按数值平滑。
    leximin_rho: float = 0.1
    # 实验性的结构约束：在分配远程上报之前，先为每个可服务目标放一条本地观测。
    # 这把"远程补充本地"编码进去，而不需要一个拟合出来的标量奖励。
    require_local_anchor: bool = False
    # 资源上限。
    max_links_per_target: int = 6
    max_total_links: int = 60
    # --- 协同辐射价格 -----------------------------------------------------
    # 感知观测期间每一个不同的在发 UAV 都会把自己的直连泄漏重新注入每个接收机。
    # 已发布的目标函数只最大化各目标的边际检测量，从不问自己唤醒了多少个节点，
    # 于是它在 3 个照射机就够的时候乐于把 25 条链路摊到 10–11 个照射机上
    # （500 m / RCS 0.2 实测：给节点数封顶把最差目标 P_D 从 0.205 抬到 ~0.79，
    # 连"随机挑 3 个节点再封顶"都优于原参照）。
    #
    # 当候选会引入一个**新的**在发节点时，``tx_penalty`` 以效用单位收这笔钱；
    # ``max_tx_nodes`` 是同一件事的硬上限对偶。UAV 间信令假设理想且瞬时
    # （见论文的假设清单）。两者默认都是"关"，所以冻结的发布路径保持逐位不变。
    tx_penalty: float = 0.0
    max_tx_nodes: int | None = None
    # 反事实对照：用来审计结果在多大程度上依赖"融合 UAV 上零上报成本的证据"。
    # ``-1`` 保持名义上的无限制策略，``0`` 禁止本地证据，正值则对每个目标
    # 分别限制其条数。
    max_local_observations_per_target: int = -1
    # 全系统 UAV 间上报的硬上限。它与观测上限相互独立，因为本地证据消耗感知/计算
    # 但不占上报时隙。``-1`` 表示名义选择器不受限。
    max_remote_reports: int = -1
    # 硬处理预算。每条被选中的 (i,j,q) 观测在 j 处占一个接收处理单元、
    # 在 f_q 处占一个融合处理单元。``-1`` 表示对应资源不限。
    max_observations_per_receiver: int = -1
    max_observations_per_fusion_uav: int = -1
    candidate_topk_per_target: int = 40
    min_marginal_D: float = 0.0
    # V1.2 目标—融合—束的列生成控制。
    bundle_shortlist_per_type: int = 4
    bundle_cg_max_iterations: int = 8
    bundle_pricing_tolerance: float = 1e-8
    bundle_exact_pricing_max_candidates: int = 10
