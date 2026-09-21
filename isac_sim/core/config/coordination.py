"""多机协同回合与主动感知（active sensing）配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Coordination:
    """**发布**选择路径上的 UAV 间辐射协同。

    已发布的选择器在给候选打分时用的感知分母里**每架** UAV 都在辐射，于是一个
    什么都不照的节点仍然把自己的直连泄漏注入每个接收机。协同把那个回路当作
    不动点来闭合::

        round 0: mask = None（人人都在辐射）        -> 选择
        round k: mask = 第 k-1 轮的照射机集合 -> 重建表 -> 重选

    这与 ``coordination.select_with_coordination`` 跑的是同一个映射，但接到了
    :func:`isac_sim.experiments.flow.simulate.run_method_on_trial` 上，好让**发布
    入口**（而不只是 ``tools/*``）能报它。实测收益很大，而且纯属协议：没有额外
    硬件增益。只在**发布**入口下引用它（600 m / RCS 0.1、G_hw = 0 dB、
    kappa = 40 dB、MC = 1000、``proposed_c2f_adaptive_pd`` +
    ``selector.max_tx_nodes = 3``）::

        P_D        0.6938 -> 0.8501
        worst P_D  0.6650 -> 0.8380     （配对、同 seed）

    ``tools/*`` 上更大的那个数（0.85 -> 0.95）是**另一套口径**，不能与上面并列
    引用 —— 见 ``COORDINATION_WIRING.md``。

    ``enable`` 默认 ``False``，好让冻结的发布路径逐位不变。打开它要求
    ``interference.sense_gate_by_active_tx``（这是**校验**出来的，不是假设）：
    没有门控时 ``compute_link_tables`` 会忽略掩码，协同就静默地成了空操作 ——
    正是这个失效模式让早先一个"协同"数字无法上报。

    ``rounds`` 是**预算**，不是承诺。掩码映射是确定性的但没有收敛性证明；
    在 6 个 seed 上实测 2–4 轮就稳定，而且移动的是**成员**而非**计数**。
    掩码重复时报为环，而不是悄悄把预算烧掉。
    """

    enable: bool = False
    rounds: int = 6

@dataclass
class ActiveSensing:
    """主动证据获取用的离散主动观测设计。

    每个模式联合声明一个感知功率乘子、一个独立 look 的绝对次数，以及是否使用
    细化的 DD 接收机。能量以归一化的 look-功率为单位。观测角定义的是一个有限、
    预先声明的集合；它们不是拿 trial 结果拟合出来的。
    """

    enable: bool = False
    mode_names: tuple[str, ...] = ("eco", "nominal", "intensive")
    power_scales: tuple[float, ...] = (0.5, 1.0, 1.25)
    looks: tuple[int, ...] = (8, 16, 32)
    refined: tuple[bool, ...] = (False, True, True)
    energy_budget_per_target: float = 64.0
    energy_budget_per_uav: float = 256.0
    max_tx_observations_per_uav: int = -1
    matched_filter_cycles_per_look: float = 1.0
    llr_cycles_per_look: float = 0.25
    dd_refine_cycles: float = 16.0
    max_candidates_per_pair: int = 8
    candidate_strategy: str = "scenario_union"
    scenario_topk_per_scenario: int = 1
    complementary_pair_topk: int = 2
    complete_pool_max_links: int = 12
    branch_node_limit: int = 200_000
    information_metric: str = "forward_kl"
    energy_price: float = 0.0
    report_price: float = 0.0
    aspect_enable: bool = True
    aspect_angles_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    aspect_floor: float = 0.20
