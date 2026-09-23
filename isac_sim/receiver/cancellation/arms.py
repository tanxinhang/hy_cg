"""cancellation 积木：对消臂的装配与打分。"""

from __future__ import annotations

from typing import Dict

import numpy as np

from isac_sim.core.config import Config
from isac_sim.receiver.cancellation.arm import _Arm
from isac_sim.receiver.cancellation.arm_ctx import ArmContext
from isac_sim.receiver.cancellation.arms_adaptive import build_adaptive_arm
from isac_sim.receiver.cancellation.arms_basic import build_basic_arms
from isac_sim.receiver.cancellation.arms_score import score_arms
from isac_sim.receiver.cancellation.containers import Observation
from isac_sim.receiver.cancellation.result import CancellationResult


def cancellation_arms(
    cfg: Config,
    obs: Observation,
    *,
    weak_target: int | None = None,
    threshold: float = 0.0,
    candidate_policy: str = "protected_only",
    only: str | None = None,
) -> Dict[str, CancellationResult]:
    """在一个 trial 上跑完第一个 TP-UIC 实验的所有对比臂。

    这些臂就是方法笔记里列的那些，每个只隔离**一个**机制，因此两臂之差是可归因的：

    ``no_ic``           原始观测（下界）
    ``plain_ls``        全网格上的无保护最小二乘
    ``ridge_ls``        无保护的正则化（MAP）最小二乘
    ``protected_ls``    目标保护 + 普通 LS（无先验）
    ``tp_uic_stage1``   目标保护 + MAP + 不确定性报告
    ``tp_uic_full``     stage 1 之后再做式 (4) 的联合精化
    ``perfect_channel`` oracle：精确减掉真值直达路径

    ``threshold`` 是接受一个目标候选进入联合阶段的 CFAR 电平；请传入经验标定过的
    H0 电平，这样候选集永远不是一个 oracle。

    ``candidate_policy`` 决定 stage 2 的联合支撑集怎么选，而 V1/V1.1 的那个门
    是错的 —— 只有追问"这个统计量**是什么意思**"时才看得出来：

    ``"protected_only"``（默认）让支撑集就等于保护集本身：stage 2 只建模接收机
    自己的信念已经宣布为"承载目标"的那些回波，不多不少。这是一条**声明出来的
    设计规则**，不是一个检验，这正是要点。旧规则拿
    ``||U_q^H r1||^2 / rank(U_q)`` 去和门限比，但不变式 ``P r1 = P y`` 说明
    ``r1`` 里被保护的那部分就是 ``y`` 里被保护的那部分、原封未动 —— 它在**两个**
    假设下都按构造保留了直达场。一个电平被假设支配的检验是无法标定的，所以那个
    "门"是一个伪装成决策的常数：它在每个 trial 上都选中每个受保护目标，而它
    打印出来的门限与虚警率毫无关系。把规则声明出来，就去掉了一个无法解释的
    统计量，同时没有改变这一臂的行为。

    ``"statistic"`` 保留旧的门限检验，存在的唯一目的是让消融能跑、差值能测。
    V1 与 V1.1 的头条表格就是用它生产的，所以基于它的结果与基于默认值的结果
    不可比。``gate_targets`` 无论策略是否采用它，都会报告那个检验选了什么。

    第三种变体 —— 门**减去**受保护目标 —— 是故意**不提供**的：在单目标场景里
    每个目标都受保护，于是 stage 2 的支撑集为空，``tp_uic_full`` 塌回
    ``tp_uic_stage1``。一个会删掉机制的规则不是规则；移除联合阶段的那次消融
    已经由 ``tp_uic_stage1`` 这一臂承担了。
    """
    if candidate_policy not in ("protected_only", "statistic"):
        raise ValueError(
            "candidate_policy must be 'protected_only' or 'statistic', got %r"
            % (candidate_policy,)
        )
    ctx = ArmContext(
        cfg, obs, weak_target=weak_target, threshold=threshold,
        candidate_policy=candidate_policy,
    )
    parts: Dict[str, _Arm] = {}
    gate, supported = build_basic_arms(ctx, parts, only=only)
    if only is None or only == "adaptive_soft_tpuic":
        soft_belief = ctx.weak_block if ctx.weak_block is not None else ctx.belief
        build_adaptive_arm(ctx, parts, soft_belief)
    if only is None:
        zero = np.zeros(ctx.n_bins, dtype=complex)
        parts["perfect_channel"] = _Arm(
            "perfect_channel", ctx.x.copy(), zero.copy(), zero.copy(),
            obs.h_true, np.zeros(ctx.X.shape[1]), ctx.empty, predict="exact",
        )
    return score_arms(ctx, parts, tuple(gate), tuple(supported))
