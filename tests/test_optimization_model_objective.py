"""优化模型一致性门禁（2/4）：目标函数形状 —— max-min，而不是加权和。

对应提案 §5、§6、§17。
"""
from __future__ import annotations

import numpy as np
import pytest

from isac_sim.core.config import Config, apply_preset
from isac_sim.core.config.presets.successors import HEADLINE_RELEASE_PRESET
from isac_sim.detection.fusion import (
    leximin_potential,
    selection_utility_from_pd,
    target_alpha,
    worst_case_pd,
)
from isac_sim.sensing.model import pd_from_deflection

from _optmodel_common import make_cfg

# ==========================================================================
# §5 / §6 / §17  目标函数形状：max-min，而不是加权和
# ==========================================================================

class TestObjectiveShape:
    """提案 §5、§6、§17：小 RCS 多目标问题应当盯住最弱目标。"""

    def test_objective_prefers_a_balanced_operating_point(self):
        """转移原则：把 P_D 从弱目标挪给强目标，目标函数必须变差。"""
        cfg = make_cfg()
        D = np.zeros(int(cfg.scale.Q))
        unbalanced = np.array([0.90, 0.50, 0.50])
        balanced = np.full(3, float(np.mean(unbalanced)))

        assert selection_utility_from_pd(cfg, D, balanced) > selection_utility_from_pd(
            cfg, D, unbalanced
        ), "目标函数不满足 Pigou–Dalton 转移原则，弱目标会被继续牺牲"

    def test_maxmin_is_gated_off_by_default(self):
        """门控默认关闭 ⇒ 发布口径仍是 softmin 势，逐位不变。"""
        cfg = make_cfg()
        assert cfg.selector.maxmin_objective is False

        # 手算 softmin 势，钉住"没被静默改掉"。
        D = np.array([1.0, 4.0, 25.0])
        pd = pd_from_deflection(cfg, D)
        req = float(cfg.detect.pd_required)
        eff = np.minimum(pd, req)
        tau = float(cfg.selector.softmin_tau)
        z = -eff / tau
        softmin = -float(cfg.scale.Q) * tau * (z.max() + np.log(np.exp(z - z.max()).sum()))
        deficit = np.maximum(req - pd, 0.0)
        expected = softmin - float(cfg.selector.mu_deficit) * float(
            np.sum(deficit * deficit)
        ) / (2.0 * req)
        assert selection_utility_from_pd(cfg, D, pd) == pytest.approx(expected, abs=1e-12)

    def test_the_worst_case_quantity_is_the_exact_minimum(self):
        """§5/(P1)：被接受的 ``t`` 是精确 ``min_q P_D,q``，不含任何平滑。"""
        cfg = make_cfg(**{"selector.maxmin_objective": True})
        D = np.zeros(int(cfg.scale.Q))
        weak_better = np.array([0.70, 0.70, 0.70])
        weak_worse = np.array([0.99, 0.99, 0.65])

        assert worst_case_pd(cfg, weak_better) == pytest.approx(0.70)
        assert worst_case_pd(cfg, weak_worse) == pytest.approx(0.65)
        assert leximin_potential(cfg, weak_better, 0.0) == pytest.approx(0.70), (
            "rho=0 时 leximin 必须退化为精确的 min"
        )

    def test_the_potential_prefers_a_balanced_operating_point(self):
        """转移原则：最弱目标更好的方案必须胜出，无论强目标多好。"""
        cfg = make_cfg(**{"selector.maxmin_objective": True})
        D = np.zeros(int(cfg.scale.Q))
        weak_better = np.array([0.70, 0.70, 0.70])
        weak_worse = np.array([0.99, 0.99, 0.65])

        # 实测对比：默认 softmin 口径下 U(均衡)=1.672 < U(不均衡)=1.874（反向），
        # leximin 下 0.777 > 0.759 —— 方向被纠正。
        assert selection_utility_from_pd(cfg, D, weak_better) > selection_utility_from_pd(
            cfg, D, weak_worse
        )

    def test_the_potential_is_schur_concave_not_just_minimising(self):
        """势必须 Schur-凹：把 P_D 从弱的挪给强的必然变差（不只盯最小值）。"""
        cfg = make_cfg(**{"selector.maxmin_objective": True})
        rho = float(cfg.selector.leximin_rho)
        balanced = np.array([0.60, 0.60, 0.60])
        # 一次"反向转移"：把最弱目标的资源挪给最强目标
        transfer = np.array([0.62, 0.62, 0.56])
        assert leximin_potential(cfg, balanced, rho) > leximin_potential(
            cfg, transfer, rho
        )

    def test_the_marginal_is_largest_for_the_weakest_target(self):
        """边际必须按"第几弱"递减 —— 这是"盯住最弱目标"的可执行形式。"""
        cfg = make_cfg(**{"selector.maxmin_objective": True})
        D = np.array([1.0, 4.0, 25.0])
        pd = pd_from_deflection(cfg, D)
        alpha = target_alpha(cfg, D)
        order = np.argsort(pd)  # 由弱到强

        # 纯 min（rho=0）只让最弱目标带动；leximin 让次弱也带动，但量级递减。
        assert alpha[order[0]] > alpha[order[1]] > alpha[order[2]], (
            "边际没有按弱者优先排序"
        )
        tied = target_alpha(cfg, np.array([4.0, 4.0, 4.0]))
        assert tied[0] == pytest.approx(tied[2]), "平局时不得偏向字典序靠前的目标"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "残留缺口 §5：max-min 能力已落地（selector.maxmin_objective），但头条口径 "
            "target-local-v1 尚未启用它，默认仍是 softmin(tau=0.10) + 二次 deficit 罚。"
            "把它打开 = 换目标函数形状 = 新假设，必须先跑一次主场景对照再决定。"
        ),
    )
    def test_headline_preset_enables_maxmin(self):
        cfg = apply_preset(Config(), HEADLINE_RELEASE_PRESET)
        assert cfg.selector.maxmin_objective is True

    def test_hard_budget_successor_prices_nothing_into_the_objective(self):
        """§6：预算口径里，通信/时延应当是约束，不是目标里的价格项。"""
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        assert cfg.selector.lambda_c == 0.0
        assert cfg.selector.use_delay_price is False

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "缺口 §6：当前头条口径 target-local-v1 仍带着 selector.lambda_c=0.005 的"
            "时延价格（use_delay_price=True），也就是把通信开销线性折算进检测效用。"
            "这正是提案要求回避的「为什么 lambda=0.005」型审稿问题。"
        ),
    )
    def test_headline_objective_carries_no_scalar_price(self):
        cfg = apply_preset(Config(), HEADLINE_RELEASE_PRESET)
        assert cfg.selector.lambda_c == 0.0
        assert cfg.selector.use_delay_price is False


# ==========================================================================
# §7 – §13  约束 C1 – C8
# ==========================================================================
