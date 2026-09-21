"""优化模型一致性门禁（4/4）：信息集、交替与接受、字典序次级目标。

对应提案 §15、§18、§19、§20。
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from isac_sim.detection.fusion import deflection_for_links
from isac_sim.detection.oracle import lexicographic_gap_components
from isac_sim.receiver import cancellation as cx
from isac_sim.sensing.model import pd_from_deflection

from _optmodel_common import make_cfg, make_world, make_sources
from experiments.coordination import _release_selector
from experiments.selection import select_lagrangian
from tools.run_joint_tpuic_coordination import (
    FORMAL_CAPABILITY_SOURCE,
    FORMAL_RETENTION_SOURCE,
    strict_joint_update,
)

# ==========================================================================
# §15  信息集限制：只用 belief
# ==========================================================================

class TestBeliefOnlyInformationSet:
    """提案 §15：决策必须对 belief σ-代数可测，不得含真值。"""

    def test_selector_consumes_no_truth_geometry(self):
        """选择器只拿 (cfg, base, tables, plan) —— 没有真值几何入口。"""
        params = set(inspect.signature(select_lagrangian).parameters)
        assert not params & {"geom", "geom_true", "truth", "oracle", "base_truth"}
        assert {"cfg", "base", "tables"} <= params

    def test_scheduler_certificates_are_declared_belief_only(self):
        """调度器消费的能力/存活证书必须是 predicted / predicted_risk。"""
        assert FORMAL_CAPABILITY_SOURCE == "predicted"
        assert FORMAL_RETENTION_SOURCE == "predicted_risk"

    def test_belief_is_stated_not_guessed(self):
        """信念是调用方给出的对象本身，不是重建出来的。"""
        cfg = make_cfg()
        geom, base, _tables, _plan = make_world(cfg)
        oracle_ctx = cx.ReceiverContext.from_trial(cfg, geom, geom, base)
        assert oracle_ctx.belief_is_truth is True

        perturbed_rng = np.random.default_rng([cfg.run.seed, 3])
        from isac_sim.scenario.prior import perturbed_geometry

        belief = perturbed_geometry(
            cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, perturbed_rng
        )
        ctx = cx.ReceiverContext.from_trial(cfg, geom, belief, base)
        assert ctx.geom_belief is belief
        assert ctx.belief_is_truth is False


# ==========================================================================
# §18 / §19  交替上升与接受原则
# ==========================================================================


def _state(cfg, mask, selected, value: float):
    """``strict_joint_update`` 只认这三个字段的最小状态对象。"""
    return SimpleNamespace(
        mask=np.asarray(mask, dtype=bool),
        selected=selected,
        belief_objective=float(value),
        measure_seconds=0.0,
    )


class TestAlternationAndAcceptance:
    """提案 §18/§19：交替更新 + 只有目标不下降才接受。"""

    def test_acceptance_rule_is_monotone_with_an_epsilon_gate(self):
        """§19：``Phi^{r+1} >= Phi^r + eps`` 才接受，否则回滚。"""
        cfg = make_cfg()
        m_rx = int(cfg.scale.M)
        # 初值刻意不等于 selected 的照射集，好让第一轮真的去评估候选
        mask0 = np.zeros(m_rx, dtype=bool)
        mask0[[0, 1, 2]] = True
        selected = {0: [(0, 1), (1, 2)], 1: [(1, 3)], 2: [(0, 4)]}
        start = _state(cfg, mask0, selected, 1.0)

        def make_evaluate(value: float):
            def evaluate(mask):
                return _state(cfg, mask, selected, value)

            return evaluate

        accepted, hist, _term = strict_joint_update(
            cfg, start, make_evaluate(1.5), rounds=3, epsilon=0.01
        )
        assert hist[0]["accepted"] is True
        assert accepted.belief_objective == pytest.approx(1.5)

        tiny, hist_tiny, term_tiny = strict_joint_update(
            cfg, start, make_evaluate(1.001), rounds=3, epsilon=0.01
        )
        assert hist_tiny[0]["accepted"] is False
        assert tiny is start, "增益不足时必须回滚，而不是留下一个更差的解"
        assert term_tiny == "strict_reject"

        worse, hist_worse, _ = strict_joint_update(
            cfg, start, make_evaluate(0.5), rounds=3, epsilon=0.01
        )
        assert hist_worse[0]["accepted"] is False
        assert worse.belief_objective == pytest.approx(1.0)

    def test_the_accepted_quantity_can_be_the_worst_target_probability(self):
        """§19：接受量可以是精确的最差目标 P_D（不含价格、不含平滑）。"""
        from audits.theory import worst_case_objective

        cfg = make_cfg(
            **{
                "selector.maxmin_objective": True,
                "selector.use_delay_price": False,
                "selector.lambda_c": 0.0,
            }
        )
        _geom, base, tables, plan = make_world(cfg)
        selected, _D = select_lagrangian(cfg, base, tables, plan)
        objective = float(worst_case_objective(cfg, tables, selected, plan, base))
        pds = np.array(
            [
                float(
                    pd_from_deflection(
                        cfg,
                        deflection_for_links(
                            cfg, tables, q, selected.get(q, []), plan=plan, base=base
                        ),
                    )
                )
                for q in range(int(cfg.scale.Q))
            ]
        )
        # 目标函数按 pd_required 封顶（见 fusion.maxmin 的 saturate 说明）。
        expected = float(np.min(np.minimum(pds, float(cfg.detect.pd_required))))
        assert objective == pytest.approx(expected, abs=1e-12)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "残留缺口 §19：能力已具备（selector.maxmin_objective 打开后 "
            "task_objective 就等于 Phi），但发布协议默认仍单调化 "
            "selection_utility - lambda_c*cost（audits/theory.py:task_objective）。"
            "切换前需要在主场景上跑一次「max-min 口径 vs 现状」的对照，"
            "确认最差目标 P_D 与总开销的变化方向。"
        ),
    )
    def test_headline_accepted_quantity_is_the_worst_target_probability(self):
        from audits.theory import task_objective

        cfg = make_cfg()
        _geom, base, tables, plan = make_world(cfg)
        selected, _D = select_lagrangian(cfg, base, tables, plan)
        objective = float(task_objective(cfg, tables, selected, plan, base))
        pds = np.array(
            [
                float(
                    pd_from_deflection(
                        cfg,
                        deflection_for_links(
                            cfg, tables, q, selected.get(q, []), plan=plan, base=base
                        ),
                    )
                )
                for q in range(int(cfg.scale.Q))
            ]
        )
        assert objective == pytest.approx(float(np.min(pds)))

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "缺口 §18 后半：交替只发生在 x 侧（照射掩码 -> 链路表 -> 重选，"
            "experiments/coordination.py）。按目标 deficit 更新保护集 z "
            "（delta_q = [P_D_req - P_D,q]_+，取 top-K_j^protect）尚未实现；"
            "发布选择器没有保护态入参，无法闭环。"
        ),
    )
    def test_protection_is_updated_by_target_deficit(self):
        params = set(inspect.signature(_release_selector).parameters)
        assert params & {"protected", "protection", "deficit", "z"}


# ==========================================================================
# §20  字典序次级目标
# ==========================================================================


# ==========================================================================
# §20  字典序次级目标
# ==========================================================================

class TestLexicographicSecondary:
    """提案 §20：同 worst-case P_D 时再比开销，而不是加权相减。"""

    def test_the_objective_vector_is_kept_component_wise(self):
        """四个分量必须分开报，不许预先塌成标量。"""
        from isac_sim.detection.oracle import lexicographic_objective_vector

        names = set(inspect.signature(lexicographic_objective_vector).parameters)
        assert {"cfg", "base", "tables", "plan", "selected"} <= names

        gaps = lexicographic_gap_components(
            {
                "worst_detection_deficit": 0.10,
                "total_detection_deficit": 0.30,
                "remote_reports": 2.0,
                "processing_load": 5.0,
            },
            {
                "worst_detection_deficit": 0.05,
                "total_detection_deficit": 0.30,
                "remote_reports": 2.0,
                "processing_load": 5.0,
            },
        )
        assert set(gaps) == {
            "delta_worst_detection_deficit",
            "delta_total_detection_deficit",
            "delta_remote_reports",
            "delta_processing_load",
            "exact_lexicographic_match",
        }
        assert gaps["exact_lexicographic_match"] == 0.0
