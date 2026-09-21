"""优化模型一致性门禁（3/4）：资源是约束，不是目标里的惩罚项。

对应提案 §7–§14（C1–C8、C9）。
"""
from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from isac_sim.cooperation.primitives import feasible_links_for_target
from isac_sim.cooperation.reporting import is_local_observation, report_dest
from isac_sim.detection.fusion import deflection_for_links
from isac_sim.receiver.cancellation.protection import protected_target_ids
from isac_sim.sensing.model import pd_from_deflection

from _optmodel_common import make_cfg, make_world, make_sources
from experiments.coordination import _release_selector
from experiments.selection import select_lagrangian

# ==========================================================================
# §7 – §13  约束 C1 – C8
# ==========================================================================

class TestResourceConstraints:
    """提案 §7–§13：资源是约束，不是目标里的惩罚项。"""

    def test_c1_per_target_observation_budget(self):
        """C1：``sum_ij x_ijq <= K_q^max``。"""
        cfg = make_cfg(**{"selector.max_links_per_target": 2})
        _geom, base, tables, plan = make_world(cfg)
        selected, _D = select_lagrangian(cfg, base, tables, plan)
        assert all(len(v) <= 2 for v in selected.values())

    @pytest.mark.xfail(
        strict=True,
        reason="缺口 §7：只有 K_q^max，没有 K_q^min（下界冗余）。提案建议 K_min 只在鲁棒性实验里用。",
    )
    def test_c1_lower_bound_is_available_for_robustness_runs(self):
        from isac_sim.core.config.selector import Selector

        assert "min_links_per_target" in {f.name for f in dataclasses.fields(Selector)}

    def test_c2_total_observation_budget(self):
        """C2：``sum_q sum_ij x_ijq <= K_tot``。"""
        cfg = make_cfg(**{"selector.max_total_links": 4})
        _geom, base, tables, plan = make_world(cfg)
        selected, _D = select_lagrangian(cfg, base, tables, plan)
        assert sum(len(v) for v in selected.values()) <= 4

    def test_c3_illuminator_budget(self):
        """C3：发射 UAV 数量封顶（提案建议用硬上限而非价格）。"""
        cfg = make_cfg(**{"selector.max_tx_nodes": 2})
        _geom, base, tables, plan = make_world(cfg)
        selected, _D = select_lagrangian(cfg, base, tables, plan)
        tx = {int(i) for links in selected.values() for (i, _j) in links}
        assert len(tx) <= 2

    def test_c4_receiver_and_fusion_processing_budgets(self):
        """C4：每个接收机 / 每个融合 UAV 的处理单元上限。"""
        cfg = make_cfg(
            **{
                "selector.max_observations_per_receiver": 2,
                "selector.max_observations_per_fusion_uav": 2,
            }
        )
        _geom, base, tables, plan = make_world(cfg)
        selected, _D = select_lagrangian(cfg, base, tables, plan)

        rx_count: dict[int, int] = {}
        fuse_count: dict[int, int] = {}
        for q, links in selected.items():
            for link in links:
                rx_count[int(link[1])] = rx_count.get(int(link[1]), 0) + 1
                dest = int(report_dest(plan, link, q))
                fuse_count[dest] = fuse_count.get(dest, 0) + 1
        assert max(rx_count.values(), default=0) <= 2
        assert max(fuse_count.values(), default=0) <= 2

    def test_c5_protection_budget(self):
        """C5：``sum_q z_jq <= K_j^protect`` —— TP-UIC 最关键的约束。"""
        for cap in (1, 2, 3):
            cfg = make_cfg(**{"cancellation.max_protected_targets": cap})
            ids = protected_target_ids(cfg, make_sources())
            assert len(ids) <= cap, f"cap={cap} 时保护了 {len(ids)} 个目标"

    def test_c5_zero_means_unconstrained_and_must_say_so(self):
        """C5 的口径陷阱：``0`` 是"保护全部"，不是"不保护任何人"。

        提案写的是 ``sum_q z_jq <= K``，按字面 K=0 应为空集；实现里 0 是
        无约束变体（保留给那次"保护一切导致对消塌到零深度"的消融）。这条断言
        把该约定钉住，防止有人按字面语义"修"掉它而静默改变发布数字。
        """
        cfg = make_cfg(**{"cancellation.max_protected_targets": 0})
        ids = protected_target_ids(cfg, make_sources())
        assert len(ids) == 5, "0 表示无约束保护；改成空集会静默改变已发布结果"

    def test_c7_c8_reporting_feasibility_and_budget(self):
        """C7/C8：上报可行性逐链路成立，且远程上报数受硬预算约束。"""
        cfg = make_cfg(**{"selector.max_remote_reports": 1})
        _geom, base, tables, plan = make_world(cfg)
        selected, _D = select_lagrangian(cfg, base, tables, plan)

        remote = sum(
            1
            for q, links in selected.items()
            for link in links
            if not is_local_observation(plan, link, q)
        )
        assert remote <= 1

        for q, links in selected.items():
            feasible = set(feasible_links_for_target(cfg, base, tables, q, plan))
            assert set(links) <= feasible, f"目标 {q} 选中了上报不可行的链路"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "缺口 §12/(C6')：selection 与 protection 之间没有耦合。发布选择器签名是 "
            "(cfg, base, tables, plan, mask) —— 没有保护态 z，也没有 eta_min 门限，"
            "于是「选中的观测是否至少允许该目标被保护」这件事不由系统决定。"
        ),
    )
    def test_c6_selection_and_protection_are_coupled(self):
        params = set(inspect.signature(_release_selector).parameters)
        assert params & {"protected", "protection", "protect_mask", "z", "eta_min"}

    def test_c9_every_target_is_scored_at_one_pfa(self):
        """C9：所有 P_D 在同一个 P_FA* 下比较。"""
        cfg = make_cfg()
        D = np.array([1.0, 10.0, 100.0])
        pd = pd_from_deflection(cfg, D)
        assert float(pd_from_deflection(cfg, 0.0)) == pytest.approx(
            cfg.detect.Pfa_target, abs=1e-4
        )
        assert np.all(np.diff(pd) > 0.0)


# ==========================================================================
# §15  信息集限制：只用 belief
# ==========================================================================
