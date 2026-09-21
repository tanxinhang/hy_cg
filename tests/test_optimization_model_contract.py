"""优化模型一致性门禁（1/4）：变量、接收机接口、检测信息、P_D 映射。

提案骨架

    (xi_hat, P_q) -> (I_res_bar, eta_under) -> gamma_eff -> P_D,q -> selection/protection

核心命题（P0/P1）是

    max_{x,z} min_q  P_D,q(x,z)        s.t. 资源预算

本门禁把编号 §1–§20 逐条翻译成断言：**跑得通的**是正常通过项，
**跑不通的**一律写成 ``xfail(strict=True)`` 并写明缺口 —— 一旦哪天被实现，
它会变成 XPASS 而报错，逼你把这个标记删掉、把缺口转成常态断言。

因此有两种阅读方式：

* ``pytest tests/test_optimization_model_*`` —— 门禁；
* 直接读每个 ``xfail`` 的 ``reason`` —— 当前实现与提案的差距清单。

四个文件按条款分组（每组 ≤350 行）：
1 变量 / 接口 / 信息 / P_D（本文件）
2 ``test_optimization_model_objective.py`` —— 目标函数形状
3 ``test_optimization_model_resources.py`` —— 资源约束
4 ``test_optimization_model_alternation.py`` —— 信息集 / 交替 / 字典序
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from isac_sim.core.config import apply_overrides
from isac_sim.cooperation.primitives import feasible_links_for_target
from isac_sim.detection.fusion import deflection_for_links
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver.cancellation.protection import protected_target_ids
from isac_sim.sensing.model import (
    compute_link_tables,
    pd_from_deflection,
    qfunc,
    threshold_from_pfa,
)
from isac_sim.receiver.cancellation.protection import protected_target_ids
from isac_sim.types import CertificateView

from _optmodel_common import make_cfg, make_world, make_sources
from experiments.selection import select_lagrangian


# ==========================================================================
# §1  优化变量：只有 x_{ijq} 与 z_{jq}
# ==========================================================================

class TestOptimisationVariables:
    """提案 §1：第一版只保留观测选择 x 与接收端保护 z 两个变量。"""

    def test_selection_variable_is_a_binary_membership_set(self):
        """``x_{ijq}`` in {0,1}：一条链路对一个目标要么选、要么不选。"""
        cfg = make_cfg()
        _geom, base, tables, plan = make_world(cfg)
        selected, _D = select_lagrangian(cfg, base, tables, plan)

        assert set(selected) == set(range(int(cfg.scale.Q)))
        for q, links in selected.items():
            # 二值性在这里的具体含义：不允许重复计数（否则就是 x=2）
            assert len(links) == len(set(links)), f"目标 {q} 的链路集出现重复项"
            for link in links:
                i, j = link
                assert i != j, "自环不是一次双站观测"

    def test_protection_variable_is_a_binary_target_set(self):
        """``z_{jq}`` in {0,1}：一个接收机保护或不保护某个目标。"""
        cfg = make_cfg()
        ids = protected_target_ids(cfg, make_sources())

        assert isinstance(ids, frozenset), "保护变量必须是一个集合，而不是权重向量"
        assert all(isinstance(t, int) for t in ids)
        assert all(0 <= t < 5 for t in ids)

    def test_first_version_optimises_no_power_beam_or_cpi(self):
        """§1 明确要求：不要把 power / beam / CPI 也塞进优化变量。"""
        from isac_sim.core.config.selector import Selector

        names = {f.name for f in dataclasses.fields(Selector)}
        forbidden = names & {"power", "beam", "beamforming", "waveform", "cpi", "n_cpi"}
        assert not forbidden, f"选择器仍在优化资源变量：{sorted(forbidden)}"


# ==========================================================================
# §2  接收机 → 调度器接口只有 (I_res, eta)，且按式 (1) 进入 SINR
# ==========================================================================


# ==========================================================================
# §2  接收机 → 调度器接口只有 (I_res, eta)，且按式 (1) 进入 SINR
# ==========================================================================

class TestReceiverSchedulerInterface:
    """提案 §2：唯一的接口是两个保守量，并直接落到有效 SINR 上。"""

    def test_certificate_is_exactly_two_scalars(self):
        """接口不许悄悄长大：调度器只见 ``I_res`` 与 ``eta``。"""
        fields = [f.name for f in dataclasses.fields(CertificateView)]
        assert fields == ["residual_power", "target_retention"]

    def test_survival_scales_the_numerator(self):
        """式 (1) 的分子：``eta`` 线性缩放有效信号。"""
        cfg = make_cfg()
        _geom, base, tables, plan = make_world(cfg)
        m_rx = int(cfg.scale.M)

        half = compute_link_tables(
            cfg, base, target_retention_by_receiver=np.full(m_rx, 0.5)
        )
        mask = tables.gamma_sense > 0.0
        assert np.count_nonzero(mask) > 0
        np.testing.assert_allclose(
            half.gamma_sense[mask], 0.5 * tables.gamma_sense[mask], rtol=1e-9
        )

    def test_residual_sits_in_the_denominator(self):
        """式 (1) 的分母：``I_res`` 越大，有效 SINR 越低（单调，不是等式）。"""
        cfg = make_cfg()
        _geom, base, tables, plan = make_world(cfg)
        m_rx, n_q = int(cfg.scale.M), int(cfg.scale.Q)
        mask = tables.gamma_sense > 0.0

        levels = []
        for value in (1e-15, 1e-13, 1e-11):
            t = compute_link_tables(
                cfg,
                base,
                residual_power_by_receiver_target=np.full((m_rx, n_q), value),
            )
            levels.append(t.gamma_sense[mask])

        for lo, hi in zip(levels, levels[1:]):
            assert np.all(hi <= lo * (1.0 + 1e-9)), "残余干扰没有进入分母"
        assert np.any(levels[-1] < levels[0] * 0.5), "残余干扰对分母没有实质影响"

    def test_certificate_is_belief_side_not_a_truth_measurement(self):
        """§2 强调：证书由 belief 得到，不是真值测量 —— 换观测实现不许动它。"""
        cfg = make_cfg(**{"cancellation.enable": True})
        geom, base, _tables, _plan = make_world(cfg)
        belief_ctx = cx.ReceiverContext.from_trial(cfg, geom, geom, base)

        first = cx.measure_receiver_context(
            belief_ctx, rng=np.random.default_rng([cfg.run.seed, 51])
        )
        second = cx.measure_receiver_context(
            belief_ctx, rng=np.random.default_rng([cfg.run.seed, 52])
        )
        np.testing.assert_array_equal(first.predicted_fraction, second.predicted_fraction)
        assert not np.array_equal(first.i_in, second.i_in), "实测输入场必须随观测实现变化"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "缺口 §2（生产方）：CertificateView 这两个量只能由调用方注入，而生产链路 "
            "experiments/ 里零处注入 —— cancellation.mode 默认 'off'，端到端数字里的 "
            "(I_res, eta) 是常数 kappa 占位，不是 TP-UIC 证书。接口形态达标、内容未接线，"
            "所以论文里「TP-UIC 通过 (I_res, eta) 改变观测价值」目前没有实现支撑。"
        ),
    )
    def test_certificate_has_a_production_path(self):
        """式 (1) 的两个保守量必须有一个**生产方**，不能只有接口。

        扫描 ``experiments/``：只要出现对 ``residual_power_by_receiver_target`` /
        ``target_retention_by_receiver`` 的注入，这条就会 XPASS。
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        keys = ("residual_power_by_receiver_target", "target_retention_by_receiver")
        hits = []
        for path in (root / "experiments").rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(k in text for k in keys):
                hits.append(path.name)
        assert hits, "式(1) 的证书没有生产方：experiments/ 零处注入"


# ==========================================================================
# §3  检测信息：D_q = sum_ij x_ijq * J(gamma_ijq)
# ==========================================================================


# ==========================================================================
# §3  检测信息：D_q = sum_ij x_ijq * J(gamma_ijq)
# ==========================================================================

class TestDetectionInformation:
    """提案 §3：先建信息量，再谈检测；且在独立口径下可加。"""

    def test_information_is_additive_over_independent_observations(self):
        """式 (2)：观测条件独立时，融合信息等于各观测信息之和。"""
        cfg = make_cfg()
        _geom, base, tables, plan = make_world(cfg)
        assert cfg.corr.enable is False, "本断言只描述独立口径"

        q = 0
        links = feasible_links_for_target(cfg, base, tables, q, plan)[:3]
        assert len(links) == 3

        per_link = [
            deflection_for_links(
                cfg, tables, q, [link], weight_mode="deflection", plan=plan, base=base
            )
            for link in links
        ]
        joint = deflection_for_links(
            cfg, tables, q, links, weight_mode="deflection", plan=plan, base=base
        )
        assert joint == pytest.approx(float(np.sum(per_link)), rel=1e-9)

    def test_per_link_information_is_derived_not_hand_tuned(self):
        """§3 要求 J 来自 LLR/KL/偏转，而不是手工标定的 surrogate。"""
        cfg = make_cfg()
        assert cfg.detect.soft_stat_model == "llr", "主口径应当是派生出来的软统计量"

        _geom, base, tables, plan = make_world(cfg)
        link = feasible_links_for_target(cfg, base, tables, 0, plan)[0]
        before = deflection_for_links(cfg, tables, 0, [link], plan=plan, base=base)

        tuned = apply_overrides(cfg, {"detect.soft_mu_scale": 40.0})
        after = deflection_for_links(tuned, tables, 0, [link], plan=plan, base=base)
        assert after == pytest.approx(before), "手工标定的软均值尺度仍在影响检测信息"


# ==========================================================================
# §4 / C9  固定 P_FA 下的 P_D = G(D, P_FA*)
# ==========================================================================


# ==========================================================================
# §4 / C9  固定 P_FA 下的 P_D = G(D, P_FA*)
# ==========================================================================

class TestPdMapping:
    """提案 §4 与 §14：所有 P_D 都在同一个 P_FA* 下比较。"""

    def test_pd_is_a_function_of_information_at_the_fixed_pfa(self):
        """式 (3)：``P_D = Q(eta_PFA - sqrt(D))``，只由 D 与设计虚警决定。"""
        cfg = make_cfg()
        eta = threshold_from_pfa(cfg)
        for D in (0.0, 1.0, 4.0, 25.0):
            assert float(pd_from_deflection(cfg, D)) == pytest.approx(
                float(qfunc(eta - np.sqrt(D))), abs=1e-12
            )

    def test_the_threshold_depends_on_the_design_pfa_and_nothing_else(self):
        """C9：改别的物理旋钮不许动门限，否则 P_D 之间不可比。"""
        cfg = make_cfg()
        eta = threshold_from_pfa(cfg)
        for over in (
                {"detect.target_rcs": 1.0},
                {"detect.n_looks": 64},
                {"scale.M": 8},
                {"detect.pd_required": 0.5},
        ):
            assert threshold_from_pfa(apply_overrides(cfg, over)) == eta

    def test_pd_is_monotone_in_information(self):
        """信息越多检测概率越高 —— 这是把优化建立在 D 上的前提。"""
        cfg = make_cfg()
        grid = np.linspace(0.0, 40.0, 41)
        pd = pd_from_deflection(cfg, grid)
        assert np.all(np.diff(pd) >= -1e-12)
        # 工作点是 0.0499952 而不是标称 0.05：门限表 ``threshold_from_pfa``
        # 用的是四位常数（0.05 -> 1.6449），Q(1.6449) = 0.0499952。差 5e-6，
        # 对结论无影响，但"标称 P_FA"与"实现 P_FA"不是同一个数，报数时要分清。
        assert pd[0] == pytest.approx(cfg.detect.Pfa_target, abs=1e-4)
        assert pd[0] != pytest.approx(cfg.detect.Pfa_target, abs=1e-9)

# ==========================================================================
# §5 / §6 / §17  目标函数形状：max-min，而不是加权和
# ==========================================================================
