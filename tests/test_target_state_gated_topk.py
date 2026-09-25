"""``gated_topk`` 求解器的阶段 A 不变式（缓存 / 筛选 / NMS / 多峰细化）。

与 ``test_target_state_map.py`` 的分工：那边钉"信息隔离与真值泄漏"，这边钉
"搜索策略"。策略类性质用**解析目标函数**测（把搜索器和仿真器分开），只有
"两个求解器在现实场景上是否一致"才动用真观测。

阶段 A 门控默认关（``accepted`` 恒 True），所以这里不测门限 —— 门限要等阶段 D
在 40 个 0 m 场景上标定之后才能写成断言。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pytest

from _targetstate_common import (
    RECEIVERS, TARGET, make_cfg, make_noiseless_look, make_pair, make_scorers,
    make_world,
)
from isac_sim.receiver.target_state.coarse import coarse_to_fine_search
from isac_sim.receiver.target_state.fit import fit_shared_target_offset
from isac_sim.receiver.target_state.gated_topk import fit_gated_topk
from isac_sim.receiver.target_state.nms import non_maximum_suppression
from isac_sim.receiver.target_state.objective import ObjectiveEvaluator
from isac_sim.receiver.target_state.screen import receiver_order
from isac_sim.receiver.target_state.views import (
    ReceiverView,
    make_receiver_views,
)

STEP = 75.0      # σ=150 时的粗网格步长（σ/2）
SPAN = 600.0     # search_sigma=4


# --------------------------------------------------------------------------
# 解析目标函数：把"搜索策略"从"仿真器"里剥离出来测
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Src:
    uav: int = 0
    target: int = TARGET


@dataclass(frozen=True)
class _Geom:
    p_tgt: np.ndarray
    p_uav: np.ndarray


class _BumpEvaluator:
    """若干高斯峰之和（**不带先验项**）—— 只考察搜索器能否找到最高的那个峰。

    峰半高宽只有几十米，粗网格间距 75 m：同一个峰在网格上只被采到一两个点，
    格点第一名经常不是真峰。这正是 ``gated_topk`` 要解决的问题。
    """

    def __init__(self, peaks, receivers=(0, 1, 2, 3), target=TARGET):
        self.peaks = tuple(
            (np.asarray(c, float), float(h), float(w)) for c, h, w in peaks)
        self.target = int(target)
        self.evaluations = 0
        self.receiver_evaluations = 0
        self.belief_geometry = _Geom(
            p_tgt=np.zeros((target + 1, 6)),
            p_uav=np.array([[1000.0 * (i + 1), 500.0 * i, 300.0, 0, 0, 0]
                            for i in range(len(receivers))]),
        )
        self.views = [ReceiverView(int(rx), None, (_Src(),)) for rx in receivers]

    def _height(self, delta) -> float:
        d = np.asarray(delta, dtype=float).reshape(2)
        return float(sum(
            h * math.exp(-float(np.dot(d - c, d - c)) / (2.0 * w * w))
            for c, h, w in self.peaks))

    def gains(self, delta, views=None):
        n = len(self.views if views is None else views)
        self.receiver_evaluations += n
        return np.full(n, self._height(delta))

    def objective(self, delta, views=None):
        self.evaluations += 1
        return self._height(delta), self.gains(delta, views)


def _fit_bumps(peaks, span=SPAN, **options):
    evaluator = _BumpEvaluator(peaks)
    out = fit_gated_topk(evaluator, evaluator.views, span, STEP, options or None)
    return out, evaluator


def test_topk_recovers_the_offset_when_the_true_peak_is_sampled():
    """真峰更高且被网格采到时，Top-K 细化必须收敛到真峰（≤12 m，细化步长 9.375）。"""
    truth = np.array([130.0, -60.0])
    out, _ = _fit_bumps([(truth, 10.0, 35.0), ((-150.0, 150.0), 6.0, 30.0)])
    assert float(np.linalg.norm(out["best"] - truth)) <= 12.0, out["best"]
    assert out["value"] > 9.5  # 必须吃到峰顶，不是停在格点上
    assert out["coarse_candidates"] == 289  # [-600,600] / 75 m 的 17×17 网格
    assert out["full_candidates"] >= 2


def test_topk_finds_the_true_peak_when_the_coarse_winner_is_a_decoy():
    """粗网格第一名是假峰时，Top-3 仍要找到真峰 —— coarse-to-fine 会留在假峰。

    这是 Gate-0 那个 335 m 跑飞的抽象版：格点第一名（原点处的窄包，9.0）高于真峰
    在最近格点上的采样值（7.75），单峰细化从假峰出发永远走不出去。
    """
    truth = np.array([130.0, -60.0])
    peaks = [(np.zeros(2), 9.0, 15.0), (truth, 10.0, 35.0)]
    out, evaluator = _fit_bumps(peaks)
    assert float(np.linalg.norm(out["best"] - truth)) <= 12.0, out["best"]
    coarse, _, _, _ = coarse_to_fine_search(evaluator, evaluator.views, SPAN, STEP)
    assert float(np.linalg.norm(coarse)) <= 20.0, coarse  # 旧求解器被假峰留住
    assert out["peak_gap"] > 0.0  # 次优局部最优（假峰）确实更矮


def test_boundary_hit_flags_a_solution_pinned_to_the_search_box_edge():
    """解被推到盒的边上时必须被标记 —— 阶段 D 的门控靠它拒绝更新。

    这是 §9 那 3 个恶化折的判别量：贴边解的状态误差 370--602 m，而先验惩罚只有
    0.5·(450/150)² = 4.5，拦不住它跑到边界上。
    """
    # span=450 是正式实验的口径（search_sigma=3 × prior_sigma 150）
    pinned, _ = _fit_bumps([((440.0, 0.0), 10.0, 40.0)], span=450.0)
    assert pinned["boundary_hit"] is True
    assert float(np.max(np.abs(pinned["best"]))) >= 450.0 - STEP
    inside, _ = _fit_bumps([((0.0, 0.0), 10.0, 40.0)], span=450.0)
    assert inside["boundary_hit"] is False
    assert float(np.max(np.abs(inside["best"]))) <= STEP


def test_nms_never_returns_two_neighbours_of_the_same_peak():
    """少了 NMS，Top-K 就是"同一个峰的前 K 名"，多峰细化退化成单峰细化。"""
    points = np.array([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0),
                       (200.0, 0.0), (0.0, 200.0), (200.0, 200.0)])
    values = np.array([9.0, 8.9, 8.8, 8.7, 7.0, 6.0, 5.0])
    kept, kept_values = non_maximum_suppression(points, values, 3, 75.0)
    assert len(kept) == 3
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            assert float(np.linalg.norm(kept[i] - kept[j])) >= 75.0 - 1e-9
    assert np.allclose(kept[0], (0.0, 0.0))  # 第一名必须留下
    assert kept_values == sorted(kept_values, reverse=True)


# --------------------------------------------------------------------------
# 按接收机缓存
# --------------------------------------------------------------------------


class _Flat:
    def energy(self, sources: Sequence, *, tested_only: bool = False) -> float:
        return 0.0


@pytest.fixture(scope="module")
def six_views():
    """六个接收机的真观测 + 空打分器（只测缓存/顺序，不需要真的匹配收益）。"""
    cfg = make_cfg()
    truth, _, base, bgeom = make_world(cfg, radius_m=150.0)
    receivers = tuple(range(6))
    observations = [make_pair(cfg, truth, bgeom, base, rx)[0] for rx in receivers]
    views = make_receiver_views(receivers, observations, [_Flat()] * 6, TARGET)
    return cfg, bgeom, views


def test_cache_does_not_re_evaluate_when_a_candidate_gains_receivers(six_views):
    """候选从 2 个接收机升级到 6 个时只补算新增的 4 个，不重算已有的 2 个。

    successive-halving 的全部收益都押在这条上：否则"先用少数接收机筛"只是把同一
    份活分两遍做。
    """
    cfg, bgeom, views = six_views
    evaluator = ObjectiveEvaluator(cfg, views, TARGET, bgeom, 150.0)
    delta = np.array([75.0, -75.0])
    evaluator.objective(delta, views[:2])
    assert evaluator.receiver_evaluations == 2
    assert evaluator.evaluations == 1
    evaluator.objective(delta, views)
    assert evaluator.receiver_evaluations == 6  # 只补算了 4 个
    assert evaluator.evaluations == 1           # 候选数不因升级而涨
    evaluator.objective(np.array([150.0, 0.0]), views)
    assert evaluator.evaluations == 2
    assert evaluator.receiver_evaluations == 12


def test_receiver_order_is_belief_only_and_order_independent(six_views):
    """顺序只由信念几何决定：与 views 的给出次序无关，且是接收机的一个排列。"""
    _, bgeom, views = six_views
    forward = receiver_order(views, bgeom, TARGET)
    backward = receiver_order(list(reversed(views)), bgeom, TARGET)
    assert forward == backward
    assert sorted(forward) == sorted(int(v.receiver) for v in views)
    assert len(set(forward)) == len(views)


# --------------------------------------------------------------------------
# 现实场景上的一致性与确定性
# --------------------------------------------------------------------------


def _shifted_geometry(known):
    """信念位置 = 真值 - known，于是真偏移 = +known。"""
    cfg = make_cfg(target_rcs=2.0)
    truth, belief, base, bgeom = make_world(cfg, radius_m=0.0)
    p_tgt = np.array(bgeom.p_tgt, dtype=float, copy=True)
    p_tgt[TARGET, 0] -= float(known[0])
    p_tgt[TARGET, 1] -= float(known[1])
    return cfg, truth, belief, base, replace(bgeom, p_tgt=p_tgt)


def _fit_real(solver, solver_options=None):
    cfg, truth, belief, base, bgeom = _shifted_geometry((120.0, -90.0))
    look = make_noiseless_look(cfg, truth, bgeom, base, RECEIVERS)
    return fit_shared_target_offset(
        cfg, look, TARGET, belief, receivers=RECEIVERS, belief_geometry=bgeom,
        scorers=make_scorers(cfg, look), prior_sigma_m=150.0, solver=solver,
        solver_options=solver_options)


def test_both_solvers_agree_on_a_noiseless_scene():
    """新搜索策略不许把"简单场景"做坏：与冻结的 coarse-to-fine 落在同一个峰上。"""
    known = np.array([120.0, -90.0])
    old = _fit_real("coarse_to_fine")
    new = _fit_real("gated_topk")
    assert float(np.linalg.norm(old.delta_xy_m - known)) <= 75.0
    assert float(np.linalg.norm(new.delta_xy_m - known)) <= 75.0
    assert float(np.linalg.norm(new.delta_xy_m - old.delta_xy_m)) <= STEP
    assert new.solver == "gated_topk" and old.solver == "coarse_to_fine"
    assert new.accepted is True             # 阶段 A 门控恒接受
    assert np.array_equal(new.delta_xy_m, new.raw_delta_xy_m)
    assert new.receiver_evaluations >= new.evaluations
    assert old.receiver_evaluations == old.evaluations * len(RECEIVERS)


def test_gated_topk_is_deterministic():
    """固定种子下逐次一致：偏移、目标函数、两个评价计数全等。"""
    first, second = _fit_real("gated_topk"), _fit_real("gated_topk")
    assert np.array_equal(first.delta_xy_m, second.delta_xy_m)
    assert first.objective == second.objective
    assert first.evaluations == second.evaluations
    assert first.receiver_evaluations == second.receiver_evaluations
    assert first.screen_receiver_count == second.screen_receiver_count
