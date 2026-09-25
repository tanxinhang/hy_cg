"""共享目标偏移 cross-fitted MAP 的不变式（方案 §11）。

这里的每一条都对应方案里"必须被钉住"的一条性质。信息隔离与真值泄漏是**功能**
测试（推翻它们就会 FAIL），不是文档。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from _targetstate_common import (
    RECEIVERS,
    TARGET,
    make_cfg,
    make_noiseless_look,
    make_pair,
    make_scorers,
    make_world,
)
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.receiver.target_state import (
    apply_target_offset,
    build_scorer,
    crossfit_target_state_score,
    fit_shared_target_offset,
)


# --------------------------------------------------------------------------
# 打分器与参考检测器必须逐位一致
# --------------------------------------------------------------------------


def test_scorer_matches_reference_detector_bit_for_bit():
    cfg = make_cfg()
    truth, belief, base, bgeom = make_world(cfg, radius_m=0.0)
    obs, _ = make_pair(cfg, truth, bgeom, base, 0)
    results = cx.cancellation_arms(cfg, obs, weak_target=TARGET, only="tp_uic_full")
    model = gl.residual_model(cfg, obs, "tp_uic_full", results)
    scorer = build_scorer(cfg, obs, model, TARGET)
    sources = [s for s in obs.targets_belief if int(s.target) == TARGET]
    mine = scorer.energy(sources)
    reference = gl.target_neighbourhood_glrt(
        cfg, obs, results["tp_uic_full"], model, target=TARGET,
        radius_bins=0.0, grid_points=1, aggregation="max").statistic
    assert abs(mine - reference) <= 1e-9 * max(abs(reference), 1.0)


def test_fast_search_objective_matches_full_template():
    """搜索只建被测目标那几列（15 -> 5，快 ~3 倍），必须给出同一个值。

    其余目标的源在 ``shifted_target_sources`` 里原样不动，字典列与 ``delta``
    无关且已被 ``q_neg`` 正交掉；实测最坏相对差 1.5e-14。
    """
    cfg = make_cfg()
    truth, belief, base, bgeom = make_world(cfg, radius_m=250.0)
    obs, _ = make_pair(cfg, truth, bgeom, base, 0)
    model = gl.residual_model(
        cfg, obs, "tp_uic_full",
        cx.cancellation_arms(cfg, obs, weak_target=TARGET, only="tp_uic_full"))
    scorer = build_scorer(cfg, obs, model, TARGET)
    full = scorer.energy(obs.targets_belief)
    fast = scorer.energy(obs.targets_belief, tested_only=True)
    assert abs(fast - full) <= 1e-12 * max(abs(full), 1.0)


# --------------------------------------------------------------------------
# §5 字典重建：只动接收机可知的东西
# --------------------------------------------------------------------------


def test_zero_offset_is_the_identity():
    cfg = make_cfg()
    truth, belief, base, bgeom = make_world(cfg, radius_m=200.0)
    obs, _ = make_pair(cfg, truth, bgeom, base, 0)
    same = apply_target_offset(cfg, obs, TARGET, np.zeros(2), receiver=0,
                               belief_geometry=bgeom)
    assert np.array_equal(same.y, obs.y)
    assert np.array_equal(same.A, obs.A)
    assert np.array_equal(same.basis_belief, obs.basis_belief)
    assert np.array_equal(same.x_direct, obs.x_direct)
    assert np.array_equal(same.s_target, obs.s_target)


def test_offset_leaves_truth_only_fields_untouched():
    cfg = make_cfg()
    truth, belief, base, bgeom = make_world(cfg, radius_m=200.0)
    obs, _ = make_pair(cfg, truth, bgeom, base, 0)
    moved = apply_target_offset(cfg, obs, TARGET, np.array([150.0, -75.0]),
                                receiver=0, belief_geometry=bgeom)
    for name in ("y", "X", "x_direct", "s_target", "h_true", "alpha_true", "sigma2"):
        assert np.array_equal(np.asarray(getattr(moved, name)),
                              np.asarray(getattr(obs, name))), name
    assert moved.direct_est is obs.direct_est
    assert moved.targets is obs.targets
    # 信念侧必须真的变了
    assert not np.array_equal(moved.A, obs.A)
    before = [s for s in obs.targets_belief if int(s.target) == TARGET]
    after = [s for s in moved.targets_belief if int(s.target) == TARGET]
    assert any(abs(float(a.delay_bin) - float(b.delay_bin)) > 1e-9
               for a, b in zip(before, after))
    # 其它目标一个字节都不许动
    others_before = [s for s in obs.targets_belief if int(s.target) != TARGET]
    others_after = [s for s in moved.targets_belief if int(s.target) != TARGET]
    assert all(
        float(a.delay_bin) == float(b.delay_bin)
        and float(a.doppler_bin) == float(b.doppler_bin)
        for a, b in zip(others_before, others_after)
    )


def test_target_map_does_not_read_truth_geometry():
    """改真值侧字段、保持观测与信念不变 -> 估计结果必须一字不变。"""
    cfg = make_cfg()
    truth, belief, base, bgeom = make_world(cfg, radius_m=250.0)
    look = [make_pair(cfg, truth, bgeom, base, rx)[0] for rx in RECEIVERS]
    scorers = make_scorers(cfg, look)
    fit = fit_shared_target_offset(
        cfg, look, TARGET, belief, receivers=RECEIVERS, belief_geometry=bgeom,
        scorers=scorers, prior_sigma_m=150.0)

    spoiled = [
        replace(obs, x_direct=obs.x_direct * 3.0, s_target=obs.s_target * 0.0,
                alpha_true=np.zeros_like(obs.alpha_true),
                h_true=obs.h_true * 0.0, targets=list(reversed(obs.targets)),
                basis_truth=np.zeros_like(obs.basis_truth))
        for obs in look
    ]
    fit2 = fit_shared_target_offset(
        cfg, spoiled, TARGET, belief, receivers=RECEIVERS, belief_geometry=bgeom,
        scorers=scorers, prior_sigma_m=150.0)
    assert np.array_equal(fit.delta_xy_m, fit2.delta_xy_m)
    assert fit.objective == fit2.objective


# --------------------------------------------------------------------------
# 共享状态 / 求解器 / 先验
# --------------------------------------------------------------------------


def test_shared_map_returns_one_state_for_all_receivers():
    cfg = make_cfg()
    truth, belief, base, bgeom = make_world(cfg, radius_m=200.0)
    look = [make_pair(cfg, truth, bgeom, base, rx)[0] for rx in RECEIVERS]
    scorers = make_scorers(cfg, look)
    fit = fit_shared_target_offset(
        cfg, look, TARGET, belief, receivers=RECEIVERS, belief_geometry=bgeom,
        scorers=scorers, prior_sigma_m=150.0)
    assert np.shape(fit.delta_xy_m) == (2,)
    assert np.shape(fit.estimated_position_m) == (3,)
    assert np.shape(fit.receiver_gain) == (len(RECEIVERS),)
    assert fit.evaluations >= 25


def test_prior_pulls_to_zero_when_the_data_carry_no_information():
    """匹配收益恒定时，MAP 必须精确回到 delta = 0（先验行为，方案 §11）。"""
    import isac_sim.receiver.target_state.fit as tsm

    class _Flat:
        def energy(self, sources, *, tested_only=False):
            return 0.0

    class _View:
        def __init__(self, receiver, obs, target):
            self.receiver = receiver
            self.scorer = _Flat()
            self.sources = tuple(s for s in obs.targets_belief
                                 if int(s.target) == target)

    cfg = make_cfg()
    truth, belief, base, bgeom = make_world(cfg, radius_m=0.0)
    look = [make_pair(cfg, truth, bgeom, base, rx)[0] for rx in RECEIVERS]
    original = tsm.make_receiver_views
    tsm.make_receiver_views = (
        lambda receivers, observations, scorers, target:
        [_View(int(rx), obs, int(target))
         for rx, obs in zip(receivers, observations)]
    )
    try:
        fit = fit_shared_target_offset(
            cfg, look, TARGET, belief, receivers=RECEIVERS,
            belief_geometry=bgeom, scorers=[_Flat()] * len(RECEIVERS),
            prior_sigma_m=150.0)
    finally:
        tsm.make_receiver_views = original
    assert np.allclose(fit.delta_xy_m, 0.0)
    assert fit.prior_penalty == 0.0


def test_prior_penalty_uses_the_declared_covariance():
    """惩罚项必须按**声明**的 sigma_p 算，而不是按实际注入的误差。"""
    cfg = make_cfg(prior_sigma=150.0)
    truth, belief, base, bgeom = make_world(cfg, radius_m=400.0)
    assert abs(float(belief.P[TARGET, 0, 0]) - 150.0 ** 2) < 1e-9
    look = [make_pair(cfg, truth, bgeom, base, rx)[0] for rx in RECEIVERS]
    fit = fit_shared_target_offset(
        cfg, look, TARGET, belief, receivers=RECEIVERS, belief_geometry=bgeom,
        scorers=make_scorers(cfg, look), prior_sigma_m=150.0)
    expected = 0.5 * float(np.dot(fit.delta_xy_m, fit.delta_xy_m)) / 150.0 ** 2
    assert fit.prior_penalty == pytest.approx(expected, abs=1e-12)
    assert np.max(np.abs(fit.delta_xy_m)) <= 4.0 * 150.0 + 1e-9


# --------------------------------------------------------------------------
# §3 信息隔离
# --------------------------------------------------------------------------


def _crossfit_with_spy(cfg, belief, bgeom, look_a, look_b):
    """跑一次双折，并记录「拟合看到了哪些 y」与「打分看到了哪些 y」。"""
    # 必须打在 ``crossfit`` 的名字上：它 ``from ... import fit_shared_target_offset``，
    # 打在 ``fit`` 模块上不会影响 crossfit 里已绑定的引用。
    import isac_sim.receiver.target_state.crossfit as tsm

    fitted, scored = [], []
    original = tsm.fit_shared_target_offset

    def spy(cfg_, observations, target, belief_, **kwargs):
        fitted.append([np.asarray(o.y) for o in observations])
        return original(cfg_, observations, target, belief_, **kwargs)

    def score_fn(observations):
        scored.append([np.asarray(o.y) for o in observations])
        return float(len(observations))

    tsm.fit_shared_target_offset = spy
    try:
        result = crossfit_target_state_score(
            cfg, look_a, look_b, TARGET, belief, receivers=RECEIVERS,
            belief_geometry=bgeom, scorers_a=make_scorers(cfg, look_a),
            scorers_b=make_scorers(cfg, look_b), score_fn=score_fn,
            prior_sigma_m=150.0)
    finally:
        tsm.fit_shared_target_offset = original
    return result, fitted, scored


def test_fold_a_state_is_used_only_on_fold_b():
    """第一折：state 只在 A 上估；被改动的字典只出现在 B 上。"""
    cfg = make_cfg()
    truth, belief, base, bgeom = make_world(cfg, radius_m=250.0)
    look_a = [make_pair(cfg, truth, bgeom, base, rx, look=0)[0] for rx in RECEIVERS]
    look_b = [make_pair(cfg, truth, bgeom, base, rx, look=1)[0] for rx in RECEIVERS]
    _, fitted, scored = _crossfit_with_spy(cfg, belief, bgeom, look_a, look_b)
    assert len(fitted) == 2 and len(scored) == 2
    for j, obs in enumerate(look_a):
        assert np.array_equal(fitted[0][j], obs.y)
        assert np.array_equal(scored[0][j], look_b[j].y)
    # A 的观测绝不能出现在自己那一折的 held-out 打分里
    assert not np.array_equal(scored[0][0], look_a[0].y)


def test_fold_b_state_is_used_only_on_fold_a():
    """第二折：镜像规则。"""
    cfg = make_cfg()
    truth, belief, base, bgeom = make_world(cfg, radius_m=250.0)
    look_a = [make_pair(cfg, truth, bgeom, base, rx, look=0)[0] for rx in RECEIVERS]
    look_b = [make_pair(cfg, truth, bgeom, base, rx, look=1)[0] for rx in RECEIVERS]
    _, fitted, scored = _crossfit_with_spy(cfg, belief, bgeom, look_a, look_b)
    for j, obs in enumerate(look_b):
        assert np.array_equal(fitted[1][j], obs.y)
        assert np.array_equal(scored[1][j], look_a[j].y)
    assert not np.array_equal(scored[1][0], look_b[0].y)


# --------------------------------------------------------------------------
# 无噪声恢复
# --------------------------------------------------------------------------


def _shifted_belief_geometry(bgeom, known):
    """信念位置 = 真值 - known，于是真偏移 = +known。"""
    out = replace(bgeom, p_tgt=np.array(bgeom.p_tgt, dtype=float, copy=True))
    out.p_tgt[TARGET, 0] -= float(known[0])
    out.p_tgt[TARGET, 1] -= float(known[1])
    return out


def _recover(cfg, truth, base, known):
    bgeom2 = _shifted_belief_geometry(make_world(cfg, radius_m=0.0)[3], known)
    look = make_noiseless_look(cfg, truth, bgeom2, base, RECEIVERS)
    fit = fit_shared_target_offset(
        cfg, look, TARGET, make_world(cfg, radius_m=0.0)[1], receivers=RECEIVERS,
        belief_geometry=bgeom2, scorers=make_scorers(cfg, look),
        prior_sigma_m=150.0)
    return float(np.linalg.norm(fit.delta_xy_m - np.asarray(known, float))), fit


def test_shared_map_recovers_known_offset_without_noise():
    cfg = make_cfg(target_rcs=2.0)
    truth, _, base, _ = make_world(cfg, radius_m=0.0, target=TARGET)
    known = np.array([120.0, -90.0])
    err, _ = _recover(cfg, truth, base, known)
    assert err < np.linalg.norm(known)
    assert err <= 75.0


def test_solver_recovers_offsets_from_many_directions():
    """粗网格间距必须够密，否则"搜索"变成在噪声包里挑最大的那个。

    方案 §6 的 ±4σ/±2σ/0（300 m）在实测峰半高宽 25--75 m 面前等于没采样：
    Gate-0 五个场景里两个跑飞 335 m，而真值处的目标函数明明高 6.4 倍。这条
    用四个方向钉住修正后的网格，同时用评价次数钉住"网格确实加密了"。
    """
    cfg = make_cfg(target_rcs=2.0)
    truth, _, base, _ = make_world(cfg, radius_m=0.0)
    errs, evals = [], []
    for known in ((120.0, -90.0), (-100.0, 140.0), (200.0, -60.0),
                  (-180.0, -120.0)):
        err, fit = _recover(cfg, truth, base, known)
        errs.append(err)
        evals.append(int(fit.evaluations))
    assert max(errs) <= 60.0, errs
    assert min(evals) >= 150, evals


# --------------------------------------------------------------------------
# §8 配对随机流：误差半径不得进入任何随机流
# --------------------------------------------------------------------------


def test_error_radius_does_not_move_the_physical_random_stream():
    cfg = make_cfg()
    truth, _, base, _ = make_world(cfg, radius_m=0.0)
    geom_a = make_world(cfg, radius_m=0.0)[3]
    geom_b = make_world(cfg, radius_m=400.0)[3]
    obs_a, _ = make_pair(cfg, truth, geom_a, base, 0)
    obs_b, _ = make_pair(cfg, truth, geom_b, base, 0)
    assert np.array_equal(obs_a.y - obs_a.s_target, obs_b.y - obs_b.s_target)
    assert np.array_equal(obs_a.x_direct, obs_b.x_direct)
    assert np.array_equal(obs_a.X, obs_b.X)
