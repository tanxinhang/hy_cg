"""残余干扰记账口径的门控（方向 1 处方，2026-09-21）。

背景：``i_res = structural + estimation`` 里的 ``estimation = ||f(n)||^2`` 是
**被减掉的**噪声 —— 实测占 ``||n||^2`` 的 0.12%，残差保留 99.93% 的噪声 ——
却被当成"没消掉的干扰"，占了 ``i_res`` 的 **99.8%**（比值 506）。于是报出的
深度 38 dB 描述的是噪声记账，而真正的对消能力是 65 dB（>
需求 52.25 dB）。见 ``studies/direction1/README.md`` §3.1。

本模块把口径做成可切换的门：``cancellation.residual_accounting``。

钉住的四件事：
  1. 默认 ``measured`` 逐位不变（= ``structural + estimation``）；
  2. ``structural`` 只改**记账**，不动算法 —— residual / h_hat / eta_* 逐位相同；
  3. 解析侧同步同口径，否则 ``calibration_error_db`` 会跨口径相比；
  4. ``structural`` 在 delta=0 下给出的是**上界**（字典完备），不是可达值
     ⇒ 必须配 ``direct_estimation_sigma_* > 0`` 才能当结果报。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.core.config.validate import validate_config
from isac_sim.receiver import cancellation as cx
from isac_sim.sensing.model import build_base_gains, generate_geometry

ARM = "tp_uic_full"


def make_cfg(**overrides) -> Config:
    """``cancellation__x`` 写成双下划线只是因为 kwargs 不能带点，这里转成点分路径。"""
    cfg = apply_preset(Config(), "small-uav-compact-800m")
    base = {
        "geometry.area_xy": 600.0,
        "detect.target_rcs": 0.1,
        "scale.M": 6,
        "scale.Q": 3,
        "run.seed": 2026,
        "run.verbose": False,
        "cancellation.enable": True,
    }
    for key, value in overrides.items():
        base[key.replace("__", ".")] = value
    return apply_overrides(cfg, base)


def make_obs(cfg: Config, seed: int = 0):
    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    b = build_base_gains(cfg, geom, rng)
    from isac_sim.scenario.prior import perturbed_geometry

    belief = perturbed_geometry(
        cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
    )
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    return cx.build_observation(
        cfg, geom, belief, b, 0, rng=rng, sense_power=sense, radiated_power=sense,
        processing_gain=cfg.waveform.N * cfg.waveform.L, hw_gain=1.0,
    )


def run(cfg: Config, obs):
    return cx.cancellation_arms(cfg, obs, only=ARM)[ARM]


# ---------------------------------------------------------------------------
# 1. 默认口径逐位不变
# ---------------------------------------------------------------------------
def test_default_accounting_is_bitwise_the_two_term_sum():
    """默认 ``measured`` 必须精确等于 ``structural + estimation``。

    这是一切已发布数字的地基：门一旦不再逐位复现旧口径，就不是"加了选项"，
    而是把基线改了。
    """
    cfg = make_cfg()
    res = run(cfg, make_obs(cfg, seed=1))
    assert res.i_res == res.i_res_structural + res.i_res_estimate
    cfg_explicit = make_cfg(cancellation__residual_accounting="measured")
    res_e = run(cfg_explicit, make_obs(cfg_explicit, seed=1))
    assert res_e.i_res == res.i_res
    assert res_e.i_res_pred == res.i_res_pred


def test_default_accounting_matches_the_published_kappa_ballpark():
    """默认口径的 kappa 应落在既有实测（36.5~37.2 dB）附近。

    不钉具体值（几何是随机的），只防止记账被悄悄改成另一个口径。
    """
    cfg = make_cfg()
    vals = [run(cfg, make_obs(cfg, seed=s)).kappa_db for s in (0, 1)]
    assert all(30.0 < v < 45.0 for v in vals), f"measured kappa out of range: {vals}"


# ---------------------------------------------------------------------------
# 2. structural 只改记账，不动算法
# ---------------------------------------------------------------------------
def test_structural_accounting_equals_the_structural_term():
    cfg = make_cfg(cancellation__residual_accounting="structural")
    res = run(cfg, make_obs(cfg, seed=1))
    assert res.i_res == res.i_res_structural
    assert res.i_res_estimate > 0.0, "estimation term must still be measured"


def test_accounting_mode_does_not_move_the_algorithm():
    """同一个观测、两种口径：除了记账字段，一切都得逐位相同。

    这是本门控的核心安全性质 —— 它改的是"怎么记"，不是"怎么消"。若哪天
    residual 也跟着动了，就说明口径渗进了求解器。
    """
    obs = make_obs(make_cfg(), seed=2)
    a = run(make_cfg(), obs)
    b = run(make_cfg(cancellation__residual_accounting="structural"), obs)
    assert np.array_equal(np.asarray(a.residual), np.asarray(b.residual))
    assert np.array_equal(np.asarray(a.h_hat), np.asarray(b.h_hat))
    assert np.array_equal(np.asarray(a.c_h_diag), np.asarray(b.c_h_diag))
    assert a.i_in == b.i_in
    assert a.i_res_structural == b.i_res_structural
    assert a.i_res_estimate == b.i_res_estimate
    assert a.eta_survive == b.eta_survive
    assert a.eta_survive_q == b.eta_survive_q
    assert a.protect_dim == b.protect_dim
    assert a.noise_enhance_db == b.noise_enhance_db


def test_structural_accounting_is_at_least_as_deep():
    """少记一项只可能让 i_res 变小 ⇒ kappa 只可能变深。"""
    obs = make_obs(make_cfg(), seed=2)
    a = run(make_cfg(), obs).kappa_db
    b = run(make_cfg(cancellation__residual_accounting="structural"), obs).kappa_db
    assert b >= a, f"structural ({b:.2f}) must be >= measured ({a:.2f})"


# ---------------------------------------------------------------------------
# 3. 解析侧同步
# ---------------------------------------------------------------------------
def test_prediction_side_follows_the_accounting_mode():
    """``i_res_pred`` 必须跟着口径走，否则 calibration_error_db 跨口径相比。"""
    obs = make_obs(make_cfg(), seed=2)
    a = run(make_cfg(), obs)
    b = run(make_cfg(cancellation__residual_accounting="structural"), obs)
    assert math.isfinite(a.calibration_error_db)
    assert math.isfinite(b.calibration_error_db)
    assert b.i_res_pred <= a.i_res_pred
    if b.i_res_retained > 0.0:
        # 能拆出独立结构项时就必须只留它；候选分支 retained=0 时解析侧拆不开，
        # 保持原预测值（裁剪只会造出 NaN）。
        assert b.i_res_pred == b.i_res_retained


# ---------------------------------------------------------------------------
# 4. delta=0 的 structural 只是上界
# ---------------------------------------------------------------------------
def test_zero_delta_structural_is_an_upper_bound_not_an_achievable_depth():
    """delta=0 时字典完备把结构残差压成 0 ⇒ structural 口径给出的是上界。

    这条是防误用的：那个数（~68 dB）常被当成"TP-UIC 能消 68 dB"来报，实际上
    它是"字典恰好包含直连场"的构造产物。配 delta>0 才是由数据决定的深度。
    """
    d0 = make_cfg(
        cancellation__residual_accounting="structural",
        cancellation__direct_estimation_sigma_delay_bins=0.0,
    )
    d1 = make_cfg(
        cancellation__residual_accounting="structural",
        cancellation__direct_estimation_sigma_delay_bins=3e-3,
    )
    k0 = [run(d0, make_obs(d0, seed=s)).kappa_db for s in (0, 1)]
    k1 = [run(d1, make_obs(d1, seed=s)).kappa_db for s in (0, 1)]
    m0, m1 = float(np.mean(k0)), float(np.mean(k1))
    assert m0 - m1 > 10.0, (
        f"delta=0 must read far deeper than delta>0 (upper bound): "
        f"{m0:.2f} vs {m1:.2f} dB"
    )


def test_delta_moves_the_production_kappa_only_under_structural():
    """同一件事的另一面：默认口径看不见 delta（噪声增强项把它压住了）。

    这正是"模型版 / 真实版当前数值几乎相同"的机制，方向 2 报数时必须写明。
    """
    base = dict(cancellation__direct_estimation_sigma_delay_bins=3e-3)
    a = make_cfg()
    b = make_cfg(**base)
    ka = float(np.mean([run(a, make_obs(a, seed=s)).kappa_db for s in (0, 1)]))
    kb = float(np.mean([run(b, make_obs(b, seed=s)).kappa_db for s in (0, 1)]))
    assert abs(ka - kb) < 2.0, (
        f"measured accounting should be near-blind to delta: {ka:.2f} vs {kb:.2f} dB"
    )


# ---------------------------------------------------------------------------
# 5. 校验
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "Structural", "sum", "measure"])
def test_unknown_accounting_mode_is_rejected(bad):
    cfg = make_cfg(cancellation__residual_accounting=bad)
    with pytest.raises(ValueError, match="residual_accounting"):
        validate_config(cfg)
