"""直连参数估计误差 delta 的门控与敏感性（方向 1，2026-09-21）。

背景：干扰字典此前用**真值**分数 DD bin 构造，于是 ``span(X)`` 完整包含直连场，
``structural = ||x - f(x)||^2`` 被构造压成 ~0，深度是"字典完备"的产物而不是
算法功绩（见 ``studies/direction1/README.md`` §3.1；同目录 ``docs/`` 下的审计
正文已于 2026-09-21 确认编码损坏，不作为引用来源）。

本模块的修法不是改记账，而是让模型变真实：``X`` 用"接收机以为的" bin，
``x_direct`` 仍用真值 bin。于是字典外能量由 delta 决定，kappa 由数据决定。

钉住的三件事：
  1. 门关着（sigma = 0）时逐位不变，且**不消耗 rng**；
  2. delta 只改字典，不改物理场（``y`` / ``x_direct`` 逐位相同）；
  3. 深度随 delta 单调退化 —— 这是"真实版"能压过"模型版"的前提。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver.cancellation.build_direct import (
    build_direct_sources,
    perturb_direct_sources,
)
from isac_sim.receiver.cancellation.dictionaries import direct_dictionary
from isac_sim.sensing.model import build_base_gains, generate_geometry


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


def make_obs(cfg: Config, seed: int = 0, belief_error: bool = False):
    """构造一个观测。两次调用只要 cfg 相同，rng 序列就相同。"""
    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    b = build_base_gains(cfg, geom, rng)
    if belief_error:
        from isac_sim.scenario.prior import perturbed_geometry

        belief = perturbed_geometry(
            cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
        )
    else:
        belief = geom
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    return cx.build_observation(
        cfg, geom, belief, b, 0, rng=rng, sense_power=sense, radiated_power=sense,
        processing_gain=cfg.waveform.N * cfg.waveform.L, hw_gain=1.0,
    )


def _sources(cfg: Config, seed: int = 0):
    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    b = build_base_gains(cfg, geom, rng)
    from isac_sim.receiver.cancellation.build_ctx import BuildContext

    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    ctx = BuildContext(
        cfg, geom, geom, b, 0, sense_power=sense, radiated_power=sense,
        processing_gain=cfg.waveform.N * cfg.waveform.L, hw_gain=1.0,
    )
    return build_direct_sources(ctx)


def _kappa(cfg: Config, obs, arm: str = "tp_uic_full"):
    return cx.cancellation_arms(cfg, obs, only=arm)[arm]


def _struct_db(res) -> float:
    """结构性深度 ``10log10(i_in / i_res_structural)``。

    结果对象只暴露总的 ``kappa_db``（含噪声记账那一项），而本模块要盯的正是
    被字典完备掩盖的**结构**残差，所以在这里自己算。
    """
    if res.i_res_structural <= 0.0 or res.i_in <= 0.0:
        return float("inf")
    return 10.0 * math.log10(res.i_in / res.i_res_structural)


# ---------------------------------------------------------------------------
# 1. 门关着 => 逐位不变
# ---------------------------------------------------------------------------
def test_gate_off_returns_the_same_list_object():
    """sigma = 0 时必须返回**同一个对象**：调用方据此跳过第二次字典构造。

    这不是风格问题 —— 逐位不变与算力都挂在这个 ``is`` 上。
    """
    cfg = make_cfg()
    srcs = _sources(cfg)
    assert perturb_direct_sources(cfg, srcs, rng=np.random.default_rng(0)) is srcs
    cfg0 = make_cfg(
        cancellation__direct_estimation_sigma_delay_bins=0.0,
        cancellation__direct_estimation_sigma_doppler_bins=0.0,
    )
    assert perturb_direct_sources(cfg0, srcs, rng=np.random.default_rng(0)) is srcs


def test_gate_off_does_not_consume_rng():
    """关着的门不能悄悄吃掉随机数 —— 否则它会移动 h_true 与 noise。"""
    cfg = make_cfg()
    srcs = _sources(cfg)
    rng = np.random.default_rng(7)
    before = rng.bit_generator.state["state"]["state"]
    perturb_direct_sources(cfg, srcs, rng=rng)
    after = rng.bit_generator.state["state"]["state"]
    assert before == after


def test_gate_off_is_bitwise_identical_to_the_published_path():
    """默认配置下 Observation 的每个数组都逐位不变。"""
    obs_default = make_obs(make_cfg(), seed=3, belief_error=True)
    obs_explicit = make_obs(
        make_cfg(
            cancellation__direct_estimation_sigma_delay_bins=0.0,
            cancellation__direct_estimation_sigma_doppler_bins=0.0,
        ),
        seed=3,
        belief_error=True,
    )
    for field in ("y", "X", "x_direct", "s_target", "h_true", "alpha_true"):
        a = np.asarray(getattr(obs_default, field))
        b = np.asarray(getattr(obs_explicit, field))
        assert np.array_equal(a, b), f"{field} drifted with the gate off"
    assert obs_default.sigma2 == obs_explicit.sigma2


# ---------------------------------------------------------------------------
# 2. delta 只改字典，不改物理场
# ---------------------------------------------------------------------------
def test_positive_delta_changes_only_the_dictionary():
    """``y`` / ``x_direct`` 必须逐位相同：接收机估错了字典，不改变到达天线口的场。

    这条保证了 delta 扫描是**配对**的 —— 同一份噪声、同一份真值直连场，只有
    字典不同，所以 kappa 的变化只可能来自失配。
    """
    sig = 1e-2
    obs0 = make_obs(make_cfg(), seed=5, belief_error=True)
    obs1 = make_obs(
        make_cfg(cancellation__direct_estimation_sigma_delay_bins=sig),
        seed=5,
        belief_error=True,
    )
    assert np.array_equal(np.asarray(obs0.y), np.asarray(obs1.y)), "y must not move"
    assert np.array_equal(
        np.asarray(obs0.x_direct), np.asarray(obs1.x_direct)
    ), "the true direct field must not depend on the receiver's error"
    assert not np.array_equal(
        np.asarray(obs0.X), np.asarray(obs1.X)
    ), "the dictionary must move"


def test_positive_delta_puts_the_direct_field_outside_the_dictionary_span():
    """直连场的一部分落在 ``span(X)`` 之外 —— 这就是"消不掉的干扰"。"""
    sig = 1e-2
    obs0 = make_obs(make_cfg(), seed=5)
    obs1 = make_obs(
        make_cfg(cancellation__direct_estimation_sigma_delay_bins=sig), seed=5
    )
    x = np.asarray(obs1.x_direct, dtype=complex)
    gram = np.asarray(obs1.X).conj().T @ np.asarray(obs1.X)
    rhs = np.asarray(obs1.X).conj().T @ x
    coef = np.linalg.solve(
        gram + 1e-18 * np.eye(gram.shape[0]), rhs
    )
    off = float(np.vdot(x - np.asarray(obs1.X) @ coef, x - np.asarray(obs1.X) @ coef).real)
    total = float(np.vdot(x, x).real)
    assert off / total > 1e-6, "delta > 0 must leave out-of-span energy"
    # 门关着时该分数应为数值零：这就是被"字典完备"掩盖的那个量。
    x0 = np.asarray(obs0.x_direct, dtype=complex)
    g0 = np.asarray(obs0.X).conj().T @ np.asarray(obs0.X)
    c0 = np.linalg.solve(g0 + 1e-18 * np.eye(g0.shape[0]), np.asarray(obs0.X).conj().T @ x0)
    off0 = float(np.vdot(x0 - np.asarray(obs0.X) @ c0, x0 - np.asarray(obs0.X) @ c0).real)
    assert off0 / total < 1e-12, "delta = 0 must be a perfect fit (span contains x)"


# ---------------------------------------------------------------------------
# 3. 深度随 delta 退化
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [1e-3, 1e-2])
def test_structural_depth_degrades_with_delta(sigma):
    """结构残差必须随 delta 上升 —— 否则这个开关没有建模任何东西。

    用多个 seed 平均抑制单次抽取的波动；只断言"显著变深"，不钉具体 dB。
    """
    k0, k1 = [], []
    for seed in (0, 1, 2):
        o0 = make_obs(make_cfg(), seed=seed)
        o1 = make_obs(
            make_cfg(cancellation__direct_estimation_sigma_delay_bins=sigma), seed=seed
        )
        k0.append(_struct_db(_kappa(make_cfg(), o0)))
        cfg1 = make_cfg(cancellation__direct_estimation_sigma_delay_bins=sigma)
        k1.append(_struct_db(_kappa(cfg1, o1)))
    mean0, mean1 = float(np.mean(k0)), float(np.mean(k1))
    # sigma = 1e-3 时理论退化 ~-20log10(1e-3)= 60 dB 量级，取 5 dB 作保守下界。
    assert mean1 < mean0 - 5.0, (
        f"structural depth must degrade with delta: {mean0:.2f} -> {mean1:.2f} dB"
    )


def test_deeper_delta_is_worse_than_shallower_delta():
    """单调性：delta 越大越差。用中位数避免单次抽取把顺序搅乱。"""
    meds = []
    for sigma in (1e-4, 1e-3, 1e-2):
        vals = []
        for seed in (0, 1, 2, 3):
            cfg = make_cfg(cancellation__direct_estimation_sigma_delay_bins=sigma)
            vals.append(_struct_db(_kappa(cfg, make_obs(cfg, seed=seed))))
        meds.append(float(np.median(vals)))
    assert meds[0] > meds[1] > meds[2], f"depth must fall monotonically: {meds}"
