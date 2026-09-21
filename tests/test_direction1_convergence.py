"""方向 1 收敛：把审计**已实测**的结论钉成断言（2026-09-21）。

背景：方向 1 跑出了一批结论，但它们一度只存在于 ``studies/direction1/docs/`` 的审计文档和
一次性探针里 —— 文档不会失败，探针不会在 CI 里跑。本模块把它们搬进断言，
这样将来任何一次改动只要推翻了其中一条，测试就会响。

每条断言都对应一份实测证据（数字见 ``studies/direction1/README.md`` §3，
原始产物在 ``studies/direction1/data/``）：

  1. ``estimation`` 主导残余 ⇒ 动 ``structural`` 的旋钮在默认口径下必然看不见；
  2. 默认口径**看不见** delta（0.17 dB），structural 口径**看得见**（>15 dB）；
     这一对照是"模型版/真实版"报数时必须带上的限定；
  3. kappa_struct(delta) 服从平方反比律 —— 用两档标定、**预测**第三档；
  4. delta 只动分母，不动分子（回波存活率）；
  5. delta=0 是上界，不是可达值；
  6. "同步误差"在分子侧与分母侧必须是同一个数（防两个键各说各话）。

⚠️ 本模块**只钉机制，不钉端到端 P_D**：MC=20 下 P_D 的 SE≈0.05，
同一配置换 rng 可得 0.7167 / 0.8000。端到端数字留给带 MC 标注的扫描脚本。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.receiver import cancellation as cx
from isac_sim.sensing.model import build_base_gains, generate_geometry

SEEDS = (0, 1, 2, 3)


def make_cfg(**overrides) -> Config:
    """kwargs 不能带点，所以 ``cancellation__x`` 在这里转成点分路径。"""
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
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    return cx.build_observation(
        cfg, geom, geom, b, 0, rng=rng, sense_power=sense, radiated_power=sense,
        processing_gain=cfg.waveform.N * cfg.waveform.L, hw_gain=1.0,
    )


def arm(cfg: Config, obs, only: str = "tp_uic_full"):
    return cx.cancellation_arms(cfg, obs, only=only)[only]


def struct_db(res) -> float:
    """结构性深度 ``10log10(i_in / i_res_structural)``。

    结果对象只暴露总的 ``kappa_db``（含噪声记账那一项），而方向 1 要盯的正是
    被字典完备掩盖的**结构**残差，所以在这里自己算。
    """
    if res.i_res_structural <= 0.0 or res.i_in <= 0.0:
        return float("inf")
    return 10.0 * math.log10(res.i_in / res.i_res_structural)


def median_struct_db(sigma: float) -> float:
    cfg = make_cfg(cancellation__direct_estimation_sigma_delay_bins=sigma)
    vals = [struct_db(arm(cfg, make_obs(cfg, seed=s))) for s in SEEDS]
    return float(np.median(vals))


def median_total_db(sigma: float) -> float:
    cfg = make_cfg(cancellation__direct_estimation_sigma_delay_bins=sigma)
    vals = [arm(cfg, make_obs(cfg, seed=s)).kappa_db for s in SEEDS]
    return float(np.median(vals))


# ---------------------------------------------------------------------------
# 1. 谁主导残余预算
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_estimation_dominates_the_residual_budget(seed):
    """``estimation = ||f(n)||^2`` 必须占 ``i_res`` 的绝大多数。

    实测 ``estimation / structural ≈ 506``（structural 只占 0.2%）。这条是
    方向 1"靶子错"的量化根据：任何只动 structural 的旋钮，在默认口径下
    必然看不见效果 —— 除非同时切换记账口径。这里只钉"主导"（>0.9），
    不钉死具体比例，避免把一次 MC 的噪声写成契约。
    """
    cfg = make_cfg()
    res = arm(cfg, make_obs(cfg, seed=seed))
    assert res.i_res > 0.0, "residual must be positive"
    share = float(res.i_res_estimate) / float(res.i_res)
    assert share > 0.9, f"estimation must dominate the budget: {share:.4f}"


# ---------------------------------------------------------------------------
# 2. 两个口径对 delta 的敏感度必须相反
# ---------------------------------------------------------------------------
def test_production_kappa_is_nearly_blind_to_delta():
    """默认（``measured``）口径下 delta 几乎不改变 kappa。

    实测只动 **0.62 dB**（本 preset，4 seeds 中位）；δ 扫描那一轮的口径下是
    **0.17 dB**。两者都远小于 structural 口径看到的 ~10 dB，所以这里给 1.5 dB
    的界，真正的对照交给下面的比值断言。

    这条是报"模型版/真实版"时**必须带上**的限定：在默认口径下两个版本的
    数字会几乎相同，那不是门控失效，是噪声增强项把 delta 压住了。
    """
    k0 = median_total_db(0.0)
    k3 = median_total_db(3e-3)
    assert abs(k3 - k0) < 1.5, (
        f"production accounting must be delta-blind: {k0:.2f} -> {k3:.2f} dB")


def test_structural_kappa_is_highly_sensitive_to_delta():
    """``structural`` 口径下 delta 必须显著改变深度 —— 与上一条构成对照。

    实测 **62.78 -> 52.99 dB**（本 preset），δ 扫描口径下是 63.53 -> 48.44。
    两条一起钉住，才说明"两版数字"有意义：门控真的建模了东西，
    只是默认口径看不见它。
    """
    k0 = median_struct_db(0.0)
    k3 = median_struct_db(3e-3)
    assert k0 - k3 > 8.0, (
        f"structural accounting must see delta: {k0:.2f} -> {k3:.2f} dB")


def test_structural_is_an_order_of_magnitude_more_sensitive():
    """两个口径对 delta 的敏感度必须差一个数量级（比值型，与 preset 无关）。

    实测 structural 掉 9.80 dB 而 production 只掉 0.62 dB ⇒ 比值 ≈16。
    用比值而不是两个绝对阈值，是为了让这条在换 preset / 换 seed 后仍然成立 ——
    钉的是"记账口径决定你能不能看见 delta"这个机制，不是某一次的 dB 数。
    """
    d_struct = abs(median_struct_db(3e-3) - median_struct_db(0.0))
    d_total = abs(median_total_db(3e-3) - median_total_db(0.0))
    assert d_total > 0.0, "production must move at least a little, or the test is vacuous"
    assert d_struct / d_total > 5.0, (
        f"structural must be far more sensitive: {d_struct:.2f} vs {d_total:.2f} dB")


# ---------------------------------------------------------------------------
# 3. 深度随 delta 的规律（可预测，不是拟合）
# ---------------------------------------------------------------------------
def test_structural_depth_follows_inverse_square_law():
    """``kappa_struct(delta) = -10log10(A*delta^2 + floor)``：每十倍 delta 掉 20 dB。

    用 delta=0 定地板、delta=3e-3 定系数，然后**预测** delta=1e-3。
    预测对了才叫规律；只报"单调下降"不足以支撑"达标需要 delta<=X"这类结论。
    实测拟合误差 ≤0.12 dB，这里给 2.5 dB 容差（换 preset / 换 seed 都该过）。
    """
    k0 = median_struct_db(0.0)
    k3 = median_struct_db(3e-3)
    k1 = median_struct_db(1e-3)
    floor = 10.0 ** (-k0 / 10.0)
    coef = (10.0 ** (-k3 / 10.0) - floor) / (3e-3 ** 2)
    assert coef > 0.0, "delta must add out-of-span energy"
    pred = -10.0 * math.log10(coef * (1e-3 ** 2) + floor)
    assert abs(pred - k1) <= 2.5, (
        f"inverse-square law must predict the middle point: "
        f"pred {pred:.2f} dB vs measured {k1:.2f} dB")


def test_depth_degrades_twenty_db_per_decade():
    """平方律的直接推论：delta 每大十倍，深度掉 20 dB（δ² 项主导时）。"""
    k_small = median_struct_db(1e-3)
    k_large = median_struct_db(1e-2)
    drop = k_small - k_large
    assert 15.0 <= drop <= 25.0, (
        f"expected ~20 dB per decade, got {drop:.2f} dB")


# ---------------------------------------------------------------------------
# 4. delta 只动分母
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS[:3])
def test_delta_does_not_move_the_numerator(seed):
    """回波存活率随 delta 的变化必须**可忽略**（实测 4e-5 量级）。

    ⚠️ 这里钉的是"可忽略"，不是"逐位相同" —— 字典变了，保护子空间跟着变，
    存活率**确实会动一点点**（实测相对变化 3.8e-5 ~ 5.1e-5）。所以严格说
    "delta 只动分母"是**不精确**的：它动分子，只是动得比分母小三个数量级。

    端到端（n=20）三档 delta 下"分子固定 vs 分子随臂"的 P_D 差**全为 0**，
    说明这 4e-5 不足以改变任何检测判决。若哪天这条挂了，
    "真实版"的收益就要按分子损失重新打折。
    """
    cfg0 = make_cfg()
    cfg1 = make_cfg(cancellation__direct_estimation_sigma_delay_bins=3e-3)
    # 物理场与噪声完全一致，只有字典不同（由 test_direct_estimation_error 钉住）。
    e0 = float(arm(cfg0, make_obs(cfg0, seed=seed)).eta_survive)
    e1 = float(arm(cfg1, make_obs(cfg1, seed=seed)).eta_survive)
    denom = max(abs(e0), 1e-12)
    assert abs(e1 - e0) / denom < 1e-4, (
        f"delta must barely move the numerator: {e0:.12f} -> {e1:.12f}")


# ---------------------------------------------------------------------------
# 5. delta=0 是上界
# ---------------------------------------------------------------------------
def test_delta_zero_structural_depth_is_an_upper_bound():
    """``delta=0`` 给出最深的结构残差 ⇒ 它是**上界**，不是可达值。

    字典用真值 bin 构造 ⇒ ``span(X)`` 完整包含直连场 ⇒ 结构残差被构造压成 0。
    谁把这个数当可达值报，谁就在重复当年 kappa=40 dB 的错误。
    """
    ks = [median_struct_db(s) for s in (0.0, 1e-3, 3e-3)]
    assert ks[0] > ks[1] > ks[2], f"delta=0 must be the deepest: {ks}"


# ---------------------------------------------------------------------------
# 6. 同一个"同步误差"不能有两个值
# ---------------------------------------------------------------------------
def test_sync_error_keys_agree_by_default():
    """分子侧与分母侧的同步误差必须是同一个数（当前默认都是 0.0）。

    ``waveform_impairments.sync_delay_bins`` 罚回波**分子**（sinc² 捕获损耗），
    ``cancellation.direct_estimation_sigma_delay_bins`` 抬对消**分母**（字典失配）。
    两者描述的是同一个硬件量。当前都默认为 0，一致；一旦标定，必须同步设值 ——
    否则会出现"分子说同步误差 0、分母说同步误差 3e-3"的自相矛盾。
    这条把该要求写成断言：只改其中一个会 FAIL。
    """
    cfg = make_cfg()
    assert (cfg.waveform_impairments.sync_delay_bins
            == cfg.cancellation.direct_estimation_sigma_delay_bins), (
        "delay sync error must have one value, not two")
    assert (cfg.waveform_impairments.sync_doppler_bins
            == cfg.cancellation.direct_estimation_sigma_doppler_bins), (
        "doppler sync error must have one value, not two")
