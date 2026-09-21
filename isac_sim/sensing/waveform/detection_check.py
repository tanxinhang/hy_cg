"""波形自校验：从 OTFS 匹配滤波输出反推有限 look 的 LLR，并与解析预测对照。

用**物理 PSF**（而非解析 eta）驱动检测模型：目标捕获、干扰泄漏、杂波、多径全部
由真实 PSF 投影得到，因此同时验证「解析桥」与「波形实现」是否自洽。

返回四组量：``empirical_*`` 抽样实测、``predicted_pd`` 矩匹配解析预测、
``exact_*`` 精确混合尾概率、``calibrated_*`` 对真实 H0 混合分布二分标定的结果。

⚠️ 随机数调用顺序是逐位契约，见 :mod:`isac_sim.sensing.waveform.sampling`。
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.waveform.analytic import (
    failure_variance, fused_moments_and_threshold, scaled_llr,
)
from isac_sim.sensing.waveform.budget import interference_variance, link_budget
from isac_sim.sensing.waveform.exact_reference import (
    calibrated_singleton_threshold, exact_singleton_reference,
)
from isac_sim.sensing.waveform.projection import psf_projections
from isac_sim.sensing.waveform.sampling import (
    apply_report_channel, draw_matched_filter_energy,
)


def waveform_llr_detection_check(
    cfg: Config,
    *,
    raw_gamma: float = 0.5,
    interference_gamma: float = 0.0,
    report_success: float = 1.0,
    target_offset: tuple[float, float] = (0.31, -0.27),
    interferer_offset: tuple[float, float] = (-0.18, 0.22),
    num_interferers: int = 1,
    clutter_gamma: float = 0.0,
    multipath_ratio: float = 0.0,
    sync_error: tuple[float, float] = (0.0, 0.0),
    n_trials: int = 30_000,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    """对 OTFS 匹配滤波输出做有限 look 的 LLR 自校验。

    按配置的局部窗口切出「期望目标的分数 DD PSF」与「位移干扰体的 PSF」，
    两者的复重叠决定匹配滤波之后的真实干扰；随后逐 look 抽复数高斯目标系数
    与接收噪声。上报以 ``report_success`` 的概率成功，否则替换为零均值高斯
    失效代理（与系统仿真器用的是同一条代理）。

    解析预测使用当前矩匹配检测器，``empirical_*`` 来自波形派生的蒙特卡洛样本。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_trials = max(int(n_trials), 1)
    n_looks = max(int(cfg.detect.n_looks), 1)
    chi = float(np.clip(report_success, 0.0, 1.0))
    g_raw = max(float(raw_gamma), 0.0)
    g_int = max(float(interference_gamma), 0.0)
    g_clutter = max(float(clutter_gamma), 0.0)
    multipath_ratio = max(float(multipath_ratio), 0.0)

    # 非整数目标 + 一个未分辨的邻近干扰体：两者的局部 PSF 大幅重叠，
    # 所以这是真正的多目标泄漏检验，而不是等效正交的第二个目标。
    proj = psf_projections(
        cfg, target_offset=target_offset, interferer_offset=interferer_offset,
        sync_error=sync_error, num_interferers=num_interferers,
    )
    background_variance, desired_variance, gamma_effective = link_budget(
        raw_gamma=g_raw, interference_gamma=g_int, clutter_gamma=g_clutter,
        multipath_ratio=multipath_ratio, captured_energy=proj["captured_energy"],
        leakage_projection=proj["leakage_projection"],
        clutter_projection=proj["clutter_projection"],
        multipath_projection=proj["multipath_projection"],
    )
    interference_var = interference_variance(
        interference_gamma=g_int, clutter_gamma=g_clutter,
        leakage_projection=proj["leakage_projection"],
        clutter_projection=proj["clutter_projection"],
    )

    # 精确的匹配滤波投影：上面的完整 PSF 同时决定了目标捕获能量与干扰复重叠。
    x0, x1 = draw_matched_filter_energy(
        rng, n_trials=n_trials, n_looks=n_looks,
        desired_variance=desired_variance, background_variance=background_variance,
        interference_variance=interference_var,
    )
    a, llr0, llr1, delta, local_v0, local_v1 = scaled_llr(gamma_effective, n_looks, x0, x1)
    failure_v = failure_variance(cfg, local_v0)
    received0, received1 = apply_report_channel(
        rng, n_trials=n_trials, report_success=chi,
        llr0=llr0, llr1=llr1, failure_variance=failure_v,
    )
    _mean1, var0, _var1, threshold, predicted_pd = fused_moments_and_threshold(
        cfg, n_looks=n_looks, report_success=chi, a=a, delta=delta,
        local_v0=local_v0, local_v1=local_v1, failure_v=failure_v,
    )

    # 上报成功时有限 look 能量精确服从 Gamma：Erlang 存活函数给出不依赖抽样的精确参照。
    exact = exact_singleton_reference(
        n_looks=n_looks, chi=chi, a=a, gamma_effective=gamma_effective,
        failure_v=failure_v, threshold=threshold,
    )
    # 直接对真实的单站 H0 混合分布二分标定门限，而不是只用它的前三阶矩 ——
    # 这样能看出丢包不可忽略时 Cornish--Fisher 近似到底损失了多少性能。
    cal_thr, cal_pfa, cal_pd = calibrated_singleton_threshold(
        cfg, n_looks=n_looks, chi=chi, a=a, gamma_effective=gamma_effective,
        failure_v=failure_v, var0=var0,
    )

    return {
        "raw_gamma": g_raw,
        "interference_gamma": g_int,
        "num_interferers": proj["num_interferers"],
        "clutter_gamma": g_clutter,
        "multipath_ratio": multipath_ratio,
        "sync_error_k": proj["sync_error_k"],
        "sync_error_l": proj["sync_error_l"],
        "report_success": chi,
        "target_k": proj["target_k"],
        "target_l": proj["target_l"],
        "interferer_k": proj["interferer_k"],
        "interferer_l": proj["interferer_l"],
        "captured_energy": proj["captured_energy"],
        "local_window_energy": proj["local_window_energy"],
        "leakage_projection": proj["leakage_projection"],
        "clutter_projection": proj["clutter_projection"],
        "multipath_projection": proj["multipath_projection"],
        "gamma_effective": gamma_effective,
        "predicted_pd": predicted_pd,
        "exact_success_pd": exact["exact_success_pd"],
        "exact_success_pfa": exact["exact_success_pfa"],
        "exact_mixture_pd": exact["exact_mixture_pd"],
        "exact_mixture_pfa": exact["exact_mixture_pfa"],
        "empirical_pd": float(np.mean(received1 > threshold)),
        "empirical_pfa": float(np.mean(received0 > threshold)),
        "threshold": float(threshold),
        "calibrated_threshold": float(cal_thr),
        "calibrated_exact_pd": cal_pd,
        "calibrated_exact_pfa": cal_pfa,
        "calibrated_empirical_pd": float(np.mean(received1 > cal_thr)),
        "calibrated_empirical_pfa": float(np.mean(received0 > cal_thr)),
        "n_trials": float(n_trials),
    }
