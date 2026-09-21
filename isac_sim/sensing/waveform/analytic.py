"""波形自校验里的解析矩与门限（确定性，不用随机数）。

有限 look 的能量在 H0 下精确服从 ``Gamma(n_looks, 1)``、H1 下服从
``Gamma(n_looks, 1 + gamma)``。这里按一阶/二阶矩给出解析预测，
再用 Cornish--Fisher 把 H0 偏度折进门限。
"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model import EPS, qfunc, threshold_from_pfa


def scaled_llr(
    gamma_effective: float, n_looks: int, x0: np.ndarray, x1: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, float, float, float]:
    """把能量型统计量缩成 LLR，并给出该缩放下的一阶/二阶量。

    返回 ``(a, llr0, llr1, delta, local_v0, local_v1)``。
    """
    a = gamma_effective / (1.0 + gamma_effective)
    llr0 = a * (x0 - n_looks)
    llr1 = a * (x1 - n_looks)
    delta = n_looks * gamma_effective ** 2 / (1.0 + gamma_effective)
    local_v0 = n_looks * gamma_effective ** 2 / (1.0 + gamma_effective) ** 2
    local_v1 = n_looks * gamma_effective ** 2
    return a, llr0, llr1, delta, local_v0, local_v1


def failure_variance(cfg: Config, local_v0: float) -> float:
    """上报失败时注入的代理方差（高斯替换）。"""
    return cfg.detect.soft_error_sigma_scale ** 2 * local_v0


def fused_moments_and_threshold(
    cfg: Config,
    *,
    n_looks: int,
    report_success: float,
    a: float,
    delta: float,
    local_v0: float,
    local_v1: float,
    failure_v: float,
) -> tuple[float, float, float, float, float]:
    """上报信道之后的融合矩与 Cornish--Fisher 门限。

    返回 ``(mean1, var0, var1, threshold, predicted_pd)``。
    """
    chi = report_success
    mean1 = chi * delta
    var0 = chi * local_v0 + (1.0 - chi) * failure_v
    var1 = (
        chi * (local_v1 + (delta - mean1) ** 2)
        + (1.0 - chi) * (failure_v + mean1 ** 2)
    )
    mu3_h0 = chi * 2.0 * n_looks * a ** 3
    skew0 = mu3_h0 / max(var0, EPS) ** 1.5
    z = threshold_from_pfa(cfg)
    z_cf = z + (skew0 / 6.0) * (z * z - 1.0)
    threshold = z_cf * np.sqrt(max(var0, EPS))
    predicted_pd = float(qfunc((threshold - mean1) / np.sqrt(max(var1, EPS))))
    return mean1, var0, var1, threshold, predicted_pd
