"""把 PSF 投影折算成有效检测信噪比 ``gamma_effective``。

背景方差 = 热噪声 + 干扰泄漏 + 杂波；期望方差 = 目标捕获能量 + 多径贡献。
两者都按「单位能量模板」归一，所以相除得到的 gamma 与波形细节无关。
"""

from __future__ import annotations

import numpy as np

from isac_sim.sensing.model import EPS


def link_budget(
    *,
    raw_gamma: float,
    interference_gamma: float,
    clutter_gamma: float,
    multipath_ratio: float,
    captured_energy: float,
    leakage_projection: float,
    clutter_projection: float,
    multipath_projection: float,
) -> tuple[float, float, float]:
    """返回 ``(background_variance, desired_variance, gamma_effective)``。"""
    background_variance = (
        1.0 + interference_gamma * leakage_projection + clutter_gamma * clutter_projection
    )
    desired_variance = raw_gamma * (
        captured_energy + multipath_ratio * multipath_projection
    )
    gamma_effective = desired_variance / max(background_variance, EPS)
    return background_variance, desired_variance, gamma_effective


def interference_variance(
    *, interference_gamma: float, clutter_gamma: float,
    leakage_projection: float, clutter_projection: float,
) -> float:
    """进入匹配滤波的那部分干扰功率（不含热噪声项）。"""
    return interference_gamma * leakage_projection + clutter_gamma * clutter_projection
