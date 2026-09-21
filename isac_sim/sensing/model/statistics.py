"""标量统计工具：Q 函数、高斯密度、门限反解、检测概率及其导数、二项置信区间。"""

from __future__ import annotations

import numpy as np
import math
from typing import Tuple
from isac_sim.core.config import Config


def qfunc(x: np.ndarray | float) -> np.ndarray | float:
    """标准正态的右尾概率 ``Q(x)``。"""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return 0.5 * math.erfc(float(arr.item()) / math.sqrt(2.0))
    vals = [0.5 * math.erfc(float(v) / math.sqrt(2.0)) for v in arr.ravel()]
    return np.array(vals, dtype=float).reshape(arr.shape)

def normal_pdf(x: np.ndarray | float) -> np.ndarray | float:
    """标准正态密度（已经中心化并归一）。"""
    arr = np.asarray(x, dtype=float)
    out = np.exp(-0.5 * arr * arr) / math.sqrt(2.0 * math.pi)
    if out.ndim == 0:
        return float(out.item())
    return out

def threshold_from_pfa(cfg: Config) -> float:
    """由 ``detect.Pfa_target`` 反解门限 ``eta``，使 ``Q(eta) = P_FA``。

    常用的几个 P_FA 走内置常数表（省掉每次调用都做一遍二分），其余二分 80 次。
    """
    in_built = {0.10: 1.2816, 0.05: 1.6449, 0.01: 2.3263, 0.001: 3.0902}
    for pfa, thr in in_built.items():
        if abs(cfg.detect.Pfa_target - pfa) < 1e-12:
            return thr
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if qfunc(mid) > cfg.detect.Pfa_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def pd_from_deflection(cfg: Config, D: np.ndarray | float) -> np.ndarray:
    """矩匹配检测概率 ``P_D = Q(eta - sqrt(D))``。"""
    eta = threshold_from_pfa(cfg)
    D_arr = np.asarray(D, dtype=float)
    return np.asarray(qfunc(eta - np.sqrt(np.maximum(D_arr, 0.0))), dtype=float)

def d_pd_d_D(cfg: Config, D: np.ndarray | float) -> np.ndarray:
    """``Q(eta - sqrt(D))`` 对 ``D`` 的导数（选择器的边际增益要用）。"""
    eta = threshold_from_pfa(cfg)
    D_arr = np.asarray(D, dtype=float)
    sqrtD = np.sqrt(np.maximum(D_arr, 1e-4))
    z = eta - sqrtD
    return np.asarray(normal_pdf(z), dtype=float) / (2.0 * sqrtD)

def binomial_ci95(success: int, total: int) -> Tuple[float, float, float]:
    """二项比例的 Wilson 95% 区间，返回 ``(下界, 上界, 半宽)``。"""
    if total <= 0:
        return 0.0, 0.0, 0.0
    z = 1.96
    n = float(total)
    p = success / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi, 0.5 * (hi - lo)
