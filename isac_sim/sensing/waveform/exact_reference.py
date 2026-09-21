"""不依赖蒙特卡洛的精确参照：Erlang 存活函数 + 对真实混合分布做二分标定。

上报成功时能量精确服从 Gamma；失败时是零点原子。把两者加权就得到 H0/H1 的
**精确**尾概率。再用它直接二分出门限，就能看出 Cornish--Fisher 近似在
丢包不可忽略时到底损失了多少性能。
"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model import EPS, qfunc


def erlang_sf(x: float, order: int) -> float:
    """Erlang（整数阶 Gamma）存活函数 ``P(X > x)``，逐项求和避免下溢。"""
    if x <= 0.0:
        return 1.0
    term = series = 1.0
    for k in range(1, order):
        term *= x / k
        series += term
    return float(np.exp(-x) * series)


def _mixture_tails(
    candidate: float,
    *,
    n_looks: int,
    a: float,
    gamma_effective: float,
    chi: float,
    failure_v: float,
) -> tuple[float, float]:
    """在给定门限下返回 ``(P_FA, P_D)`` 的精确混合值。"""
    energy_cut = n_looks + candidate / max(a, EPS)
    success_pfa = erlang_sf(energy_cut, n_looks)
    success_pd = erlang_sf(energy_cut / (1.0 + gamma_effective), n_looks)
    if failure_v > EPS:
        fail_tail = float(qfunc(candidate / np.sqrt(failure_v)))
    else:
        fail_tail = float(candidate < 0.0)
    return (
        chi * success_pfa + (1.0 - chi) * fail_tail,
        chi * success_pd + (1.0 - chi) * fail_tail,
    )


def exact_singleton_reference(
    *,
    n_looks: int,
    chi: float,
    a: float,
    gamma_effective: float,
    failure_v: float,
    threshold: float,
) -> dict[str, float]:
    """在 Cornish--Fisher 门限处给出精确 H0/H1 值（用于量化近似误差）。"""
    if a > EPS:
        energy_threshold = n_looks + threshold / a
        exact_success_pfa = erlang_sf(energy_threshold, n_looks)
        exact_success_pd = erlang_sf(energy_threshold / (1.0 + gamma_effective), n_looks)
    else:
        energy_threshold = float("inf")
        exact_success_pfa = 0.0
        exact_success_pd = 0.0

    if failure_v > EPS:
        failure_tail = float(qfunc(threshold / np.sqrt(failure_v)))
    else:
        failure_tail = float(threshold < 0.0)

    return {
        "exact_success_pfa": exact_success_pfa,
        "exact_success_pd": exact_success_pd,
        "exact_mixture_pfa": chi * exact_success_pfa + (1.0 - chi) * failure_tail,
        "exact_mixture_pd": chi * exact_success_pd + (1.0 - chi) * failure_tail,
    }


def calibrated_singleton_threshold(
    cfg: Config,
    *,
    n_looks: int,
    chi: float,
    a: float,
    gamma_effective: float,
    failure_v: float,
    var0: float,
) -> tuple[float, float, float]:
    """直接对真实 H0 混合分布二分，返回 ``(门限, 精确 P_FA, 精确 P_D)``。"""
    scale = np.sqrt(max(var0, failure_v, EPS))
    lo, hi = -20.0 * scale, 20.0 * scale
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _mixture_tails(
            mid, n_looks=n_looks, a=a, gamma_effective=gamma_effective,
            chi=chi, failure_v=failure_v,
        )[0] > cfg.detect.Pfa_target:
            lo = mid
        else:
            hi = mid
    calibrated_threshold = 0.5 * (lo + hi)
    pfa, pd = _mixture_tails(
        calibrated_threshold, n_looks=n_looks, a=a, gamma_effective=gamma_effective,
        chi=chi, failure_v=failure_v,
    )
    return calibrated_threshold, pfa, pd
