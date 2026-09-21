"""单站真擦除信道下的精确混合门限（用二分求分位）。

失败的上报是一个**零点原子**，没有高斯替换方差；成功分支是 Gamma−Gaussian 混合。
这里直接用该分布的精确尾概率 ``h0_tail`` 做二分，而不是 Cornish--Fisher 近似。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import EPS, qfunc


def singleton_mixture_threshold(
    cfg: Config,
    tables,
    q: int,
    links: List[Link],
    weights: Dict[Link, float],
    plan: "object | None",
    var0: float,
    fallback: float,
) -> float:
    """单站（``len(links) == 1``）、真擦除信道下的精确混合门限。"""
    from isac_sim.cooperation.reporting import report_chi
    from isac_sim.sensing.soft_channel import local_moments

    link = links[0]
    i, j = link
    gamma = max(float(tables.gamma_sense[i, j, q]), 0.0)
    if gamma <= EPS:
        return float(fallback)
    n_looks = max(int(cfg.detect.n_looks), 1)
    a = gamma / (1.0 + gamma)
    local = local_moments(cfg, tables, link, q)
    chi = (
        float(np.clip(report_chi(tables, plan, link, q), 0.0, 1.0))
        if cfg.detect.enable_comm_error_pollution else 1.0
    )
    # 这条分支只在真擦除信道进入。失败的单站上报是零点原子，没有高斯替换方差。
    failure_v = 0.0
    weight = float(weights[link])

    def erlang_sf(x: float) -> float:
        if x <= 0.0:
            return 1.0
        term = series = 1.0
        for k in range(1, n_looks):
            term *= x / k
            series += term
        return float(np.exp(-x) * series)

    def h0_tail(threshold: float) -> float:
        unweighted = threshold / max(weight, EPS)
        success = erlang_sf(n_looks + unweighted / a)
        failure = (
            float(qfunc(unweighted / np.sqrt(failure_v)))
            if failure_v > EPS else float(unweighted < 0.0)
        )
        return chi * success + (1.0 - chi) * failure

    scale = np.sqrt(max(var0, weight * weight * failure_v, EPS))
    lo, hi = -20.0 * scale, 20.0 * scale
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if h0_tail(mid) > cfg.detect.Pfa_target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))
