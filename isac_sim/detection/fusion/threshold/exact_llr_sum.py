"""精确 LLR 和阈值：独立观测下对同一条已实现分布做分位标定。

相关（非独立）观测被显式拒绝 —— 当前的相关抽象没有识别高阶联合律，
硬用这条规则会给出没有依据的门限。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from isac_sim.core.config import Config, Link


def exact_llr_sum_threshold(
    cfg: Config,
    tables,
    q: int,
    links: List[Link],
    weights: Dict[Link, float],
    plan: "object | None" = None,
) -> float:
    """``statistic_mode="exact_llr_sum"`` 分支。"""
    if cfg.corr.enable and len(links) > 1:
        raise ValueError("exact LLR sum currently requires independent observations")
    n_samples = int(cfg.detect.fused_calibration_samples)
    n_looks = max(int(cfg.detect.n_looks), 1)
    rng = np.random.default_rng(0xE11E57)
    fused = np.zeros(n_samples, dtype=float)
    from isac_sim.cooperation.reporting import report_chi
    for link in links:
        i, j = link
        gamma = max(float(tables.gamma_sense[i, j, q]), 0.0)
        a = gamma / (1.0 + gamma)
        chi = (
            float(np.clip(report_chi(tables, plan, link, q), 0.0, 1.0))
            if cfg.detect.enable_comm_error_pollution else 1.0
        )
        exact = -n_looks * np.log1p(gamma) + a * rng.gamma(
            n_looks, 1.0, n_samples
        )
        received = np.where(rng.random(n_samples) < chi, exact, 0.0)
        fused += float(weights[link]) * received
    return float(np.quantile(fused, 1.0 - cfg.detect.Pfa_target, method="higher"))
