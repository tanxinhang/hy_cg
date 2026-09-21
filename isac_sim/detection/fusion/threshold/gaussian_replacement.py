"""高斯替换信道下的精确混合门限。

仅当「软统计量模型 = llr」且「通信误差模型 = gaussian_replacement」且不相关时才走这条：
失败上报不是零点原子，而是带局部方差的高斯替换，所以门限必须对这条混合律做分位。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from isac_sim.core.config import Config, Link


def gaussian_replacement_threshold(
    cfg: Config,
    tables,
    q: int,
    links: List[Link],
    weights: Dict[Link, float],
    plan: "object | None" = None,
) -> float:
    """``exact_gaussian_replacement_threshold`` 分支。"""
    from isac_sim.cooperation.reporting import report_chi
    from isac_sim.sensing.soft_channel import local_moments

    n_samples = int(cfg.detect.fused_calibration_samples)
    n_looks = max(int(cfg.detect.n_looks), 1)
    rng = np.random.default_rng(0x6A551A)
    fused = np.zeros(n_samples, dtype=float)
    for link in links:
        i, j = link
        gamma = max(float(tables.gamma_sense[i, j, q]), 0.0)
        a = gamma / (1.0 + gamma)
        local = local_moments(cfg, tables, link, q)
        chi = (
            float(np.clip(report_chi(tables, plan, link, q), 0.0, 1.0))
            if cfg.detect.enable_comm_error_pollution else 1.0
        )
        success = a * (rng.gamma(n_looks, 1.0, n_samples) - n_looks)
        failure = rng.normal(
            0.0,
            cfg.detect.soft_error_sigma_scale * np.sqrt(max(local.v0, 0.0)),
            n_samples,
        )
        received = np.where(rng.random(n_samples) < chi, success, failure)
        fused += float(weights[link]) * received
    return float(np.quantile(fused, 1.0 - cfg.detect.Pfa_target, method="higher"))
