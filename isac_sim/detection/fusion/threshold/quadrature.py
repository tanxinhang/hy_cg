"""多站求和门限的确定性求积。

复用同一条固定标准变差流，把蒙特卡洛变成**确定性求积**：门限可复现，
配对方法之间的差异不会来自标定用的随机数。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from isac_sim.core.config import Config, Link


def multi_link_quadrature_threshold(
    cfg: Config,
    tables,
    q: int,
    links: List[Link],
    weights: Dict[Link, float],
    plan: "object | None" = None,
) -> float:
    """多站（``len(links) > 1``）、真擦除信道下的确定性求积分支。"""
    from isac_sim.cooperation.reporting import report_chi
    from isac_sim.sensing.soft_channel import local_moments

    n_samples = int(cfg.detect.fused_calibration_samples)
    n_looks = max(int(cfg.detect.n_looks), 1)
    # 复用固定的标准变差流 ⇒ 确定性求积：门限可复现，配对比较不会因标定 RNG 抖动而不同。
    rng = np.random.default_rng(0xC0FFEE)
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
        # 这条分支只在真擦除信道进入：失败上报对融合统计量的贡献恰好为零。
        received = np.where(rng.random(n_samples) < chi, success, 0.0)
        fused += float(weights[link]) * received
    return float(np.quantile(fused, 1.0 - cfg.detect.Pfa_target, method="higher"))
