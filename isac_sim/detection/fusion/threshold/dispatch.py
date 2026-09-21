"""标定融合门限：按「统计量口径 + 信道模型 + 相关性」分派到四条标定路径。

    exact_llr_sum                    -> 独立观测的精确 LLR 和分位
    gaussian_replacement（精确）     -> 高斯替换信道的精确混合分位
    erasure 且多站                   -> 确定性求积
    erasure 且单站                   -> 精确混合尾概率二分

其余情形（非 erasure、或相关多站）保留既有的 Cornish--Fisher 规则 ——
它们的联合高阶律不被当前的相关抽象识别。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from isac_sim.core.config import Config, Link
from isac_sim.detection.fusion.h0 import fused_h0_skewness, fused_h0_variance
from isac_sim.detection.fusion.threshold.exact_llr_sum import exact_llr_sum_threshold
from isac_sim.detection.fusion.threshold.gaussian_replacement import (
    gaussian_replacement_threshold,
)
from isac_sim.detection.fusion.threshold.quadrature import multi_link_quadrature_threshold
from isac_sim.detection.fusion.threshold.singleton import singleton_mixture_threshold
from isac_sim.sensing.model import EPS, BaseGains, threshold_from_pfa


def calibrated_fused_threshold(
    cfg: Config,
    tables,
    q: int,
    links: List[Link],
    weights: Dict[Link, float],
    plan: "object | None" = None,
    base: BaseGains | None = None,
    statistic_mode: str = "centered",
) -> float:
    """Calibrated finite-look LLR-mixture threshold.

    A singleton uses its exact Gamma--Gaussian mixture CDF. Independent
    multi-report sums use deterministic Monte-Carlo quadrature of that same
    implemented distribution. Correlated and non-erasure modes retain the
    established Cornish--Fisher rule because their joint higher-order law is
    not identified by the current correlation abstraction.
    """
    if statistic_mode == "exact_llr_sum":
        return exact_llr_sum_threshold(cfg, tables, q, links, weights, plan)

    if (
        cfg.detect.exact_gaussian_replacement_threshold
        and cfg.detect.soft_stat_model.lower() == "llr"
        and cfg.detect.comm_error_model == "gaussian_replacement"
        and not cfg.corr.enable
    ):
        return gaussian_replacement_threshold(cfg, tables, q, links, weights, plan)

    var0 = fused_h0_variance(cfg, tables, q, links, weights, plan, base)
    z0 = threshold_from_pfa(cfg)
    skew0 = fused_h0_skewness(cfg, tables, q, links, weights, plan, base)
    fallback = (z0 + (skew0 / 6.0) * (z0 * z0 - 1.0)) * np.sqrt(max(var0, EPS))
    if (
        cfg.detect.soft_stat_model.lower() != "llr"
        or cfg.detect.comm_error_model != "erasure"
        or (cfg.corr.enable and len(links) > 1)
    ):
        return float(fallback)

    if len(links) > 1:
        return multi_link_quadrature_threshold(cfg, tables, q, links, weights, plan)

    return singleton_mixture_threshold(cfg, tables, q, links, weights, plan, var0, fallback)
