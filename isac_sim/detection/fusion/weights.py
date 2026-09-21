"""weights（自 ``isac_sim/detection/fusion/weights.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import (
    EPS,
    BaseGains,
    d_pd_d_D,
    pd_from_deflection,
    qfunc,
    threshold_from_pfa,
)
from isac_sim.sensing.soft_channel import (
    local_moments,
    received_full_llr_moments,
    received_h0_third_central,
    received_moments,
)
from isac_sim.detection.fusion.link_stats import _per_link_arrays, deflection_variance_for_link, effective_h1_mean_for_link


def compute_weights(
    cfg: Config,
    tables,
    q: int,
    links: List[Link],
    mode: str = "deflection",
    plan: "object | None" = None,
    base: BaseGains | None = None,
) -> Dict[Link, float]:
    if not links:
        return {}

    if mode == "equal":
        return {link: 1.0 / len(links) for link in links}
    if mode == "exact_llr_sum":
        return {link: 1.0 for link in links}

    if mode == "deflection" and cfg.corr.enable and base is not None and len(links) > 1:
        from isac_sim.detection.corr import correlation_aware_weights

        delta, sigma = _per_link_arrays(cfg, tables, q, links, plan)
        w = correlation_aware_weights(cfg, links, delta, sigma, base=base, q=q)
        return {link: float(w[k]) for k, link in enumerate(links)}

    vals: List[float] = []
    for link in links:
        if mode == "deflection":
            mu_eff = effective_h1_mean_for_link(cfg, tables, link, q, plan)
            # 除数守卫用 ``max`` 而不是 ``+ EPS``。两者在方差远大于 EPS 时几乎
            # 相同，但只有 ``max`` 让"偏转最优权 = mu/var"精确成立，从而让
            # 独立观测的信息可加性成为恒等式。
            #
            # 这不是假想的风险：``interference.direct_cancellation_db`` 被删除后
            # 默认口径的统计量整体下移了约 40 dB，``var`` 落到 1e-8 量级，此时
            # ``var + EPS``（EPS = 1e-12）会把最优权扭曲 ~1e-4，可加性断言在
            # rel=1e-9 下失败。**绝对**常数守卫是尺度盲的，而这里恰恰会跨 5 个
            # 数量级地换尺度。
            var = deflection_variance_for_link(cfg, tables, link, q, plan)
            vals.append(max(mu_eff, 0.0) / max(var, EPS))
        else:
            raise ValueError(mode)

    total = float(np.sum(vals))
    if total <= EPS:
        return {link: 1.0 / len(links) for link in links}
    return {link: v / total for link, v in zip(links, vals)}


def deflection_for_links(
    cfg: Config,
    tables,
    q: int,
    links: List[Link],
    weight_mode: str = "deflection",
    plan: "object | None" = None,
    base: BaseGains | None = None,
) -> float:
    if not links:
        return 0.0

    weights = compute_weights(cfg, tables, q, links, mode=weight_mode, plan=plan, base=base)

    if weight_mode == "deflection" and cfg.corr.enable and base is not None and len(links) > 1:
        from isac_sim.detection.corr import correlated_deflection, correlation_aware_weights

        delta, sigma = _per_link_arrays(cfg, tables, q, links, plan)
        w = np.array([weights[link] for link in links], dtype=float)
        return correlated_deflection(cfg, links, delta, sigma, w, base=base, q=q)

    mean_gap = 0.0
    var0 = 0.0
    for link, w in weights.items():
        if weight_mode == "exact_llr_sum":
            moments = received_full_llr_moments(cfg, tables, link, q, plan)
            mean_gap += w * moments.gap
            var0 += (w ** 2) * moments.v0
        else:
            mean_gap += w * effective_h1_mean_for_link(cfg, tables, link, q, plan)
            var0 += (w ** 2) * deflection_variance_for_link(cfg, tables, link, q, plan)

    # 同上面的权：除数守卫用 ``max`` 不用 ``+ EPS``。这里是**偏转系数本身**的
    # 分母，加一个绝对常数会直接按比例压低 D —— 在本场景下（var0 ~ 1e-8）
    # 压低约 1.8e-4，足以让"信息可加性"这类恒等式在 rel=1e-9 下失败。
    return float((max(mean_gap, 0.0) ** 2) / max(var0, EPS))
