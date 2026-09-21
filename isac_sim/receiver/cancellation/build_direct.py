"""cancellation 积木：直达源列表与字典。"""

from __future__ import annotations

from dataclasses import replace
from typing import List

import numpy as np

from isac_sim.receiver.cancellation.link_offsets import direct_link_offset
from isac_sim.receiver.cancellation.sources import DirectSource


def build_direct_sources(ctx) -> List[DirectSource]:
    """逐激活照射源构造直达源。

    直达路径**不依赖信念** —— 只有目标字典和保护基依赖，这正是实验要定价的
    那条不对称。接收机自身不照射自己，未激活的源也不贡献直达。
    """
    direct: List[DirectSource] = []
    for i in range(ctx.cfg.scale.M):
        if i == ctx.receiver or not ctx.active_mask[i]:
            continue
        k_bin, l_bin = direct_link_offset(ctx.cfg, ctx.geom_true, i, ctx.receiver)
        u_i = 0.0
        if ctx.m_rx > 1:
            v = np.asarray(ctx.geom_true.p_uav[i], dtype=float)[:3] - ctx.p_rx
            n = float(np.linalg.norm(v))
            u_i = float(v[ctx.axis] / n) if n > 0.0 else 0.0
        direct.append(
            DirectSource(
                uav=i,
                power=float(ctx.radiated_power[i]),
                gain=float(ctx.base.direct_gain[i, ctx.receiver]),
                doppler_bin=float(k_bin),
                delay_bin=float(l_bin),
                u=float(u_i),
            )
        )
    return direct


def perturb_direct_sources(
    cfg,
    sources: List[DirectSource],
    rng: np.random.Generator | None = None,
) -> List[DirectSource]:
    """把直连 DD 参数的**估计误差**加到源列表上，得到"接收机以为的"字典源。

    真值源与估计源的差别就是方向 1 要定价的那个量：字典用估计源构造、直连场
    用真值源构造，于是直连场有一小部分落在字典张成空间**之外**，
    ``structural = ||x - f(x)||^2`` 由数据决定，而不再被"字典完备"这个构造
    压成 0。

    门关着（两个 sigma 都为 0）时**返回同一个列表对象**：调用方用 ``is`` 判断
    即可跳过第二次字典构造，于是既逐位不变也不多花算力。随机数从 ``rng.spawn``
    派生子流，因此主 rng 的状态不受影响 —— h_true 与 noise 在 delta 扫描的各个
    档位上完全对齐，配对比较才干净。
    """
    c = getattr(cfg, "cancellation", None)
    s_l = float(getattr(c, "direct_estimation_sigma_delay_bins", 0.0) or 0.0)
    s_k = float(getattr(c, "direct_estimation_sigma_doppler_bins", 0.0) or 0.0)
    if (not s_l > 0.0) and (not s_k > 0.0):
        return sources
    if not sources:
        return sources
    if rng is None:
        raise ValueError(
            "direct estimation error requires an rng once the sigma is positive"
        )
    child = rng.spawn(1)[0]
    out: List[DirectSource] = []
    for src in sources:
        d_l = float(child.normal() * s_l) if s_l > 0.0 else 0.0
        d_k = float(child.normal() * s_k) if s_k > 0.0 else 0.0
        out.append(
            replace(
                src,
                delay_bin=float(src.delay_bin) + d_l,
                doppler_bin=float(src.doppler_bin) + d_k,
            )
        )
    return out
