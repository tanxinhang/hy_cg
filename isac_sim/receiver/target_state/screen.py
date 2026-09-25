"""积木：接收机筛选顺序（belief-only）与粗网格点。

顺序只能用**接收机已知**的量：真值几何、H1 统计量、残差能量一律不许进来，
否则"筛选"就变成拿答案挑接收机。可用的只有信念几何里的 UAV 坐标，加上被测目标
源自带的发射机索引 ``src.uav``，于是顺序由

    quality(rx) = 1 / (R_tx^2 * R_rx^2)      # 双基地雷达方程的形状项
    azimuth(rx) = atan2(p_tgt - p_uav[rx])    # 观测方位

决定：第一个取链路质量最好的；之后每次取"到已选集合**最小**方位差"最大的那个，
避免挑进两个看同一方向的节点。平局一律按接收机号升序 —— 确定性优先于"更优"。
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["receiver_order", "coarse_grid"]

_MIN_RANGE = 1e-9


def _circular_gap(a: float, b: float) -> float:
    """两个方位角的最小夹角（0..π）。"""
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def receiver_order(views, belief_geometry, target: int) -> list[int]:
    """返回接收机索引的一个**固定**顺序：同一折内中途不许改。"""
    p_tgt = np.asarray(belief_geometry.p_tgt, dtype=float)[int(target), :3]
    p_uav = np.asarray(belief_geometry.p_uav, dtype=float)
    azimuth, quality = {}, {}
    for view in views:
        rx = int(view.receiver)
        d_xy = p_tgt[:2] - np.asarray(p_uav[rx], dtype=float)[:2]
        azimuth[rx] = math.atan2(float(d_xy[1]), float(d_xy[0]))
        tx = next((int(s.uav) for s in view.sources
                   if int(s.target) == int(target)), None)
        r_tx = (float(np.linalg.norm(p_tgt - np.asarray(p_uav[tx], float)[:3]))
                if tx is not None else 1.0)
        r_rx = float(np.linalg.norm(p_tgt - np.asarray(p_uav[rx], float)[:3]))
        quality[rx] = 1.0 / max(r_tx * r_rx, _MIN_RANGE) ** 2
    order, remaining = [], sorted(int(v.receiver) for v in views)
    while remaining:
        if not order:
            pick = max(remaining, key=lambda rx: (quality[rx], -rx))
        else:
            pick = max(remaining, key=lambda rx: (
                min(_circular_gap(azimuth[rx], azimuth[s]) for s in order), -rx))
        order.append(pick)
        remaining.remove(pick)
    return order


def coarse_grid(span: float, step: float) -> list:
    """``[-span, span]`` 上的方形网格（含端点），与 coarse-to-fine 同一个 arange。"""
    grid = np.arange(-span, span + 0.5 * step, step)
    return [np.array([float(dx), float(dy)], dtype=float)
            for dx in grid for dy in grid]
