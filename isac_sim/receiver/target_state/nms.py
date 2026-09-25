"""积木：非极大值抑制（NMS）。

G(delta) 的相关峰半高宽只有 25--75 m（实测：峰值 1463 在 ±50 m 处掉到 167），
粗网格上同一个峰会点亮好几个相邻格点。不做 NMS，Top-K 就只是"同一个峰的前 K
名"，多峰细化退化成单峰细化 —— 这恰恰是 Top-K 想解决的那个问题。
"""
from __future__ import annotations

import numpy as np

__all__ = ["non_maximum_suppression"]


def non_maximum_suppression(points, values, keep: int, radius_m: float):
    """按排名贪心：从高到低遍历，与**已选**所有点的距离都 ≥ radius 才收。

    同分按索引升序，保证确定性。返回 ``(kept_points, kept_values)``，仍按分数降序。
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    vals = np.asarray(values, dtype=float).reshape(-1)
    order = sorted(range(vals.size), key=lambda i: (-float(vals[i]), i))
    kept: list[int] = []
    for i in order:
        far = all(float(np.linalg.norm(pts[i] - pts[j])) >= float(radius_m) - 1e-12
                  for j in kept)
        if far:
            kept.append(i)
            if len(kept) >= int(keep):
                break
    return ([np.asarray(pts[i], dtype=float) for i in kept],
            [float(vals[i]) for i in kept])
