"""积木：Top-K 多峰局部细化（模式搜索）。

粗网格第一名经常是假峰：Gate-0 五个场景里两个因此跑飞 335 m，而真值处的目标函数
明明高 6.4 倍。所以**不信第一名** —— NMS 后的前 K 个峰各细化一次，最后比 K 个
局部最优。走模式搜索而不是 Nelder--Mead：边界裁剪、评价计数、确定性都更好写。
"""
from __future__ import annotations

import numpy as np

__all__ = ["pattern_search"]

_NEIGHBOURS = tuple((dx, dy) for dx in (-1.0, 0.0, 1.0)
                    for dy in (-1.0, 0.0, 1.0) if (dx, dy) != (0.0, 0.0))


def pattern_search(evaluator, views, seed, steps, span: float):
    """8 邻域 × 每层步长，每层取 ``max([best, *cands])``。

    返回 ``(best, value, moved)``；``moved`` 是**最细一层**是否移动，语义对齐
    coarse-to-fine 的 ``converged``。
    """
    best = np.asarray(seed, dtype=float).reshape(2)
    best_value, _ = evaluator.objective(best, views)
    moved = False
    for sub in steps:
        sub = float(sub)
        cand, cand_value = best, best_value
        for dx, dy in _NEIGHBOURS:
            point = best + np.array([sub * dx, sub * dy], dtype=float)
            if np.max(np.abs(point)) > float(span) + 1e-9:
                continue
            value, _ = evaluator.objective(point, views)
            if value > cand_value:
                cand_value, cand = value, point
        moved = bool(cand_value > best_value)
        if moved:
            best_value, best = cand_value, cand
    return best, float(best_value), moved
