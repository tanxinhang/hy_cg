"""积木：coarse-to-fine 求解（论文基线 / 回归参考，行为冻结）。

这一段是 Gate 1/2 的口径，**逐字搬**自 ``fit.py`` 第一版：改它就会动到发布数字。
新搜索策略一律走 :mod:`gated_topk`，靠 ``solver`` 派发，不共用这段代码。
"""
from __future__ import annotations

import numpy as np

__all__ = ["REFINE_FRACTIONS", "coarse_to_fine_search"]

#: 细化步长 = 粗网格步长 × 这些分数（σ=150、step=75 -> 37.5 / 18.75 / 9.375 m）
REFINE_FRACTIONS = (0.5, 0.25, 0.125)


def coarse_to_fine_search(evaluator, views, span: float, step: float):
    """粗网格 argmax + 三层 3×3 细化，返回 ``(best, value, gains, converged)``。"""
    if span <= 0.0:
        value, gains = evaluator.objective(np.zeros(2), views)
        return np.zeros(2), float(value), gains, True
    grid = np.arange(-span, span + 0.5 * step, step)
    best, best_value = np.zeros(2), -np.inf
    for dx in grid:
        for dy in grid:
            value, _ = evaluator.objective(np.array([dx, dy], dtype=float),
                                           views)
            if value > best_value:
                best_value, best = value, np.array([dx, dy], dtype=float)
    converged = False
    for fraction in REFINE_FRACTIONS:
        sub = float(fraction) * step
        moved = False
        for dx in (-sub, 0.0, sub):
            for dy in (-sub, 0.0, sub):
                candidate = best + np.array([dx, dy], dtype=float)
                if np.max(np.abs(candidate)) > span + 1e-9:
                    continue
                value, _ = evaluator.objective(candidate, views)
                if value > best_value:
                    best_value, best = value, candidate
                    moved = True
        # 只有**最后一层**的结论被保留：最细一层不再移动才算收敛。
        converged = not moved
    value, gains = evaluator.objective(best, views)
    return best, float(value), gains, bool(converged)
