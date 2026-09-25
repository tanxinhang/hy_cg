"""积木：状态更新的四个诊断量（阶段 A **只算不判**）。

门限要靠训练折上的分布来定（阶段 D，40 个 0 m 场景采分布取分位数），所以阶段 A
先把四个量算出来落盘，``accepted`` 恒 ``True``：既能分清"搜索错了"还是"门控拒绝
了"，也避免把"搜索策略"和"门限"两条主线混在一起调。

``objective_at_zero`` 是**额外**的一次候选评价 —— 所以只有 ``gated_topk`` 会填
这些字段，``coarse_to_fine`` 保持原有的评价集合不变（bit-exact，铁律 1）。
"""
from __future__ import annotations

import numpy as np

__all__ = ["state_diagnostics"]


def state_diagnostics(evaluator, views, best, best_value, second_value,
                      span, step) -> dict:
    """``gain_over_zero`` / ``peak_gap`` / ``boundary_hit`` / ``receiver_support``。

    ``peak_gap`` 必须比**次优局部最优**，不是次优格点：格点间距 75 m 小于峰宽时
    相邻格点的差几乎恒为 0，用它会把真峰误判成平顶。只有一个峰时记 ``-inf``。
    """
    zero_value, zero_gains = evaluator.objective(np.zeros(2), views)
    gains = evaluator.gains(best, views)
    return {
        "objective_at_zero": float(zero_value),
        "objective_second": float(second_value),
        "gain_over_zero": float(best_value - zero_value),
        "peak_gap": float(best_value - second_value),
        "boundary_hit": bool(float(span) - float(np.max(np.abs(best)))
                             <= float(step) + 1e-9),
        "receiver_support": int(np.sum(gains > zero_gains)),
        "boundary_expanded": False,
        "boundary_unresolved": False,
    }
