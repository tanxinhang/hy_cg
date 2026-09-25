"""积木：``gated_topk`` 求解器的编排 —— screen -> halving -> NMS -> refine -> 诊断。

成本（σ=150、span=600、step=75、6 接收机，单位：次接收机评价）：

    coarse_to_fine   289×6 + 3×8×6                    ≈ 1878
    gated_topk       289×2 + 40×2 + 12×2 + 3×3×8×6    ≈ 1114   ⇒ ~1.7×

⚠️ 加速**不是**主要收益（细化一项就占 ~40%）。主要收益是多峰细化带来的状态误差
p90 改善，以及阶段 D 起才生效的门控。
"""
from __future__ import annotations

import numpy as np

from isac_sim.receiver.target_state.gate import state_diagnostics
from isac_sim.receiver.target_state.nms import non_maximum_suppression
from isac_sim.receiver.target_state.refine import pattern_search
from isac_sim.receiver.target_state.screen import coarse_grid, receiver_order

__all__ = ["DEFAULT_OPTIONS", "fit_gated_topk"]

DEFAULT_OPTIONS = {
    "screen_receivers": 2,
    "screen_keep": 40,
    "verify_receivers": 4,
    "verify_keep": 12,
    "top_k": 3,
    "refine_fractions": (0.5, 0.25, 0.125),
}
#: 最后一档 NMS 半径放宽到 4/3 个粗网格步长：此时只留 3 个 seed，宁可漏一个弱峰，
#: 也不想把同一个峰的两个格点当成两个 seed。
_FINAL_RADIUS_FRAC = 4.0 / 3.0


def fit_gated_topk(evaluator, views, span: float, step: float,
                   options: dict | None = None) -> dict:
    """返回 ``dict(best, value, gains, converged, **diagnostics)``。"""
    opts = dict(DEFAULT_OPTIONS)
    opts.update(options or {})
    span, step = float(span), float(step)
    stages = (
        (int(opts["screen_receivers"]), int(opts["screen_keep"]), step),
        (int(opts["verify_receivers"]), int(opts["verify_keep"]), step),
        (len(views), int(opts["top_k"]), _FINAL_RADIUS_FRAC * step),
    )
    order = receiver_order(views, evaluator.belief_geometry, evaluator.target)
    ordered = sorted(views, key=lambda v: order.index(int(v.receiver)))
    if span <= 0.0:
        points = [np.zeros(2)]
        coarse_candidates = 0
    else:
        points = coarse_grid(span, step)
        coarse_candidates = len(points)
    screen_receivers = 0
    for n_rx, keep, radius in stages:
        n_rx = max(1, min(int(n_rx), len(ordered)))
        screen_receivers = screen_receivers or n_rx
        if span > 0.0:
            values = np.array(
                [evaluator.objective(p, ordered[:n_rx])[0] for p in points])
            points, _ = non_maximum_suppression(points, values, keep, radius)
        if not points:
            points = [np.zeros(2)]
            break
    steps = tuple(float(f) * step for f in opts["refine_fractions"])
    refined = [pattern_search(evaluator, views, p, steps, span) for p in points]
    rank = sorted(range(len(refined)), key=lambda i: (-float(refined[i][1]), i))
    best, best_value, moved = refined[rank[0]]
    second = float(refined[rank[1]][1]) if len(rank) > 1 else -np.inf
    diagnostics = state_diagnostics(evaluator, views, best, best_value,
                                    second, span, step)
    value, gains = evaluator.objective(best, views)
    return dict(
        best=np.asarray(best, dtype=float), value=float(value), gains=gains,
        converged=bool(not moved), screen_receiver_count=int(screen_receivers),
        coarse_candidates=int(coarse_candidates),
        full_candidates=int(len(refined)), **diagnostics,
    )
