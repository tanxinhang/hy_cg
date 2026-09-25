"""积木：共享偏移的目标函数与 (候选, 接收机) 级缓存。

``J(delta) = sum_j G_j(delta) - 0.5 delta^T P^{-1} delta``。缓存粒度从"候选"降到
"候选 × 接收机"，是因为 successive-halving 要先用 2 个接收机筛几百个点、再用
4 个复筛、最后才用全部接收机细化：粒度到接收机，候选升级时只补算**新增**的那
几个，不重算已有的（``tests/test_target_state_gated_topk.py`` 钉住）。

``evaluations`` 记**去重后的候选数**（与 coarse-to-fine 时代的 ``len(cache)``
同义，现有断言 ``evaluations >= 150`` 依赖这个语义）；真正的成本记在
``receiver_evaluations``（候选 × 接收机）里。
"""
from __future__ import annotations

import numpy as np

from isac_sim.receiver.target_state.geometry import (
    shift_belief_geometry,
    shifted_target_sources,
)

__all__ = ["ObjectiveEvaluator"]

_EPS = 1e-12


class ObjectiveEvaluator:
    """一份训练 CPI 上的目标函数、缓存与评价计数。

    ``delta`` 必须先落到几何上再重建源：:func:`shifted_target_sources` 只认几何，
    不认偏移。漏掉这一步目标函数就与 ``delta`` 无关，于是 argmax 恒为先验中心
    0 —— 一个"看起来收敛了"的静默错误（本模块第一版就犯过）。
    """

    def __init__(self, cfg, views, target, belief_geometry, prior_sigma_m,
                 tested_only: bool = True, cache: dict | None = None):
        self.cfg = cfg
        self.views = list(views)
        self.target = int(target)
        self.belief_geometry = belief_geometry
        self.sigma = float(prior_sigma_m)
        self.tested_only = bool(tested_only)
        self.cache = {} if cache is None else cache
        self._seen: set = set()
        self.evaluations = 0            # 去重后的候选数
        self.receiver_evaluations = 0   # 候选 × 接收机：真正的成本

    def delta_key(self, delta) -> tuple[float, float]:
        d = np.asarray(delta, dtype=float).reshape(2)
        return (round(float(d[0]), 9), round(float(d[1]), 9))

    def receiver_gain(self, delta, view) -> float:
        key = self.delta_key(delta) + (int(view.receiver),)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        geom = shift_belief_geometry(self.belief_geometry, self.target, delta)
        gain = float(view.scorer.energy(
            shifted_target_sources(self.cfg, geom, view.receiver, self.target,
                                   view.sources),
            tested_only=self.tested_only))
        self.cache[key] = gain
        self.receiver_evaluations += 1
        return gain

    def gains(self, delta, views=None) -> np.ndarray:
        views = self.views if views is None else views
        dkey = self.delta_key(delta)
        if dkey not in self._seen:
            self._seen.add(dkey)
            self.evaluations += 1
        return np.array([self.receiver_gain(delta, v) for v in views],
                        dtype=float)

    def penalty(self, delta) -> float:
        d = np.asarray(delta, dtype=float).reshape(2)
        return 0.5 * float(np.dot(d, d)) / max(self.sigma, _EPS) ** 2

    def objective(self, delta, views=None):
        """``(J(delta), G(delta))``；``views`` 缺省用全部接收机。"""
        gains = self.gains(delta, views)
        return float(np.sum(gains)) - self.penalty(delta), gains
