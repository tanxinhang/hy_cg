"""积木：偏移搜索用的缓存打分器。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from isac_sim.core.config import Config
from isac_sim.receiver.cancellation.containers import Observation
from isac_sim.receiver.cancellation.dictionaries import target_dictionary
from isac_sim.receiver.cancellation.manifold import orthonormalise
from isac_sim.receiver.cancellation_glrt.linalg import (
    _project_out,
    _projector_energy,
)

__all__ = ["TargetStateScorer", "build_scorer", "neighbourhood_energy"]


@dataclass(frozen=True)
class TargetStateScorer:
    """一份 (CPI, 接收机) 上**与偏移无关**的量，供搜索反复复用。

    ``z`` 是 TP-UIC 之后的白化残差，``q_neg`` 是其它目标中心列白化后的标准
    正交基。方案 §2 把 ``z`` 定义成不随 ``delta`` 变化的量，于是两者只需各算
    一次：一次 arm 装配（~0.7 s）摊到几十次候选评价上。

    投影链与 :func:`target_conditioned_glrt` 逐字一致（``signal_transfer`` ->
    ``whiten_matrix`` -> ``_project_out`` -> ``_projector_energy``）；一致性由
    ``tests/test_target_state_map.py`` 的逐位对照钉住。
    """

    cfg: Config
    model: object
    target: int
    z: np.ndarray
    q_neg: np.ndarray
    nuisance_columns: int = 0

    def energy(self, sources: Sequence, *, tested_only: bool = False) -> float:
        """``G_j(delta)`` —— 白化残差在偏移后目标字典上的投影能量。

        ``tested_only=True`` 只建被测目标那几列：其余目标的源在
        :func:`shifted_target_sources` 里原样不动，其字典列与 ``delta`` 无关，
        且早已被 ``q_neg`` 正交掉 —— 实测 120 次评价最坏相对差 1.5e-14
        （``tests/test_target_state_map.py`` 钉住）。模板从 15 列降到 5 列，
        单次评价 25.8 ms -> ~8 ms；搜索用它，最终 held-out 统计量仍走完整的
        :func:`target_neighbourhood_glrt`。
        """
        if tested_only:
            sources = [s for s in sources if int(s.target) == int(self.target)]
        template = target_dictionary(
            self.cfg, sources, tangent_order=0, covariance_expanded=False
        )
        if template.shape[1] == 0:
            return 0.0
        w = self.model.cov.whiten_matrix(self.model.signal_transfer(template))
        return float(_projector_energy(_project_out(self.q_neg, w), self.z))


def build_scorer(cfg: Config, obs: Observation, model,
                 target: int) -> TargetStateScorer:
    """从一个**名义**信念的观测与其残差模型造出打分器。"""
    ids = np.asarray(obs.A_target_ids)
    mask = np.asarray(obs.A_centre_mask, dtype=bool)
    A_neg = np.asarray(obs.A)[:, (ids != int(target)) & mask]
    w_neg = model.cov.whiten_matrix(model.signal_transfer(A_neg))
    return TargetStateScorer(
        cfg=cfg,
        model=model,
        target=int(target),
        z=model.cov.whiten(model.residual(obs.y)),
        q_neg=orthonormalise(w_neg).U,
        nuisance_columns=int(A_neg.shape[1]),
    )


def neighbourhood_energy(scorer: TargetStateScorer, sources: Sequence,
                         offsets: Sequence[float]) -> float:
    """局部 DD 网格上的最大值聚合，对齐 ``target_neighbourhood_glrt``。"""
    best = -np.inf
    for dl in offsets:
        for dk in offsets:
            shifted = [
                replace(src,
                        delay_bin=float(src.delay_bin) + float(dl),
                        doppler_bin=float(src.doppler_bin) + float(dk))
                for src in sources
            ]
            value = scorer.energy(shifted)
            if value > best:
                best = value
    return float(best if np.isfinite(best) else 0.0)
