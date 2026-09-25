"""积木：双折 cross-fit（方案 §3）。"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from isac_sim.core.config import Config
from isac_sim.receiver.target_state.apply import apply_target_offset
from isac_sim.receiver.target_state.fit import fit_shared_target_offset
from isac_sim.receiver.target_state.results import CrossfitTargetStateResult
from isac_sim.receiver.target_state.scorer import TargetStateScorer

__all__ = ["crossfit_target_state_score"]


def crossfit_target_state_score(
    cfg: Config,
    look_a: Sequence,
    look_b: Sequence,
    target: int,
    belief,
    *,
    receivers: Sequence[int],
    belief_geometry,
    scorers_a: Sequence[TargetStateScorer],
    scorers_b: Sequence[TargetStateScorer],
    score_fn,
    prior_sigma_m: float | None = None,
    search_sigma: float = 4.0,
    solver: str = "coarse_to_fine",
    coarse_step_m: float | None = None,
    tested_only: bool = True,
    solver_options: dict | None = None,
    frozen_gate: dict | None = None,
) -> CrossfitTargetStateResult:
    """A 上估状态 -> 在 B 上打分；B 上估状态 -> 在 A 上打分；两者相加。

    ``score_fn(observations)`` 把一折上的一组（已应用偏移的）观测折成一个标量
    检测统计量；它由调用方提供，于是"held-out 统计量"可以换成任意臂，而信息
    隔离规则（A 的状态永不进 A 的检测）只在这一个地方实现。

    ``solver`` / ``solver_options`` 只透传给 :func:`fit_shared_target_offset`：
    换搜索策略不许碰到这里的统计逻辑与信息隔离。
    """
    started = time.perf_counter()
    statistics, fits, objectives, evaluations = [], [], [], 0
    converged = True
    folds = ((look_a, look_b, scorers_a), (look_b, look_a, scorers_b))
    for train, _held, scorers in folds:
        fit = fit_shared_target_offset(
            cfg, train, target, belief, receivers=receivers,
            belief_geometry=belief_geometry, scorers=scorers,
            prior_sigma_m=prior_sigma_m, search_sigma=search_sigma,
            solver=solver, coarse_step_m=coarse_step_m,
            tested_only=tested_only, solver_options=solver_options,
        )
        fits.append(fit)
        objectives.append(float(fit.objective))
        evaluations += int(fit.evaluations)
        converged = converged and bool(fit.converged)

    raw_deltas = [np.asarray(f.delta_xy_m, dtype=float) for f in fits]
    accepted = [True, True]
    if frozen_gate is not None:
        rule = frozen_gate["accept_rule"]
        fold_distance = float(np.linalg.norm(raw_deltas[0] - raw_deltas[1]))
        accepted = [bool(
            fit.gain_over_zero > float(rule["gain_over_zero_strictly_gt"])
            and fit.receiver_support >= int(rule["receiver_support_ge"])
            and fit.boundary_hit is bool(rule["boundary_hit_must_equal"])
            and fold_distance <= float(rule["cross_fold_distance_m_le"])
        ) for fit in fits]
    deltas = [raw if keep else np.zeros(2, dtype=float)
              for raw, keep in zip(raw_deltas, accepted)]
    for (_train, held, _scorers), delta in zip(folds, deltas):
        applied = [
            apply_target_offset(cfg, obs, target, delta,
                                receiver=int(rx), belief_geometry=belief_geometry)
            for rx, obs in zip(receivers, held)
        ]
        statistics.append(float(score_fn(applied)))
    return CrossfitTargetStateResult(
        target=int(target),
        statistic=float(np.sum(statistics)),
        fold_statistics=(float(statistics[0]), float(statistics[1])),
        delta_a_xy_m=np.asarray(deltas[0], dtype=float),
        delta_b_xy_m=np.asarray(deltas[1], dtype=float),
        objective_a=float(objectives[0]),
        objective_b=float(objectives[1]),
        evaluations=int(evaluations),
        seconds=float(time.perf_counter() - started),
        converged=bool(converged),
        accepted_a=accepted[0],
        accepted_b=accepted[1],
        raw_delta_a_xy_m=raw_deltas[0],
        raw_delta_b_xy_m=raw_deltas[1],
    )
