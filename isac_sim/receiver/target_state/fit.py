"""积木：一个 CPI 上共享偏移 MAP 的求解器派发（方案 §6）。

两个求解器共用 :class:`ObjectiveEvaluator`，于是候选缓存可以跨阶段复用：

- ``coarse_to_fine`` —— 论文基线 + 回归参考（Gate 1/2 的口径），行为冻结在
  :mod:`coarse`，**评价集合一字不改**；
- ``gated_topk`` —— 新搜索策略（:mod:`gated_topk`），默认关。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from isac_sim.core.config import Config
from isac_sim.receiver.target_state.coarse import coarse_to_fine_search
from isac_sim.receiver.target_state.gated_topk import fit_gated_topk
from isac_sim.receiver.target_state.wls import fit_jacobian_wls
from isac_sim.receiver.target_state.objective import ObjectiveEvaluator
from isac_sim.receiver.target_state.results import TargetStateMAPResult
from isac_sim.receiver.target_state.scorer import TargetStateScorer
from isac_sim.receiver.target_state.views import make_receiver_views

__all__ = ["fit_shared_target_offset", "make_receiver_views", "SOLVERS"]

SOLVERS = ("coarse_to_fine", "gated_topk", "jacobian_wls")

# 方案 §6 写的是"±4σ/±2σ/0 的 5×5 粗网格"，即 **300 m** 间距（σ=150）。
# 实测峰半高宽只有 25--75 m（studies/direction4/scripts/measure_correlation_peak_width.py：峰值 1463 在
# ±50 m 处掉到 167），300 m 间距落在峰内的概率≈0，粗选阶段退化成"在噪声包里
# 挑最大的那个"—— Gate-0 五个场景里两个因此跑飞 335 m（真值处目标函数明明
# 高 6.4 倍）。所以粗网格间距改成**由 DD 相关峰宽定标**：默认 σ/2，
# 细化步长按粗网格步长的分数给。
_COARSE_STEP_FRAC = 0.5
_EPS = 1e-12


def fit_shared_target_offset(
    cfg: Config,
    observations: Sequence,
    target: int,
    belief,
    *,
    receivers: Sequence[int],
    belief_geometry,
    scorers: Sequence[TargetStateScorer],
    prior_sigma_m: float | None = None,
    search_sigma: float = 4.0,
    solver: str = "coarse_to_fine",
    coarse_step_m: float | None = None,
    tested_only: bool = True,
    solver_options: dict | None = None,
) -> TargetStateMAPResult:
    """在**一个** CPI 上、用所有接收机共同估计一个二维共享偏移。

    ``belief`` 只读先验协方差 ``P[q, 0:2, 0:2]``（声明的那一档，不是实际注入的
    误差）；``belief_geometry`` 是接收机已知的那份几何。``scorers`` 与
    ``observations`` 逐个对齐，且必须由**名义**信念的 arm 装配产出。

    ``coarse_step_m`` 是粗网格间距（米）。默认 ``σ/2``，因为 G(δ) 的相关峰宽
    只有几十米；给 ``None`` 之外的较小值只会更慢，不会更准。

    ``tested_only`` 让搜索只建被测目标的字典列（同值、快 ~3 倍），见
    :meth:`TargetStateScorer.energy`。``solver_options`` 只被 ``gated_topk`` 读。
    """
    if str(solver) not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}")
    views = make_receiver_views(receivers, observations, scorers, target)
    if not views:
        raise ValueError("no receiver view to fit")

    sigma = (
        float(np.sqrt(max(float(np.asarray(belief.P)[int(target), 0, 0]), 0.0)))
        if prior_sigma_m is None else float(prior_sigma_m)
    )
    span = float(search_sigma) * sigma
    step = float(_COARSE_STEP_FRAC * sigma if coarse_step_m is None
                 else coarse_step_m)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(f"bad coarse_step_m {coarse_step_m!r}")
    belief_pos = np.asarray(belief_geometry.p_tgt)[int(target), :3].astype(float)
    evaluator = ObjectiveEvaluator(cfg, views, target, belief_geometry, sigma,
                                   tested_only)
    extra: dict = {}
    if str(solver) == "coarse_to_fine":
        best, value, gains, converged = coarse_to_fine_search(
            evaluator, views, span, step)
        # Zero belongs to every symmetric coarse grid.  Reading it again hits
        # ObjectiveEvaluator's cache, so these deployable diagnostics do not
        # alter the frozen candidate set or receiver computations.
        zero_value, zero_gains = evaluator.objective(np.zeros(2), views)
        extra = {
            "objective_at_zero": float(zero_value),
            "gain_over_zero": float(value - zero_value),
            "boundary_hit": bool(
                span - float(np.max(np.abs(best))) <= step + 1e-9),
            "receiver_support": int(np.sum(
                np.asarray(gains) > np.asarray(zero_gains))),
        }
    elif str(solver) == "gated_topk":
        out = fit_gated_topk(evaluator, views, span, step, solver_options)
        best, value = out["best"], out["value"]
        gains, converged = out["gains"], out["converged"]
        extra = {k: v for k, v in out.items()
                 if k not in ("best", "value", "gains", "converged")}
    else:
        out = fit_jacobian_wls(evaluator, views, sigma, solver_options)
        best, value = out["best"], out["value"]
        gains, converged = out["gains"], out["converged"]
        extra = {k: v for k, v in out.items()
                 if k not in ("best", "value", "gains", "converged")}
    penalty = 0.5 * float(np.dot(best, best)) / max(float(sigma), _EPS) ** 2
    return TargetStateMAPResult(
        target=int(target),
        delta_xy_m=np.asarray(best, dtype=float),
        estimated_position_m=np.array([
            belief_pos[0] + float(best[0]),
            belief_pos[1] + float(best[1]),
            belief_pos[2],
        ], dtype=float),
        objective=float(value),
        prior_penalty=float(penalty),
        receiver_gain=np.asarray(gains, dtype=float),
        converged=bool(converged),
        evaluations=int(evaluator.evaluations),
        solver=str(solver),
        raw_delta_xy_m=np.asarray(best, dtype=float),
        receiver_evaluations=int(evaluator.receiver_evaluations),
        **extra,
    )
