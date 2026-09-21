"""臂表：把 :mod:`arm_plan` 的形状填成线性探针直接能用的字典。"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from isac_sim.receiver.cancellation import (
    CancellationResult,
    Observation,
    orthonormalise,
)
from isac_sim.receiver.cancellation_glrt.arm_plan import (
    ArmPlan,
    _belief_subspace,
    _empty_subspace,
)


def arm_plans(
    cfg, obs: Observation, results: Dict[str, CancellationResult] | None = None
) -> Dict[str, ArmPlan]:
    """臂表，整理成线性探针需要的形状。

    ``adaptive_soft_tpuic`` 是唯一一条形状取决于结果的臂：它保护的是被测目标
    自己的子空间加上一个软惩罚，所以只有结果里真的有它才登记。
    """
    n_bins = int(obs.y.size)
    empty = _empty_subspace(n_bins)
    belief = _belief_subspace(obs)
    prior = cfg.cancellation.prior_variance if cfg.cancellation.prior_variance else None
    candidates: Tuple[int, ...] = ()
    if results is not None and "tp_uic_full" in results:
        candidates = tuple(int(c) for c in results["tp_uic_full"].candidates)
    plans = {
        "no_ic": ArmPlan("no_ic", "none", empty, None),
        "plain_ls": ArmPlan("plain_ls", "estimator", empty, None),
        "ridge_ls": ArmPlan("ridge_ls", "estimator", empty, prior),
        "protected_ls": ArmPlan("protected_ls", "estimator", belief, None),
        "tp_uic_stage1": ArmPlan("tp_uic_stage1", "estimator", belief, prior),
        "tp_uic_full": ArmPlan("tp_uic_full", "estimator", belief, prior, candidates),
        "perfect_channel": ArmPlan("perfect_channel", "oracle", empty, None),
    }
    if results is not None and "adaptive_soft_tpuic" in results:
        adaptive = results["adaptive_soft_tpuic"]
        if adaptive.soft_mu is None:
            plans["adaptive_soft_tpuic"] = ArmPlan(
                "adaptive_soft_tpuic", "estimator", belief, prior, candidates
            )
        else:
            ids = np.asarray(obs.A_target_ids)
            tested = int(obs.weak_index)
            target_subspace = orthonormalise(obs.A[:, ids == tested])
            plans["adaptive_soft_tpuic"] = ArmPlan(
                "adaptive_soft_tpuic", "estimator", target_subspace, prior,
                soft_mu=float(adaptive.soft_mu),
            )
    return plans
