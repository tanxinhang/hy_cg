"""cancellation 积木：自适应软保护臂（信念侧 Pareto 安全折中）。

候选选择只读解析残差矩与信念侧风险存活率。真值回波/噪声会影响被选中臂的
**实现**输出，但从不参与乘子的选择。若没有软点能在两份合同上同时不劣于硬臂，
自适应臂就是硬臂的精确回退。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

import numpy as np

from isac_sim.receiver.cancellation.arm import _Arm
from isac_sim.receiver.cancellation.arm_operator import run_arm
from isac_sim.receiver.cancellation.prior_quantile import (
    _residual_quantile_from_descriptors,
)


def power_not_worse(value: float, reference: float) -> bool:
    """相对容差比较。

    典型场景下的残差功率是 O(1e-13)。固定的 1e-12 容差会让每个候选都"安全"，
    等于重犯本模块别处已经守住的那个量纲 EPS bug。
    """
    scale = max(abs(float(reference)), np.finfo(float).tiny)
    return float(value) <= float(reference) + 1e-10 * scale


def build_adaptive_arm(ctx, parts: Dict[str, _Arm], soft_belief) -> None:
    """在软惩罚网格上挑一个 Pareto 安全的点；挑不到就精确回退到硬臂。"""
    cfg = ctx.cfg
    if not cfg.cancellation.adaptive_soft_enable:
        return
    hard = parts["tp_uic_full"]
    slack = float(cfg.cancellation.adaptive_soft_risk_slack)
    quantile = float(cfg.cancellation.adaptive_soft_residual_quantile)
    soft_candidates = [
        run_arm(ctx, "adaptive_soft_tpuic", soft_belief, ctx.prior, soft_mu=float(mu))
        for mu in cfg.cancellation.adaptive_soft_mu_grid
    ]
    hard_tail = _residual_quantile_from_descriptors(
        hard.residual_direct_gram, hard.residual_noise_eigenvalues, quantile
    )
    soft_tail = {
        id(arm): _residual_quantile_from_descriptors(
            arm.residual_direct_gram, arm.residual_noise_eigenvalues, quantile
        )
        for arm in soft_candidates
    }
    feasible = [
        arm for arm in soft_candidates
        if (
            arm.eta_survive_risk_q + slack >= hard.eta_survive_risk_q - 1e-12
            and power_not_worse(arm.i_res_moment_mean, hard.i_res_moment_mean)
            and power_not_worse(soft_tail[id(arm)], hard_tail)
        )
    ]
    if feasible:
        parts["adaptive_soft_tpuic"] = min(
            feasible,
            key=lambda arm: (
                soft_tail[id(arm)],
                arm.i_res_moment_mean,
                -arm.eta_survive_risk_q,
                float(arm.soft_mu or 0.0),
            ),
        )
    else:
        parts["adaptive_soft_tpuic"] = replace(
            hard, name="adaptive_soft_tpuic", soft_mu=None
        )
