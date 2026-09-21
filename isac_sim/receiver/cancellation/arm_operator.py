"""cancellation 积木：单臂算子（把共享上下文 + 臂选择装配成一个 ``_Arm``）。"""

from __future__ import annotations

import numpy as np

from isac_sim.receiver.cancellation.arm import _Arm
from isac_sim.receiver.cancellation.arm_moments import arm_moments
from isac_sim.receiver.cancellation.arm_risk import arm_risk
from isac_sim.receiver.cancellation.arm_solve import _ArmSolveMixin


def run_arm(ctx, name: str, subspace, pv,
            candidates: tuple = (), soft_mu: float | None = None) -> _Arm:
    """装配并求解一臂的便捷入口。"""
    return _ArmOperator(ctx, name, subspace, pv, candidates, soft_mu).build()


class _ArmOperator(_ArmSolveMixin):
    """一臂 = 共享上下文 + 该臂自己的（子空间，先验，候选集，软惩罚）。

    ``solve``/``removed``/``survival`` 来自 :class:`_ArmSolveMixin`；矩来自
    :mod:`arm_moments`；sigma 点风险来自 :mod:`arm_risk`。
    """

    def __init__(self, ctx, name: str, subspace, pv,
                 candidates: tuple = (), soft_mu: float | None = None) -> None:
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.X, self.A = ctx.X, ctx.A
        self.y, self.x, self.s, self.n = ctx.y, ctx.x, ctx.s, ctx.n
        self.sigma2 = ctx.sigma2
        self.n_bins = ctx.n_bins
        self.name = name
        self.subspace = subspace
        self.pv = pv
        self.candidates = candidates
        self.soft_mu = soft_mu
        self.mx = subspace.complement_matrix(self.X)
        self.full_subtraction = bool(candidates or soft_mu is not None)

    def build(self) -> _Arm:
        """对各个分量分别求解，再汇总成这一臂的中间结果。"""
        h_y, c_y = self.solve(self.y)
        h_x, _ = self.solve(self.x)
        h_s, _ = self.solve(self.s)
        h_n, _ = self.solve(self.n)
        h_q = self.solve(self.ctx.s_q)[0] if self.ctx.s_q is not None else None
        moment_mean, moment_var, direct_gram, spectrum = arm_moments(self)
        eta_pred_q = self.survival(self.ctx.belief_q)
        eta_risk_q = arm_risk(self, eta_pred_q)
        if self.full_subtraction:
            sub_q = None if h_q is None else self.X @ h_q
            direct_action = (self.X @ h_x, self.X @ h_s, self.X @ h_n)
        else:
            sub_q = None if h_q is None else self.mx @ h_q
            direct_action = (self.mx @ h_x, self.mx @ h_s, self.mx @ h_n)
        return _Arm(
            self.name, direct_action[0], direct_action[1], direct_action[2],
            h_y, c_y, self.subspace, self.candidates,
            sub_target_q=sub_q, eta_survive_pred_q=eta_pred_q,
            eta_survive_risk_q=eta_risk_q,
            i_res_moment_mean=moment_mean,
            i_res_moment_var=moment_var,
            residual_direct_gram=np.asarray(direct_gram, dtype=complex),
            residual_noise_eigenvalues=np.asarray(spectrum, dtype=float),
            soft_mu=None if self.soft_mu is None else float(self.soft_mu),
        )
