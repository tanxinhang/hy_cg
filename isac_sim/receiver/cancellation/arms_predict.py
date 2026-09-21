"""cancellation 积木：解析残差预测（打分阶段的 ``i_res_pred``）。"""

from __future__ import annotations

import numpy as np

from isac_sim.receiver.cancellation.arm import _Arm
from isac_sim.receiver.cancellation.joint import joint_interference_covariance
from isac_sim.receiver.cancellation.accounting import (
    residual_accounting,
    soft_residual_accounting,
)


def predicted_powers(ctx, arm: _Arm, prior):
    """返回 ``(retained, pred_estimate, i_res_pred)``。

    ``none``：什么都没减，残余就是输入场。
    ``exact``：该臂已知直达分量，于是它的深度是一个**定义**而非预测，标定误差
    按构造为零，不是碰巧为零。
    """
    if arm.predict != "estimator":
        retained = pred_estimate = 0.0
        i_res_pred = float(np.vdot(ctx.x, ctx.x).real) if arm.predict == "none" else None
        return retained, pred_estimate, i_res_pred
    if arm.soft_mu is not None:
        retained, pred_estimate = soft_residual_accounting(
            ctx.X, arm.subspace, ctx.sigma2, prior, float(arm.soft_mu)
        )
    elif arm.candidates:
        cov_h = joint_interference_covariance(
            ctx.X, ctx.A[:, list(arm.candidates)], ctx.sigma2, prior, prior or 1.0
        )
        retained = 0.0
        pred_estimate = float(np.real(np.trace(cov_h @ (ctx.X.conj().T @ ctx.X))))
    else:
        retained, pred_estimate = residual_accounting(
            ctx.X, arm.subspace, ctx.sigma2, prior
        )
    return retained, pred_estimate, retained + pred_estimate
