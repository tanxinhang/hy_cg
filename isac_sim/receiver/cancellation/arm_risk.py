"""cancellation 积木：延迟-多普勒失配的 sigma 点风险。

一小簇**确定性** sigma 点可以在没有真值目标的前提下给 DD 失配定价：每个点把该
目标的每条信念路径沿链路局部的 95% 边际偏移挪一下，取最小存活率作为**保守能力**
—— 这不是校准过的联合置信界。
"""

from __future__ import annotations

import math

import numpy as np

from isac_sim.receiver.cancellation.manifold import kernel_vector
from isac_sim.receiver.cancellation.steering import lift_dictionary

Z95 = 1.6448536269514722          # 一维 95% 分位
RADIUS95_4D = 3.080215745168048   # sqrt(chi2_4(.95))，四维 95% 半径


def _point_matrix(op, cols, bearings):
    return lift_dictionary(op.cfg, np.column_stack(cols), bearings)


def marginal_sigma_points(op, risk_values) -> None:
    """逐条路径独立地按链路局部 95% 边际偏移挪动（k / l / u 三轴各正负一次）。"""
    for axis, sign in (("k", 1.0), ("k", -1.0),
                       ("l", 1.0), ("l", -1.0),
                       ("u", 1.0), ("u", -1.0)):
        cols, bearings = [], []
        for src in op.ctx.sources_q:
            dk = sign * Z95 * float(src.sigma_doppler_bin or 0.0) if axis == "k" else 0.0
            dl = sign * Z95 * float(src.sigma_delay_bin or 0.0) if axis == "l" else 0.0
            du = (
                sign * Z95 * float(src.sigma_bearing_u or 0.0)
                if axis == "u" else 0.0
            )
            cols.append(
                math.sqrt(max(float(src.power), 0.0))
                * kernel_vector(
                    op.cfg, float(src.doppler_bin) + dk, float(src.delay_bin) + dl
                )
            )
            bearings.append(float(np.clip(float(src.u) + du, -1.0, 1.0)))
        if cols:
            risk_values.append(op.survival(_point_matrix(op, cols, bearings)))


def state_sigma_points(op, risk_values) -> None:
    """相关状态 sigma 点：一次水平位置/速度扰动会**相干地**挪动该目标每条
    双基地路径的 DD 与方位。半径取四维高斯的 95% 包围；场景模型里 z 与 v_z
    是确定性的，因此不带虚构协方差。"""
    cfg = op.cfg
    scale = np.asarray([
        float(cfg.prior.belief_sigma_pos_m), float(cfg.prior.belief_sigma_pos_m), 0.0,
        float(cfg.prior.belief_sigma_vel_mps), float(cfg.prior.belief_sigma_vel_mps), 0.0,
    ])
    for dim in (0, 1, 3, 4):
        if scale[dim] <= 0.0:
            continue
        for sign in (1.0, -1.0):
            delta = sign * RADIUS95_4D * scale[dim]
            cols, bearings = [], []
            for src in op.ctx.sources_q:
                gk, gl, gu = (
                    src.jacobian_doppler_state,
                    src.jacobian_delay_state,
                    src.jacobian_bearing_state,
                )
                dk = 0.0 if gk is None else float(gk[dim]) * delta
                dl = 0.0 if gl is None else float(gl[dim]) * delta
                du = 0.0 if gu is None else float(gu[dim]) * delta
                cols.append(
                    math.sqrt(max(float(src.power), 0.0))
                    * kernel_vector(
                        op.cfg, float(src.doppler_bin) + dk, float(src.delay_bin) + dl
                    )
                )
                bearings.append(float(np.clip(float(src.u) + du, -1.0, 1.0)))
            if cols:
                risk_values.append(op.survival(_point_matrix(op, cols, bearings)))


def arm_risk(op, eta_pred_q: float) -> float:
    """该臂的保守存活率：所有 sigma 点里最坏的一个。

    ``eta_pred_q`` 由调用方在矩之后算好再传进来，以保持原闭包的求值顺序。
    """
    risk_values = [eta_pred_q]
    marginal_sigma_points(op, risk_values)
    state_sigma_points(op, risk_values)
    return float(min(risk_values))
