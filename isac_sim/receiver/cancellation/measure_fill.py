"""cancellation 积木：把一次对消结果写进测量缓冲区。"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.receiver.cancellation.arms import cancellation_arms
from isac_sim.receiver.cancellation.containers import Observation
from isac_sim.receiver.cancellation.measure_buffers import MeasureBuffers
from isac_sim.receiver.cancellation.prior_quantile import residual_power_prior_quantile
from isac_sim.receiver.cancellation.result import CancellationResult


def fill_receiver(buf: MeasureBuffers, j: int, result: CancellationResult,
                  residual_quantile_probability: float | None) -> None:
    """把被测接收机的整场口径写进缓冲区。

    观测从未承载过的目标，其存活率是未定义的，而 ``1.0`` 是乘性分子唯一
    的中性值 —— 填 0 会被读成"回波被摧毁了"。
    """
    buf.i_in[j] = float(result.i_in)
    buf.i_res[j] = float(result.i_res)
    buf.i_res_est[j] = float(result.i_res_estimate)
    buf.i_res_pred[j] = float(result.i_res_pred)
    buf.i_res_moment_mean[j] = float(result.i_res_moment_mean)
    buf.i_res_pred_var[j] = float(result.i_res_pred_var)
    if buf.prior_quantile_fraction is not None and result.i_in_pred > 0.0:
        buf.prior_quantile_fraction[j] = (
            residual_power_prior_quantile(
                result, float(residual_quantile_probability)
            ) / float(result.i_in_pred)
        )
    buf.i_in_pred[j] = float(result.i_in_pred)
    buf.noise_db[j] = float(result.noise_enhance_db)
    if result.soft_mu is not None:
        buf.selected_soft_mu[j] = float(result.soft_mu)
    if result.i_in > 0.0:
        buf.fraction[j] = float(result.i_res / result.i_in)
    buf.eta_field[j] = float(result.eta_survive)
    buf.eta_q[j] = float(result.eta_survive_q)
    buf.eta_protect[j] = float(result.eta_protect)
    if result.s_q_energy <= 0.0:
        buf.eta_q[j] = 1.0


def fill_targets(cfg: Config, buf: MeasureBuffers, j: int, obs: Observation,
                 result: CancellationResult, arm: str,
                 residual_quantile_probability: float | None) -> None:
    """逐目标口径：对每个目标各跑一次对消，取该目标自己的那份证书。"""
    if buf.eta_by_target is None:
        return
    q0 = int(obs.weak_index)
    for q in range(int(cfg.scale.Q)):
        if q == q0:
            rq = result
        else:
            prune = bool(getattr(cfg.cancellation, "measure_prune_arms", True))
            only = arm if prune and arm != "perfect_channel" else None
            rq = cancellation_arms(
                cfg, obs, weak_target=q, only=only
            )[arm]
        buf.eta_by_target[j, q] = (
            float(rq.eta_survive_q) if rq.s_q_energy > 0.0 else 1.0
        )
        buf.i_res_by_target[j, q] = float(rq.i_res)
        buf.i_res_pred_by_target[j, q] = float(rq.i_res_pred)
        if rq.i_in > 0.0:
            buf.fraction_by_target[j, q] = float(rq.i_res / rq.i_in)
        if rq.i_in_pred > 0.0:
            buf.predicted_fraction_by_target[j, q] = float(
                rq.i_res_pred / rq.i_in_pred
            )
            buf.model_fraction_by_target[j, q] = float(rq.i_res / rq.i_in_pred)
        buf.moment_mean_by_target[j, q] = float(rq.i_res_moment_mean)
        buf.moment_var_by_target[j, q] = float(rq.i_res_pred_var)
        if (
            buf.prior_quantile_fraction_by_target is not None
            and rq.i_in_pred > 0.0
        ):
            buf.prior_quantile_fraction_by_target[j, q] = (
                residual_power_prior_quantile(
                    rq, float(residual_quantile_probability)
                ) / float(rq.i_in_pred)
            )
        buf.predicted_eta_by_target[j, q] = float(rq.eta_survive_pred_q)
        buf.risk_eta_by_target[j, q] = float(rq.eta_survive_risk_q)
        if rq.soft_mu is not None:
            buf.selected_soft_mu_by_target[j, q] = float(rq.soft_mu)
