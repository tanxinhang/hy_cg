"""cancellation 积木：接收机测量。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from isac_sim.receiver.cancellation.arms import cancellation_arms
from isac_sim.receiver.cancellation.build import build_observation
from isac_sim.receiver.cancellation.constants import EPS
from isac_sim.receiver.cancellation.context import ReceiverContext
from isac_sim.receiver.cancellation.measure_buffers import MeasureBuffers
from isac_sim.receiver.cancellation.measure_fill import fill_receiver, fill_targets
from isac_sim.receiver.cancellation.measurement import ReceiverMeasurement


def measure_receiver_context(
    ctx: ReceiverContext,
    *,
    rng: np.random.Generator,
    receivers: Sequence[int] | None = None,
    weak_index: int | None = None,
    all_targets: bool = False,
    residual_quantile_probability: float | None = None,
) -> ReceiverMeasurement:
    """在给定上下文的每个接收机上跑一遍估计器。

    信念几何就是调用方放进上下文的那一份。把真值当信念传进来得到的是
    **oracle-belief 深度** —— 那是 TP-UIC 能力的上界，不是对声明系统的测量；
    声明系统写了 ``sigma_p``/``sigma_v``，必须用该 trial 的调度器实际看到的
    扰动几何去测。

    子积木：缓冲区在 :mod:`measure_buffers`，逐接收机 / 逐目标的写入在
    :mod:`measure_fill`。
    """
    cfg = ctx.cfg
    m = int(cfg.scale.M)
    if receivers is None:
        receivers = range(m)
    buf = MeasureBuffers.create(
        cfg, all_targets=all_targets,
        residual_quantile_probability=residual_quantile_probability,
    )

    for j in receivers:
        obs = build_observation(
            cfg, ctx.geom_true, ctx.geom_belief, ctx.base, int(j), rng=rng,
            sense_power=ctx.sense_power, radiated_power=ctx.radiated_power,
            processing_gain=ctx.processing_gain, hw_gain=ctx.hw_gain,
            active_mask=ctx.active_mask, base_belief=ctx.base_belief,
            include_echo=True,
            weak_index=int(weak_index if weak_index is not None else 0),
        )
        prune = bool(getattr(cfg.cancellation, "measure_prune_arms", True))
        only = ctx.arm if prune and ctx.arm != "perfect_channel" else None
        result = cancellation_arms(cfg, obs, only=only)[ctx.arm]
        fill_receiver(buf, int(j), result, residual_quantile_probability)
        fill_targets(
            cfg, buf, int(j), obs, result, ctx.arm, residual_quantile_probability
        )
    with np.errstate(divide="ignore"):
        kappa = -10.0 * np.log10(np.clip(buf.fraction, EPS, None))
    predicted_fraction = np.ones(m, dtype=float)
    positive_input = buf.i_in_pred > 0.0
    predicted_fraction[positive_input] = (
        buf.i_res_pred[positive_input] / buf.i_in_pred[positive_input]
    )
    with np.errstate(divide="ignore"):
        predicted_kappa = -10.0 * np.log10(np.clip(predicted_fraction, EPS, None))
    model_fraction = np.ones(m, dtype=float)
    model_fraction[positive_input] = (
        buf.i_res[positive_input] / buf.i_in_pred[positive_input]
    )
    return ReceiverMeasurement(
        fraction=buf.fraction,
        kappa_db=kappa,
        eta_survive=buf.eta_field,
        eta_survive_q=buf.eta_q,
        eta_protect=buf.eta_protect,
        i_res=buf.i_res,
        i_in=buf.i_in,
        i_res_estimate=buf.i_res_est,
        i_res_pred=buf.i_res_pred,
        i_res_moment_mean=buf.i_res_moment_mean,
        i_res_pred_var=buf.i_res_pred_var,
        i_in_pred=buf.i_in_pred,
        predicted_fraction=predicted_fraction,
        predicted_kappa_db=predicted_kappa,
        model_fraction=model_fraction,
        noise_enhance_db=buf.noise_db,
        selected_soft_mu=buf.selected_soft_mu,
        fraction_by_target=buf.fraction_by_target,
        predicted_fraction_by_target=buf.predicted_fraction_by_target,
        model_fraction_by_target=buf.model_fraction_by_target,
        i_res_by_target=buf.i_res_by_target,
        i_res_pred_by_target=buf.i_res_pred_by_target,
        moment_mean_by_target=buf.moment_mean_by_target,
        moment_var_by_target=buf.moment_var_by_target,
        prior_quantile_fraction=buf.prior_quantile_fraction,
        prior_quantile_fraction_by_target=buf.prior_quantile_fraction_by_target,
        selected_soft_mu_by_target=buf.selected_soft_mu_by_target,
        eta_survive_by_target=buf.eta_by_target,
        predicted_retention_by_target=buf.predicted_eta_by_target,
        risk_retention_by_target=buf.risk_eta_by_target,
        arm=ctx.arm,
        belief_is_truth=ctx.geom_belief is ctx.geom_true,
    )
