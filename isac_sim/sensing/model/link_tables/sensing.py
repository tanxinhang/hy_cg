"""感知块：逐 (i,j,q) 的感知 SINR 与软统计量矩。

对级量（残余、RINR、sigma0、有效感知功率）交给
:mod:`~isac_sim.sensing.model.link_tables.pair_terms`，这里只做目标维的循环。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isac_sim.core.config import Config
from isac_sim.detection.llr import soft_mean as _soft_mean, soft_var0 as _soft_var0
from isac_sim.sensing.model.containers import BaseGains
from isac_sim.sensing.model.link_tables.fields import InterferenceFields
from isac_sim.sensing.model.link_tables.inputs import ReceiverInputs
from isac_sim.sensing.model.link_tables.pair_terms import pair_terms
from isac_sim.sensing.model.link_tables.power import PowerSplit
from isac_sim.sensing.model.link_tables.residual import ResidualModel
from isac_sim.sensing.model.mathkit import radar_hardware_gain


@dataclass
class SensingTables:
    """感知侧六张表。"""

    raw_gamma_sense: np.ndarray
    gamma_sense: np.ndarray
    rinr: np.ndarray
    mu_soft: np.ndarray
    sigma0: np.ndarray
    var0_q: np.ndarray


def sensing_tables(
    cfg: Config,
    base: BaseGains,
    power: PowerSplit,
    fields: InterferenceFields,
    residual: ResidualModel,
    inputs: ReceiverInputs,
    chi_comm: np.ndarray,
    active_tx_mask: np.ndarray | None,
    dd_used: np.ndarray,
) -> SensingTables:
    """建感知侧的表。"""
    M, Q = cfg.scale.M, cfg.scale.Q
    d = cfg.detect
    n0, eps_den = power.n0, power.eps_den
    retention = inputs.retention

    raw_gamma_sense = np.zeros((M, M, Q))
    gamma_sense = np.zeros((M, M, Q))
    rinr = np.zeros((M, M))
    mu_soft = np.zeros((M, M, Q))
    var0_q = np.zeros((M, M, Q))
    sigma0 = np.full((M, M), d.soft_sigma0)

    G_proc = (
        cfg.waveform.N * cfg.waveform.L
        if d.sensing_processing_gain is None
        else d.sensing_processing_gain
    )
    G_hw = radar_hardware_gain(cfg)

    for i in range(M):
        for j in range(M):
            if i == j:
                continue

            pt = pair_terms(
                cfg, base, power, fields, residual, chi_comm, active_tx_mask, M, i, j
            )
            rinr[i, j] = pt.rinr
            sigma0[i, j] = pt.sigma0
            residual_total = pt.residual_total
            residual_direct_by_target = pt.residual_direct_by_target
            effective_sensing_power = pt.effective_sensing_power

            for q in range(Q):
                if cfg.dd.use_otfs_bin_validity and (not base.valid_dd[i, j, q]):
                    continue

                # 原始感知 SINR 基线必须与所选 ISAC 功率模型保持一致。
                raw_signal = effective_sensing_power * base.target_gain[i, j, q] * G_proc * G_hw
                raw_gamma_sense[i, j, q] = raw_signal / (n0 + eps_den)

                collision_penalty = 1.0 / (
                    max(base.dd_collision_count[i, j, q], 1.0) ** cfg.dd.dd_collision_alpha
                )
                dd_loss = dd_used[i, j, q] if cfg.dd.enable_dd_fractional_penalty else 1.0

                # 接收机仿射映射下的回波存活率 —— 对消器的**分子**那一半，
                # 见 :func:`compute_link_tables` 的文档。它只作用在 ``gamma_sense``
                # 上：``raw_gamma_sense`` 是原始 SINR 基线，按构造描述的是接收机
                # 处理**之前**的链路，所以不能动。
                eta_ret = 1.0
                if retention is not None:
                    eta_ret = float(retention[j] if retention.ndim == 1 else retention[j, q])

                signal = (
                    effective_sensing_power * base.target_gain[i, j, q] * G_proc * G_hw
                    * collision_penalty * dd_loss * eta_ret
                )
                waveform_capture = 1.0
                waveform_inr = 0.0
                if cfg.waveform_impairments.enable:
                    wi = cfg.waveform_impairments
                    waveform_capture = float(
                        np.sinc(wi.sync_delay_bins) ** 2
                        * np.sinc(wi.sync_doppler_bins) ** 2
                    )
                    unresolved = max(base.dd_collision_count[i, j, q] - 1.0, 0.0)
                    waveform_inr = float(
                        wi.clutter_inr
                        + wi.multipath_inr
                        + wi.unresolved_target_inr * unresolved
                    )
                residual_total_q = residual_total
                if residual_direct_by_target is not None:
                    residual_total_q = (
                        pt.residual_self + float(residual_direct_by_target[q])
                        + pt.residual_multi
                    )
                gamma = (
                    signal * waveform_capture
                    / ((n0 + residual_total_q + eps_den) * (1.0 + waveform_inr))
                )
                gamma_sense[i, j, q] = gamma
                # 配置模型下的软统计量矩：旧的 ``kappa_mu * log(1+gamma)`` 高斯均值，
                # 或中心化局部 LLR 均值 ``L*gamma^2/(1+gamma)``。经 llr.py 分派，
                # 好让两者在 "gaussian" 模式逐位兼容。
                mu_soft[i, j, q] = _soft_mean(cfg, gamma)
                # H0 方差：旧模型广播对级 sigma0^2，LLR 模型由 SINR 反推。
                var0_q[i, j, q] = _soft_var0(cfg, gamma, float(sigma0[i, j]) ** 2)

    return SensingTables(
        raw_gamma_sense=raw_gamma_sense,
        gamma_sense=gamma_sense,
        rinr=rinr,
        mu_soft=mu_soft,
        sigma0=sigma0,
        var0_q=var0_q,
    )
