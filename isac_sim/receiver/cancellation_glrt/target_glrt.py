"""目标条件化检测用的模板积木：字典、中心列掩码、其他目标的流形列。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from isac_sim.receiver.cancellation import (
    Observation,
    TargetSource,
    lift_dictionary,
    steering_derivative,
    tangent_columns,
    target_dictionary,
)
from isac_sim.receiver.cancellation_glrt.chi2 import glrt_p_value


@dataclass
class TargetGLRT:
    """V1.1 检测器对一组 (臂, 观测, 目标) 的输出。"""

    arm: str
    target: int
    statistic: float
    dof_real: int
    threshold: float
    p_fa: float
    detected: bool
    rho: np.ndarray  # 逐模板，落在 [0, 1]
    g_self: np.ndarray  # 逐模板，白化后的匹配滤波能量
    ncp_unit: float  # 单位幅度回波的期望偏移
    ncp_best: float  # 单位范数回波向量上的最好情况偏移
    rho_weighted: float  # 能量加权的逃逸比例，落在 [0, 1]
    xi_q: float  # lambda_min(B_q^H B_q)
    xi_rel_q: float  # 无标度版本，落在 [0, 1]
    n_templates: int
    n_nuisance_columns: int
    nuisance_rank: int
    residual_power: float
    whitened_residual_power: float = 0.0

    @property
    def rho_min(self) -> float:
        """最坏那个模板的可辨识度。"""
        return float(self.rho.min()) if self.rho.size else float("nan")

    @property
    def p_value(self) -> float:
        """该统计量在原假设下的尾概率。"""
        return glrt_p_value(self.statistic, self.dof_real)


def _dictionary(cfg, obs: Observation, which: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(字典, 列 -> 目标 id)``，取自信念或真值。"""
    if which == "belief":
        return obs.A, np.asarray(obs.A_target_ids)
    if which == "truth":
        return target_dictionary(
            cfg, obs.targets, covariance_expanded=False
        ), np.asarray(obs.A_true_target_ids)
    raise ValueError("dictionary must be 'belief' or 'truth', got %r" % (which,))


def _centre_mask(cfg, n_cols: int) -> np.ndarray:
    """挑出每个源块里的中心列。

    ``tangent_columns`` 把中心列放在最前，后面跟着 ``2*order`` 个分数 DD 导数；
    字典按源重复这个布局。正切列是**导数**，即"回波到底在哪"的不确定性，不是
    又一条回波签名：把两者混进被测块会用不带独立信号的方向虚增自由度，并因为
    与可辨识性无关的理由把 ``rho`` 压低。
    """
    n_basis = 1 + 2 * int(cfg.cancellation.tangent_order)
    mask = np.zeros(int(n_cols), dtype=bool)
    if n_basis > 0:
        mask[::n_basis] = True
    return mask


def _dictionary_centre_mask(cfg, obs: Observation, which: str, n_cols: int) -> np.ndarray:
    """优先用观测自带的中心列掩码，没有时按 :func:`_centre_mask` 推断。"""
    explicit = (
        obs.A_centre_mask if which == "belief" else obs.A_true_centre_mask
    )
    if explicit is None:
        return _centre_mask(cfg, n_cols)
    mask = np.asarray(explicit, dtype=bool)
    if mask.shape != (int(n_cols),):
        raise ValueError(
            "%s dictionary centre mask has shape %r, expected (%d,)"
            % (which, mask.shape, int(n_cols))
        )
    return mask


def _manifold_columns(
    cfg,
    obs: Observation,
    target: int,
    order: int,
    step: float | None = None,
    dictionary: str = "belief",
    subset: Sequence[int] | None = None,
) -> np.ndarray:
    """**其他**目标的分数 DD 正切列。"""
    sources: Sequence[TargetSource] = (
        (obs.targets_belief if dictionary == "belief" else obs.targets)
        or obs.targets_belief
        or obs.targets
    )
    h = float(cfg.cancellation.tangent_step_bins if step is None else step)
    allowed = None if subset is None else {int(v) for v in subset}
    blocks = []
    for src in sources:
        if int(src.target) == int(target):
            continue
        if allowed is not None and int(src.target) not in allowed:
            continue
        amp = math.sqrt(max(float(src.power), 0.0))
        dd_block = amp * tangent_columns(
            cfg, src.doppler_bin, src.delay_bin, order, h
        )
        blocks.append(
            lift_dictionary(cfg, dd_block, [float(src.u)] * dd_block.shape[1])
        )
        m_rx = int(cfg.aperture.m_rx) if cfg.aperture.enable else 1
        if m_rx > 1:
            centre = amp * tangent_columns(
                cfg, src.doppler_bin, src.delay_bin, 0, h
            )[:, 0]
            blocks.append(
                np.kron(centre, steering_derivative(m_rx, float(src.u)))[:, None]
            )
    if not blocks:
        return np.zeros((int(obs.y.size), 0), dtype=complex)
    return np.concatenate(blocks, axis=1)
