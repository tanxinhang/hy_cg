"""发射功率划分与整张表共用的常数。

功率模型把每架 UAV 的功率分成**感知分量** ``rho * P`` 与**通信分量**
``(1-rho) * P``。``rho_by_uav`` 允许逐机给不同的划分比例。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model.mathkit import EPS, bandwidth, denominator_guard, noise_power


@dataclass
class PowerSplit:
    """一张链路表的全部功率与常数。"""

    #: 每架 UAV 的发射功率 (M,)
    P: np.ndarray
    #: 感知功率占比（标量或 (M,)）
    rho: Any
    #: 感知分量功率 (M,)
    P_sense: np.ndarray
    #: 通信分量功率 (M,)
    P_comm: np.ndarray
    #: 噪声功率
    n0: float
    #: 分母守卫项
    eps_den: float
    #: 带宽
    B: float
    #: 达到 R_min 所需的最小 SINR
    gamma_req: float


def power_split(
    cfg: Config, sensing_power_scale_by_uav: np.ndarray | None = None
) -> PowerSplit:
    """算功率划分与 ``n0 / eps_den / B / gamma_req``。

    ``sensing_power_scale_by_uav`` 在发射端施加主动模式功率：抬高某架 UAV 的
    感知功率既增强它自己的回波，也增加其它接收机遇见的干扰。
    默认全 1 时与已发布的被动模型完全一致。
    """
    M = cfg.scale.M
    r, c = cfg.radio, cfg.comm

    P = np.full(M, r.P_default, dtype=float)
    rho = r.rho
    if r.rho_by_uav is not None:
        rho = np.asarray(r.rho_by_uav, dtype=float)
        if rho.shape != (M,) or not np.all(np.isfinite(rho)) or np.any((rho <= 0) | (rho >= 1)):
            raise ValueError(f"rho_by_uav must contain {M} finite fractions in (0, 1)")
    P_sense = rho * P
    if sensing_power_scale_by_uav is not None:
        power_scale = np.asarray(sensing_power_scale_by_uav, dtype=float)
        if power_scale.shape != (M,) or not np.all(np.isfinite(power_scale)):
            raise ValueError(f"sensing power scale must be finite with shape ({M},)")
        if np.any(power_scale <= 0.0):
            raise ValueError("sensing power scales must be strictly positive")
        P_sense = P_sense * power_scale
    P_comm = (1.0 - rho) * P

    n0 = noise_power(cfg)
    eps_den = denominator_guard(cfg, n0)
    B = bandwidth(cfg)
    gamma_req = max(2.0 ** (c.R_min / B) - 1.0, EPS)
    return PowerSplit(
        P=P, rho=rho, P_sense=P_sense, P_comm=P_comm,
        n0=n0, eps_den=eps_den, B=B, gamma_req=gamma_req,
    )
