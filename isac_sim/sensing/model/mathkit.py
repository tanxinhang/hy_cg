"""初等数学与链路预算的基本量：波长、带宽、噪声功率、雷达硬件增益、分母守卫。"""

from __future__ import annotations

import numpy as np
from typing import Tuple
from isac_sim.core.config import Config


EPS = 1e-12

# ==========================================================================
# 初等数学辅助
# ==========================================================================
def wavelength(cfg: Config) -> float:
    return cfg.waveform.c / cfg.waveform.fc

def bandwidth(cfg: Config) -> float:
    return cfg.waveform.N * cfg.waveform.delta_f

def noise_power(cfg: Config) -> float:
    psd_w_hz = 10.0 ** ((cfg.radio.noise_psd_dbm_hz - 30.0) / 10.0)
    nf_linear = 10.0 ** (cfg.radio.noise_figure_db / 10.0)
    return psd_w_hz * bandwidth(cfg) * nf_linear

def radar_hardware_gain(cfg: Config) -> float:
    """雷达收发增益与系统损耗给出的**线性**期望回波增益。

    ``target_gain`` 故意只保留传播/RCS 项，好让 RCS 保持"平方米"这个物理单位。
    硬件增益单独进入接收回波，默认值为 1，以便历史结果精确可复现。
    """
    r = cfg.radio
    net_db = (
        r.radar_net_gain_db
        if r.radar_net_gain_db is not None
        else r.radar_tx_gain_dbi + r.radar_rx_gain_dbi - r.radar_system_loss_db
    )
    return 10.0 ** (net_db / 10.0)

def denominator_guard(cfg: Config, n0: float) -> float:
    """SINR 分母上的加性守卫。

    守卫的唯一作用是避免除零，因此它相对噪声地板必须是可忽略的。
    ``model.EPS = 1e-12`` **不是** —— 对默认射频参数它等于 26 倍 ``n0``，
    会把每个感知 SINR 压低 14.33 dB，并把感知干扰项整个盖掉。
    在 ``radio.eps_mode="noise_relative"`` 下，守卫放在噪声功率下方
    ``eps_rel_db`` 处，与发射功率无关、任何功率下都无害。

    ``eps_mode`` 认不出时报错而不是静默兜底：拼错一个字母会悄悄复现旧行为，
    让整轮扫描作废，而且在任何地方都不留症状。
    """
    mode = cfg.radio.eps_mode
    if mode == "noise_relative":
        return n0 * 10.0 ** (cfg.radio.eps_rel_db / 10.0)
    if mode == "legacy":
        return EPS
    raise ValueError(
        f"Unknown radio.eps_mode={mode!r}; expected 'legacy' or 'noise_relative'"
    )

def path_gain(d: np.ndarray, cfg: Config) -> np.ndarray:
    lam = wavelength(cfg)
    d = np.maximum(d, 1.0)
    return (lam / (4.0 * np.pi)) ** 2 / (d ** cfg.detect.path_loss_exp)

def rician_power_gain(shape: Tuple[int, ...], K_db: float, rng: np.random.Generator) -> np.ndarray:
    """单位均值莱斯功率增益（LOS 分量功率归一为 1）。"""
    K = 10.0 ** (K_db / 10.0)
    los = np.sqrt(K / (K + 1.0))
    nlos = np.sqrt(1.0 / (K + 1.0)) * (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ) / np.sqrt(2.0)
    return np.abs(los + nlos) ** 2
