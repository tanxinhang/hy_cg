"""物理设定：规模、部署盒、波形与波形损伤、以及接收机射频参数。"""

from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------
# 物理设定
# --------------------------------------------------------------------------
@dataclass
class Scale:
    """Numbers of cooperating UAVs and targets."""

    M: int = 15
    Q: int = 10

@dataclass
class Geometry:
    """Deployment box and kinematic ranges."""

    area_xy: float = 4000.0
    h_uav_min: float = 800.0
    h_uav_max: float = 1200.0
    h_target_min: float = 700.0
    h_target_max: float = 1500.0
    comm_range: float = 2500.0
    uav_speed_min: float = 20.0
    uav_speed_max: float = 60.0
    target_speed_min: float = 30.0
    target_speed_max: float = 150.0

@dataclass
class Waveform:
    """OTFS delay-Doppler grid and carrier."""

    N: int = 64
    L: int = 64
    delta_f: float = 30e3
    T: float = 1.0 / 30e3
    fc: float = 5.9e9
    c: float = 3e8

@dataclass
class WaveformImpairments:
    """匹配滤波之后的损伤比，用于鲁棒性实验。

    所有字段默认都是理想接收机，因此论文/V1 preset 不变。
    INR 值相对于既有的"噪声 + 残余"分母。同步误差以时延/多普勒格为单位。
    """

    enable: bool = False
    # belief 模式下若为 false：真值/检测仍然看到损伤，但调度器在理想波形模型下
    # 建自己的感知表。这就是显式的"不知情调度器"反事实。
    scheduler_aware: bool = True
    clutter_inr: float = 0.0
    multipath_inr: float = 0.0
    unresolved_target_inr: float = 0.0
    sync_delay_bins: float = 0.0
    sync_doppler_bins: float = 0.0
