"""cancellation 积木：直达源与目标源容器。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class DirectSource:
    """一个激活照射源在接收机处的直达路径项。"""

    uav: int
    power: float  # 到达直达路径的辐射感知功率，W
    gain: float  # 直达路径功率增益 g_ij（无量纲）
    doppler_bin: float  # UAV-UAV 链路的分数多普勒格
    delay_bin: float  # UAV-UAV 链路的分数延迟格
    # 接收机看到的方位（沿阵列轴的方向余弦）。默认 0.0 让每个既有构造点继续
    # 有效；只有升维路径（``aperture.enable``）会读它。在协同网络里直达路径是
    # **已知**的，所以这是真值方位，不是信念。
    u: float = 0.0

    @property
    def power_at_receiver(self) -> float:
        return self.power * self.gain


@dataclass(frozen=True)
class TargetSource:
    """接收机观察到的一个信念/真值目标回波。"""

    uav: int
    target: int
    power: float  # P_sense * target_gain * G_proc * G_hw，即回波功率
    doppler_bin: float
    delay_bin: float
    # 接收机看到的方位（沿阵列轴的方向余弦）。真值字典带真值方位，信念字典带
    # 信念方位，于是升维路径给方位误差定价的方式与它已经给延迟/多普勒误差定价
    # 的方式相同。默认 0.0 让既有构造点继续有效。
    u: float = 0.0
    # 信念协方差传播到这条双基地链路的 DD 坐标上。``None`` 保留手工构造源时
    # 历史上那个全局界回退；可执行观测总是填满这两个字段。
    sigma_doppler_bin: float | None = None
    sigma_delay_bin: float | None = None
    sigma_bearing_u: float | None = None
    jacobian_doppler_state: np.ndarray | None = None
    jacobian_delay_state: np.ndarray | None = None
    jacobian_bearing_state: np.ndarray | None = None
