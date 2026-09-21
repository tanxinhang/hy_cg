"""cancellation 积木：导向矢量与升维字典。"""

from __future__ import annotations

from isac_sim.core.config import Config
import math
import numpy as np
from typing import Dict, Sequence

def _u_of(points, p_rx: np.ndarray, q_count: int, axis: int) -> np.ndarray:
    """``points`` 从 ``p_rx`` 看去、沿 ``axis`` 的 ``(Q,)`` 方向余弦。"""
    us = np.zeros(int(q_count), dtype=float)
    for q in range(int(q_count)):
        v = np.asarray(points[q], dtype=float)[:3] - p_rx
        n = float(np.linalg.norm(v))
        us[q] = float(v[axis] / n) if n > 0.0 else 0.0
    return us


def _n_obs(cfg: Config) -> int:
    """一次观测的维度：DD 格数 × 接收阵元数。

    一个阵元就是已发布的 DD-only 模型，所以那里它就是 K，既有数字不会移动。
    """
    m_rx = int(cfg.aperture.m_rx) if cfg.aperture.enable else 1
    return int(cfg.waveform.N * cfg.waveform.L) * max(int(m_rx), 1)


def steering_vector(m_rx: int, u: float) -> np.ndarray:
    """半波长 ULA 的单位范数阵列响应 ``a(u)``。

    取单位范数是**故意**的：这样阵列只改变两个模板之间的**相干性**，别的什么
    都不改，于是一个阵元与十六个阵元下观测的 SNR 相同，任何测到的变化都是
    **选择性**而不是**增益**。它诱导出的内积恰好就是
    :func:`isac_sim.sensing.aperture.array_factor`。
    """
    if m_rx <= 1:
        return np.ones(1, dtype=complex)
    idx = np.arange(int(m_rx), dtype=float)
    return np.exp(-1j * math.pi * float(u) * idx) / math.sqrt(float(m_rx))


def steering_derivative(m_rx: int, u: float) -> np.ndarray:
    """单位范数半波 ULA 响应的导数 ``d a(u) / d u``。"""
    if m_rx <= 1:
        return np.zeros(1, dtype=complex)
    idx = np.arange(int(m_rx), dtype=float)
    return (-1j * math.pi * idx) * steering_vector(m_rx, u)


def lift_dictionary(cfg: Config, matrix: np.ndarray, us: Sequence[float]) -> np.ndarray:
    """``(K, n) -> (K*m, n)``：第 ``c`` 列变成 ``kron(column_c, a(u_c))``。

    这就是 P1-2 的全部：一个 ``m`` 阵元接收机的观测空间是 DD 网格与阵列的张量
    积，而方位 ``u`` 上的散射体激发的是 ``a_DD (x) a(u)``。下游的一切 ——
    对消器的子空间拟合、残差协方差、白化统计量、逃逸分数 ``rho`` —— 都是与维度
    无关的线性代数，所以只要字典被升维，阵列就同时出现在它们所有环节里。

    内积按 ``<a_DD_i, a_DD_j> * A(u_i, u_j)`` 分解（角度探针正是建在这个
    Kronecker 恒等式上），这就是为什么两个共享同一个 DD 格的目标仍然可以有
    接近 1 的 ``rho``。
    """
    m_rx = int(cfg.aperture.m_rx) if cfg.aperture.enable else 1
    if m_rx <= 1:
        return matrix
    k_bins, n_col = matrix.shape
    if n_col == 0:
        return np.zeros((k_bins * m_rx, 0), dtype=complex)
    out = np.empty((k_bins * m_rx, n_col), dtype=complex)
    cache: Dict[float, np.ndarray] = {}
    for c in range(n_col):
        u = float(us[c])
        a = cache.get(u)
        if a is None:
            a = steering_vector(m_rx, u)
            cache[u] = a
        out[:, c] = np.kron(matrix[:, c], a)
    return out
