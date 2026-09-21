"""接收阵列模型：只有一份实现，探针与模型共用。

这里的一切都是**单位模**的：导向向量已归一化，所以更大的口径买到的只是
**空间选择性**，没有别的。这正是把它单独放在一个模块里的全部理由。

一个 ``m`` 元阵的**汇集增益**（在期望回波上 ``10 log10 m``）是与
``radio.radar_net_gain_db`` 完全同类的**链路预算**量：它让一切都变响，信号和
干扰一起变响，因此这里**不建模**。角度探针的全部结论都建立在把两者分开之上 ——
报一个来自口径的检测增益却不说清是哪一种产生的，就是在重复把预算当机制来读的
``G_hw = 15 dB`` 错误。如果平台要被记一份汇集增益，它属于链路预算，并且必须
被明确声明。
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "beamwidth_u",
    "array_factor",
    "array_escape",
    "bearings",
    "masking_fraction",
]


def beamwidth_u(m_rx: int) -> float:
    """半波长 ULA 的 3 dB 波束宽度，以 ``u = sin(phi)`` 为单位。

    ``0.886 / m`` 是 ``|sin(m x)/(m sin x)|`` 主瓣的双边半功率宽度。本项目里所有
    间隔都用这个单位报，而不是用角度，因为整个角度结果只是 ``m * delta_u`` 的
    函数：用角度的话，同一份几何在每个口径下读数都不同，而"相隔几个波束宽度"
    这个量会随它本该去指导的硬件选择一起漂移。
    """
    return 0.886 / float(m_rx)


def array_factor(m_rx: int, delta_u: float) -> float:
    """``|<a(u), a(u + du)>|``，单位范数导向、间距 ``lambda/2``。"""
    if m_rx <= 1:
        return 1.0
    x = math.pi * delta_u / 2.0
    if x % math.pi == 0.0:
        return 1.0
    return abs(math.sin(m_rx * x) / (m_rx * math.sin(x)))


def array_escape(m_rx: int, delta_u: float) -> float:
    """``rho = 1 - |A|^2``：一个导向向量逃出另一个张成空间的比例 —— 即 DD 逃逸
    比例的"纯阵列"对应物。

    ``rho`` 是一种**相干性**而不是增益：两个目标被看在同一方向时它为 0，一旦
    二者相隔超过一个波束宽度就趋于 1。它对任何一条回波有多响不置一词。
    """
    if m_rx <= 1:
        return 0.0
    a = array_factor(m_rx, delta_u)
    return 1.0 - a * a


def bearings(geom, receiver: int, q_count: int, axis: int) -> np.ndarray:
    """``(Q,)`` 从 ``receiver`` 看到的目标方向余弦。

    投影到**一个**机体轴上（``0`` = x，``1`` = y），因为一维阵只沿一个轴分辨。
    一维阵还有前后模糊（``|A|`` 在 ``delta_u = 2`` 处回到 1），所以这是一个
    **被声明的硬件假设**而不是自由选择：轴在配置里固定，绝不看完数字再挑。
    """
    us = np.zeros(q_count, dtype=float)
    p_rx = np.asarray(geom.p_uav[receiver], dtype=float)
    for q in range(q_count):
        v = np.asarray(geom.p_tgt[q], dtype=float)[:3] - p_rx[:3]
        n = float(np.linalg.norm(v))
        us[q] = float(v[axis] / n) if n > 0.0 else 0.0
    return us


def masking_fraction(m_rx: int, delta_u: float) -> float:
    """一条回波被另一条掩蔽掉多少：``1 - rho = |A|^2``。

    这正是生产链路上碰撞惩罚所用的量。单元阵时它精确等于 1.0（即 DD-only 模型：
    任何共 bin 目标都完全掩蔽），并随阵列把这一对分开而减小 —— 这正是碰撞惩罚
    能在不引入硬门限的前提下"感知阵列"的原因。
    """
    a = array_factor(m_rx, delta_u)
    return a * a
