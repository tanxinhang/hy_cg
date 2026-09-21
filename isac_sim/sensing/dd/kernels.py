"""DD 域核函数（自 ``isac_sim/sensing/dd.py`` 拆出）。"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple
import numpy as np


EPS = 1e-12


def dirichlet_kernel(N: int, x) -> np.ndarray:
    """周期 Dirichlet 核 ``D_N(x) = sin(pi x) / (N sin(pi x / N))``。

    在 ``x == 0 mod N`` 处归一到 1。约定与兄弟包
    ``gate_otfs_collision/kernels.py`` 一致，故两个 DD 模型在数值精度内相符。
    """
    x = np.asarray(x, dtype=float)
    denom = N * np.sin(np.pi * x / N)
    numer = np.sin(np.pi * x)
    out = np.empty_like(x, dtype=float)
    small = np.abs(denom) < 1e-10
    out[small] = 1.0
    out[~small] = numer[~small] / denom[~small]
    return out


def _quantise_centre(x: float, step: float = 1e-3) -> float:
    """把一个分数 DD 中心量化，以便缓存复用。

    窗口求和得到的泄漏随中心平滑变化，因此 1e-3 的量化能去掉无谓的重算，又不
    带来可见的数值漂移。
    """
    return float(round(x / step) * step)


@lru_cache(maxsize=65536)
def leakage_1d(N: int, centre: float) -> np.ndarray:
    """分数 DD 源在一条周期 N-bin 轴上的能量分布。

    返回长度为 ``N``、和为 1 的向量。``p[b]`` 是该源泄漏进物理 bin ``b`` 的能量
    份额。整数中心塌缩成近似 Dirac 的单 bin 分布；分数中心把能量摊到几个邻居上，
    这就是 OTFS 的多普勒间/延迟间干扰建模。
    """
    centre = _quantise_centre(centre)
    idx = np.arange(N, dtype=float)
    # 从 ``centre`` 到整数 bin ``idx`` 的带符号圆周偏移。
    dx = ((idx - centre + N / 2.0) % N) - N / 2.0
    p = np.abs(dirichlet_kernel(N, dx)) ** 2
    return p / (float(np.sum(p)) + EPS)


def window_bins(
    centre_k: float,
    centre_l: float,
    half_w: int,
    N_k: int,
    N_l: int,
) -> List[Tuple[int, int]]:
    """局部 ``(2*half_w+1) x (2*half_w+1)`` DD 窗内的 bin 索引。

    窗口以离 ``(centre_k, centre_l)`` 最近的 bin 为中心。bin 索引按 ``N_k`` /
    ``N_l`` 取模回绕，于是任何中心下窗口都留在 OTFS 帧内。
    """
    ck = int(np.round(centre_k)) % N_k
    cl = int(np.round(centre_l)) % N_l
    out: List[Tuple[int, int]] = []
    for dk in range(-half_w, half_w + 1):
        for dl in range(-half_w, half_w + 1):
            out.append(((ck + dk) % N_k, (cl + dl) % N_l))
    return out
