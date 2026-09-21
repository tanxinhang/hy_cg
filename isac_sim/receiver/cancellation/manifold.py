"""cancellation 积木：核向量、切向列与子空间。"""

from __future__ import annotations

from dataclasses import dataclass
from isac_sim.core.config import Config
from isac_sim.sensing.waveform import full_otfs_kernel
import numpy as np

from isac_sim.receiver.cancellation.constants import EPS

def kernel_vector(cfg: Config, doppler_bin: float, delay_bin: float) -> np.ndarray:
    """单位幅度散射体的单位范数全网格 DD 响应。

    包在 :func:`isac_sim.sensing.waveform.full_otfs_kernel` 外面 —— 它本身已经
    做了 L2 归一化并带缓存。返回的向量按 C 序在 ``(doppler, delay)`` 上展平，
    与 :func:`otfs_demodulate` 的输出布局一致。
    """
    w = cfg.waveform
    grid = full_otfs_kernel(
        int(w.N),
        int(w.L),
        float(doppler_bin),
        float(delay_bin),
        float(w.delta_f),
        int(round(w.fc / 1e6)),
    )
    return np.asarray(grid, dtype=complex).reshape(-1)


def tangent_columns(
    cfg: Config,
    doppler_bin: float,
    delay_bin: float,
    order: int,
    step: float,
) -> np.ndarray:
    """``[a, da/df_delay, da/df_doppler, ...]``，形状 ``(K, 1 + 2*order)`` 的块。

    对**精确** OTFS 响应做中心有限差分。单侧差分或解析导数会更小，但核已经带
    缓存，所以差分不花代价，而且对数值模型保持诚实。
    """
    centre = kernel_vector(cfg, doppler_bin, delay_bin)
    if order <= 0:
        return centre[:, None]
    h = float(max(step, EPS))
    d_delay = (kernel_vector(cfg, doppler_bin, delay_bin + h)
               - kernel_vector(cfg, doppler_bin, delay_bin - h)) / (2.0 * h)
    d_doppler = (kernel_vector(cfg, doppler_bin + h, delay_bin)
                 - kernel_vector(cfg, doppler_bin - h, delay_bin)) / (2.0 * h)
    columns = [centre, d_delay, d_doppler]
    return np.stack(columns, axis=1)


@dataclass
class Subspace:
    """正交归一的基，以及它的两个投影算子（隐式施用）。"""

    U: np.ndarray  # (K, r) 正交归一的列
    rank: int

    def project(self, x: np.ndarray) -> np.ndarray:
        """``P x`` —— 落在"承载目标"子空间里的那个分量。"""
        if self.rank == 0:
            return np.zeros_like(x)
        return self.U @ (self.U.conj().T @ x)

    def complement(self, x: np.ndarray) -> np.ndarray:
        """``M x = (I - P) x`` —— 可用于干扰学习的那个分量。"""
        if self.rank == 0:
            return x
        return x - self.U @ (self.U.conj().T @ x)

    def complement_matrix(self, B: np.ndarray) -> np.ndarray:
        """矩阵 ``B`` 的 ``M B``，形状 ``(K, m)``；逐列但批量完成。"""
        if self.rank == 0:
            return B
        return B - self.U @ (self.U.conj().T @ B)


def orthonormalise(
    columns: np.ndarray, tol: float = 1e-6, max_rank: int | None = None
) -> Subspace:
    """经 Gram 矩阵做的、带秩裁剪的正交归一化。

    走 Gram 路线（``m x m`` 特征分解，``m`` = 候选列数）而不是对 ``K x m`` 矩阵
    做 QR/SVD，是因为 ``K`` 是 4096 而 ``m`` 只有几百。

    列在特征值裁剪**之前**先归一化，这不是一个数值细节：DD 切向列是有限差分，
    其范数量级是 ``1/step``，也就是中心列的几百倍。因此若把裁剪放在**未归一化**
    Gram 的最大特征值之上，就会放进一批"唯一区别是有限差分噪声"的方向，报出的
    保护秩会膨胀 3 倍（600 m 场景实测：195 而不是 62）。对"这个子空间是否可能
    承载目标证据"而言要紧的是**相关性**而不是**幅度**，所以裁剪是加在相关矩阵
    上、用绝对门限做的。
    """
    B = np.asarray(columns, dtype=complex)
    if B.ndim == 1:
        B = B[:, None]
    if B.shape[1] == 0:
        return Subspace(U=np.zeros((B.shape[0], 0), dtype=complex), rank=0)
    norms = np.linalg.norm(B, axis=0)
    keep_cols = norms > EPS
    if not np.any(keep_cols):
        return Subspace(U=np.zeros((B.shape[0], 0), dtype=complex), rank=0)
    B = B[:, keep_cols] / norms[keep_cols][None, :]
    gram = B.conj().T @ B
    gram = 0.5 * (gram + gram.conj().T)
    w, V = np.linalg.eigh(gram)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    keep = w > float(tol)
    if max_rank is not None:
        keep &= np.arange(w.size) < int(max_rank)
    w, V = w[keep], V[:, keep]
    if w.size == 0:
        return Subspace(U=np.zeros((B.shape[0], 0), dtype=complex), rank=0)
    U = (B @ V) / np.sqrt(w)[None, :]
    return Subspace(U=U, rank=int(U.shape[1]))
