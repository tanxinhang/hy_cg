"""小线代数积木：PSD 开方、秩安全投影、投影能量、广义最小特征值。

这些都是"低秩协方差 + 目标条件化检测"会反复用到的原子操作，一律走秩安全
路径（``lstsq``/``eigh``），避免在秩亏时静默给出错误答案。
"""
from __future__ import annotations

import numpy as np

from isac_sim.receiver.cancellation import EPS


def _psd_sqrt(C: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """``L``，满足 ``L L^H = C``（``C`` 为 Hermitian 半正定），走特征分解。"""
    if C.size == 0:
        return np.zeros_like(C)
    w, V = np.linalg.eigh(0.5 * (C + C.conj().T))
    if w.size and w.min() < -tol * max(abs(w.max()), EPS):
        raise RuntimeError(
            "estimator covariance is not PSD (min eigenvalue %.3e)" % (w.min(),)
        )
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)[None, :]).astype(complex)


def _project_out(B_neg: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``(I - B_neg B_neg^dagger) V``，用最小二乘实现（秩安全）。"""
    if B_neg.shape[1] == 0:
        return V
    coef = np.linalg.lstsq(B_neg, V, rcond=None)[0]
    return V - B_neg @ coef


def _projector_energy(B: np.ndarray, v: np.ndarray) -> float:
    """``v^H P_B v``，``B`` 允许秩亏。"""
    if B.shape[1] == 0:
        return 0.0
    coef = np.linalg.lstsq(B, v, rcond=None)[0]
    fitted = B @ coef
    return float(np.vdot(fitted, fitted).real)


def _numerical_rank(B: np.ndarray, rtol: float = 1e-8) -> int:
    """``B`` 的数值秩（相对最大奇异值）。"""
    if B.size == 0 or B.shape[1] == 0:
        return 0
    sv = np.linalg.svd(B, compute_uv=False)
    if sv.size == 0 or sv[0] <= 0.0:
        return 0
    return int(np.sum(sv > rtol * sv[0]))


def _generalised_min_eigen(G_keep: np.ndarray, G_full: np.ndarray) -> float:
    """``G_keep`` 值域上的 ``lambda_min(G_full, G_keep)``。

    这是 ``rho`` 的矩阵版本：模板块最坏的那个方向还保留多少白化匹配滤波能量。
    计算为 ``G_keep^{-1/2} G_full G_keep^{-1/2}`` 的最小特征值；该矩阵对称，且
    因为 ``G_full`` 是 ``G_keep`` 的单调压缩，结果可夹到 ``[0, 1]``。
    """
    if G_keep.size == 0 or G_keep.shape[0] == 0:
        return 0.0
    w, V = np.linalg.eigh(0.5 * (G_keep + G_keep.conj().T))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    if w.size == 0 or w.max() <= EPS:
        return 0.0
    keep = w > max(w.max() * 1e-12, EPS)
    if not np.any(keep):
        return 0.0
    Vk = V[:, keep]
    inv_sqrt = Vk @ np.diag(w[keep] ** -0.5) @ Vk.conj().T
    M = inv_sqrt @ G_full @ inv_sqrt
    M = 0.5 * (M + M.conj().T)
    return float(max(0.0, min(1.0, np.linalg.eigvalsh(M).min())))
