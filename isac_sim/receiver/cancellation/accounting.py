"""cancellation 积木：残差核算。"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from isac_sim.receiver.cancellation.joint import _positive_sigma
from isac_sim.receiver.cancellation.manifold import Subspace

def soft_residual_accounting(
    X: np.ndarray,
    subspace: Subspace,
    sigma2: float,
    prior_variance: float | None,
    mu: float,
) -> Tuple[float, float]:
    """软 TP-UIC 的结构性功率与拟合噪声功率的期望值。"""
    PX = subspace.project(X)
    gram = X.conj().T @ X + float(mu) * (PX.conj().T @ PX)
    sg = _positive_sigma(sigma2)
    if prior_variance is not None and prior_variance > 0.0:
        gram = gram + (sg / float(prior_variance)) * np.eye(gram.shape[0])
    H = np.linalg.pinv(gram, rcond=1e-12) @ X.conj().T
    F_small = H @ X
    residual_dictionary = X @ (np.eye(X.shape[1]) - F_small)
    direct_variance = float(prior_variance or 1.0)
    structural = direct_variance * float(
        np.real(np.vdot(residual_dictionary, residual_dictionary))
    )
    removal = X @ H
    estimate = sg * float(np.real(np.vdot(removal, removal)))
    return structural, estimate


def residual_accounting(
    X: np.ndarray,
    subspace: Subspace,
    sigma2: float,
    prior_variance: float | None,
) -> Tuple[float, float]:
    """式 (3) 的**总功率**版本，全程不构造任何 ``K x K`` 矩阵。

    因为投影算子是幂等的，
    ``tr(P X R_h X^H P) = sum_i R_h[i] * ||(X^H U)[i, :]||^2``；
    因为迹是可轮换的，
    ``tr(M X C_h X^H M) = tr(C_h X^H M X)``。两者都归约成小矩阵乘积。

    这里**不除以** ``K``：它的实测对应量 ``||P x||^2`` 是一个总功率，
    ``I_sense_field`` 也是总功率。方法笔记式 (3) 里的逐元素平均，是同一句话在
    两边同时除了一下；两种口径绝不能混用，否则预测会差 ``log10(K)`` = 36 dB ——
    这恰好就是第一次运行所显示的分歧大小。
    """
    xu = subspace.U.conj().T @ X if subspace.rank else np.zeros((0, X.shape[1]), dtype=complex)
    r_h = (np.full(X.shape[1], float(prior_variance))
           if prior_variance is not None and prior_variance > 0.0
           else np.zeros(X.shape[1]))
    retained = float(np.sum(r_h * np.sum(np.abs(xu) ** 2, axis=0)))
    mx = subspace.complement_matrix(X)
    gram = mx.conj().T @ mx
    eye = np.eye(gram.shape[0])
    inv_r_h = (eye / float(prior_variance)
               if prior_variance is not None and prior_variance > 0.0
               else np.zeros_like(eye))
    c_h = np.linalg.inv(gram / _positive_sigma(sigma2) + inv_r_h)
    # tr(M X C_h X^H M) = tr(C_h X^H M X)：一个小的迹，而不是对角乘积之和。
    # 只要 Gram 不是对角的，两者就不相等，而本场景它不是对角的（条件数 ~ 2e3），
    # 这正是逐元素形式会**高估**估计残差的原因。无先验的 LS 情形必须精确退化为
    # sigma^2 * rank(M X)：tr(C_h G) 做到了，逐元素形式做不到。
    estimate = float(np.real(np.trace(c_h @ gram)))
    return retained, estimate
