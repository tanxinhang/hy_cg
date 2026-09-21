"""cancellation 积木：联合精化与联合协方差。"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from isac_sim.receiver.cancellation.constants import EPS

def _joint_dictionary(X: np.ndarray, A_cand: np.ndarray) -> np.ndarray:
    blocks = [X]
    if A_cand.shape[1] > 0:
        blocks.append(A_cand)
    return np.concatenate(blocks, axis=1)


def _joint_regulariser(X, A_cand, sigma2, prior_variance, target_prior):
    """式 (4a) 的对角正则：干扰块用 ``sigma2/pv``，目标块用 ``sigma2/target_prior``。"""
    lam = np.zeros(X.shape[1] + A_cand.shape[1])
    if prior_variance is not None and prior_variance > 0.0:
        lam[: X.shape[1]] = float(sigma2) / float(prior_variance)
    if A_cand.shape[1] > 0:
        lam[X.shape[1]:] = float(sigma2) / max(float(target_prior), EPS)
    return lam


def joint_system(X, A_cand, sigma2, prior_variance, target_prior):
    """联合臂里**与被拟合向量无关**的那四个量。

    返回 ``(D, normal, gram, prior_precision)``，其中 ``normal`` 是
    ``joint_refine`` 要解的正规方程矩阵，``gram = D^H D``。

    为什么值得单独拎出来：一条臂要解 5 个分量（``y``/``x``/``s``/``n``/``s_q``）
    外加十几个 sigma 点，每一次都重建 ``D``（4096 行的拼接）并重算 ``D^H D``。
    剖析显示 ``joint_refine`` + ``joint_interference_covariance`` 约占接收机测量
    的 40%，其中绝大部分就是这一次又一次的同一次 ``D^H D``。

    缓存是**数值恒等**的，不是近似：矩阵由同样的运算、同样的顺序算出，随后
    ``np.linalg.solve`` / ``pinv`` 拿到的输入逐位相同。
    """
    D = _joint_dictionary(X, A_cand)
    gram = D.conj().T @ D
    lam = _joint_regulariser(X, A_cand, sigma2, prior_variance, target_prior)
    prior_precision = np.zeros(gram.shape[0], dtype=float)
    if prior_variance is not None and prior_variance > 0.0:
        prior_precision[: X.shape[1]] = 1.0 / float(prior_variance)
    if A_cand.shape[1] > 0:
        prior_precision[X.shape[1]:] = 1.0 / max(float(target_prior), EPS)
    return D, gram + np.diag(lam), gram, prior_precision


def joint_refine(
    y: np.ndarray,
    X: np.ndarray,
    A_cand: np.ndarray,
    sigma2: float,
    prior_variance: float | None,
    target_prior: float,
    system=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """式 (4a)：在候选支撑集上同时拟合干扰与目标。

    两个块都带高斯先验，于是正规方程始终是 ``(d + m)`` 阶方阵，不出现任何
    ``K x K`` 求逆。目标块只被用来**解释**候选支撑上的观测，它不会被减掉。

    ``system`` 是 :func:`joint_system` 的结果；给了它就跳过矩阵重建（逐位
    等价）。不给则按需构造，行为与以前完全一致。
    """
    if system is None:
        system = joint_system(X, A_cand, sigma2, prior_variance, target_prior)
    D, normal, _gram, _pp = system
    rhs = D.conj().T @ y
    theta = np.linalg.solve(normal, rhs)
    return theta[: X.shape[1]], theta[X.shape[1]:]
def marginalized_interference_map(
    y: np.ndarray,
    X: np.ndarray,
    A_cand: np.ndarray,
    sigma2: float,
    prior_variance: float | None,
    target_prior: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Bayesian interference MAP after marginalising target amplitudes.

    The Schur form equals ``joint_refine`` without constructing the
    observation-sized covariance ``sigma2 I + A R_alpha A^H``.
    """
    sg = _positive_sigma(sigma2)
    d = int(X.shape[1])
    H_hh = X.conj().T @ X
    if prior_variance is not None and prior_variance > 0.0:
        H_hh = H_hh + (sg / float(prior_variance)) * np.eye(d)
    b_h = X.conj().T @ y

    if A_cand.shape[1] == 0:
        h_hat = np.linalg.solve(H_hh, b_h)
        cov_h = sg * np.linalg.pinv(H_hh, rcond=1e-12)
        return h_hat, np.asarray(cov_h, dtype=complex)

    H_aa = A_cand.conj().T @ A_cand
    H_aa = H_aa + (sg / max(float(target_prior), EPS)) * np.eye(A_cand.shape[1])
    H_ha = X.conj().T @ A_cand
    b_a = A_cand.conj().T @ y
    eliminated_cross = np.linalg.solve(H_aa, H_ha.conj().T)
    eliminated_rhs = np.linalg.solve(H_aa, b_a)
    schur = H_hh - H_ha @ eliminated_cross
    rhs = b_h - H_ha @ eliminated_rhs
    schur = 0.5 * (schur + schur.conj().T)  # restore analytic Hermitian form
    h_hat = np.linalg.solve(schur, rhs)
    cov_h = sg * np.linalg.pinv(schur, rcond=1e-12)
    return h_hat, np.asarray(cov_h, dtype=complex)
def joint_interference_covariance(
    X: np.ndarray,
    A_cand: np.ndarray,
    sigma2: float,
    prior_variance: float | None,
    target_prior: float,
    system=None,
) -> np.ndarray:
    """联合干扰/目标拟合下 ``h`` 的后验协方差。

    由目标块引入的相关性被保留下来。完整臂减掉的是 ``X h_hat``，所以这个协方差
    必须经 ``X`` 传播，而不是经 stage-1 的投影字典 ``M X`` 传播。
    """
    if system is None:
        system = joint_system(X, A_cand, sigma2, prior_variance, target_prior)
    _D, _normal, gram, prior_precision = system
    cov = np.linalg.pinv(
        gram / _positive_sigma(sigma2) + np.diag(prior_precision), rcond=1e-12
    )
    return np.asarray(cov[: X.shape[1], : X.shape[1]], dtype=complex)


def _positive_sigma(sigma2: float) -> float:
    """守住一个方差，别让它落到模块级 ``EPS`` 上。

    ``EPS = 1e-12`` 在这里**不是**一个安全的除数守卫：默认射频的噪声功率是
    ``3.83e-14`` W，比 EPS **小** 26 倍，于是 ``max(sigma2, EPS)`` 会静默地把真实
    方差换成一个大 26 倍的值。预测残差随之大 26 倍（实测：365 而不是 14 个噪声
    单位），看起来像一个建模错误，而它并不是。``model.py`` 对
    ``radio.eps_mode`` 记录的正是这同一种失效模式；这里用"拒绝一个非正方差"
    而不是"把它夹住"来避开同一个陷阱。
    """
    value = float(sigma2)
    if not value > 0.0:
        raise ValueError(f"noise variance must be positive, got {sigma2!r}")
    return value
