"""cancellation 积木：受保护 MAP 估计。"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from isac_sim.receiver.cancellation.joint import _positive_sigma
from isac_sim.receiver.cancellation.manifold import Subspace

def protected_map(
    y: np.ndarray,
    X: np.ndarray,
    subspace: Subspace,
    sigma2: float,
    prior_variance: float | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """式 (2)：受保护的（岭正则化的）最小二乘。

    返回 ``(h_hat, c_diag, MX)``，其中 ``MX`` 是投影后的字典 ``M X``，供相减步骤
    复用。``prior_variance=None`` 丢掉先验，剩下普通的受保护 LS —— 也就是模块
    docstring 引用的那个 ``R_h^{-1} -> 0`` 极限。

    这个估计对 ``y`` 是线性的，下面的核算正是利用这一点：同一个例程被分别作用在
    直达分量、回波分量与噪声分量上，于是每一臂对每一个分量的作用是**精确已知**
    的，而不是从总量里推断出来的。
    """
    MX = subspace.complement_matrix(X)
    gram = MX.conj().T @ MX
    rhs = MX.conj().T @ y
    sg = _positive_sigma(sigma2)
    if prior_variance is not None and prior_variance > 0.0:
        gram_r = gram + (sg / float(prior_variance)) * np.eye(gram.shape[0])
        cov = np.linalg.inv((gram / sg)
                            + (1.0 / float(prior_variance)) * np.eye(gram.shape[0]))
        h_hat = np.linalg.solve(gram_r, rhs)
    else:
        cov = np.linalg.pinv(gram, rcond=1e-10) * sg
        h_hat = np.linalg.lstsq(gram, rhs, rcond=1e-10)[0]
    return h_hat, np.real(np.diag(cov)), MX


def soft_protected_map(
    y: np.ndarray,
    X: np.ndarray,
    subspace: Subspace,
    sigma2: float,
    prior_variance: float | None,
    mu: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""软目标保持 MAP 估计。

    求解 ``min_h ||y-Xh||^2 + mu ||P Xh||^2 + lambda_h ||h||^2``。

    相减的是 ``X h_hat``。``mu=0`` 恰好就是岭 LS；与硬投影不同，有限 ``mu`` 把
    对消深度与"目标子空间被削减"之间做成连续折中。
    """
    weight = float(mu)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("soft protection weight must be finite and non-negative")
    PX = subspace.project(X)
    gram = X.conj().T @ X + weight * (PX.conj().T @ PX)
    sg = _positive_sigma(sigma2)
    precision = gram / sg
    gram_r = gram.copy()
    if prior_variance is not None and prior_variance > 0.0:
        precision = precision + (1.0 / float(prior_variance)) * np.eye(gram.shape[0])
        gram_r = gram_r + (sg / float(prior_variance)) * np.eye(gram.shape[0])
    cov = np.linalg.pinv(precision, rcond=1e-12)
    h_hat = np.linalg.pinv(gram_r, rcond=1e-12) @ (X.conj().T @ y)
    return h_hat, np.asarray(cov, dtype=complex)
