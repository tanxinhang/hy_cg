"""线性探针：把一条臂的消除映射恢复成 ``F = basis small basis^H``。

``cancellation`` 的每条臂都是仿射映射 ``removed = F y + c``，因为
``protected_map``/``joint_refine`` 解的是一个右端关于 ``y`` 线性的小型正规方程。
``F`` 靠**探针**恢复：把映射作用在覆盖其值域与行空间的一组标准正交基上，得到
的小矩阵就能精确复现它 —— 探针误差是被测量并断言的，不是被假设的。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from isac_sim.receiver.cancellation import (
    EPS,
    Observation,
    joint_refine,
    protected_map,
    soft_protected_map,
)
from isac_sim.receiver.cancellation_glrt.arm_plan import ArmPlan


def _removed_vector(cfg, obs: Observation, plan: ArmPlan, v: np.ndarray) -> np.ndarray:
    """一个计划下的估计器从 ``v`` 里移除的向量 —— 即 V1 的 ``operator``。"""
    if plan.kind != "estimator":
        return np.zeros(int(v.size), dtype=complex)
    X, A = obs.X, obs.A
    sigma2 = float(obs.sigma2)
    if plan.candidates:
        A_cand = A[:, list(plan.candidates)]
        h, _a = joint_refine(v, X, A_cand, sigma2, plan.prior, plan.prior or 1.0)
        return X @ h
    if plan.soft_mu is not None:
        h, _cov = soft_protected_map(
            v, X, plan.subspace, sigma2, plan.prior, plan.soft_mu
        )
        return X @ h
    h, _c, mx = protected_map(v, X, plan.subspace, sigma2, plan.prior)
    return mx @ h


def _probe_basis(obs: Observation, plan: ArmPlan) -> np.ndarray:
    """一组同时覆盖消除映射值域与行空间的基。

    受保护的估计器做的是 ``M X h(M y)``，完全落在 ``span(M X)`` 里；联合臂做的是
    ``X h``，而 ``h`` 只依赖 ``[X, A_cand]^H y``，所以 ``span([X, A_cand])``
    同时盖住像空间与行空间。在一组同时覆盖两者的基上，探针能精确重建该映射。
    """
    if plan.kind != "estimator":
        return np.zeros((int(obs.y.size), 0), dtype=complex)
    if plan.candidates:
        return np.concatenate([obs.X, obs.A[:, list(plan.candidates)]], axis=1)
    if plan.soft_mu is not None:
        return obs.X
    return plan.subspace.complement_matrix(obs.X)


def _low_rank_form(
    cfg, obs: Observation, plan: ArmPlan, tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, float]:
    """恢复 ``(basis, small, error)``，使得 ``F = basis @ small @ basis^H``。

    在覆盖子空间的标准正交基 ``Q`` 上探测可直接得到 ``F Q``；若 ``P = Q R``，
    则小矩阵为 ``(Q^H F Q) (Q^H P)^+``。``error`` 是探针列上**实测**的重建
    残差 —— 返回而不是假设，这样"秩悄悄亏掉的探针"不能冒充精确模型。

    ``tol`` 刻意取松（1e-6）而不是机器精度：联合臂的覆盖基是 ``[X, A_cand]``，
    其列可能近似共线，``pinv(R)`` 会放大舍入误差，某些 trial 上实测误差在 1e-8
    量级。这比任何能改动所报抵消深度的量级还低八个数量级，而真正非仿射的臂
    会暴露在 O(1e-2) 上。
    """
    n_bins = int(obs.y.size)
    if plan.kind != "estimator":
        return np.zeros((n_bins, 0), dtype=complex), np.zeros((0, 0), dtype=complex), 0.0
    P = _probe_basis(obs, plan)
    if P.shape[1] == 0:
        return np.zeros((n_bins, 0), dtype=complex), np.zeros((0, 0), dtype=complex), 0.0
    Q, R = np.linalg.qr(P)
    removed = np.stack(
        [_removed_vector(cfg, obs, plan, P[:, j]) for j in range(P.shape[1])], axis=1
    )
    small = (Q.conj().T @ removed) @ np.linalg.pinv(R, rcond=1e-12)
    back = Q @ (small @ (Q.conj().T @ P))
    denom = max(float(np.linalg.norm(removed)), EPS)
    error = float(np.linalg.norm(back - removed) / denom)
    if error > tol:
        raise RuntimeError(
            "arm %r: the linear probe does not reproduce the arm (relative error "
            "%.3e > %.1e) -- the arm is not affine, or the covering subspace is "
            "too small" % (plan.name, error, tol)
        )
    return Q, np.asarray(small), error
