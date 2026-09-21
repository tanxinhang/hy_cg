"""先验与估计器协方差：残余里那两个"非噪声"项各自有多大。"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from isac_sim.receiver.cancellation import Observation, joint_interference_covariance
from isac_sim.receiver.cancellation_glrt.arm_plan import ArmPlan


def _direct_prior(cfg, obs: Observation, plan: ArmPlan) -> Tuple[str, np.ndarray]:
    """``(label, R_diag)`` —— 接收机对直联系数 ``h`` 声明的先验。

    先验对每条臂都相同，oracle 臂也不例外，这不是疏忽：oracle 臂留下的是
    ``direct_gain * x``，其协方差是 ``direct_gain^2 X R_h X^H``，用的是**同一个**
    ``R_h``。在这里返回零等于告诉检测器"`no_ic`` 的残余是纯噪声"，也就是直连
    场已经被抵消掉了 —— 而那恰恰是这条臂明确不做的事。
    """
    d = int(obs.X.shape[1])
    value = cfg.cancellation.prior_variance
    return "prior", np.full(d, float(value) if value else 1.0)


def _oracle_gain(cfg, name: str) -> float:
    """一条 oracle 臂在残余里留下多少直连场。"""
    if name == "no_ic":
        return 1.0
    if name == "perfect_channel":
        return 0.0
    raise KeyError(name)


def _estimator_covariance(cfg, obs: Observation, plan: ArmPlan) -> np.ndarray:
    """干扰系数的后验协方差 ``C_h``。

    这里镜像 ``protected_map``/``joint_refine`` 的正规方程，而不是把 ``c_diag``
    读回来 —— 对角丢掉了各照射源之间的相关性，而联合臂根本不上报协方差。参考
    预算以 ``1 / n_cpi`` 进入：``n_cpi`` 个相干 CPI 正是"系数误差比单 CPI 自拟合
    小这么多"这句话本身。
    """
    d = int(obs.X.shape[1])
    if plan.kind != "estimator" or d == 0:
        return np.zeros((d, d), dtype=complex)
    sigma2 = float(obs.sigma2)
    pv = plan.prior
    if plan.candidates:
        A_cand = obs.A[:, list(plan.candidates)]
        return joint_interference_covariance(
            obs.X, A_cand, sigma2, pv, pv or 1.0
        )
    if plan.soft_mu is not None:
        px = plan.subspace.project(obs.X)
        gram = obs.X.conj().T @ obs.X + float(plan.soft_mu) * (
            px.conj().T @ px
        )
        precision = gram / sigma2
        if pv:
            precision = precision + (1.0 / float(pv)) * np.eye(d)
        return np.asarray(np.linalg.pinv(precision, rcond=1e-12), dtype=complex)
    mx = plan.subspace.complement_matrix(obs.X)
    gram = mx.conj().T @ mx
    if pv:
        cov = np.linalg.inv((gram / sigma2) + (1.0 / float(pv)) * np.eye(d))
    else:
        cov = np.linalg.pinv(gram, rcond=1e-10) * sigma2
    return np.asarray(cov, dtype=complex)
