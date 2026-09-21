"""cancellation 积木：残差功率的矩与噪声谱。

``||B h||^2 + ||F n||^2`` 的分布：物理直达系数是独立的单位模相位，噪声是圆对称
复高斯。两部分的矩都归结到小 Gram 矩阵上，全程不构造 KxK 协方差。
"""

from __future__ import annotations

import numpy as np

from isac_sim.receiver.cancellation.joint import _positive_sigma


def structural_moments(op):
    """直达场被削减后剩下的那部分的均值与方差。"""
    physical_x = op.X[:, op.ctx.direct_centres]
    left = physical_x - op.removed(physical_x)
    gram = left.conj().T @ left
    mean = float(np.real(np.trace(gram)))
    var = float(
        np.sum(np.abs(gram) ** 2) - np.sum(np.abs(np.diag(gram)) ** 2)
    )
    return mean, var, gram


def noise_moments(op):
    """估计误差引入的噪声增强：均值、方差与有效谱。"""
    sg = _positive_sigma(op.sigma2)
    cc_h, subtraction_dictionary = _estimator_gram(op, sg)
    output_gram = subtraction_dictionary.conj().T @ subtraction_dictionary
    noise_core = cc_h @ output_gram
    mean = sg * float(np.real(np.trace(noise_core)))
    var = sg * sg * float(np.real(np.trace(noise_core @ noise_core)))
    cc_eval, cc_evec = np.linalg.eigh((cc_h + cc_h.conj().T) / 2.0)
    cc_eval = np.maximum(np.real(cc_eval), 0.0)
    sqrt_cc = (cc_evec * np.sqrt(cc_eval)) @ cc_evec.conj().T
    spectrum = sg * np.maximum(
        np.real(np.linalg.eigvalsh(
            (sqrt_cc @ output_gram @ sqrt_cc
             + (sqrt_cc @ output_gram @ sqrt_cc).conj().T) / 2.0
        )),
        0.0,
    )
    top = float(np.max(spectrum, initial=0.0))
    spectrum = (
        spectrum[spectrum > 1e-12 * top] if top > 0.0
        else np.zeros(0, dtype=float)
    )
    return mean, var, spectrum


def _estimator_gram(op, sg: float):
    """估计器的系数协方差 ``C_h`` 与该臂实际相减所用的字典。"""
    X, pv = op.X, op.pv
    if op.soft_mu is not None:
        px = op.subspace.project(X)
        normal = X.conj().T @ X + float(op.soft_mu) * (px.conj().T @ px)
        if pv is not None and pv > 0.0:
            normal = normal + (sg / float(pv)) * np.eye(normal.shape[0])
        inv_normal = np.linalg.pinv(normal, rcond=1e-12)
        return inv_normal @ (X.conj().T @ X) @ inv_normal.conj().T, X
    if op.candidates:
        d_joint = np.concatenate([X, op.A[:, list(op.candidates)]], axis=1)
        gram_joint = d_joint.conj().T @ d_joint
        lam = np.zeros(gram_joint.shape[0])
        if pv is not None and pv > 0.0:
            lam[: X.shape[1]] = sg / float(pv)
            lam[X.shape[1]:] = sg / float(pv)
        inv_joint = np.linalg.pinv(
            gram_joint + np.diag(lam), rcond=1e-12
        )[: X.shape[1], :]
        return inv_joint @ gram_joint @ inv_joint.conj().T, X
    mx = op.mx
    gram_protected = mx.conj().T @ mx
    normal = gram_protected.copy()
    if pv is not None and pv > 0.0:
        normal = normal + (sg / float(pv)) * np.eye(normal.shape[0])
    inv_normal = np.linalg.pinv(normal, rcond=1e-12)
    return inv_normal @ gram_protected @ inv_normal.conj().T, mx


def arm_moments(op):
    """合成该臂报告用的矩：均值、方差、直达 Gram、噪声谱。"""
    s_mean, s_var, direct_gram = structural_moments(op)
    n_mean, n_var, spectrum = noise_moments(op)
    return (
        max(s_mean + n_mean, 0.0),
        max(s_var + n_var, 0.0),
        np.asarray(direct_gram, dtype=complex),
        np.asarray(spectrum, dtype=float),
    )
