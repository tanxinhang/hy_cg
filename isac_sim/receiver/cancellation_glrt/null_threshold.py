"""失配原假设下的门限：把"建模的 C_res 与实际的 C_res 不同"兑换成门限。"""
from __future__ import annotations

import numpy as np

from isac_sim.receiver.cancellation_glrt.chi2 import glrt_threshold


def _null_cov_threshold(cov, B_q: np.ndarray, dof_real: int, p_fa: float, null_cov_model):
    """在原假设协方差另有其物时，重算 ``T_q`` 的 CFAR 门限。

    当检测器用的白化协方差与残余真实的协方差不一致时，``2 T_q`` 不再是标准
    卡方，而是**加权**卡方：权重是零模型协方差在被测子空间上的特征值。特征值
    全相等时可以直接缩放门限；不等时改用固定数值求积 —— 用固定种子的指数抽样
    估计分位数，使门限对"评估 H0 的那几组抽样"保持确定、可复现。
    """
    rank_q = dof_real // 2
    q_basis = np.linalg.qr(B_q)[0][:, :rank_q]
    raw_directions = cov.whiten_matrix(q_basis)
    null_small = raw_directions.conj().T @ (
        null_cov_model.cov.apply_matrix(raw_directions)
    )
    null_eigenvalues = np.maximum(
        np.real(np.linalg.eigvalsh(0.5 * (null_small + null_small.conj().T))),
        0.0,
    )
    if np.allclose(null_eigenvalues, null_eigenvalues[0], rtol=1e-10, atol=0.0):
        return float(null_eigenvalues[0]) * glrt_threshold(p_fa, dof_real)
    # 复高斯二次型是若干单位指数变量的加权和。固定数值求积让门限确定，
    # 且独立于被评估的那些 H0 抽样。
    rng = np.random.default_rng(0x43464152)
    draws = rng.exponential(
        scale=1.0, size=(262144, int(null_eigenvalues.size))
    ) @ null_eigenvalues
    return float(np.quantile(draws, 1.0 - float(p_fa), method="higher"))
