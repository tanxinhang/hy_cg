"""一条臂的仿射作用，以及它留下的残余协方差。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from isac_sim.receiver.cancellation import EPS
from isac_sim.receiver.cancellation_glrt.covariance import LowRankCovariance


@dataclass
class ResidualModel:
    """一条臂的仿射动作加上它留下的协方差。

    ``removed(y) = basis small basis^H y + const``；``direct_gain`` 缩放残留下来的
    直连场（估计器臂为 ``1``，oracle 臂为 ``0``）；
    ``cov`` 是式 (2) 的 ``C_res``。
    """

    name: str
    basis: np.ndarray
    small: np.ndarray
    const: np.ndarray
    direct_gain: float
    sigma2: float
    cov: LowRankCovariance
    probe_error: float = 0.0
    direct_variance: str = "prior"
    r_diag: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def rank(self) -> int:
        """消除映射 ``F`` 的秩。"""
        return int(self.basis.shape[1])

    def remove(self, v: np.ndarray) -> np.ndarray:
        """``F v`` —— 只含与 ``y`` 有关的那部分消除量。"""
        if self.rank == 0:
            return np.zeros_like(v)
        return self.basis @ (self.small @ (self.basis.conj().T @ v))

    def remove_matrix(self, V: np.ndarray) -> np.ndarray:
        """``F V`` —— 逐列的矩阵版本。"""
        if self.rank == 0:
            return np.zeros_like(V)
        return self.basis @ (self.small @ (self.basis.conj().T @ V))

    def residual(self, v: np.ndarray) -> np.ndarray:
        """``v - F v - c`` —— 接收机最后拿到的东西。"""
        return v - self.remove(v) - self.const

    def signal_transfer(self, A: np.ndarray) -> np.ndarray:
        """``T_e A``：这条臂如何衰减一组回波模板。"""
        return A - self.remove_matrix(A)

    def transfer_frobenius(self) -> float:
        """``tr((I-F)(I-F)^H)`` 的闭式，供协方差自检使用。"""
        n_bins = int(self.basis.shape[0])
        if self.rank == 0:
            return float(n_bins)
        s = self.small
        return float(
            n_bins
            - 2.0 * np.real(np.trace(s))
            + np.real(np.trace(s.conj().T @ s))
        )
