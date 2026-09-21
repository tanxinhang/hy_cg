"""``sigma^2 I +`` 低秩形式的协方差，以及它的精确白化。

``C = sigma2 I + W diag(lam) W^H``，``W`` 标准正交。保留特征形式正是白化所需::

    C^{-1/2} v = sigma^{-1} (v + W ((1 + lam/sigma2)^{-1/2} - 1) * (W^H v))

等于 ``sigma2`` 的特征值干脆不进 ``W``。这样 ``K x K`` 矩阵永远不会被显式构造，
4096 个 bin 上的白化才付得起。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from isac_sim.receiver.cancellation import EPS


@dataclass
class LowRankCovariance:
    """``C = sigma2 I + W diag(lam) W^H``，``W`` 标准正交。"""

    sigma2: float
    W: np.ndarray
    lam: np.ndarray
    min_ratio: float

    @property
    def rank(self) -> int:
        """低秩部分的维数（不含被省略的 ``sigma2`` 平台）。"""
        return int(self.W.shape[1])

    @property
    def trace(self) -> float:
        """``tr(C)``：闭式，不构造矩阵。"""
        n_bins = int(self.W.shape[0])
        return float(self.sigma2 * (n_bins - self.rank) + np.sum(self.sigma2 + self.lam))

    def apply_matrix(self, V: np.ndarray) -> np.ndarray:
        """``C @ V``。"""
        WtV = self.W.conj().T @ V
        return self.sigma2 * V + self.W @ (self.lam[:, None] * WtV)

    def apply(self, v: np.ndarray) -> np.ndarray:
        """``C @ v``。"""
        return self.apply_matrix(np.asarray(v)[:, None])[:, 0]

    def whiten_matrix(self, V: np.ndarray) -> np.ndarray:
        """``C^{-1/2} @ V``。"""
        factor = (1.0 + self.lam / self.sigma2) ** -0.5 - 1.0
        return (V + self.W @ (factor[:, None] * (self.W.conj().T @ V))) / math.sqrt(
            self.sigma2
        )

    def whiten(self, v: np.ndarray) -> np.ndarray:
        """``C^{-1/2} @ v``。"""
        return self.whiten_matrix(np.asarray(v)[:, None])[:, 0]

    def inverse_matrix(self, V: np.ndarray) -> np.ndarray:
        """``C^{-1} @ V``。"""
        factor = 1.0 / (1.0 + self.lam / self.sigma2) - 1.0
        return (V + self.W @ (factor[:, None] * (self.W.conj().T @ V))) / self.sigma2

    def inverse(self, v: np.ndarray) -> np.ndarray:
        """``C^{-1} @ v``。"""
        return self.inverse_matrix(np.asarray(v)[:, None])[:, 0]


def _check_covariance_trace(
    cov: "LowRankCovariance",
    blocks: list,
    sigma2: float,
    n_bins: int,
    name: str,
) -> None:
    """特征形式必须复现解析迹。

    不一致意味着限制到 ``span(U)`` 时漏掉了一部分扰动，那会让之后**每一个**
    统计量都被静默地错误白化。
    """
    analytic = sigma2 * float(n_bins) + sum(
        float(np.sum(np.abs(b) ** 2)) for b in blocks
    )
    if abs(cov.trace - analytic) > 1e-6 * max(abs(analytic), EPS):
        raise RuntimeError(
            "arm %r: residual-covariance trace mismatch (eigen %.6e vs "
            "analytic %.6e)" % (name, cov.trace, analytic)
        )
