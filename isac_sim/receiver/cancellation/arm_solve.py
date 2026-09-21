"""cancellation 积木：单臂的线性算子（解算 / 削减 / 存活率）。"""

from __future__ import annotations

import numpy as np

from isac_sim.receiver.cancellation.joint import (
    joint_interference_covariance,
    joint_refine,
    joint_system,
)
from isac_sim.receiver.cancellation.protect import protected_map, soft_protected_map


class _ArmSolveMixin:
    """把同一个线性估计器分别作用在各个分量上。

    字段由 :class:`isac_sim.receiver.cancellation.arm_operator._ArmOperator`
    提供：``X``/``A``/``sigma2``/``n_bins`` 来自共享上下文，``subspace``/
    ``pv``/``candidates``/``soft_mu`` 是这一臂自己的选择。
    """

    def _joint(self):
        """这一臂的联合系统（``D`` / 正规方程 / Gram / 先验精度），只算一次。

        ``solve`` 会被调用二十几次（5 个分量 + ``arm_moments`` 一次 + 十几个
        sigma 点），而系统只依赖 ``(X, A_cand, sigma2, pv)`` —— 它们在这一臂
        构造时就定死了。缓存它是**数值恒等**的：同样的运算、同样的顺序，
        解算器拿到的输入逐位相同。
        """
        cached = self.__dict__.get("_joint_cache")
        if cached is None:
            A_cand = self.A[:, list(self.candidates)]
            system = joint_system(self.X, A_cand, self.sigma2, self.pv,
                                  self.pv or 1.0)
            # ``D`` 的前 d 列就是 ``X``，后面是候选块；切片与
            # ``A[:, list(candidates)]`` 逐位相同，但省掉一次花式索引拷贝。
            A_view = system[0][:, self.X.shape[1]:]
            cov = joint_interference_covariance(
                self.X, A_view, self.sigma2, self.pv, self.pv or 1.0,
                system=system,
            )
            # 后验协方差同样与右端无关 —— 20/21 次 ``pinv`` 是重复的。
            cached = (system, A_view, np.real(np.diag(cov)))
            self._joint_cache = cached
        return cached

    def solve(self, v: np.ndarray):
        """解出系数与协方差对角。三条互斥路径：软保护 / 联合精化 / 受保护 MAP。"""
        if self.soft_mu is not None:
            h, cov_h = soft_protected_map(v, self.X, self.subspace, self.sigma2,
                                          self.pv, self.soft_mu)
            return h, np.real(np.diag(cov_h))
        if self.candidates:
            system, A_cand, cov_diag = self._joint()
            h, _a = joint_refine(v, self.X, A_cand, self.sigma2, self.pv,
                                 self.pv or 1.0, system=system)
            return h, cov_diag
        h, c, _mx = protected_map(v, self.X, self.subspace, self.sigma2, self.pv)
        return h, c

    def removed(self, v: np.ndarray) -> np.ndarray:
        """该臂实际减掉的那部分。受保护 MAP 只减 ``M`` 里的分量，联合/软臂减全部。"""
        if not v.shape[1]:
            return np.zeros_like(v)
        h_v = self.solve(v)[0]
        return self.X @ h_v if self.full_subtraction else self.mx @ h_v

    def survival(self, v: np.ndarray) -> float:
        """能量存活率。分母为 0 时返回 1.0（乘性分子的中性值）。"""
        energy = float(np.real(np.vdot(v, v)))
        if energy <= 0.0:
            return 1.0
        left = v - self.removed(v)
        return float(np.real(np.vdot(left, left)) / energy)
