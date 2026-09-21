"""cancellation 积木：对消臂的共享上下文。"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.receiver.cancellation.containers import Observation
from isac_sim.receiver.cancellation.dictionaries import target_dictionary
from isac_sim.receiver.cancellation.manifold import Subspace, orthonormalise


class ArmContext:
    """一次 trial 上所有对消臂共享的量。

    原先这些变量被 ``cancellation_arms`` 里的闭包 ``operator`` 隐式捕获；
    提成显式对象后，算子、矩、sigma 点风险三块才能各自独立成积木。
    字段与原局部变量一一对应，语义不变。
    """

    def __init__(
        self, cfg: Config, obs: Observation, *, weak_target: int | None,
        threshold: float, candidate_policy: str,
    ) -> None:
        self.cfg = cfg
        self.obs = obs
        self.threshold = threshold
        self.candidate_policy = candidate_policy
        X, A, y = obs.X, obs.A, obs.y
        x, s = obs.x_direct, obs.s_target
        self.X, self.A, self.y = X, A, y
        self.x, self.s = x, s
        self.n = y - x - s
        self.sigma2 = float(obs.sigma2)
        self.n_bins = int(y.size)
        self.wt = int(obs.weak_index if weak_target is None else weak_target)
        # 弱目标统计量是"该目标列张成块子空间内的能量 / 秩"，于是它的 H0 均值
        # 就是每格噪声方差，CFAR 门限可以在不知道几何的前提下设定。单列模板在
        # 这里是错的：一个目标被多架 UAV 照射，接收机并不知道哪条路径占优。
        self.weak_block = None
        if obs.A_target_ids is not None and obs.A.shape[1]:
            cols = A[:, np.asarray(obs.A_target_ids) == self.wt]
            if cols.shape[1]:
                self.weak_block = orthonormalise(cols)
        self.belief = Subspace(
            U=obs.basis_belief,
            rank=0 if obs.basis_belief is None else obs.basis_belief.shape[1],
        )
        self.empty = Subspace(U=np.zeros((self.n_bins, 0), dtype=complex), rank=0)
        self.prior = (
            cfg.cancellation.prior_variance if cfg.cancellation.prior_variance else None
        )
        # ``h_true`` 在每个直达切向块的中心列上放一个单位功率随机相位，导数列
        # 系数为零。这个期望才是解析残差预测的匹配分母：拿期望残差除以**实现**
        # 出来的 ``||X h_true||^2`` 会把随机相位相消重新引入一个本应在观测前就
        # 稳定的能力证书。
        block = 1 + 2 * int(cfg.cancellation.interference_tangent_order)
        self.direct_centres = np.arange(0, X.shape[1], block, dtype=int)
        self.i_in_pred = float(
            np.vdot(X[:, self.direct_centres], X[:, self.direct_centres]).real
        ) if self.direct_centres.size else 0.0
        # 被测目标自己的回波，从真值里隔离出来。下面一切都是线性的，所以把某个
        # 臂单独作用在 ``s_q`` 上，得到的就是总削减里落在**这个目标**上的那一份。
        self.s_q = None
        if obs.alpha_true is not None and obs.A_true_target_ids is not None:
            ids_t = np.asarray(obs.A_true_target_ids)
            alpha = np.asarray(obs.alpha_true)
            A_true = target_dictionary(cfg, obs.targets, covariance_expanded=False)
            if ids_t.size == alpha.size == A_true.shape[1]:
                sel_q = ids_t == self.wt
                if np.any(sel_q):
                    self.s_q = A_true[:, sel_q] @ alpha[sel_q]
        self.sources_q = [
            src for src in (obs.targets_belief or ()) if int(src.target) == self.wt
        ]
        self.belief_q = target_dictionary(
            cfg, self.sources_q, tangent_order=0, covariance_expanded=False
        )
