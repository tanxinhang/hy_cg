"""臂表的数据形状：一条臂"跑的是哪个估计器"，供线性探针重建它的仿射形式。

这里刻意镜像 ``isac_sim.receiver.cancellation.cancellation_arms`` 的臂表。
候选支撑集是**从已记录的结果里读回**的，而不是重算一遍 —— 唯一被复制的逻辑
是下面这张很小的子空间/先验表，而 ``tests/test_cancellation_glrt.py`` 会在观测
本身上比对"重建的仿射形式"与"真实的臂"，一旦 V1 改了臂表而这里没跟上，探针
就复现不出那条臂，每个白化统计量都在悄悄描述一个并不存在的抵消器。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from isac_sim.receiver.cancellation import Observation, Subspace

#: 臂的固定顺序。扫描与出图都按它排序，避免"字典序"悄悄改变结论的排列。
ARM_ORDER: Tuple[str, ...] = (
    "no_ic",
    "plain_ls",
    "ridge_ls",
    "protected_ls",
    "tp_uic_stage1",
    "tp_uic_full",
    "perfect_channel",
)


@dataclass
class ArmPlan:
    """一条臂跑的是哪个估计器，从而可以重建它的线性形式。"""

    name: str
    kind: str  # "none" | "oracle" | "estimator"
    subspace: Subspace
    prior: float | None
    candidates: Tuple[int, ...] = ()
    soft_mu: float | None = None


def _empty_subspace(n_bins: int) -> Subspace:
    """空子空间：不保护任何东西的估计器，以及根本不做消除的臂。"""
    return Subspace(U=np.zeros((n_bins, 0), dtype=complex), rank=0)


def _belief_subspace(obs: Observation) -> Subspace:
    """信念保护子空间：由被保护目标的正切列张成。"""
    U = obs.basis_belief
    return Subspace(U=U, rank=0 if U is None else int(U.shape[1]))
