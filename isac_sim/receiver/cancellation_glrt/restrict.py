"""单目标机制验证观测：把"多目标可辨识性"这个问题先摘掉。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from isac_sim.receiver.cancellation import (
    Observation,
    _protection_basis,
    target_dictionary,
)
from isac_sim.receiver.cancellation_glrt.target_glrt import _centre_mask


def restrict_to_target(cfg, obs: Observation, target: int) -> Observation:
    """同一份几何，但只剩一条回波和一个单目标保护基。

    这是**机制验证**用的观测：它回答"干扰消除到底能不能传导到检测"这一个问题，
    把多目标可辨识性的疑问摘掉，除此之外不做别的。它保持 ``X`` 与直连场不动，
    把其他回波置零，把 ``A``/``A_true`` 与 id 限制到被测目标，并从该目标单独
    重建两个保护基，于是抵消器永远不会被要求去保护幻影。
    """
    ids = np.asarray(obs.A_true_target_ids)
    sel = ids == int(target)
    if not np.any(sel):
        raise ValueError("target %d has no columns in the true dictionary" % (target,))

    true_sources = [s for s in obs.targets if int(s.target) == int(target)]
    belief_sources = [
        s for s in (obs.targets_belief or []) if int(s.target) == int(target)
    ]
    if not true_sources:
        raise ValueError("target %d has no true source" % (target,))

    A_true = target_dictionary(
        cfg, true_sources, covariance_expanded=False
    )
    alpha_true = np.asarray(obs.alpha_true)[sel].copy()
    s_target = A_true @ alpha_true
    if belief_sources:
        A = target_dictionary(cfg, belief_sources)
        A_ids = np.full(A.shape[1], int(target), dtype=int)
        centre_blocks = []
        for src in belief_sources:
            width = target_dictionary(cfg, [src]).shape[1]
            block = np.zeros(width, dtype=bool)
            if width:
                block[0] = True
            centre_blocks.append(block)
        A_centres = np.concatenate(centre_blocks)
    else:  # 手工构造观测的调用方
        A, A_ids = obs.A, np.full(int(obs.A.shape[1]), int(target), dtype=int)
        A_centres = _centre_mask(cfg, A.shape[1])
    true_centres = _centre_mask(cfg, A_true.shape[1])
    noise = obs.y - obs.x_direct - obs.s_target
    n_bins = int(obs.y.size)
    return replace(
        obs,
        y=obs.x_direct + s_target + noise,
        A=A,
        A_target_ids=A_ids,
        A_true_target_ids=np.full(A_true.shape[1], int(target), dtype=int),
        A_centre_mask=A_centres,
        A_true_centre_mask=true_centres,
        s_target=s_target,
        alpha_true=alpha_true,
        targets=true_sources,
        targets_belief=belief_sources or true_sources,
        basis_belief=_protection_basis(cfg, belief_sources or true_sources, n_bins),
        basis_truth=_protection_basis(cfg, true_sources, n_bins),
        weak_index=int(target),
    )
