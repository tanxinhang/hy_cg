"""目标条件化白化 GLRT：式 (1) 的 ``T_q``。"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from isac_sim.receiver.cancellation import EPS, CancellationResult, Observation
from isac_sim.receiver.cancellation_glrt.chi2 import glrt_threshold
from isac_sim.receiver.cancellation_glrt.linalg import (
    _generalised_min_eigen,
    _numerical_rank,
    _project_out,
    _projector_energy,
)
from isac_sim.receiver.cancellation_glrt.null_threshold import _null_cov_threshold
from isac_sim.receiver.cancellation_glrt.residual_form import ResidualModel
from isac_sim.receiver.cancellation_glrt.target_glrt import (
    TargetGLRT,
    _dictionary,
    _dictionary_centre_mask,
    _manifold_columns,
)


def target_conditioned_glrt(
    cfg,
    obs: Observation,
    result: CancellationResult,
    model: ResidualModel,
    *,
    target: int | None = None,
    p_fa: float = 0.05,
    nuisance_manifold: int = 0,
    manifold_step: float | None = None,
    dictionary: str = "belief",
    template_override: np.ndarray | None = None,
    centre_only: bool = True,
    nuisance_targets: Sequence[int] | None = None,
    null_cov_model: ResidualModel | None = None,
    threshold_override: float | None = None,
) -> TargetGLRT:
    """``T_q`` 以及与它配套的可辨识度数字。

    ``nuisance_manifold`` 用**其他**目标的分数 DD 正切列去扩张干扰块，也就是让
    检测器说"另外九个只在某个连续的延迟/多普勒偏移内已知"。这是该问题诚实的
    版本；比较 order 0 与 order 1 下的 ``rho``，正是把"网格让它们撞在一起"和
    "它们真是同一个子空间"区分开的手段。
    """
    A, ids = _dictionary(cfg, obs, dictionary)
    tgt = int(obs.weak_index if target is None else target)
    sel_tgt = np.asarray(ids) == tgt
    if not np.any(sel_tgt):
        raise ValueError(
            "target %d has no column in the %s dictionary" % (tgt, dictionary)
        )
    base = (
        _dictionary_centre_mask(cfg, obs, dictionary, A.shape[1])
        if centre_only else np.ones(A.shape[1], dtype=bool)
    )
    sel_q = sel_tgt & base
    if not np.any(sel_q):
        raise ValueError("target %d has no centre column" % (tgt,))

    if template_override is not None:
        A_q = np.asarray(template_override, dtype=complex)
        if A_q.ndim == 1:
            A_q = A_q[:, None]
    else:
        A_q = A[:, sel_q]
    A_neg = np.where(
        np.isin(np.asarray(ids), list(nuisance_targets)) if nuisance_targets is not None
        else ~sel_tgt,
        base,
        False,
    )
    A_neg = A[:, A_neg]
    if nuisance_manifold > 0:
        extra = _manifold_columns(
            cfg, obs, tgt, int(nuisance_manifold), manifold_step, dictionary,
            subset=nuisance_targets,
        )
        if extra.shape[1]:
            A_neg = np.concatenate([A_neg, extra], axis=1)

    cov = model.cov
    w_q = cov.whiten_matrix(model.signal_transfer(A_q))
    w_neg = cov.whiten_matrix(model.signal_transfer(A_neg))
    r_whitened = cov.whiten(model.residual(obs.y))

    B_q = _project_out(w_neg, w_q)
    statistic = _projector_energy(B_q, r_whitened)
    dof_real = 2 * _numerical_rank(B_q)
    threshold = (
        float(threshold_override)
        if threshold_override is not None else glrt_threshold(p_fa, dof_real)
    )
    if threshold_override is None and null_cov_model is not None and dof_real > 0:
        threshold = _null_cov_threshold(cov, B_q, dof_real, p_fa, null_cov_model)

    g_self = np.real(np.sum(np.abs(w_q) ** 2, axis=0))
    g_keep = np.real(np.sum(np.abs(B_q) ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(g_self > 0.0, g_keep / np.maximum(g_self, EPS), 0.0)
    rho = np.clip(rho, 0.0, 1.0)
    gram_q = B_q.conj().T @ B_q if B_q.shape[1] else np.zeros((0, 0), dtype=complex)
    # 两个偏移参数，因为它们回答的是不同的问题。回波系数是单位幅度、随机相位，
    # 所以**期望**偏移是非相干和 sum_c rho_c g_c；``ncp_best`` 是相干界（最大
    # 奇异值的平方），即接收机若知道相位能达到的上界 —— 两者的差距就是"不知道
    # 相位"的代价。
    sv_b = np.linalg.svd(B_q, compute_uv=False) if B_q.shape[1] else np.zeros(0)
    total_self = float(np.sum(g_self))

    return TargetGLRT(
        arm=model.name,
        target=tgt,
        statistic=float(statistic),
        dof_real=dof_real,
        threshold=float(threshold),
        p_fa=float(p_fa),
        detected=bool(statistic > threshold) if np.isfinite(threshold) else False,
        rho=np.asarray(rho, dtype=float),
        g_self=np.asarray(g_self, dtype=float),
        ncp_unit=float(np.sum(rho * g_self)),
        ncp_best=float(sv_b[0] ** 2) if sv_b.size else 0.0,
        rho_weighted=float(np.sum(rho * g_self) / total_self) if total_self > 0 else 0.0,
        xi_q=float(np.linalg.eigvalsh(gram_q).min()) if gram_q.shape[0] else 0.0,
        xi_rel_q=_generalised_min_eigen(w_q.conj().T @ w_q, gram_q),
        n_templates=int(A_q.shape[1]),
        n_nuisance_columns=int(A_neg.shape[1]),
        nuisance_rank=_numerical_rank(w_neg),
        residual_power=float(np.vdot(model.residual(obs.y), model.residual(obs.y)).real),
        whitened_residual_power=float(np.vdot(r_whitened, r_whitened).real),
    )
