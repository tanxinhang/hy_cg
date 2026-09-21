"""连续 DD 可辨识性审计：目标 ``q`` 到底能不能被分开。"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from isac_sim.receiver.cancellation import CancellationResult, Observation, TargetSource
from isac_sim.receiver.cancellation import tangent_columns
from isac_sim.receiver.cancellation_glrt.glrt import target_conditioned_glrt
from isac_sim.receiver.cancellation_glrt.ident import IdentifiabilityAudit
from isac_sim.receiver.cancellation_glrt.residual_form import ResidualModel
from isac_sim.receiver.cancellation_glrt.target_glrt import _dictionary


def identifiability_audit(
    cfg,
    obs: Observation,
    model: ResidualModel,
    result: CancellationResult,
    *,
    target: int | None = None,
    dictionary: str = "belief",
    p_fa: float = 0.05,
    centre_only: bool = True,
    deltas: Iterable[Tuple[float, float]] = (
        (0.4, 0.0),
        (-0.4, 0.0),
        (0.0, 0.4),
        (0.0, -0.4),
        (0.4, 0.4),
        (-0.4, -0.4),
        (0.5, 0.5),
    ),
) -> IdentifiabilityAudit:
    """在网格上和连续模型里，目标 ``q`` 是否可分？

    ``rho_on_grid`` 用其他目标的**在网格上**的列作干扰，``rho_manifold`` 再加上
    它们的一阶正切族，``off_grid`` 则把目标 ``q`` 自己的模板按分数延迟/多普勒
    偏移挪动，而其他目标保留各自的流形。一个在**每一种**偏移下都被掩蔽的目标
    是**本征地**共位的，此时该做的是提高物理分辨率，而不是换一个更好的抵消器。
    """
    A, ids = _dictionary(cfg, obs, dictionary)
    tgt = int(obs.weak_index if target is None else target)
    on_grid = target_conditioned_glrt(
        cfg, obs, result, model, target=tgt, p_fa=p_fa,
        nuisance_manifold=0, dictionary=dictionary, centre_only=centre_only,
    )
    manifold = target_conditioned_glrt(
        cfg, obs, result, model, target=tgt, p_fa=p_fa,
        nuisance_manifold=1, dictionary=dictionary, centre_only=centre_only,
    )
    manifold2 = target_conditioned_glrt(
        cfg, obs, result, model, target=tgt, p_fa=p_fa,
        nuisance_manifold=2, dictionary=dictionary, centre_only=centre_only,
    )

    sources: Sequence[TargetSource] = (
        obs.targets_belief if dictionary == "belief" else obs.targets
    ) or obs.targets
    per_target = [s for s in sources if int(s.target) == tgt]
    step = float(cfg.cancellation.tangent_step_bins)
    base_A_q = A[:, np.asarray(ids) == tgt]

    off_grid: Dict[Tuple[float, float], np.ndarray] = {}
    for dk, dl in deltas:
        if dk == 0.0 and dl == 0.0:
            continue
        cols = [
            math.sqrt(max(float(src.power), 0.0))
            * tangent_columns(
                cfg, float(src.doppler_bin) + float(dk),
                float(src.delay_bin) + float(dl), 0, step,
            )
            for src in per_target
        ]
        template = np.concatenate(cols, axis=1) if cols else base_A_q
        shifted = target_conditioned_glrt(
            cfg, obs, result, model, target=tgt, p_fa=p_fa,
            nuisance_manifold=1, dictionary=dictionary, template_override=template,
            centre_only=centre_only,
        )
        off_grid[(float(dk), float(dl))] = shifted.rho

    return IdentifiabilityAudit(
        arm=model.name,
        target=tgt,
        dictionary=dictionary,
        rho_on_grid=on_grid.rho,
        rho_manifold=manifold.rho,
        rho_manifold2=manifold2.rho,
        xi_rel_on_grid=on_grid.xi_rel_q,
        xi_rel_manifold=manifold.xi_rel_q,
        n_templates=on_grid.n_templates,
        n_nuisance_on_grid=on_grid.n_nuisance_columns,
        n_nuisance_manifold=manifold.n_nuisance_columns,
        ncp_unit_on_grid=on_grid.ncp_unit,
        ncp_unit_manifold=manifold.ncp_unit,
        dof_on_grid=on_grid.dof_real,
        dof_manifold=manifold.dof_real,
        off_grid=off_grid,
    )
