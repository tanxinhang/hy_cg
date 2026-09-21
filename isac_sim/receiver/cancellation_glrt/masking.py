"""掩蔽曲线：要多少个共位散射体才把 ``rho`` 压下去。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from isac_sim.receiver.cancellation import CancellationResult, Observation, TargetSource
from isac_sim.receiver.cancellation_glrt.glrt import target_conditioned_glrt
from isac_sim.receiver.cancellation_glrt.residual_form import ResidualModel
from isac_sim.receiver.cancellation_glrt.target_glrt import _dictionary


@dataclass
class MaskingCurve:
    """``rho`` 随"把多少个其他目标算作干扰"变化的曲线。

    这条曲线的**形状本身就是机制**。若 ``rho`` 随邻居增加而**逐渐**崩塌，掩蔽
    就是"子空间并集"（patch covering）效应，起约束作用的是共位散射体的**个数**；
    若它在第一个邻居处就崩塌，被测目标就是被那一个目标单独掩蔽的，该改的是
    观测几何而不是分辨率。
    """

    arm: str
    target: int
    dictionary: str
    counts: List[int]
    rho_weighted: List[float]
    rho_min: List[float]
    ncp_unit: List[float]
    distance_bins: List[float]
    neighbours: List[int]


def masking_curve(
    cfg,
    obs: Observation,
    model: ResidualModel,
    result: CancellationResult,
    *,
    target: int | None = None,
    dictionary: str = "belief",
    p_fa: float = 0.05,
    counts: Iterable[int] = (0, 1, 2, 3, 5, 9),
    centre_only: bool = True,
) -> MaskingCurve:
    """``rho`` 对"最近的若干个共位目标被算作干扰"的个数。"""
    A, ids = _dictionary(cfg, obs, dictionary)
    tgt = int(obs.weak_index if target is None else target)
    sources: Sequence[TargetSource] = (
        (obs.targets_belief if dictionary == "belief" else obs.targets)
        or obs.targets_belief
        or obs.targets
    )
    positions: Dict[int, List[Tuple[float, float]]] = {}
    for src in sources:
        positions.setdefault(int(src.target), []).append(
            (float(src.doppler_bin), float(src.delay_bin))
        )
    if tgt not in positions:
        raise ValueError("target %d has no source" % (tgt,))
    mine = np.mean(np.asarray(positions[tgt], dtype=float), axis=0)
    others = sorted(q for q in positions if q != tgt)
    dist = {q: float(np.linalg.norm(np.mean(np.asarray(positions[q], dtype=float), axis=0) - mine))
            for q in others}
    order = sorted(others, key=lambda q: dist[q])

    rho_w: List[float] = []
    rho_m: List[float] = []
    ncp: List[float] = []
    reach: List[float] = []
    used: List[int] = []
    for n in counts:
        n = int(min(max(n, 0), len(order)))
        subset = order[:n]
        got = target_conditioned_glrt(
            cfg, obs, result, model, target=tgt, p_fa=p_fa,
            nuisance_manifold=0, dictionary=dictionary, centre_only=centre_only,
            nuisance_targets=subset,
        )
        rho_w.append(got.rho_weighted)
        rho_m.append(got.rho_min)
        ncp.append(got.ncp_unit)
        reach.append(max((dist[q] for q in subset), default=0.0))
        used.append(n)
    return MaskingCurve(
        arm=model.name,
        target=tgt,
        dictionary=dictionary,
        counts=used,
        rho_weighted=rho_w,
        rho_min=rho_m,
        ncp_unit=ncp,
        distance_bins=reach,
        neighbours=[int(q) for q in order],
    )
