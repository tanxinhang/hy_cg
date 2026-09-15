"""Per-target diagnostics that separate sensing, capacity, communication and fusion headroom."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np

from .config import Config, Link
from .fusion import deflection_for_links, predicted_pd_for_links
from .model import BaseGains, LinkTables
from .reporting import ReportingPlan, is_local_observation
from .selection import feasible_links_for_target


@dataclass(frozen=True)
class FusionHeadroom:
    target: int
    pd_local: float
    pd_best_k2: float
    pd_best_ksafe: float
    pd_perfect_comm_ksafe: float
    pd_fixed_k2: float
    best_fusion_k2: int
    best_fusion_ksafe: int
    fixed_fusion: int
    links_local: tuple[Link, ...]
    links_k2: tuple[Link, ...]
    links_ksafe: tuple[Link, ...]

    @property
    def observation_headroom(self) -> float:
        return self.pd_best_ksafe - self.pd_best_k2

    @property
    def communication_headroom(self) -> float:
        return self.pd_perfect_comm_ksafe - self.pd_best_ksafe

    @property
    def fusion_location_headroom(self) -> float:
        return self.pd_best_k2 - self.pd_fixed_k2


def _single_target_plan(cfg: Config, fusion: int) -> ReportingPlan:
    return ReportingPlan(
        mode="explicit", f_q=np.full(cfg.scale.Q, int(fusion), dtype=int)
    )


def _shortlist(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    q: int,
    fusion: int,
    per_type: int,
) -> tuple[ReportingPlan, list[Link]]:
    plan = _single_target_plan(cfg, fusion)
    scored = [
        (
            deflection_for_links(
                cfg, tables, q, [link], weight_mode="deflection",
                plan=plan, base=base,
            ),
            link,
        )
        for link in feasible_links_for_target(cfg, base, tables, q, plan)
    ]
    local = sorted(
        (item for item in scored if is_local_observation(plan, item[1], q)),
        reverse=True,
    )[:per_type]
    remote = sorted(
        (item for item in scored if not is_local_observation(plan, item[1], q)),
        reverse=True,
    )[:per_type]
    return plan, list(dict.fromkeys(link for _, link in local + remote))


def _pd(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    q: int,
    plan: ReportingPlan,
    links: list[Link] | tuple[Link, ...],
) -> float:
    if not links:
        return 0.0
    return predicted_pd_for_links(
        cfg, tables, q, list(links), weight_mode="exact_llr_sum",
        plan=plan, base=base,
    )


def _best_bundle(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    q: int,
    fusion: int,
    max_links: int,
    per_type: int,
) -> tuple[float, tuple[Link, ...]]:
    plan, candidates = _shortlist(cfg, base, tables, q, fusion, per_type)
    limit = min(max(int(max_links), 0), len(candidates))
    if limit <= 2:
        best_pd, best = 0.0, ()
        for r in range(1, limit + 1):
            for subset in itertools.combinations(candidates, r):
                value = _pd(cfg, base, tables, q, plan, subset)
                if value > best_pd:
                    best_pd, best = value, tuple(subset)
        return float(best_pd), best

    chosen: list[Link] = []
    current = 0.0
    while len(chosen) < limit:
        best_step: tuple[float, Link] | None = None
        for link in candidates:
            if link in chosen:
                continue
            value = _pd(cfg, base, tables, q, plan, chosen + [link])
            if best_step is None or value > best_step[0]:
                best_step = (value, link)
        if best_step is None or best_step[0] <= current + 1e-12:
            break
        current = best_step[0]
        chosen.append(best_step[1])
    return float(current), tuple(chosen)


def fusion_headroom_diagnostic(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    fixed_plan: ReportingPlan,
    *,
    k_safe: int = 6,
    shortlist_per_type: int = 4,
) -> list[FusionHeadroom]:
    """Compute a tractable upper-resource diagnostic for every target.

    The K=2 problems are enumerated exactly over a declared local/remote
    shortlist.  The K-safe problems use exact-detector marginal greedy growth;
    they are therefore diagnostics, not claims of a global combinatorial
    optimum.  All global receiver/report/fusion load constraints are relaxed.
    """
    if fixed_plan.mode != "explicit" or fixed_plan.f_q is None:
        raise ValueError("fusion headroom requires an explicit fixed plan")
    if cfg.detect.soft_stat_model.lower() != "llr" or cfg.detect.comm_error_model != "erasure":
        raise ValueError("fusion headroom requires LLR statistics and true erasures")
    if cfg.corr.enable:
        raise ValueError("fusion headroom exact-LLR diagnostic requires independent observations")

    perfect = replace(tables, chi_comm=np.ones_like(tables.chi_comm, dtype=float))
    rows: list[FusionHeadroom] = []
    for q in range(cfg.scale.Q):
        best_local = (0.0, -1, ())
        best_k2 = (0.0, -1, ())
        best_ksafe = (0.0, -1, ())
        best_perfect = (0.0, -1, ())
        for fusion in range(cfg.scale.M):
            plan, candidates = _shortlist(
                cfg, base, tables, q, fusion, shortlist_per_type
            )
            local_candidates = [
                link for link in candidates if is_local_observation(plan, link, q)
            ]
            for link in local_candidates:
                value = _pd(cfg, base, tables, q, plan, [link])
                if value > best_local[0]:
                    best_local = (value, fusion, (link,))
            k2_pd, k2_links = _best_bundle(
                cfg, base, tables, q, fusion, 2, shortlist_per_type
            )
            if k2_pd > best_k2[0]:
                best_k2 = (k2_pd, fusion, k2_links)
            kinf_pd, kinf_links = _best_bundle(
                cfg, base, tables, q, fusion, k_safe, shortlist_per_type
            )
            if kinf_pd > best_ksafe[0]:
                best_ksafe = (kinf_pd, fusion, kinf_links)
            perfect_pd, perfect_links = _best_bundle(
                cfg, base, perfect, q, fusion, k_safe, shortlist_per_type
            )
            if perfect_pd > best_perfect[0]:
                best_perfect = (perfect_pd, fusion, perfect_links)

        fixed_fusion = int(fixed_plan.f_q[q])
        fixed_pd, _ = _best_bundle(
            cfg, base, tables, q, fixed_fusion, 2, shortlist_per_type
        )
        rows.append(FusionHeadroom(
            target=q,
            pd_local=float(best_local[0]),
            pd_best_k2=float(best_k2[0]),
            pd_best_ksafe=float(best_ksafe[0]),
            pd_perfect_comm_ksafe=float(best_perfect[0]),
            pd_fixed_k2=float(fixed_pd),
            best_fusion_k2=int(best_k2[1]),
            best_fusion_ksafe=int(best_ksafe[1]),
            fixed_fusion=fixed_fusion,
            links_local=tuple(best_local[2]),
            links_k2=tuple(best_k2[2]),
            links_ksafe=tuple(best_ksafe[2]),
        ))
    return rows
