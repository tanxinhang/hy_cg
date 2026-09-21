"""Exact fixed-observation reporting minimization for the V1 model.

This is a planning step, before packets are sent. It reuses only the selected
observations' fine sensing estimates. It adds no waveform refinement or physical
mechanism. The guarantee concerns predicted PD, not realized detection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isac_sim.core.config import Config, Link
from isac_sim.detection.fusion import predicted_pd_for_links
from isac_sim.sensing.model import BaseGains, LinkTables
from isac_sim.cooperation.reporting import ReportingPlan, ReportingView


@dataclass
class FusionPolishResult:
    plan: ReportingPlan
    predicted_before: np.ndarray
    predicted_after: np.ndarray
    reports_before: np.ndarray
    reports_after: np.ndarray
    report_lower_bound: np.ndarray
    candidate_evaluations: int


@dataclass
class FusionPdPolishResult:
    """Certificate for fixed-observation fusion-destination PD polishing."""

    plan: ReportingPlan
    predicted_before: np.ndarray
    predicted_after: np.ndarray
    reports_before: np.ndarray
    reports_after: np.ndarray
    candidate_evaluations: int


def minimize_fixed_set_reports(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    selected: dict[int, list[Link]],
    plan: ReportingPlan,
) -> FusionPolishResult:
    """Minimize reports subject to each target's initial predicted PD floor.

    Enumerate destinations in decreasing selected-receiver multiplicity. Stop
    at the first feasible destination satisfying the PD floor. Destinations
    with strictly more local observations than the incumbent are the only possible
    improvements. Ties retain the incumbent, avoiding gratuitous relocation.

    Caller supplies scheduler tables, never truth tables. Coupled fusion
    capacities and endogenous interference are rejected: separability would
    otherwise fail. A system-wide remote-report cap is preserved because each
    target's report count can only decrease.
    """
    if not plan.is_explicit():
        raise ValueError("fixed-set fusion polish requires an explicit plan")
    if cfg.comm.mac_model != "serial" or cfg.comm.interference_model != "orthogonal":
        raise ValueError("fixed-set fusion polish requires orthogonal serial reporting")
    if cfg.corr.enable or cfg.interference.sense_gate_by_active_tx:
        raise ValueError("fixed-set fusion polish requires independent fixed sensing")
    if (cfg.fusion.max_targets_per_uav >= 0
            or cfg.selector.max_observations_per_fusion_uav >= 0
            or cfg.fusion.cpu_rate_cycles_per_s >= 0):
        raise ValueError("coupled fusion capacities invalidate target separability")
    if cfg.selector.require_local_anchor:
        raise ValueError("local-anchor variants are outside the V1 polish contract")
    Q, M = cfg.scale.Q, cfg.scale.M
    original = np.asarray(plan.f_q)
    if original.shape != (Q,):
        raise ValueError("fusion assignment must have one entry per target")
    if set(selected) - set(range(Q)):
        raise ValueError("selected observations contain an unknown target")
    updated = ReportingPlan(mode="explicit", f_q=original.copy())
    before = np.zeros(Q)
    after = np.zeros(Q)
    reports0 = np.zeros(Q, dtype=int)
    reports1 = np.zeros(Q, dtype=int)
    lower = np.zeros(Q, dtype=int)
    evaluations = 0
    for q in range(Q):
        links = selected.get(q, [])
        if not links:
            continue
        f0 = int(original[q])
        if not 0 <= f0 < M or len(set(links)) != len(links):
            raise ValueError("invalid initial assignment or duplicate observations")
        for i, j in links:
            if not (0 <= i < M and 0 <= j < M and i != j):
                raise ValueError("invalid bistatic observation")
            if cfg.dd.use_otfs_bin_validity and not base.valid_dd[i, j, q]:
                raise ValueError("initial observation is not DD-valid")
        counts = np.bincount([j for _, j in links], minlength=M)
        reports0[q] = len(links) - counts[f0]
        reports1[q] = reports0[q]
        lower[q] = len(links) - int(counts.max())
        local_cap = cfg.selector.max_local_observations_per_target
        if local_cap >= 0 and counts[f0] > local_cap:
            raise ValueError("initial set violates the local observation cap")
        initial_view = ReportingView(cfg, base, tables, updated)
        if not all(initial_view.feasible(link, q) for link in links):
            raise ValueError("initial set has an infeasible report")
        before[q] = predicted_pd_for_links(cfg, tables, q, links, plan=updated, base=base)
        if not np.isfinite(before[q]):
            raise ValueError("initial detector prediction is nonfinite")
        after[q] = before[q]
        # Report-count ordering provides an exact stopping certificate; no
        # monotonicity in channel reliability or submodularity is assumed.
        for f in sorted(range(M), key=lambda m: (-counts[m], m)):
            if counts[f] <= counts[f0]:
                break
            if local_cap >= 0 and counts[f] > local_cap:
                continue
            updated.f_q[q] = f
            view = ReportingView(cfg, base, tables, updated)
            if not all(view.feasible(link, q) for link in links):
                continue
            pd = predicted_pd_for_links(cfg, tables, q, links, plan=updated, base=base)
            evaluations += 1
            if np.isfinite(pd) and pd >= before[q]:
                after[q] = pd
                reports1[q] = len(links) - counts[f]
                break
        else:
            updated.f_q[q] = f0
        # Rejected trials must not leak their destination into the next target.
        if reports1[q] == reports0[q]:
            updated.f_q[q] = f0
    return FusionPolishResult(updated, before, after, reports0, reports1, lower, evaluations)


def maximize_fixed_set_pd(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    selected: dict[int, list[Link]],
    plan: ReportingPlan,
) -> FusionPdPolishResult:
    """Maximize each target's predicted PD by moving only its fusion node.

    The sensing observations, their illuminators, and the already-built link
    tables remain fixed.  Consequently this changes neither the TP-UIC state
    nor the coordination mask.  The search is target-separable only when all
    cross-target fusion/processing/report capacities are disabled; coupled
    configurations are rejected instead of being approximately scored.

    Candidate destinations must carry every selected receiver's report and
    obey the per-target local-observation constraint.  Predicted PD is the
    primary objective, fewer remote reports break numerical ties, and the
    incumbent destination wins a remaining tie.  Thus every returned target
    has a directly checkable non-decreasing scheduler-side PD certificate.
    """
    if not plan.is_explicit():
        raise ValueError("fixed-set fusion PD polish requires an explicit plan")
    if (
        cfg.fusion.max_targets_per_uav >= 0
        or cfg.selector.max_remote_reports >= 0
        or cfg.selector.max_observations_per_fusion_uav >= 0
        or cfg.fusion.cpu_rate_cycles_per_s >= 0
    ):
        raise ValueError("coupled fusion capacities invalidate target separability")
    Q, M = int(cfg.scale.Q), int(cfg.scale.M)
    original = np.asarray(plan.f_q, dtype=int)
    if original.shape != (Q,):
        raise ValueError("fusion assignment must have one entry per target")
    if set(selected) - set(range(Q)):
        raise ValueError("selected observations contain an unknown target")

    updated = ReportingPlan(mode="explicit", f_q=original.copy())
    before = np.zeros(Q, dtype=float)
    after = np.zeros(Q, dtype=float)
    reports0 = np.zeros(Q, dtype=int)
    reports1 = np.zeros(Q, dtype=int)
    evaluations = 0
    tol = 1e-12
    local_cap = int(cfg.selector.max_local_observations_per_target)

    for q in range(Q):
        links = list(selected.get(q, []))
        f0 = int(original[q])
        if not 0 <= f0 < M or len(set(links)) != len(links):
            raise ValueError("invalid initial assignment or duplicate observations")
        counts = np.bincount([j for _, j in links], minlength=M)
        reports0[q] = len(links) - int(counts[f0])
        reports1[q] = reports0[q]
        incumbent = ReportingPlan(mode="explicit", f_q=updated.f_q.copy())
        view0 = ReportingView(cfg, base, tables, incumbent)
        if not all(view0.feasible(link, q) for link in links):
            raise ValueError("initial set has an infeasible report")
        before[q] = predicted_pd_for_links(
            cfg, tables, q, links, plan=incumbent, base=base
        ) if links else 0.0
        if not np.isfinite(before[q]):
            raise ValueError("initial detector prediction is nonfinite")
        after[q] = before[q]
        best_f, best_pd, best_reports = f0, float(before[q]), int(reports0[q])

        for f in range(M):
            if f == f0:
                continue
            if local_cap >= 0 and counts[f] > local_cap:
                continue
            if cfg.selector.require_local_anchor and counts[f] == 0:
                continue
            candidate = ReportingPlan(mode="explicit", f_q=updated.f_q.copy())
            candidate.f_q[q] = f
            view = ReportingView(cfg, base, tables, candidate)
            if not all(view.feasible(link, q) for link in links):
                continue
            pd = predicted_pd_for_links(
                cfg, tables, q, links, plan=candidate, base=base
            ) if links else 0.0
            evaluations += 1
            if not np.isfinite(pd):
                continue
            reports = len(links) - int(counts[f])
            if pd > best_pd + tol or (
                abs(pd - best_pd) <= tol and reports < best_reports
            ):
                best_f, best_pd, best_reports = int(f), float(pd), int(reports)

        updated.f_q[q] = best_f
        after[q] = best_pd
        reports1[q] = best_reports

    if np.any(after + tol < before):
        raise RuntimeError("fusion PD polish violated its monotonicity certificate")
    return FusionPdPolishResult(
        updated, before, after, reports0, reports1, evaluations
    )
