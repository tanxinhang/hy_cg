"""Restricted target--fusion--observation-bundle master problem.

This module closes the main sequential-decision gap in the V1.1 scheduler:
fusion assignment, observation selection, and shared hard-resource allocation
are represented by one binary bundle choice per target.  The implementation is
intended first for small-system correctness/oracle studies; a later column-
generation layer can use the same :class:`ObservationBundle` interface.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .config import Config, Link
from .fusion import predicted_pd_for_links
from .model import BaseGains, LinkTables, rescale_sensing_tables_for_rcs
from .reporting import ReportingPlan, is_local_observation
from .selection import feasible_links_for_target


@dataclass(frozen=True)
class ObservationBundle:
    """One feasible observation-set column for a target/fusion pair."""

    target: int
    fusion: int
    links: Tuple[Link, ...]
    predicted_pd: float
    remote_reports: int
    receiver_load: Tuple[int, ...]

    @property
    def processing_load(self) -> int:
        return len(self.links)


@dataclass(frozen=True)
class BundleMasterResult:
    """Selected columns and the achieved true lexicographic objective."""

    plan: ReportingPlan
    selected: Dict[int, List[Link]]
    bundles: Tuple[ObservationBundle, ...]
    objective: Dict[str, float]
    column_count: int = 0
    pricing_iterations: int = 0
    lp_worst_deficit_bound: float = float("nan")


@dataclass(frozen=True)
class _PricingDuals:
    assignment: Tuple[float, ...]
    deficit: Tuple[float, ...]
    target_capacity: Tuple[float, ...]
    receiver: Tuple[float, ...]
    fusion: Tuple[float, ...]
    cpu: Tuple[float, ...]
    report: float
    total: float
    worst_deficit_bound: float


def bundle_cpu_cycles(cfg: Config, bundle: ObservationBundle) -> float:
    """Processing work ``c0 + c1|b| + c2|b|^3`` for a non-empty bundle."""
    size = bundle.processing_load
    if size == 0:
        return 0.0
    return float(
        cfg.fusion.cpu_fixed_cycles
        + cfg.fusion.cpu_per_observation_cycles * size
        + cfg.fusion.cpu_cubic_cycles * size ** 3
    )


def fusion_cpu_budget(cfg: Config) -> float | None:
    """Return ``F_f T_proc`` or ``None`` when CPU budgeting is disabled."""
    rate = float(cfg.fusion.cpu_rate_cycles_per_s)
    if rate < 0.0:
        return None
    return rate * float(cfg.fusion.processing_window_s)


def generate_restricted_bundles(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    *,
    shortlist_per_type: int | None = None,
) -> List[ObservationBundle]:
    """Generate nondominated bundle columns for every ``(q,f)`` pair.

    Local and remote candidates are shortlisted separately so a globally high
    singleton score cannot erase the local-evidence family.  The empty column
    is retained for every pair: assigning a target and spending sensing
    resources are distinct decisions in the formal model.
    """
    Q, M = cfg.scale.Q, cfg.scale.M
    K = max(int(cfg.selector.max_links_per_target), 0)
    shortlist = (
        max(int(cfg.selector.candidate_topk_per_target), K, 1)
        if shortlist_per_type is None else max(int(shortlist_per_type), 1)
    )
    columns: List[ObservationBundle] = []
    total_combinations = 0

    for q in range(Q):
        for f in range(M):
            plan = ReportingPlan(mode="explicit", f_q=np.full(Q, f, dtype=int))
            candidates = feasible_links_for_target(cfg, base, tables, q, plan)
            scored = [
                (
                    predicted_pd_for_links(
                        cfg, tables, q, [link], weight_mode="deflection",
                        plan=plan, base=base,
                    ),
                    link,
                )
                for link in candidates
            ]
            local = sorted(
                (item for item in scored if is_local_observation(plan, item[1], q)),
                reverse=True,
            )[:shortlist]
            remote = sorted(
                (item for item in scored if not is_local_observation(plan, item[1], q)),
                reverse=True,
            )[:shortlist]
            retained = list(dict.fromkeys(link for _, link in local + remote))
            total_combinations += sum(
                _comb(len(retained), r) for r in range(min(K, len(retained)) + 1)
            )
            if total_combinations > 2_000_000:
                raise RuntimeError(
                    "restricted bundle pool is too large; reduce the shortlist "
                    "or use column generation"
                )

            # Within an identical resource vector, only the highest-PD column
            # can be useful to the master.
            best_by_resource: dict[tuple[int, Tuple[int, ...], int], ObservationBundle] = {}
            for r in range(min(K, len(retained)) + 1):
                for subset in itertools.combinations(retained, r):
                    receiver = np.zeros(M, dtype=int)
                    for _, j in subset:
                        receiver[j] += 1
                    local_count = sum(
                        is_local_observation(plan, link, q) for link in subset
                    )
                    remote_count = r - local_count
                    if not _bundle_within_caps(
                        cfg, r, local_count, remote_count, receiver
                    ):
                        continue
                    pd = (
                        predicted_pd_for_links(
                            cfg, tables, q, list(subset),
                            weight_mode="deflection", plan=plan, base=base,
                        ) if subset else 0.0
                    )
                    bundle = ObservationBundle(
                        target=q,
                        fusion=f,
                        links=tuple(subset),
                        predicted_pd=float(pd),
                        remote_reports=int(remote_count),
                        receiver_load=tuple(int(x) for x in receiver),
                    )
                    key = (bundle.remote_reports, bundle.receiver_load, r)
                    incumbent = best_by_resource.get(key)
                    if incumbent is None or bundle.predicted_pd > incumbent.predicted_pd:
                        best_by_resource[key] = bundle
            columns.extend(best_by_resource.values())

    return columns


def solve_restricted_bundle_master(
    cfg: Config,
    bundles: Sequence[ObservationBundle],
    *,
    lex_tolerance: float = 1e-5,
) -> BundleMasterResult:
    """Solve the four-stage lexicographic restricted master exactly."""
    Q, M = cfg.scale.Q, cfg.scale.M
    columns = list(bundles)
    for q in range(Q):
        if not any(bundle.target == q for bundle in columns):
            raise ValueError(f"bundle pool has no column for target {q}")

    B = len(columns)
    n = B + Q + 1
    deficit_slice = slice(B, B + Q)
    dmax_index = B + Q
    integrality = np.zeros(n, dtype=int)
    integrality[:B] = 1
    lower = np.zeros(n, dtype=float)
    upper = np.ones(n, dtype=float)
    upper[deficit_slice] = max(float(cfg.detect.pd_required), 1.0)
    upper[dmax_index] = max(float(cfg.detect.pd_required), 1.0)

    rows: List[np.ndarray] = []
    lbs: List[float] = []
    ubs: List[float] = []

    def add_row(coeff: np.ndarray, lb: float = -np.inf, ub: float = np.inf) -> None:
        rows.append(coeff)
        lbs.append(lb)
        ubs.append(ub)

    for q in range(Q):
        row = np.zeros(n)
        for b, bundle in enumerate(columns):
            row[b] = float(bundle.target == q)
        add_row(row, 1.0, 1.0)

        row = np.zeros(n)
        for b, bundle in enumerate(columns):
            if bundle.target == q:
                row[b] = -bundle.predicted_pd
        row[B + q] = -1.0
        add_row(row, ub=-float(cfg.detect.pd_required))

        row = np.zeros(n)
        row[B + q] = 1.0
        row[dmax_index] = -1.0
        add_row(row, ub=0.0)

    target_cap = int(cfg.fusion.max_targets_per_uav)
    if target_cap >= 0:
        for f in range(M):
            row = np.zeros(n)
            for b, bundle in enumerate(columns):
                row[b] = float(bundle.fusion == f)
            add_row(row, ub=float(target_cap))

    rx_cap = int(cfg.selector.max_observations_per_receiver)
    if rx_cap >= 0:
        for j in range(M):
            row = np.zeros(n)
            for b, bundle in enumerate(columns):
                row[b] = bundle.receiver_load[j]
            add_row(row, ub=float(rx_cap))

    fusion_cap = int(cfg.selector.max_observations_per_fusion_uav)
    if fusion_cap >= 0:
        for f in range(M):
            row = np.zeros(n)
            for b, bundle in enumerate(columns):
                if bundle.fusion == f:
                    row[b] = bundle.processing_load
            add_row(row, ub=float(fusion_cap))

    cpu_budget = fusion_cpu_budget(cfg)
    if cpu_budget is not None:
        for f in range(M):
            row = np.zeros(n)
            for b, bundle in enumerate(columns):
                if bundle.fusion == f:
                    row[b] = bundle_cpu_cycles(cfg, bundle)
            add_row(row, ub=cpu_budget)

    report_cap = int(cfg.selector.max_remote_reports)
    if report_cap >= 0:
        row = np.zeros(n)
        for b, bundle in enumerate(columns):
            row[b] = bundle.remote_reports
        add_row(row, ub=float(report_cap))

    total_cap = int(cfg.selector.max_total_links)
    if total_cap >= 0:
        row = np.zeros(n)
        for b, bundle in enumerate(columns):
            row[b] = bundle.processing_load
        add_row(row, ub=float(total_cap))

    stage_objectives = []
    c = np.zeros(n); c[dmax_index] = 1.0; stage_objectives.append(c)
    c = np.zeros(n); c[:B] = [
        max(float(cfg.detect.pd_required) - b.predicted_pd, 0.0)
        for b in columns
    ]; stage_objectives.append(c)
    c = np.zeros(n); c[:B] = [b.remote_reports for b in columns]; stage_objectives.append(c)
    c = np.zeros(n); c[:B] = [bundle_cpu_cycles(cfg, b) for b in columns]; stage_objectives.append(c)

    solution = None
    for stage_index, objective in enumerate(stage_objectives, start=1):
        constraint = LinearConstraint(np.vstack(rows), np.asarray(lbs), np.asarray(ubs))
        result = milp(
            objective,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=constraint,
            options={"presolve": False, "mip_rel_gap": 0.0},
        )
        if not result.success or result.x is None:
            raise ValueError(
                f"restricted bundle master stage {stage_index} is infeasible: "
                f"{result.message}"
            )
        solution = result.x
        optimum = _integer_incumbent_stage_value(
            cfg, columns, result.x, stage_index
        )
        lock_tolerance = lex_tolerance * max(1.0, abs(optimum))
        if stage_index == 1:
            # Express the locked bottleneck tier directly on the selected
            # bundle PDs.  This is equivalent to d_max <= optimum but avoids
            # carrying a near-degenerate auxiliary continuous bound into the
            # next MIP, which HiGHS can incorrectly classify as infeasible.
            minimum_pd = float(cfg.detect.pd_required) - optimum - lock_tolerance
            for q in range(Q):
                row = np.zeros(n)
                for b, bundle in enumerate(columns):
                    if bundle.target == q:
                        row[b] = -bundle.predicted_pd
                add_row(row, ub=-minimum_pd)
        else:
            add_row(objective.copy(), ub=optimum + lock_tolerance)

    assert solution is not None
    chosen = tuple(columns[b] for b in range(B) if solution[b] > 0.5)
    if len(chosen) != Q:
        raise RuntimeError("bundle master returned a non-integral target assignment")
    chosen_by_q = {bundle.target: bundle for bundle in chosen}
    f_q = np.asarray([chosen_by_q[q].fusion for q in range(Q)], dtype=int)
    selected = {q: list(chosen_by_q[q].links) for q in range(Q)}
    deficits = np.maximum(
        float(cfg.detect.pd_required)
        - np.asarray([chosen_by_q[q].predicted_pd for q in range(Q)]),
        0.0,
    )
    objective = {
        "worst_detection_deficit": float(np.max(deficits)),
        "total_detection_deficit": float(np.sum(deficits)),
        "remote_reports": float(sum(b.remote_reports for b in chosen)),
        "processing_load": float(sum(b.processing_load for b in chosen)),
        "cpu_cycles": float(sum(bundle_cpu_cycles(cfg, b) for b in chosen)),
    }
    return BundleMasterResult(
        plan=ReportingPlan(mode="explicit", f_q=f_q),
        selected=selected,
        bundles=chosen,
        objective=objective,
    )


def _integer_incumbent_stage_value(
    cfg: Config,
    columns: Sequence[ObservationBundle],
    solution: np.ndarray,
    stage_index: int,
) -> float:
    """Re-evaluate a lexicographic tier on the rounded bundle incumbent.

    HiGHS may return binary values within its integrality tolerance.  Using the
    raw continuous objective to lock the next tier can then exclude the actual
    rounded incumbent.  Resource objectives are evaluated on one maximum-value
    column per target; reliability deficits are recomputed from bundle PDs.
    """
    chosen: List[ObservationBundle] = []
    for q in range(cfg.scale.Q):
        indices = [b for b, bundle in enumerate(columns) if bundle.target == q]
        if not indices:
            raise RuntimeError(f"master incumbent has no target-{q} columns")
        best_index = max(indices, key=lambda b: solution[b])
        chosen.append(columns[best_index])
    deficits = np.maximum(
        float(cfg.detect.pd_required)
        - np.asarray([bundle.predicted_pd for bundle in chosen]),
        0.0,
    )
    if stage_index == 1:
        return float(np.max(deficits))
    if stage_index == 2:
        return float(np.sum(deficits))
    if stage_index == 3:
        return float(sum(bundle.remote_reports for bundle in chosen))
    if stage_index == 4:
        return float(sum(bundle_cpu_cycles(cfg, bundle) for bundle in chosen))
    raise ValueError(stage_index)


def joint_bundle_restricted_master(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    *,
    shortlist_per_type: int | None = None,
) -> BundleMasterResult:
    """Generate the initial column pool and solve the restricted master."""
    return solve_restricted_bundle_master(
        cfg,
        generate_restricted_bundles(
            cfg, base, tables, shortlist_per_type=shortlist_per_type
        ),
    )


def joint_bundle_column_generation(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    *,
    value_tables: LinkTables | None = None,
    allowed_fusions: Dict[int, Sequence[int]] | None = None,
    allowed_transmitters: Sequence[int] | None = None,
) -> BundleMasterResult:
    """Dual-guided C2F pricing followed by an exact integer restricted master.

    The LP pricing loop optimizes the first two reliability tiers.  Small
    target--fusion pricing problems are enumerated exactly; larger ones use a
    detector-marginal greedy oracle over separately shortlisted local/remote
    observations.  Once no improving fairness column is found (or the declared
    iteration cap is reached), the accumulated pool is solved as the exact
    four-stage integer restricted master.
    """
    bundle_tables = value_tables if value_tables is not None else tables
    pair_candidates = _shortlisted_pair_candidates(
        cfg, base, tables, allowed_fusions=allowed_fusions,
        allowed_transmitters=allowed_transmitters,
    )
    pool = _seed_bundle_pool(cfg, base, bundle_tables, pair_candidates)
    seen = {_bundle_identity(bundle) for bundle in pool}
    iterations = 0
    lp_bound = float("nan")

    # Column generation is itself lexicographic: first close the worst-deficit
    # LP, then lock that tier and close the total-deficit LP.  Pricing only the
    # second tier from an incomplete first tier would not certify fairness.
    for stage in ("worst", "total"):
        for _ in range(int(cfg.selector.bundle_cg_max_iterations)):
            duals = _fairness_lp_duals(cfg, pool, stage=stage)
            lp_bound = duals.worst_deficit_bound
            additions: List[ObservationBundle] = []
            for q in range(cfg.scale.Q):
                for f in _fusion_choices(cfg, q, allowed_fusions):
                    candidate = _price_pair_bundle(
                        cfg, base, bundle_tables, q, f,
                        pair_candidates[(q, f)], duals
                    )
                    if candidate is None:
                        continue
                    identity = _bundle_identity(candidate)
                    if identity not in seen:
                        seen.add(identity)
                        additions.append(candidate)
            iterations += 1
            if not additions:
                break
            pool.extend(additions)

    # Correlation-aware bundle values can cluster at machine-level distances
    # near saturation.  A 1e-4 reliability lock prevents HiGHS from declaring
    # the next integer lexicographic tier infeasible even though the preceding
    # incumbent is feasible; independent/released paths retain 1e-5.
    lex_tolerance = 1e-4 if cfg.corr.enable else 1e-5
    result = solve_restricted_bundle_master(
        cfg, pool, lex_tolerance=lex_tolerance
    )
    return BundleMasterResult(
        plan=result.plan,
        selected=result.selected,
        bundles=result.bundles,
        objective=result.objective,
        column_count=len(pool),
        pricing_iterations=iterations,
        lp_worst_deficit_bound=lp_bound,
    )


def rcs_robust_bundle_column_generation(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    *,
    value_tables: LinkTables | None = None,
    allowed_fusions: Dict[int, Sequence[int]] | None = None,
    allowed_transmitters: Sequence[int] | None = None,
) -> BundleMasterResult:
    """Optimize the bundle master at the RCS uncertainty-set lower endpoint.

    Under the implemented bistatic model, sensing SINR is linear in RCS and
    the fixed-bundle detector is monotone in sensing SINR. Hence the worst
    case over ``[lower_factor * mean_RCS, mean_RCS]`` occurs at the lower
    endpoint. This wrapper is therefore the exact interval-robust counterpart
    of :func:`joint_bundle_column_generation`, not an RCS-weighted heuristic.
    """
    factor = float(cfg.prior.rcs_lower_factor)
    robust_tables = rescale_sensing_tables_for_rcs(cfg, tables, factor)
    robust_value_tables = rescale_sensing_tables_for_rcs(
        cfg, value_tables if value_tables is not None else tables, factor
    )
    return joint_bundle_column_generation(
        cfg,
        base,
        robust_tables,
        value_tables=robust_value_tables,
        allowed_fusions=allowed_fusions,
        allowed_transmitters=allowed_transmitters,
    )


def _shortlisted_pair_candidates(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    *,
    allowed_fusions: Dict[int, Sequence[int]] | None = None,
    allowed_transmitters: Sequence[int] | None = None,
) -> Dict[tuple[int, int], List[Link]]:
    Q, M = cfg.scale.Q, cfg.scale.M
    limit = int(cfg.selector.bundle_shortlist_per_type)
    result: Dict[tuple[int, int], List[Link]] = {}
    allowed_tx = (
        None if allowed_transmitters is None
        else {int(node) for node in allowed_transmitters}
    )
    for q in range(Q):
        for f in _fusion_choices(cfg, q, allowed_fusions):
            plan = ReportingPlan(mode="explicit", f_q=np.full(Q, f, dtype=int))
            scored = [
                (
                    predicted_pd_for_links(
                        cfg, tables, q, [link], weight_mode="deflection",
                        plan=plan, base=base,
                    ),
                    link,
                )
                for link in feasible_links_for_target(cfg, base, tables, q, plan)
                if allowed_tx is None or int(link[0]) in allowed_tx
            ]
            local = sorted(
                (item for item in scored if is_local_observation(plan, item[1], q)),
                reverse=True,
            )[:limit]
            remote = sorted(
                (item for item in scored if not is_local_observation(plan, item[1], q)),
                reverse=True,
            )[:limit]
            result[(q, f)] = list(dict.fromkeys(
                link for _, link in local + remote
            ))
    return result


def _seed_bundle_pool(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    pair_candidates: Dict[tuple[int, int], List[Link]],
) -> List[ObservationBundle]:
    pool: List[ObservationBundle] = []
    for q in range(cfg.scale.Q):
        for f in sorted(f for qq, f in pair_candidates if qq == q):
            candidates = pair_candidates[(q, f)]
            empty = _make_bundle(cfg, base, tables, q, f, ())
            pool.append(empty)
            # Greedy PD prefixes provide resource-efficient alternatives for
            # the later report/processing tiers, rather than only a full set.
            chosen: List[Link] = []
            while len(chosen) < max(int(cfg.selector.max_links_per_target), 0):
                best: tuple[float, Link, ObservationBundle] | None = None
                current_pd = (
                    _make_bundle(cfg, base, tables, q, f, chosen).predicted_pd
                    if chosen else 0.0
                )
                for link in candidates:
                    if link in chosen:
                        continue
                    bundle = _make_bundle(cfg, base, tables, q, f, chosen + [link])
                    if bundle is None:
                        continue
                    marginal = bundle.predicted_pd - current_pd
                    if best is None or marginal > best[0]:
                        best = (marginal, link, bundle)
                if best is None or best[0] <= 0.0:
                    break
                chosen.append(best[1])
                pool.append(best[2])
            # Ensure both evidence families enter the initial RMP even when
            # one loses the unpriced greedy race.
            plan = ReportingPlan(
                mode="explicit", f_q=np.full(cfg.scale.Q, f, dtype=int)
            )
            for local_flag in (True, False):
                family = [
                    link for link in candidates
                    if is_local_observation(plan, link, q) == local_flag
                ]
                if family:
                    bundle = _make_bundle(cfg, base, tables, q, f, [family[0]])
                    if bundle is not None:
                        pool.append(bundle)
    return _deduplicate_bundles(pool)


def _fairness_lp_duals(
    cfg: Config,
    bundles: Sequence[ObservationBundle],
    *,
    stage: str,
) -> _PricingDuals:
    """Solve relaxed worst/sum-deficit tiers and expose scarcity duals."""
    columns = list(bundles)
    Q, M, B = cfg.scale.Q, cfg.scale.M, len(columns)
    n = B + Q + 1
    dmax = B + Q
    A_eq = np.zeros((Q, n), dtype=float)
    for q in range(Q):
        for b, bundle in enumerate(columns):
            A_eq[q, b] = float(bundle.target == q)
    b_eq = np.ones(Q, dtype=float)

    rows: List[np.ndarray] = []
    rhs: List[float] = []
    labels: List[tuple[str, int]] = []

    def add(coeff: np.ndarray, ub: float, label: tuple[str, int]) -> None:
        rows.append(coeff); rhs.append(ub); labels.append(label)

    for q in range(Q):
        row = np.zeros(n)
        for b, bundle in enumerate(columns):
            if bundle.target == q:
                row[b] = -bundle.predicted_pd
        row[B + q] = -1.0
        add(row, -float(cfg.detect.pd_required), ("deficit", q))
        row = np.zeros(n); row[B + q] = 1.0; row[dmax] = -1.0
        add(row, 0.0, ("dmax", q))

    target_cap = int(cfg.fusion.max_targets_per_uav)
    if target_cap >= 0:
        for f in range(M):
            row = np.zeros(n)
            for b, bundle in enumerate(columns): row[b] = float(bundle.fusion == f)
            add(row, float(target_cap), ("target", f))
    rx_cap = int(cfg.selector.max_observations_per_receiver)
    if rx_cap >= 0:
        for j in range(M):
            row = np.zeros(n)
            for b, bundle in enumerate(columns): row[b] = bundle.receiver_load[j]
            add(row, float(rx_cap), ("receiver", j))
    fusion_cap = int(cfg.selector.max_observations_per_fusion_uav)
    if fusion_cap >= 0:
        for f in range(M):
            row = np.zeros(n)
            for b, bundle in enumerate(columns):
                row[b] = bundle.processing_load if bundle.fusion == f else 0.0
            add(row, float(fusion_cap), ("fusion", f))
    cpu_budget = fusion_cpu_budget(cfg)
    if cpu_budget is not None:
        for f in range(M):
            row = np.zeros(n)
            for b, bundle in enumerate(columns):
                row[b] = bundle_cpu_cycles(cfg, bundle) if bundle.fusion == f else 0.0
            add(row, float(cpu_budget), ("cpu", f))
    report_cap = int(cfg.selector.max_remote_reports)
    if report_cap >= 0:
        row = np.zeros(n)
        for b, bundle in enumerate(columns): row[b] = bundle.remote_reports
        add(row, float(report_cap), ("report", 0))
    total_cap = int(cfg.selector.max_total_links)
    if total_cap >= 0:
        row = np.zeros(n)
        for b, bundle in enumerate(columns): row[b] = bundle.processing_load
        add(row, float(total_cap), ("total", 0))

    bounds = [(0.0, 1.0)] * B + [
        (0.0, max(float(cfg.detect.pd_required), 1.0))
    ] * (Q + 1)
    c1 = np.zeros(n); c1[dmax] = 1.0
    first = _solve_pricing_lp(
        c1, np.vstack(rows), np.asarray(rhs), A_eq, b_eq, bounds,
    )
    if not first.success:
        raise ValueError(f"bundle fairness LP is infeasible: {first.message}")
    worst = float(first.fun)
    if stage == "worst":
        active = first
    elif stage == "total":
        lock = np.zeros(n); lock[dmax] = 1.0
        rows.append(lock); rhs.append(worst + 1e-9); labels.append(("lock", 0))
        c2 = np.zeros(n); c2[B:B + Q] = 1.0
        active = _solve_pricing_lp(
            c2, np.vstack(rows), np.asarray(rhs), A_eq, b_eq, bounds,
        )
        if not active.success:
            raise ValueError(f"bundle total-deficit LP is infeasible: {active.message}")
    else:
        raise ValueError(f"unknown pricing stage {stage!r}")

    prices = {label: max(-float(marginal), 0.0) for label, marginal in zip(
        labels, active.ineqlin.marginals
    )}
    return _PricingDuals(
        assignment=tuple(float(x) for x in active.eqlin.marginals),
        deficit=tuple(prices.get(("deficit", q), 0.0) for q in range(Q)),
        target_capacity=tuple(prices.get(("target", f), 0.0) for f in range(M)),
        receiver=tuple(prices.get(("receiver", j), 0.0) for j in range(M)),
        fusion=tuple(prices.get(("fusion", f), 0.0) for f in range(M)),
        cpu=tuple(prices.get(("cpu", f), 0.0) for f in range(M)),
        report=prices.get(("report", 0), 0.0),
        total=prices.get(("total", 0), 0.0),
        worst_deficit_bound=worst,
    )


def _solve_pricing_lp(
    objective: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    bounds: Sequence[tuple[float, float]],
):
    """Solve a pricing LP with deterministic HiGHS algorithm fallbacks."""
    attempts = (
        ("highs", None),
        ("highs-ds", {"presolve": False}),
        ("highs-ipm", {"presolve": False}),
    )
    last = None
    for method, options in attempts:
        result = linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=method,
            options=options,
        )
        last = result
        if result.success:
            return result
    assert last is not None
    return last


def _price_pair_bundle(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    q: int,
    f: int,
    candidates: Sequence[Link],
    duals: _PricingDuals,
) -> ObservationBundle | None:
    max_links = min(max(int(cfg.selector.max_links_per_target), 0), len(candidates))
    exact_limit = int(cfg.selector.bundle_exact_pricing_max_candidates)
    best: tuple[float, ObservationBundle] | None = None

    def consider(links: Sequence[Link]) -> None:
        nonlocal best
        bundle = _make_bundle(cfg, base, tables, q, f, links)
        if bundle is None:
            return
        rc = _reduced_cost(bundle, duals, cfg)
        if best is None or rc < best[0]:
            best = (rc, bundle)

    if len(candidates) <= exact_limit:
        for r in range(max_links + 1):
            for subset in itertools.combinations(candidates, r):
                consider(subset)
    else:
        starts: List[List[Link]] = [[]]
        plan = ReportingPlan(
            mode="explicit", f_q=np.full(cfg.scale.Q, f, dtype=int)
        )
        local = [link for link in candidates if is_local_observation(plan, link, q)]
        remote = [link for link in candidates if not is_local_observation(plan, link, q)]
        starts.extend([[family[0]] for family in (local, remote) if family])
        for start in starts:
            chosen = list(start)
            current = _make_bundle(cfg, base, tables, q, f, chosen)
            if current is None:
                continue
            consider(chosen)
            while len(chosen) < max_links:
                next_best: tuple[float, Link, ObservationBundle] | None = None
                current_rc = _reduced_cost(current, duals, cfg)
                for link in candidates:
                    if link in chosen:
                        continue
                    bundle = _make_bundle(cfg, base, tables, q, f, chosen + [link])
                    if bundle is None:
                        continue
                    rc = _reduced_cost(bundle, duals, cfg)
                    if next_best is None or rc < next_best[0]:
                        next_best = (rc, link, bundle)
                if next_best is None or next_best[0] >= current_rc - 1e-12:
                    break
                chosen.append(next_best[1]); current = next_best[2]
                consider(chosen)

    tolerance = float(cfg.selector.bundle_pricing_tolerance)
    return best[1] if best is not None and best[0] < -tolerance else None


def _reduced_cost(bundle: ObservationBundle, duals: _PricingDuals, cfg: Config | None = None) -> float:
    q, f = bundle.target, bundle.fusion
    return float(
        -duals.assignment[q]
        - duals.deficit[q] * bundle.predicted_pd
        + duals.target_capacity[f]
        + sum(load * duals.receiver[j] for j, load in enumerate(bundle.receiver_load))
        + bundle.processing_load * duals.fusion[f]
        + (bundle_cpu_cycles(cfg, bundle) * duals.cpu[f] if cfg is not None else 0.0)
        + bundle.remote_reports * duals.report
        + bundle.processing_load * duals.total
    )


def _make_bundle(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    q: int,
    f: int,
    links: Sequence[Link],
) -> ObservationBundle | None:
    ordered = tuple(sorted(set(links)))
    receiver = np.zeros(cfg.scale.M, dtype=int)
    for _, j in ordered: receiver[j] += 1
    plan = ReportingPlan(
        mode="explicit", f_q=np.full(cfg.scale.Q, f, dtype=int)
    )
    local = sum(is_local_observation(plan, link, q) for link in ordered)
    remote = len(ordered) - local
    if not _bundle_within_caps(cfg, len(ordered), local, remote, receiver):
        return None
    pd = predicted_pd_for_links(
        cfg, tables, q, list(ordered), weight_mode="deflection",
        plan=plan, base=base,
    ) if ordered else 0.0
    return ObservationBundle(
        target=q, fusion=f, links=ordered, predicted_pd=float(pd),
        remote_reports=int(remote),
        receiver_load=tuple(int(x) for x in receiver),
    )


def _bundle_identity(bundle: ObservationBundle) -> tuple[int, int, Tuple[Link, ...]]:
    return bundle.target, bundle.fusion, bundle.links


def _deduplicate_bundles(
    bundles: Sequence[ObservationBundle],
) -> List[ObservationBundle]:
    unique: Dict[tuple[int, int, Tuple[Link, ...]], ObservationBundle] = {}
    for bundle in bundles:
        unique[_bundle_identity(bundle)] = bundle
    return list(unique.values())


def _fusion_choices(
    cfg: Config,
    q: int,
    allowed_fusions: Dict[int, Sequence[int]] | None,
) -> List[int]:
    if allowed_fusions is None:
        return list(range(cfg.scale.M))
    choices = sorted(set(int(f) for f in allowed_fusions.get(q, ())))
    if not choices or any(f < 0 or f >= cfg.scale.M for f in choices):
        raise ValueError(f"invalid allowed fusion set for target {q}: {choices}")
    return choices


def _bundle_within_caps(
    cfg: Config,
    processing: int,
    local: int,
    remote: int,
    receiver: np.ndarray,
) -> bool:
    local_cap = int(cfg.selector.max_local_observations_per_target)
    fusion_cap = int(cfg.selector.max_observations_per_fusion_uav)
    report_cap = int(cfg.selector.max_remote_reports)
    rx_cap = int(cfg.selector.max_observations_per_receiver)
    cpu_budget = fusion_cpu_budget(cfg)
    cpu_cycles = (
        0.0 if processing == 0 else
        cfg.fusion.cpu_fixed_cycles
        + cfg.fusion.cpu_per_observation_cycles * processing
        + cfg.fusion.cpu_cubic_cycles * processing ** 3
    )
    return not (
        (local_cap >= 0 and local > local_cap)
        or (fusion_cap >= 0 and processing > fusion_cap)
        or (report_cap >= 0 and remote > report_cap)
        or (rx_cap >= 0 and np.any(receiver > rx_cap))
        or (cpu_budget is not None and cpu_cycles > cpu_budget)
    )


def _comb(n: int, r: int) -> int:
    import math
    return math.comb(n, r) if 0 <= r <= n else 0
