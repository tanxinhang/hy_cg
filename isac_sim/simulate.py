"""Monte-Carlo execution and metric aggregation.

One trial = one independent geometry realisation + one detection evaluation per
method. All methods see the same geometry and the same detector-noise stream;
method-specific randomness is isolated to the selection stage. This common-
random-number design makes paired differences attributable to the selected set.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config, Link, MethodName, apply_overrides
from .fusion import (
    compute_weights,
    deflection_for_links,
    fused_h0_variance,
    fused_h0_skewness,
    fusion_weight_mode_for_method,
)
from .model import (
    BaseGains,
    LinkTables,
    EPS,
    binomial_ci95,
    build_base_gains,
    compute_link_tables,
    generate_geometry,
    threshold_from_pfa,
)
from .reporting import ReportingPlan, assign_fusion_nodes, report_dest
from .selection import (
    C2F_METHODS,
    METHOD_RNG_OFFSETS,
    METHODS,
    feasible_links_for_target,
    select_all_neighbor,
    select_c2f,
    select_c2f_adaptive,
    select_lagrangian,
    select_topk_baseline,
)


@dataclass
class MethodResult:
    """Everything one method produces on one trial."""

    name: str
    detected: int
    total_targets: int
    detected_per_target: np.ndarray
    false_alarm: int
    total_false: int
    false_alarm_overall: int
    total_false_overall: int
    overhead_bits: float
    overhead_delay_s: float
    selected_links: Dict[int, List[Link]]
    D_fuse_per_target: np.ndarray
    active_targets: int
    feasible_targets: int
    feasible_links: int
    selected_rate_mean_mbps: float
    selected_rate_min_mbps: float
    selected_rate_p10_mbps: float
    selected_chi_mean: float
    selected_chi_min: float
    selected_chi_p10: float
    selected_gamma_comm_mean_db: float
    selected_rate_satisfaction_ratio: float
    selected_chi_ge_min_ratio: float
    comm_feasible_edge_ratio: float
    fine_eval_full: float = 0.0
    fine_eval_c2f: float = 0.0
    # Belief mode only: fraction of selected links whose belief-guided DD
    # window actually captured the true target bin (1.0 outside belief mode).
    belief_capture_rate: float = 1.0
    # Trial-level assignment retained for packetization and conflict-graph
    # audits; it is intentionally excluded from scalar paper metrics.
    reporting_plan: object | None = None


# ==========================================================================
# Soft-statistic sampling
# ==========================================================================
def _is_llr(cfg: Config) -> bool:
    return cfg.detect.soft_stat_model.lower() == "llr"


def draw_h1_soft_stat(
    cfg: Config, tables: LinkTables, link: Link, q: int, rng: np.random.Generator,
    plan: "ReportingPlan | None" = None,
) -> float:
    """Draw one soft statistic under H1, including communication errors."""
    from .soft_channel import draw_received_soft_stat

    return draw_received_soft_stat(cfg, tables, link, q, rng, h1=True, plan=plan)


def draw_h0_soft_stat(
    cfg: Config, tables: LinkTables, link: Link, q: int, rng: np.random.Generator,
    plan: "ReportingPlan | None" = None,
) -> float:
    """Draw one soft statistic under H0."""
    from .soft_channel import draw_received_soft_stat

    return draw_received_soft_stat(cfg, tables, link, q, rng, h1=False, plan=plan)


# ==========================================================================
# Detection evaluation
# ==========================================================================
def evaluate_detection(
    cfg: Config,
    tables: LinkTables,
    selected: Dict[int, List[Link]],
    rng: np.random.Generator,
    method: str,
    plan: "ReportingPlan | None" = None,
    base: BaseGains | None = None,
) -> Tuple[int, int, int, int, int, int, np.ndarray]:
    """Evaluate detection and false alarms with two P_FA denominators."""
    d = cfg.detect
    detected = 0
    detected_per_target = np.zeros(cfg.scale.Q, dtype=int)
    false_alarm_active = 0
    total_targets = cfg.scale.Q
    total_false_active = 0
    total_false_overall = cfg.scale.Q * d.num_false_per_target

    base_thr = threshold_from_pfa(cfg)
    weight_mode = fusion_weight_mode_for_method(method)

    for q in range(cfg.scale.Q):
        links = selected.get(q, [])
        if not links:
            continue

        total_false_active += d.num_false_per_target
        weights = compute_weights(cfg, tables, q, links, mode=weight_mode, plan=plan, base=base)
        var0 = fused_h0_variance(cfg, tables, q, links, weights, plan=plan, base=base)
        skew0 = fused_h0_skewness(cfg, tables, q, links, weights, plan=plan, base=base)
        z_cf = base_thr + (skew0 / 6.0) * (base_thr * base_thr - 1.0)
        thr = z_cf * math.sqrt(max(var0, EPS))

        F = 0.0
        for link, w in weights.items():
            F += w * draw_h1_soft_stat(cfg, tables, link, q, rng, plan)
        if F > thr:
            detected += 1
            detected_per_target[q] = 1

        for _ in range(d.num_false_per_target):
            F0 = 0.0
            for link, w in weights.items():
                F0 += w * draw_h0_soft_stat(cfg, tables, link, q, rng, plan)
            if F0 > thr:
                false_alarm_active += 1

    false_alarm_overall = false_alarm_active
    return (
        detected,
        total_targets,
        false_alarm_active,
        total_false_active,
        false_alarm_overall,
        total_false_overall,
        detected_per_target,
    )


# ==========================================================================
# Overhead and communication metrics
# ==========================================================================
def total_overhead_bits(cfg: Config, selected: Dict[int, List[Link]]) -> float:
    from .model import packet_bits_for_target

    return float(sum(packet_bits_for_target(cfg, q) * len(links) for q, links in selected.items()))


def _report_list(
    cfg: Config, tables: LinkTables, selected: Dict[int, List[Link]], plan: "ReportingPlan | None"
) -> List[Tuple[int, int, float]]:
    """Flatten the selected links into ``(source, dest, latency_s)`` reports."""
    from .fbl import report_latency_s

    reports: List[Tuple[int, int, float]] = []
    for q, links in selected.items():
        for (i, j) in links:
            dest = report_dest(plan, (i, j), q)
            reports.append((j, dest, report_latency_s(cfg, float(tables.rate[j, dest]))))
    return reports


def total_overhead_delay_s(
    cfg: Config, tables: LinkTables, selected: Dict[int, List[Link]], plan: "ReportingPlan | None" = None
) -> float:
    """Aggregate reporting latency under the configured MAC.

    * ``serial``:    reports are time-multiplexed, ``T = sum_l T_l``.
    * ``parallel``:  all reports are concurrent, ``T = max_l T_l``.
    * ``slot``:      conflict-graph colouring (shared transmitter or receiver
                     conflicts); slots run sequentially, so
                     ``T = sum_r max_{l in S_r} T_l``.  The ``tables`` passed in
                     must already be the *slot-local* tables built by
                     :func:`build_slot_tables`, so the rate used here matches
                     the slot-local interference -- the two are one system.
    """
    from .fbl import report_latency_s
    from .reporting import slot_schedule

    reports = _report_list(cfg, tables, selected, plan)
    if not reports:
        return 0.0

    mac = cfg.comm.mac_model.lower()
    if mac == "serial":
        return float(sum(lat for _, _, lat in reports))
    if mac == "parallel":
        return float(max(lat for _, _, lat in reports))
    if mac == "slot":
        slots = slot_schedule(cfg, selected, plan)
        total = 0.0
        for slot in slots:
            slot_lat = max(
                report_latency_s(cfg, float(tables.rate[j, dest]))
                for (_, _, j, dest) in slot
            )
            total += slot_lat
        return float(total)
    raise ValueError(f"Unknown comm.mac_model={cfg.comm.mac_model!r}")


def active_target_count(selected: Dict[int, List[Link]]) -> int:
    return sum(1 for links in selected.values() if len(links) > 0)


def feasible_stats(
    cfg: Config, base: BaseGains, tables: LinkTables, plan: "ReportingPlan | None" = None
) -> Tuple[int, int]:
    counts = [len(feasible_links_for_target(cfg, base, tables, q, plan)) for q in range(cfg.scale.Q)]
    return sum(1 for c in counts if c > 0), int(np.sum(counts))


def communication_metrics_for_selection(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    selected: Dict[int, List[Link]],
    plan: "ReportingPlan | None" = None,
) -> Dict[str, float]:
    """Communication-side statistics of the selected soft-information links."""
    edge_count = int(np.sum(base.edge_mask))
    feasible_edge_count = int(np.sum(tables.feasible_comm & base.edge_mask))
    comm_feasible_edge_ratio = feasible_edge_count / max(edge_count, 1)

    # Per (reporting) leg statistics, resolved through the reporting plan.
    # Deduplicated on the reporting leg ``(source, dest)`` so a link selected
    # for several targets is counted once (the legacy code deduplicated on the
    # sensing pair ``(i, j)``, which for ``dest = i`` is the same set).
    rates: List[float] = []
    chis: List[float] = []
    gammas: List[float] = []
    seen: set = set()
    for q, links in selected.items():
        for (i, j) in links:
            dest = report_dest(plan, (i, j), q)
            leg = (j, dest)
            if leg in seen:
                continue
            seen.add(leg)
            rates.append(float(tables.rate[j, dest]))
            chis.append(float(tables.chi_comm[j, dest]))
            gammas.append(float(tables.gamma_comm[j, dest]))

    if not rates:
        return {
            "selected_rate_mean_mbps": 0.0,
            "selected_rate_min_mbps": 0.0,
            "selected_rate_p10_mbps": 0.0,
            "selected_chi_mean": 0.0,
            "selected_chi_min": 0.0,
            "selected_chi_p10": 0.0,
            "selected_gamma_comm_mean_db": 0.0,
            "selected_rate_satisfaction_ratio": 0.0,
            "selected_chi_ge_min_ratio": 0.0,
            "comm_feasible_edge_ratio": float(comm_feasible_edge_ratio),
        }

    rates_arr = np.array(rates, dtype=float)
    chis_arr = np.array(chis, dtype=float)
    gammas_arr = np.array(gammas, dtype=float)
    gamma_db = 10.0 * np.log10(np.maximum(gammas_arr, EPS))

    return {
        "selected_rate_mean_mbps": float(np.mean(rates_arr) / 1e6),
        "selected_rate_min_mbps": float(np.min(rates_arr) / 1e6),
        "selected_rate_p10_mbps": float(np.percentile(rates_arr, 10) / 1e6),
        "selected_chi_mean": float(np.mean(chis_arr)),
        "selected_chi_min": float(np.min(chis_arr)),
        "selected_chi_p10": float(np.percentile(chis_arr, 10)),
        "selected_gamma_comm_mean_db": float(np.mean(gamma_db)),
        "selected_rate_satisfaction_ratio": float(np.mean(rates_arr >= cfg.comm.R_min)),
        "selected_chi_ge_min_ratio": float(np.mean(chis_arr >= cfg.comm.chi_min)),
        "comm_feasible_edge_ratio": float(comm_feasible_edge_ratio),
    }


# ==========================================================================
# Trial driver
# ==========================================================================
def rng_for_method(cfg: Config, trial_index: int, method: str) -> np.random.Generator:
    """Method-specific stream used only by randomized selection rules."""
    offset = METHOD_RNG_OFFSETS[method]
    return np.random.default_rng([cfg.run.seed, 12345, trial_index, offset])


def rng_for_detection(cfg: Config, trial_index: int) -> np.random.Generator:
    """Common detector stream shared by every method in a paired trial."""
    return np.random.default_rng([cfg.run.seed, 54321, trial_index])


def _build_plan(cfg: Config, base: BaseGains, tables: LinkTables, geom) -> "ReportingPlan | None":
    """Reporting plan for this trial (``None`` in the legacy architecture)."""
    if cfg.fusion.mode.lower() != "explicit":
        return None
    return assign_fusion_nodes(cfg, base, tables, geom)


def build_slot_tables(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    selected: Dict[int, List[Link]],
    plan: "ReportingPlan | None" = None,
) -> LinkTables:
    """Rebuild the communication block with *slot-local* interference.

    Under ``comm.mac_model="slot"`` the selected reporting legs are grouped by a
    conflict-graph colouring (:func:`isac_sim.reporting.slot_schedule`).  Only
    reports inside the same slot are concurrent, so the communication SINR of a
    leg is recomputed counting only its co-slot transmitters as interferers.
    This makes the interference model and the latency model describe the same
    MAC: the SINR is slot-local and the latency is the sum of slot durations.

    The sensing block is untouched (sensing interference does not depend on the
    reporting MAC), so it is reused verbatim from ``tables``.
    """
    from .reporting import slot_schedule

    M, Q = cfg.scale.M, cfg.scale.Q
    slots = slot_schedule(cfg, selected, plan)

    rate = np.zeros((M, M))
    chi = np.zeros((M, M))
    gamma = np.zeros((M, M))
    for slot in slots:
        active_tx = np.zeros(M, dtype=bool)
        for (_, _, j, _) in slot:
            active_tx[j] = True
        t = compute_link_tables(cfg, base, active_tx_mask=active_tx, reuse_from=tables)
        for (_, _, j, dest) in slot:
            rate[j, dest] = t.rate[j, dest]
            chi[j, dest] = t.chi_comm[j, dest]
            gamma[j, dest] = t.gamma_comm[j, dest]

    return LinkTables(
        gamma_comm=gamma,
        rate=rate,
        chi_comm=chi,
        feasible_comm=tables.feasible_comm,
        raw_gamma_sense=tables.raw_gamma_sense,
        gamma_sense=tables.gamma_sense,
        rinr=tables.rinr,
        mu_soft=tables.mu_soft,
        sigma0=tables.sigma0,
        var0_q=tables.var0_q,
    )


def run_method_on_trial(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    method: MethodName,
    trial_index: int,
    reference_counts: Optional[Dict[int, int]] = None,
    cached_lagrangian: Optional[Tuple[Dict[int, List[Link]], np.ndarray]] = None,
    c2f_tables: Optional[LinkTables] = None,
    plan: "ReportingPlan | None" = None,
    eval_base: BaseGains | None = None,
    eval_tables: LinkTables | None = None,
    belief_dd_std: tuple[np.ndarray, np.ndarray] | None = None,
) -> MethodResult:
    """Run one method on one trial.

    ``base`` / ``tables`` are the *scheduler's* view (used for selection).  In
    belief mode these are the belief tables; ``eval_base`` / ``eval_tables`` are
    then the truth tables used for detection.  In the default (non-belief) mode
    the two coincide and ``eval_*`` are ``None``.
    """
    from .belief import truth_captured_links

    belief_mode = eval_base is not None and eval_tables is not None
    selector_rng = rng_for_method(cfg, trial_index, method)
    detector_rng = rng_for_detection(cfg, trial_index)

    if method == "proposed_lagrangian":
        selected, D = cached_lagrangian if cached_lagrangian is not None else select_lagrangian(cfg, base, tables, plan)
        fine_eval_full = 0.0
        fine_eval_c2f = 0.0
        sel_tables = tables
    elif method in {
        "proposed_c2f_adaptive",
        "proposed_c2f_adaptive_pd",
    }:
        if method == "proposed_c2f_adaptive_pd":
            matched_budget = (
                int(sum(reference_counts.values()))
                if reference_counts is not None else cfg.selector.max_total_links
            )
            select_cfg = apply_overrides(cfg, {
                "selector.score_mode": "detector_pd",
                "selector.use_delay_price": False,
                "selector.lambda_c": 0.0,
                "selector.max_total_links": matched_budget,
            })
        else:
            select_cfg = cfg
        selected, D, c2f_stats = select_c2f_adaptive(
            select_cfg, base, tables, plan=plan
        )
        fine_eval_full = float(c2f_stats["fine_eval_full"])
        fine_eval_c2f = float(c2f_stats["fine_eval_c2f"])
        sel_tables = c2f_tables if c2f_tables is not None else tables
    elif method in C2F_METHODS:
        if method == "proposed_c2f_pd":
            matched_budget = (
                int(sum(reference_counts.values()))
                if reference_counts is not None else cfg.selector.max_total_links
            )
            select_cfg = apply_overrides(cfg, {
                "selector.score_mode": "detector_pd",
                "selector.use_delay_price": False,
                "selector.lambda_c": 0.0,
                "selector.max_total_links": matched_budget,
            })
        else:
            select_cfg = cfg
        selected, D, c2f_stats = select_c2f(
            select_cfg, base, tables, apply_to_all=C2F_METHODS[method], plan=plan
        )
        fine_eval_full = float(c2f_stats["fine_eval_full"])
        fine_eval_c2f = float(c2f_stats["fine_eval_c2f"])
        sel_tables = c2f_tables if c2f_tables is not None else tables
    elif method == "all_neighbor":
        selected, D = select_all_neighbor(cfg, base, tables, plan)
        fine_eval_full = 0.0
        fine_eval_c2f = 0.0
        sel_tables = tables
    else:
        if reference_counts is None:
            ref_selected, _ = select_lagrangian(cfg, base, tables, plan)
            reference_counts = {q: len(ref_selected.get(q, [])) for q in range(cfg.scale.Q)}
        selected, D = select_topk_baseline(
            cfg, base, tables, reference_counts, method, selector_rng, plan
        )
        fine_eval_full = 0.0
        fine_eval_c2f = 0.0
        sel_tables = tables

    # Belief mode: only links whose belief-guided search window captures the
    # true delay-Doppler bin carry target evidence to the detector.
    selected_eval = (
        truth_captured_links(cfg, eval_base, base, selected, belief_dd_std)
        if belief_mode else selected
    )

    # Local DD refinement is a receiver capability, so every method is
    # evaluated on refined statistics when the canonical C2F path is enabled.
    # Methods differ in how they *select* candidates, not in whether a selected
    # report is allowed to use the same local estimator.
    tables_eval = (
        c2f_tables if cfg.refine.enable and c2f_tables is not None else sel_tables
    )
    det_base = base
    if belief_mode:
        det_base = eval_base
        tables_eval = (
            compute_link_tables(cfg, eval_base, dd_gain=eval_base.eta_fine)
            if cfg.refine.enable else eval_tables
        )
        D = np.array([
            deflection_for_links(
                cfg, eval_tables, q, selected_eval.get(q, []),
                weight_mode=fusion_weight_mode_for_method(method), plan=plan, base=eval_base,
            )
            for q in range(cfg.scale.Q)
        ])

    if cfg.comm.mac_model == "slot":
        # Slot MAC must take precedence over the coarse active-set label.  Each
        # reporting leg is evaluated against its actual co-slot transmitters.
        base_rebuild = eval_base if belief_mode else base
        reuse_from = eval_tables if belief_mode else sel_tables
        tables_eval = build_slot_tables(cfg, base_rebuild, reuse_from, selected, plan)
        D = np.array([
            deflection_for_links(
                cfg, tables_eval, q, selected_eval.get(q, []),
                weight_mode=fusion_weight_mode_for_method(method), plan=plan, base=base_rebuild,
            )
            for q in range(cfg.scale.Q)
        ])
    elif tables_eval is not sel_tables:
        # In belief mode the refined truth table must be paired with the
        # truth-captured subset and truth geometry. Reusing ``selected`` and
        # the scheduler belief here credits missed DD gates with sensing
        # information and makes the reported deflection inconsistent with the
        # detector that follows.
        D = np.array([
            deflection_for_links(
                cfg, tables_eval, q, selected_eval.get(q, []),
                weight_mode=fusion_weight_mode_for_method(method), plan=plan, base=det_base,
            )
            for q in range(cfg.scale.Q)
        ])
    elif cfg.comm.interference_model == "active_set":
        # Post-selection sensitivity evaluation only.  The selector used the
        # conservative pre-selection table, so this branch must not be described
        # as an endogenous active-set optimum.
        active_tx = np.zeros(cfg.scale.M, dtype=bool)
        for q, links in selected.items():
            for (i, j) in links:
                active_tx[j] = True
        base_rebuild = eval_base if belief_mode else base
        reuse_from = eval_tables if belief_mode else sel_tables
        tables_eval = compute_link_tables(
            cfg, base_rebuild, active_tx_mask=active_tx, reuse_from=reuse_from
        )
        D = np.array([
            deflection_for_links(
                cfg, tables_eval, q, selected_eval.get(q, []),
                weight_mode=fusion_weight_mode_for_method(method), plan=plan, base=base_rebuild,
            )
            for q in range(cfg.scale.Q)
        ])

    detection = evaluate_detection(
        cfg, tables_eval, selected_eval, detector_rng, method, plan, det_base
    )
    detected, total_targets, fa, total_false, fa_overall, total_false_overall, detected_per_target = detection
    feasible_targets, feasible_links = feasible_stats(cfg, base, tables, plan)
    comm_metrics = communication_metrics_for_selection(cfg, base, tables_eval, selected, plan)

    capture_rate = 1.0
    if belief_mode:
        from .belief import belief_capture_rate

        capture_rate = belief_capture_rate(cfg, eval_base, base, selected, belief_dd_std)

    return MethodResult(
        name=method,
        detected=detected,
        total_targets=total_targets,
        detected_per_target=detected_per_target,
        false_alarm=fa,
        total_false=total_false,
        false_alarm_overall=fa_overall,
        total_false_overall=total_false_overall,
        overhead_bits=total_overhead_bits(cfg, selected),
        overhead_delay_s=total_overhead_delay_s(cfg, tables_eval, selected, plan),
        selected_links=selected,
        D_fuse_per_target=D,
        active_targets=active_target_count(selected),
        feasible_targets=feasible_targets,
        feasible_links=feasible_links,
        fine_eval_full=fine_eval_full,
        fine_eval_c2f=fine_eval_c2f,
        belief_capture_rate=capture_rate,
        reporting_plan=plan,
        **comm_metrics,
    )


def run_one_trial(
    cfg: Config,
    trial_index: int,
    methods: Optional[List[str]] = None,
) -> Dict[str, MethodResult]:
    """Run every (or a chosen subset of) method on one shared geometry."""
    rng = np.random.default_rng([cfg.run.seed, trial_index])
    geom = generate_geometry(cfg, rng)

    # ---- Truth vs belief ------------------------------------------------
    if cfg.prior.belief_mode:
        from .belief import BeliefState, belief_dd_std_bins

        base_truth = build_base_gains(cfg, geom, rng)
        tables_truth = compute_link_tables(cfg, base_truth)

        belief = BeliefState.from_truth(cfg, geom, rng)
        geom_belief = belief.as_geometry(geom)
        # Reuse the UAV-UAV channel but expose only the configured RCS view to
        # the scheduler.  The paper-canonical path uses the mean RCS, never the
        # current-CPI truth realization.
        base_belief = build_base_gains(
            cfg, geom_belief, rng, channel=base_truth,
            rcs_view=cfg.prior.scheduler_rcs.lower(),
        )
        tables_belief = compute_link_tables(cfg, base_belief)
        belief_dd_std = belief_dd_std_bins(cfg, geom_belief, belief)

        plan = _build_plan(cfg, base_belief, tables_belief, geom_belief)

        needs_fine = cfg.refine.enable or cfg.refine.apply_to_all
        c2f_tables = None
        method_roster = methods if methods is not None else METHODS
        if needs_fine and any(m in C2F_METHODS for m in method_roster):
            c2f_tables = compute_link_tables(cfg, base_belief, dd_gain=base_belief.eta_fine)

        cached_lagrangian = select_lagrangian(cfg, base_belief, tables_belief, plan)
        if cfg.refine.enable and any(
            m in method_roster for m in (
                "proposed_c2f", "proposed_c2f_pd", "proposed_c2f_adaptive_pd"
            )
        ):
            reference_selected = select_c2f(
                cfg, base_belief, tables_belief, apply_to_all=False, plan=plan
            )[0]
        else:
            reference_selected = cached_lagrangian[0]
        reference_counts = {
            q: len(reference_selected.get(q, [])) for q in range(cfg.scale.Q)
        }

        return {
            method: run_method_on_trial(
                cfg, base_belief, tables_belief, method, trial_index,
                reference_counts, cached_lagrangian, c2f_tables, plan,
                eval_base=base_truth, eval_tables=tables_truth,
                belief_dd_std=belief_dd_std,
            )
            for method in method_roster
        }

    # ---- Default (shared state) -----------------------------------------
    base = build_base_gains(cfg, geom, rng)
    tables = compute_link_tables(cfg, base)

    plan = _build_plan(cfg, base, tables, geom)

    needs_fine = cfg.refine.enable or cfg.refine.apply_to_all
    c2f_tables = None
    method_roster = methods if methods is not None else METHODS
    if needs_fine and any(m in C2F_METHODS for m in method_roster):
        c2f_tables = compute_link_tables(cfg, base, dd_gain=base.eta_fine)

    cached_lagrangian = select_lagrangian(cfg, base, tables, plan)
    if cfg.refine.enable and any(
        m in method_roster for m in (
            "proposed_c2f", "proposed_c2f_pd", "proposed_c2f_adaptive_pd"
        )
    ):
        reference_selected = select_c2f(
            cfg, base, tables, apply_to_all=False, plan=plan
        )[0]
    else:
        reference_selected = cached_lagrangian[0]
    reference_counts = {
        q: len(reference_selected.get(q, [])) for q in range(cfg.scale.Q)
    }

    return {
        method: run_method_on_trial(
            cfg, base, tables, method, trial_index, reference_counts, cached_lagrangian, c2f_tables, plan
        )
        for method in method_roster
    }


def run_simulation(
    cfg: Config,
    methods: Optional[List[str]] = None,
    paired_reference: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run the full Monte-Carlo experiment once under the given configuration.

    ``paired_reference`` selects the method used for common-random-number
    detection differences. The historical proposed-C2F fields are preserved
    for compatibility, while generic reference-labelled fields support
    experimental successor methods without mislabelling the contrast.
    """
    method_roster = methods if methods is not None else METHODS
    if paired_reference is not None and paired_reference not in method_roster:
        raise ValueError(
            f"paired_reference={paired_reference!r} is not in the method roster"
        )
    all_results: Dict[str, List[MethodResult]] = {m: [] for m in method_roster}
    print_interval = max(1, cfg.run.num_mc // 10)

    def consume(t: int, trial: Dict[str, MethodResult]) -> None:
        for method, res in trial.items():
            all_results[method].append(res)
        if cfg.run.verbose and ((t + 1) % print_interval == 0 or (t + 1) == cfg.run.num_mc):
            watch = "proposed_lagrangian" if "proposed_lagrangian" in all_results else method_roster[0]
            prop = all_results[watch]
            det = sum(r.detected for r in prop)
            tot = sum(r.total_targets for r in prop)
            fa = sum(r.false_alarm for r in prop)
            tf = sum(r.total_false for r in prop)
            print(f"MC {t + 1:4d}/{cfg.run.num_mc}: {watch} "
                  f"P_D={det / max(tot, 1):.4f}, P_FA={fa / max(tf, 1):.4f}")

    if cfg.run.workers == 1:
        for t in range(cfg.run.num_mc):
            consume(t, run_one_trial(cfg, t, methods=method_roster))
    else:
        jobs = ((cfg, t, method_roster) for t in range(cfg.run.num_mc))
        with ProcessPoolExecutor(max_workers=cfg.run.workers) as pool:
            for t, trial in enumerate(pool.map(_run_trial_job, jobs, chunksize=1)):
                consume(t, trial)

    summary = {method: summarize(results, cfg) for method, results in all_results.items()}
    reference_method = paired_reference or (
        "proposed_c2f" if "proposed_c2f" in all_results
        else "proposed_lagrangian" if "proposed_lagrangian" in all_results
        else None
    )
    if reference_method is not None:
        prop = all_results[reference_method]
        for method, results in all_results.items():
            diffs = np.array([
                (p.detected - r.detected) / max(p.total_targets, 1)
                for p, r in zip(prop, results)
            ], dtype=float)
            mean = float(np.mean(diffs)) if diffs.size else 0.0
            half = (
                1.96 * float(np.std(diffs, ddof=1)) / np.sqrt(diffs.size)
                if diffs.size > 1 else 0.0
            )
            summary[method]["paired_reference_method"] = reference_method
            summary[method]["paired_reference_delta_P_D"] = mean
            summary[method]["paired_reference_delta_ci95_low"] = mean - half
            summary[method]["paired_reference_delta_ci95_high"] = mean + half
            if paired_reference is None:
                summary[method]["paired_proposed_delta_P_D"] = mean
                summary[method]["paired_proposed_delta_ci95_low"] = mean - half
                summary[method]["paired_proposed_delta_ci95_high"] = mean + half
    return summary


def _run_trial_job(
    job: tuple[Config, int, List[str]],
) -> Dict[str, MethodResult]:
    """Pickle-friendly worker entry point for deterministic parallel trials."""
    cfg, trial_index, methods = job
    return run_one_trial(cfg, trial_index, methods=methods)


# ==========================================================================
# Aggregation
# ==========================================================================
def summarize(results: List[MethodResult], cfg: Config) -> Dict[str, Any]:
    """Aggregate per-trial results into the scalar metrics used by tables/figures."""
    total_detected = sum(r.detected for r in results)
    total_targets = sum(r.total_targets for r in results)

    total_fa = sum(r.false_alarm for r in results)
    total_false = sum(r.total_false for r in results)
    total_fa_overall = sum(r.false_alarm_overall for r in results)
    total_false_overall = sum(r.total_false_overall for r in results)

    pd_ci = binomial_ci95(total_detected, total_targets)
    pfa_ci = binomial_ci95(total_fa, total_false)
    pfa_overall_ci = binomial_ci95(total_fa_overall, total_false_overall)

    D_arr = np.vstack([r.D_fuse_per_target for r in results])
    overhead_bits = np.array([r.overhead_bits for r in results], dtype=float)
    overhead_delay_s = np.array([r.overhead_delay_s for r in results], dtype=float)
    active = np.array([r.active_targets for r in results], dtype=float)
    feasible_targets = np.array([r.feasible_targets for r in results], dtype=float)
    feasible_links = np.array([r.feasible_links for r in results], dtype=float)
    fine_eval_full = np.array([r.fine_eval_full for r in results], dtype=float)
    fine_eval_c2f = np.array([r.fine_eval_c2f for r in results], dtype=float)
    belief_capture = np.array([r.belief_capture_rate for r in results], dtype=float)
    selected_links = np.array([sum(len(v) for v in r.selected_links.values()) for r in results], dtype=float)

    def col(attr: str) -> np.ndarray:
        return np.array([getattr(r, attr) for r in results], dtype=float)

    selected_rate_mean = col("selected_rate_mean_mbps")
    selected_rate_min = col("selected_rate_min_mbps")
    selected_rate_p10 = col("selected_rate_p10_mbps")
    selected_chi_mean = col("selected_chi_mean")
    selected_chi_min = col("selected_chi_min")
    selected_chi_p10 = col("selected_chi_p10")
    selected_gamma_db = col("selected_gamma_comm_mean_db")
    rate_sat = col("selected_rate_satisfaction_ratio")
    chi_sat = col("selected_chi_ge_min_ratio")
    feasible_edge_ratio = col("comm_feasible_edge_ratio")

    D_satisfied = D_arr >= cfg.detect.D_min
    detected_target_arr = np.vstack([r.detected_per_target for r in results])
    P_D_per_target_actual = np.mean(detected_target_arr, axis=0)

    pd = total_detected / max(total_targets, 1)
    pfa = total_fa / max(total_false, 1)
    pfa_overall = total_fa_overall / max(total_false_overall, 1)
    B_kbit = float(np.mean(overhead_bits)) / 1000.0
    T_ms = float(np.mean(overhead_delay_s)) * 1e3
    mean_D = float(np.mean(D_arr))

    return {
        "P_D": pd,
        "P_D_ci95": pd_ci[:2],
        "P_D_ci95_low": pd_ci[0],
        "P_D_ci95_high": pd_ci[1],
        "P_D_ci95_half_width": pd_ci[2],
        "P_FA": pfa,
        "P_FA_ci95": pfa_ci[:2],
        "P_FA_ci95_low": pfa_ci[0],
        "P_FA_ci95_high": pfa_ci[1],
        "P_FA_ci95_half_width": pfa_ci[2],
        "P_FA_overall": pfa_overall,
        "P_FA_overall_ci95": pfa_overall_ci[:2],
        "P_FA_overall_ci95_low": pfa_overall_ci[0],
        "P_FA_overall_ci95_high": pfa_overall_ci[1],
        "P_FA_overall_ci95_half_width": pfa_overall_ci[2],
        "B_mean_bits": float(np.mean(overhead_bits)),
        "B_std_bits": float(np.std(overhead_bits)),
        "T_mean_ms": T_ms,
        "T_std_ms": float(np.std(overhead_delay_s) * 1e3),
        "selected_links_mean": float(np.mean(selected_links)),
        "selected_links_std": float(np.std(selected_links)),
        "active_target_ratio_mean": float(np.mean(active / max(cfg.scale.Q, 1))),
        "feasible_target_ratio_mean": float(np.mean(feasible_targets / max(cfg.scale.Q, 1))),
        "feasible_links_mean": float(np.mean(feasible_links)),
        "fine_eval_full_mean": float(np.mean(fine_eval_full)),
        "fine_eval_c2f_mean": float(np.mean(fine_eval_c2f)),
        "belief_capture_rate_mean": float(np.mean(belief_capture)),
        "comm_feasible_edge_ratio_mean": float(np.mean(feasible_edge_ratio)),
        "selected_rate_mean_mbps": float(np.mean(selected_rate_mean)),
        "selected_rate_min_mbps_mean": float(np.mean(selected_rate_min)),
        "selected_rate_p10_mbps_mean": float(np.mean(selected_rate_p10)),
        "selected_chi_mean": float(np.mean(selected_chi_mean)),
        "selected_chi_min_mean": float(np.mean(selected_chi_min)),
        "selected_chi_p10_mean": float(np.mean(selected_chi_p10)),
        "selected_gamma_comm_mean_db": float(np.mean(selected_gamma_db)),
        "selected_rate_satisfaction_ratio_mean": float(np.mean(rate_sat)),
        "selected_chi_ge_min_ratio_mean": float(np.mean(chi_sat)),
        "D_mean": mean_D,
        "D_median": float(np.median(D_arr)),
        "D_p10": float(np.percentile(D_arr, 10)),
        "D_p90": float(np.percentile(D_arr, 90)),
        "D_mean_per_target": np.mean(D_arr, axis=0),
        "D_satisfied_prob_per_target": np.mean(D_satisfied, axis=0),
        "all_targets_satisfied_prob": float(np.mean(np.all(D_satisfied, axis=1))),
        "worst_target_D_mean": float(np.min(np.mean(D_arr, axis=0))),
        "worst_target_satisfied_prob": float(np.min(np.mean(D_satisfied, axis=0))),
        "P_D_per_target_actual": P_D_per_target_actual,
        "actual_mean_target_P_D": float(np.mean(P_D_per_target_actual)),
        "actual_worst_target_P_D": float(np.min(P_D_per_target_actual)),
        "actual_best_target_P_D": float(np.max(P_D_per_target_actual)),
        "P_D_per_kbit": pd / max(B_kbit, EPS),
        "P_D_per_ms": pd / max(T_ms, EPS),
        "D_per_kbit": mean_D / max(B_kbit, EPS),
        "D_per_ms": mean_D / max(T_ms, EPS),
    }
