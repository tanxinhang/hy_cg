"""Monte-Carlo execution and metric aggregation.

One trial = one independent geometry realisation + one detection evaluation per
method.  All methods see the same geometry, so differences are attributable to
the selection rule alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config, Link, MethodName
from .fusion import compute_weights, fusion_weight_mode_for_method, h0_variance_for_link
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
from .selection import (
    C2F_METHODS,
    METHOD_RNG_OFFSETS,
    METHODS,
    feasible_links_for_target,
    link_delay_s,
    select_all_neighbor,
    select_c2f,
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
    # Main P_FA uses only active detectors, i.e. targets with selected links.
    false_alarm: int
    total_false: int
    # System-level P_FA keeps inactive targets in the potential denominator.
    false_alarm_overall: int
    total_false_overall: int
    overhead_bits: float
    overhead_delay_s: float
    selected_links: Dict[int, List[Link]]
    D_fuse_per_target: np.ndarray
    active_targets: int
    feasible_targets: int
    feasible_links: int
    # Communication-side statistics of the selected soft-information links.
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
    # C2F cost accounting (window-evaluation counts; 0 for non-C2F methods).
    fine_eval_full: float = 0.0
    fine_eval_c2f: float = 0.0


# ==========================================================================
# Soft-statistic sampling
# ==========================================================================
def draw_h1_soft_stat(cfg: Config, tables: LinkTables, link: Link, q: int, rng: np.random.Generator) -> float:
    """Draw one soft statistic under H1, including communication errors."""
    i, j = link
    d = cfg.detect
    mu = float(tables.mu_soft[i, j, q])
    gamma = float(tables.gamma_sense[i, j, q])
    sigma1 = max(d.soft_sigma_floor, float(tables.sigma0[i, j]) / math.sqrt(1.0 + gamma + EPS))

    if not d.enable_comm_error_pollution:
        return float(rng.normal(mu, sigma1))

    chi = float(np.clip(tables.chi_comm[i, j], 0.0, 1.0))
    if rng.random() < chi:
        return float(rng.normal(mu, sigma1))

    sigma_err = d.soft_error_sigma_scale * float(tables.sigma0[i, j])
    if d.comm_error_model == "erasure":
        return float(rng.normal(0.0, sigma_err))
    if d.comm_error_model == "flip":
        return float(rng.normal(-d.soft_error_flip_scale * mu, sigma_err))
    if d.comm_error_model == "biased":
        return float(rng.normal(d.soft_error_bias_scale * mu, sigma_err))
    raise ValueError(d.comm_error_model)


def draw_h0_soft_stat(cfg: Config, tables: LinkTables, link: Link, rng: np.random.Generator) -> float:
    """Draw one soft statistic under H0."""
    i, j = link
    d = cfg.detect
    sigma0 = float(tables.sigma0[i, j])

    if not d.enable_comm_error_pollution:
        return float(rng.normal(0.0, sigma0))

    chi = float(np.clip(tables.chi_comm[i, j], 0.0, 1.0))
    if rng.random() < chi:
        return float(rng.normal(0.0, sigma0))

    sigma_err = d.soft_error_sigma_scale * sigma0
    if d.comm_error_model == "erasure":
        return float(rng.normal(0.0, sigma_err))
    if d.comm_error_model == "flip":
        # Under H0 there is no target-dependent sign to flip; a failed packet
        # only inflates the noise.
        return float(rng.normal(0.0, sigma_err))
    if d.comm_error_model == "biased":
        # Optional H0-side bias keeps biased-error robustness tests symmetric.
        return float(rng.normal(d.h0_error_bias_scale * sigma0, sigma_err))
    raise ValueError(d.comm_error_model)


# ==========================================================================
# Detection evaluation
# ==========================================================================
def evaluate_detection(
    cfg: Config,
    tables: LinkTables,
    selected: Dict[int, List[Link]],
    rng: np.random.Generator,
    method: str,
) -> Tuple[int, int, int, int, int, int, np.ndarray]:
    """Evaluate detection and false alarms with two P_FA denominators.

    *Main* P_FA is the active-detector rate: only targets with at least one
    selected link contribute false-alarm trials.  This is the right denominator
    for checking whether the Gaussian threshold is calibrated.

    *System-level* P_FA keeps all target hypotheses in the potential
    denominator: inactive targets produce no fused statistic and therefore
    contribute zero false alarms, while the denominator stays
    ``Q * num_false_per_target``.
    """
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
            # No soft information means no active detector for this target.
            continue

        total_false_active += d.num_false_per_target
        weights = compute_weights(cfg, tables, q, links, mode=weight_mode)
        var0 = sum((w ** 2) * h0_variance_for_link(cfg, tables, link) for link, w in weights.items())
        thr = base_thr * math.sqrt(max(var0, EPS))

        F = 0.0
        for link, w in weights.items():
            F += w * draw_h1_soft_stat(cfg, tables, link, q, rng)
        if F > thr:
            detected += 1
            detected_per_target[q] = 1

        for _ in range(d.num_false_per_target):
            F0 = 0.0
            for link, w in weights.items():
                F0 += w * draw_h0_soft_stat(cfg, tables, link, rng)
            if F0 > thr:
                false_alarm_active += 1

    # System-level numerator: inactive targets produce zero false alarms, so the
    # only observed false alarms are those from active detectors.
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


def total_overhead_delay_s(cfg: Config, tables: LinkTables, selected: Dict[int, List[Link]]) -> float:
    total = 0.0
    for q, links in selected.items():
        for link in links:
            total += link_delay_s(cfg, tables, q, link)
    return float(total)


def active_target_count(selected: Dict[int, List[Link]]) -> int:
    return sum(1 for links in selected.values() if len(links) > 0)


def feasible_stats(cfg: Config, base: BaseGains, tables: LinkTables) -> Tuple[int, int]:
    counts = [len(feasible_links_for_target(cfg, base, tables, q)) for q in range(cfg.scale.Q)]
    return sum(1 for c in counts if c > 0), int(np.sum(counts))


def communication_metrics_for_selection(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    selected: Dict[int, List[Link]],
) -> Dict[str, float]:
    """Communication-side statistics of the selected soft-information links.

    These are not throughput objectives.  They document the communication role
    in the sensing-centric ISAC model: selected links must carry soft sensing
    information, so their rate, reliability and delay matter.
    """
    selected_unique: List[Link] = []
    seen: set = set()
    for links in selected.values():
        for link in links:
            if link not in seen:
                seen.add(link)
                selected_unique.append(link)

    edge_count = int(np.sum(base.edge_mask))
    feasible_edge_count = int(np.sum(tables.feasible_comm & base.edge_mask))
    comm_feasible_edge_ratio = feasible_edge_count / max(edge_count, 1)

    if not selected_unique:
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

    rates = np.array([tables.rate[i, j] for i, j in selected_unique], dtype=float)
    chis = np.array([tables.chi_comm[i, j] for i, j in selected_unique], dtype=float)
    gammas = np.array([tables.gamma_comm[i, j] for i, j in selected_unique], dtype=float)
    gamma_db = 10.0 * np.log10(np.maximum(gammas, EPS))

    return {
        "selected_rate_mean_mbps": float(np.mean(rates) / 1e6),
        "selected_rate_min_mbps": float(np.min(rates) / 1e6),
        "selected_rate_p10_mbps": float(np.percentile(rates, 10) / 1e6),
        "selected_chi_mean": float(np.mean(chis)),
        "selected_chi_min": float(np.min(chis)),
        "selected_chi_p10": float(np.percentile(chis, 10)),
        "selected_gamma_comm_mean_db": float(np.mean(gamma_db)),
        "selected_rate_satisfaction_ratio": float(np.mean(rates >= cfg.comm.R_min)),
        "selected_chi_ge_min_ratio": float(np.mean(chis >= cfg.comm.chi_min)),
        "comm_feasible_edge_ratio": float(comm_feasible_edge_ratio),
    }


# ==========================================================================
# Trial driver
# ==========================================================================
def rng_for_method(cfg: Config, trial_index: int, method: str) -> np.random.Generator:
    offset = METHOD_RNG_OFFSETS[method]
    return np.random.default_rng([cfg.run.seed, 12345, trial_index, offset])


def run_method_on_trial(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    method: MethodName,
    trial_index: int,
    reference_counts: Optional[Dict[int, int]] = None,
    cached_lagrangian: Optional[Tuple[Dict[int, List[Link]], np.ndarray]] = None,
    c2f_tables: Optional[LinkTables] = None,
) -> MethodResult:
    rng = rng_for_method(cfg, trial_index, method)

    if method == "proposed_lagrangian":
        selected, D = cached_lagrangian if cached_lagrangian is not None else select_lagrangian(cfg, base, tables)
        fine_eval_full = 0.0
        fine_eval_c2f = 0.0
    elif method in C2F_METHODS:
        # C2F selectors need the fine-grained tables produced in run_one_trial.
        if c2f_tables is None:
            c2f_tables = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
        selected, D, c2f_stats = select_c2f(cfg, base, c2f_tables)
        fine_eval_full = float(c2f_stats["fine_eval_full"])
        fine_eval_c2f = float(c2f_stats["fine_eval_c2f"])
    elif method == "all_neighbor":
        selected, D = select_all_neighbor(cfg, base, tables)
        fine_eval_full = 0.0
        fine_eval_c2f = 0.0
    else:
        if reference_counts is None:
            ref_selected, _ = select_lagrangian(cfg, base, tables)
            reference_counts = {q: len(ref_selected.get(q, [])) for q in range(cfg.scale.Q)}
        selected, D = select_topk_baseline(cfg, base, tables, reference_counts, method, rng)
        fine_eval_full = 0.0
        fine_eval_c2f = 0.0

    detection = evaluate_detection(cfg, tables, selected, rng, method)
    detected, total_targets, fa, total_false, fa_overall, total_false_overall, detected_per_target = detection
    feasible_targets, feasible_links = feasible_stats(cfg, base, tables)
    comm_metrics = communication_metrics_for_selection(cfg, base, tables, selected)

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
        overhead_delay_s=total_overhead_delay_s(cfg, tables, selected),
        selected_links=selected,
        D_fuse_per_target=D,
        active_targets=active_target_count(selected),
        feasible_targets=feasible_targets,
        feasible_links=feasible_links,
        fine_eval_full=fine_eval_full,
        fine_eval_c2f=fine_eval_c2f,
        **comm_metrics,
    )


def run_one_trial(
    cfg: Config,
    trial_index: int,
    methods: Optional[List[str]] = None,
) -> Dict[str, MethodResult]:
    """Run every (or a chosen subset of) method on one shared geometry.

    ``methods`` lets an experiment register its own roster -- e.g. the C2F
    mode adds ``proposed_c2f`` without disturbing the main comparison.
    """
    # Multi-integer seeding avoids correlations between consecutive seeds.
    rng = np.random.default_rng([cfg.run.seed, trial_index])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    tables = compute_link_tables(cfg, base)

    cached_lagrangian = select_lagrangian(cfg, base, tables)
    reference_counts = {q: len(cached_lagrangian[0].get(q, [])) for q in range(cfg.scale.Q)}

    # Build fine-grained tables only when a C2F method is requested and the
    # refined DD gain actually differs from the coarse gain.
    needs_fine = cfg.refine.enable or cfg.refine.apply_to_all
    c2f_tables = None
    if needs_fine and any(m in C2F_METHODS for m in (methods or METHODS)):
        c2f_tables = compute_link_tables(cfg, base, dd_gain=base.eta_fine)

    method_roster = methods if methods is not None else METHODS
    return {
        method: run_method_on_trial(
            cfg,
            base,
            tables,
            method,
            trial_index,
            reference_counts,
            cached_lagrangian,
            c2f_tables,
        )
        for method in method_roster
    }


def run_simulation(
    cfg: Config,
    methods: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run the full Monte-Carlo experiment once under the given configuration.

    ``methods`` restricts the per-trial roster to a chosen subset; useful when
    a single experiment (e.g. C2F ablation) only needs a few rows.
    """
    method_roster = methods if methods is not None else METHODS
    all_results: Dict[str, List[MethodResult]] = {m: [] for m in method_roster}
    print_interval = max(1, cfg.run.num_mc // 10)

    for t in range(cfg.run.num_mc):
        for method, res in run_one_trial(cfg, t, methods=method_roster).items():
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

    return {method: summarize(results, cfg) for method, results in all_results.items()}


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
        "P_D_ci95_half_width": pd_ci[2],
        "P_FA": pfa,
        "P_FA_ci95": pfa_ci[:2],
        "P_FA_ci95_half_width": pfa_ci[2],
        "P_FA_overall": pfa_overall,
        "P_FA_overall_ci95": pfa_overall_ci[:2],
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
