"""Experiment registry.

Each experiment is described as data plus a thin runner:

* the *variants* / *sweep values* are module-level dictionaries;
* the runner simply applies overrides and calls :func:`simulate.run_simulation`.

Because variants are dotted-path override dictionaries (see
:func:`isac_sim.config.apply_overrides`), adding an ablation is a one-line
change here and requires no new configuration fields, no new CLI flags and no
changes anywhere else in the package.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from .config import Config, apply_overrides
from .report import scalar_summary_row
from .simulate import run_simulation

# --------------------------------------------------------------------------
# Sweep grids and variant definitions
# --------------------------------------------------------------------------
LAMBDA_VALUES: List[float] = [0.0005, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
R_MIN_VALUES: List[float] = [1e5, 2e5, 5e5, 1e6, 2e6]

ROBUSTNESS_VALUES: Dict[str, List[Any]] = {
    "comm_model": ["erasure", "biased", "flip"],
    "error_sigma": [1.0, 2.0, 3.0, 4.0],
    "residual_direct": [1e-5, 1e-4, 1e-3, 1e-2],
}

ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "w/o_alpha": {"selector.use_target_priority": False},
    "w/o_delay_price": {"selector.use_delay_price": False},
    "w/o_comm_error_calib": {"selector.use_comm_error_calibration": False},
    "w/o_softmin_alpha": {"selector.use_softmin_alpha": False},
}

DD_VARIANTS: Dict[str, Dict[str, Any]] = {
    "full_dd": {},
    "w/o_dd_validity": {"dd.use_otfs_bin_validity": False},
    "w/o_dd_fractional_loss": {"dd.enable_dd_fractional_penalty": False},
    "w/o_dd_collision_penalty": {"dd.enable_dd_collision_penalty": False},
    "w/o_all_dd_effects": {
        "dd.use_otfs_bin_validity": False,
        "dd.enable_dd_fractional_penalty": False,
        "dd.enable_dd_collision_penalty": False,
    },
}

# Methods recorded by the multi-method sweeps (proposed + two baselines + upper bound).
SWEEP_METHODS: List[str] = ["proposed_lagrangian", "sense_sinr", "single_best", "all_neighbor"]

FAIR_MAX_TOTAL_LINKS: int = 26


def _banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------
def lambda_sweep(cfg: Config, values: List[float] | None = None) -> List[Dict[str, Any]]:
    r"""Sweep the communication price ``lambda_c`` of the proposed selector."""
    rows: List[Dict[str, Any]] = []
    for lam in (values if values is not None else LAMBDA_VALUES):
        variant = apply_overrides(cfg, {"selector.lambda_c": float(lam)})
        _banner(f"Running lambda-cost sweep: lambda_c = {lam:g}")
        s = run_simulation(variant)["proposed_lagrangian"]
        row = {
            "lambda_cost": float(lam),
            "P_D": s["P_D"],
            "P_D_ci95_low": s["P_D_ci95"][0],
            "P_D_ci95_high": s["P_D_ci95"][1],
            "P_D_ci95_half_width": s["P_D_ci95_half_width"],
            "P_FA_active": s["P_FA"],
            "P_FA_overall": s["P_FA_overall"],
            "B_mean_bits": s["B_mean_bits"],
            "T_mean_ms": s["T_mean_ms"],
            "selected_links_mean": s["selected_links_mean"],
            "active_target_ratio_mean": s["active_target_ratio_mean"],
            "comm_feasible_edge_ratio_mean": s["comm_feasible_edge_ratio_mean"],
            "selected_rate_mean_mbps": s["selected_rate_mean_mbps"],
            "selected_rate_min_mbps_mean": s["selected_rate_min_mbps_mean"],
            "selected_chi_mean": s["selected_chi_mean"],
            "selected_chi_min_mean": s["selected_chi_min_mean"],
            "selected_rate_satisfaction_ratio_mean": s["selected_rate_satisfaction_ratio_mean"],
            "selected_chi_ge_min_ratio_mean": s["selected_chi_ge_min_ratio_mean"],
            "D_mean": s["D_mean"],
            "D_median": s["D_median"],
            "D_p10": s["D_p10"],
            "D_p90": s["D_p90"],
            "all_targets_satisfied_prob": s["all_targets_satisfied_prob"],
            "worst_target_D_mean": s["worst_target_D_mean"],
            "worst_target_satisfied_prob": s["worst_target_satisfied_prob"],
            "actual_worst_target_P_D": s["actual_worst_target_P_D"],
            "actual_mean_target_P_D": s["actual_mean_target_P_D"],
            "P_D_per_kbit": s["P_D_per_kbit"],
            "P_D_per_ms": s["P_D_per_ms"],
            "D_per_kbit": s["D_per_kbit"],
            "D_per_ms": s["D_per_ms"],
        }
        rows.append(row)
        print(f"lambda={lam:g} | P_D={row['P_D']:.4f}, T={row['T_mean_ms']:.4f} ms, "
              f"links={row['selected_links_mean']:.2f}, P_D/ms={row['P_D_per_ms']:.4f}, "
              f"worst-sat={row['worst_target_satisfied_prob']:.4f}")
    return rows


def _variant_suite(
    cfg: Config,
    variants: Dict[str, Dict[str, Any]],
    experiment: str,
    label: str,
    methods: List[str],
    extra_fields: Callable[[Config, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run each named variant and record the requested methods."""
    rows: List[Dict[str, Any]] = []
    for name, overrides in variants.items():
        variant = apply_overrides(cfg, overrides)
        _banner(f"Running {label} variant: {name}")
        summary = run_simulation(variant)
        for method in methods:
            if method not in summary:
                continue
            rows.append(scalar_summary_row(summary, method, extra_fields(variant, name)))
        if "proposed_lagrangian" in summary:
            sp = summary["proposed_lagrangian"]
            print(f"{name} | proposed P_D={sp['P_D']:.4f}, T={sp['T_mean_ms']:.4f} ms, "
                  f"links={sp['selected_links_mean']:.2f}, P_D/ms={sp['P_D_per_ms']:.4f}")
    return rows


def ablation(cfg: Config) -> List[Dict[str, Any]]:
    """Ablate each mechanism of the proposed selector, one at a time."""

    def extra(variant: Config, name: str) -> Dict[str, Any]:
        return {
            "experiment": "ablation",
            "variant": name,
            "use_target_priority": variant.selector.use_target_priority,
            "use_delay_price": variant.selector.use_delay_price,
            "use_comm_error_calibration": variant.selector.use_comm_error_calibration,
            "use_softmin_alpha": variant.selector.use_softmin_alpha,
            "lambda_cost": variant.selector.lambda_c,
            "mu_deficit": variant.selector.mu_deficit,
        }

    return _variant_suite(cfg, ABLATION_VARIANTS, "ablation", "ablation", ["proposed_lagrangian"], extra)


def fair_ablation(cfg: Config, budget: int = FAIR_MAX_TOTAL_LINKS) -> List[Dict[str, Any]]:
    """Ablation under a fixed global link budget.

    Without a shared cap, ``w/o delay price`` and ``w/o alpha`` can raise raw
    ``P_D`` simply by selecting more links.  Here every variant shares the same
    ``max_total_links``.
    """

    def extra(variant: Config, name: str) -> Dict[str, Any]:
        return {
            "experiment": "fair_ablation",
            "variant": name,
            "fair_max_total_links": variant.selector.max_total_links,
            "use_target_priority": variant.selector.use_target_priority,
            "use_delay_price": variant.selector.use_delay_price,
            "use_comm_error_calibration": variant.selector.use_comm_error_calibration,
            "use_softmin_alpha": variant.selector.use_softmin_alpha,
            "lambda_cost": variant.selector.lambda_c,
            "mu_deficit": variant.selector.mu_deficit,
        }

    capped = apply_overrides(cfg, {"selector.max_total_links": int(budget)})
    return _variant_suite(capped, ABLATION_VARIANTS, "fair_ablation", "fair ablation",
                          ["proposed_lagrangian"], extra)


def dd_ablation(cfg: Config) -> List[Dict[str, Any]]:
    """Measure the contribution of each OTFS delay-Doppler mechanism."""

    def extra(variant: Config, name: str) -> Dict[str, Any]:
        return {
            "experiment": "dd_ablation",
            "variant": name,
            "use_otfs_bin_validity": variant.dd.use_otfs_bin_validity,
            "enable_dd_fractional_penalty": variant.dd.enable_dd_fractional_penalty,
            "enable_dd_collision_penalty": variant.dd.enable_dd_collision_penalty,
        }

    return _variant_suite(cfg, DD_VARIANTS, "dd_ablation", "OTFS-DD ablation", SWEEP_METHODS, extra)


def comm_sweep(cfg: Config, values: List[float] | None = None) -> List[Dict[str, Any]]:
    """Sweep the minimum communication rate ``R_min``.

    This exposes the communication side of the sensing-centric ISAC model: as
    ``R_min`` grows, fewer UAV-to-UAV links are feasible, which moves
    selected-link reliability, exchange delay and finally ``P_D``.
    """
    rows: List[Dict[str, Any]] = []
    for r_min in (values if values is not None else R_MIN_VALUES):
        variant = apply_overrides(cfg, {"comm.R_min": float(r_min)})
        _banner(f"Running communication constraint sweep: R_min = {r_min:g} bit/s")
        summary = run_simulation(variant)
        for method in SWEEP_METHODS:
            if method not in summary:
                continue
            rows.append(scalar_summary_row(summary, method, {
                "experiment": "comm_sweep",
                "R_min_bps": float(r_min),
                "R_min_mbps": float(r_min) / 1e6,
            }))
        sp = summary["proposed_lagrangian"]
        print(f"R_min={r_min:g} | proposed P_D={sp['P_D']:.4f}, T={sp['T_mean_ms']:.2f} ms, "
              f"links={sp['selected_links_mean']:.2f}, rate={sp['selected_rate_mean_mbps']:.3f} Mbps, "
              f"chi={sp['selected_chi_mean']:.3f}, feasible-edge={sp['comm_feasible_edge_ratio_mean']:.3f}")
    return rows


def robustness(cfg: Config, axis: str = "comm_model", values: List[Any] | None = None) -> List[Dict[str, Any]]:
    """Sweep one robustness axis.

    Supported axes: ``comm_model`` (erasure/biased/flip), ``error_sigma``
    (``detect.soft_error_sigma_scale``) and ``residual_direct``
    (``radio.residual_direct_factor``).
    """
    if values is None:
        values = ROBUSTNESS_VALUES[axis]
    field_by_axis = {
        "comm_model": "detect.comm_error_model",
        "error_sigma": "detect.soft_error_sigma_scale",
        "residual_direct": "radio.residual_direct_factor",
    }
    if axis not in field_by_axis:
        raise ValueError(f"Unknown robustness axis {axis!r}. "
                         f"Choose from {sorted(field_by_axis)}.")

    rows: List[Dict[str, Any]] = []
    for value in values:
        variant = apply_overrides(cfg, {field_by_axis[axis]: value})
        _banner(f"Running robustness sweep: {axis} = {value}")
        summary = run_simulation(variant)
        for method in SWEEP_METHODS:
            if method not in summary:
                continue
            rows.append(scalar_summary_row(summary, method, {
                "experiment": "robustness",
                "robustness_type": axis,
                "condition_value": value,
                "comm_error_model": variant.detect.comm_error_model,
                "soft_error_sigma_scale": variant.detect.soft_error_sigma_scale,
                "residual_direct_factor": variant.radio.residual_direct_factor,
            }))
        sp, ss = summary["proposed_lagrangian"], summary["sense_sinr"]
        print(f"{axis}={value} | proposed P_D={sp['P_D']:.4f}, T={sp['T_mean_ms']:.2f} ms, "
              f"worst={sp['worst_target_satisfied_prob']:.4f}; "
              f"sense_sinr P_D={ss['P_D']:.4f}, T={ss['T_mean_ms']:.2f} ms")
    return rows


# --------------------------------------------------------------------------
# C2F ablation
# --------------------------------------------------------------------------
C2F_METHODS_TO_RUN: List[str] = [
    "proposed_lagrangian",
    "proposed_c2f",
    "proposed_c2f_full",
    "all_neighbor",
]


def c2f(cfg: Config) -> List[Dict[str, Any]]:
    """Compare the coarse-grained baseline, C2F and full local refinement.

    ``proposed_lagrangian`` runs the original coarse-only selector (eta^c).
    ``proposed_c2f`` implements the paper's two-stage strategy: coarse
    shortlist by eta^c, fine-stage greedy with eta^f.  ``proposed_c2f_full``
    skips the shortlist and refines every feasible link -- this is the
    *full local refinement* the paper compares against to quantify the
    ``41.1%`` fine-grid-evaluation saving.

    The experiment reports ``fine_eval_full_mean`` and ``fine_eval_c2f_mean``
    (the average number of ``(2W+1)^2`` window evaluations per trial), plus
    ``fine_eval_reduction = 1 - c2f / full`` -- the direct, paper-matching
    quantifier of the C2F saving.
    """
    if not (cfg.refine.enable or cfg.refine.apply_to_all):
        cfg = apply_overrides(cfg, {"refine.enable": True})

    _banner("Running C2F ablation (proposed_lagrangian vs proposed_c2f vs proposed_c2f_full)")
    summary = run_simulation(cfg, methods=C2F_METHODS_TO_RUN)

    fine_full = summary.get("proposed_c2f_full", {}).get("fine_eval_full_mean", 0.0)
    fine_c2f = summary.get("proposed_c2f", {}).get("fine_eval_c2f_mean", 0.0)

    rows: List[Dict[str, Any]] = []
    for method in C2F_METHODS_TO_RUN:
        if method not in summary:
            continue
        s = summary[method]
        rows.append({
            "experiment": "c2f",
            "method": method,
            "W_half_width": cfg.refine.half_width,
            "kappa_dd": cfg.refine.kappa_dd,
            "eta_min": cfg.refine.eta_min,
            "L_short": cfg.refine.shortlist_size,
            "window_kernel": cfg.refine.window_kernel,
            "P_D": s["P_D"],
            "P_D_ci95_low": s["P_D_ci95"][0],
            "P_D_ci95_high": s["P_D_ci95"][1],
            "P_FA": s["P_FA"],
            "B_mean_bits": s["B_mean_bits"],
            "T_mean_ms": s["T_mean_ms"],
            "selected_links_mean": s["selected_links_mean"],
            "active_target_ratio_mean": s["active_target_ratio_mean"],
            "D_mean": s["D_mean"],
            "worst_target_satisfied_prob": s["worst_target_satisfied_prob"],
            "actual_worst_target_P_D": s["actual_worst_target_P_D"],
            "actual_mean_target_P_D": s["actual_mean_target_P_D"],
            "P_D_per_kbit": s["P_D_per_kbit"],
            "P_D_per_ms": s["P_D_per_ms"],
            "fine_eval_full_mean": s.get("fine_eval_full_mean", 0.0),
            "fine_eval_c2f_mean": s.get("fine_eval_c2f_mean", 0.0),
        })

    if "proposed_c2f" in summary and "proposed_c2f_full" in summary:
        sc, sf = summary["proposed_c2f"], summary["proposed_c2f_full"]
        reduction = 1.0 - (fine_c2f / max(fine_full, 1e-12))
        print(
            f"C2F vs full-refine: fine-grid eval {fine_c2f:.1f} vs {fine_full:.1f} "
            f"(reduction {reduction:.1%});  P_D c2f={sc['P_D']:.4f}  full={sf['P_D']:.4f};"
            f"  T c2f={sc['T_mean_ms']:.2f} ms  full={sf['T_mean_ms']:.2f} ms"
        )

    return rows


# --------------------------------------------------------------------------
# Target-state prior sensitivity
# --------------------------------------------------------------------------
PRIOR_GRID: List[Dict[str, float]] = [
    {"sigma_pos_m": 0.0, "sigma_vel_mps": 0.0},
    {"sigma_pos_m": 5.0, "sigma_vel_mps": 1.0},
    {"sigma_pos_m": 10.0, "sigma_vel_mps": 3.0},
    {"sigma_pos_m": 20.0, "sigma_vel_mps": 5.0},
    {"sigma_pos_m": 50.0, "sigma_vel_mps": 10.0},
]

PRIOR_METHODS: List[str] = ["proposed_lagrangian", "all_neighbor"]


def prior_sweep(cfg: Config, grid: List[Dict[str, float]] | None = None) -> List[Dict[str, Any]]:
    """Measure ``P_D`` degradation under target-state prior uncertainty.

    For each ``(sigma_pos_m, sigma_vel_mps)`` in ``grid`` every Monte-Carlo
    trial regenerates the target state from a fresh random draw around the
    nominal position/velocity.  ``proposed_lagrangian`` and ``all_neighbor``
    are recorded so the reference baseline's sensitivity to prior error is
    reported alongside.
    """
    grid = grid if grid is not None else PRIOR_GRID
    rows: List[Dict[str, Any]] = []
    for entry in grid:
        sp = float(entry["sigma_pos_m"])
        sv = float(entry["sigma_vel_mps"])
        variant = apply_overrides(cfg, {
            "prior.sigma_pos_m": sp,
            "prior.sigma_vel_mps": sv,
        })
        _banner(f"Running prior sweep: sigma_pos={sp:g} m, sigma_vel={sv:g} m/s")
        summary = run_simulation(variant, methods=PRIOR_METHODS)
        for method in PRIOR_METHODS:
            if method not in summary:
                continue
            s = summary[method]
            rows.append({
                "experiment": "prior_sweep",
                "method": method,
                "sigma_pos_m": sp,
                "sigma_vel_mps": sv,
                "P_D": s["P_D"],
                "P_D_ci95_low": s["P_D_ci95"][0],
                "P_D_ci95_high": s["P_D_ci95"][1],
                "actual_worst_target_P_D": s["actual_worst_target_P_D"],
                "actual_mean_target_P_D": s["actual_mean_target_P_D"],
                "worst_target_satisfied_prob": s["worst_target_satisfied_prob"],
                "D_mean": s["D_mean"],
                "selected_links_mean": s["selected_links_mean"],
                "B_mean_bits": s["B_mean_bits"],
                "T_mean_ms": s["T_mean_ms"],
            })
        sp_row = summary["proposed_lagrangian"]
        print(f"sigma_pos={sp:g} m, sigma_vel={sv:g} m/s | "
              f"proposed P_D={sp_row['P_D']:.4f}, "
              f"worst_P_D={sp_row['actual_worst_target_P_D']:.4f}, "
              f"D_mean={sp_row['D_mean']:.2f}")
    return rows


# --------------------------------------------------------------------------
# OTFS waveform-level validation
# --------------------------------------------------------------------------
def waveform_check(
    cfg: Config,
    n_samples: int = 64,
    seed: int = 2026,
) -> List[Dict[str, Any]]:
    """Compare the analytic eta-c / eta-loc against the real OTFS PSF.

    Samples ``n_samples`` random fractional DD offsets, evaluates both the
    closed-form Dirichlet/sinc model and the end-to-end OTFS
    modulation/demodulation kernel for each, then records the pair
    alongside the absolute and relative error.  This is the cheapest way to
    answer the ``waveform-level validation`` requirement reviewers raise when
    a paper claims a closed-form DD gain.
    """
    from .waveform import sweep_compare_analytic_vs_psf

    rng = np.random.default_rng(seed)
    arr = sweep_compare_analytic_vs_psf(cfg, n_samples=n_samples, rng=rng)

    rows: List[Dict[str, Any]] = []
    for i in range(n_samples):
        rows.append({
            "experiment": "waveform_check",
            "frac_l": float(arr["frac_l"][i]),
            "frac_k": float(arr["frac_k"][i]),
            "eta_c_analytic": float(arr["eta_c_analytic"][i]),
            "eta_c_psf": float(arr["eta_c_psf"][i]),
            "eta_loc_analytic": float(arr["eta_loc_analytic"][i]),
            "eta_loc_psf": float(arr["eta_loc_psf"][i]),
        })
    _banner(f"Waveform-level validation (n={n_samples})")
    abs_err_c = np.abs(arr["eta_c_psf"] - arr["eta_c_analytic"])
    abs_err_loc = np.abs(arr["eta_loc_psf"] - arr["eta_loc_analytic"])
    rel_err_c = abs_err_c / np.maximum(arr["eta_c_psf"], 1e-6)
    rel_err_loc = abs_err_loc / np.maximum(arr["eta_loc_psf"], 1e-6)
    print(f"  eta^c    | mean |analytic-PSF|={abs_err_c.mean():.4e}  "
          f"median={np.median(abs_err_c):.4e}  "
          f"max rel={rel_err_c.max():.4e}")
    print(f"  eta^loc  | mean |analytic-PSF|={abs_err_loc.mean():.4e}  "
          f"median={np.median(abs_err_loc):.4e}  "
          f"max rel={rel_err_loc.max():.4e}")
    return rows


# --------------------------------------------------------------------------
# Oracle optimality gap (small-scale exact reference)
# --------------------------------------------------------------------------
def oracle_gap(
    cfg: Config,
    small_M: int = 3,
    small_Q: int = 3,
    max_per_target: int = 3,
    max_total: int = 6,
) -> List[Dict[str, Any]]:
    """Exact small-scale optimality gap of the greedy selector.

    Forces a small ``(M, Q)`` scenario (exhaustive search is only tractable
    there), then for each Monte-Carlo trial compares the greedy selector
    against the exact ``sum_q D_q``-maximising link set under the link
    budgets.  Returns one row per trial with the greedy and oracle objective
    and the relative gap ``(oracle - greedy) / oracle``.
    """
    from .model import build_base_gains, compute_link_tables, generate_geometry
    from .oracle import greedy_objective, oracle_exhaustive
    from .selection import select_lagrangian

    small_cfg = apply_overrides(cfg, {
        "scale.M": int(small_M),
        "scale.Q": int(small_Q),
        "selector.max_links_per_target": int(max_per_target),
        "selector.max_total_links": int(max_total),
    })

    _banner(f"Running oracle gap (M={small_M}, Q={small_Q}, "
            f"K_per={max_per_target}, K_tot={max_total})")

    rows: List[Dict[str, Any]] = []
    gaps: List[float] = []
    for t in range(cfg.run.num_mc):
        rng = np.random.default_rng([cfg.run.seed, t])
        geom = generate_geometry(small_cfg, rng)
        base = build_base_gains(small_cfg, geom, rng)
        tables = compute_link_tables(small_cfg, base)

        selected_greedy, _ = select_lagrangian(small_cfg, base, tables)
        selected_oracle, obj_oracle = oracle_exhaustive(small_cfg, base, tables)
        obj_greedy = greedy_objective(small_cfg, tables, selected_greedy)

        denom = max(obj_oracle, 1e-9)
        gap = float((obj_oracle - obj_greedy) / denom)
        gaps.append(gap)
        rows.append({
            "experiment": "oracle_gap",
            "trial": t,
            "greedy_obj": obj_greedy,
            "oracle_obj": obj_oracle,
            "gap": gap,
            "greedy_links": int(sum(len(v) for v in selected_greedy.values())),
            "oracle_links": int(sum(len(v) for v in selected_oracle.values())),
        })

    gaps_arr = np.asarray(gaps, dtype=float)
    print(f"oracle gap over {len(gaps_arr)} trials: "
          f"mean={gaps_arr.mean():.4f}, median={np.median(gaps_arr):.4f}, "
          f"max={gaps_arr.max():.4f}, p90={np.percentile(gaps_arr, 90):.4f}")
    return rows


# --------------------------------------------------------------------------
# Runtime benchmark
# --------------------------------------------------------------------------
def runtime(
    cfg: Config,
    small_M: int = 3,
    small_Q: int = 3,
    n_repeat: int = 5,
    include_oracle: bool = True,
) -> List[Dict[str, Any]]:
    """Measure the per-selection wall-clock cost of every selector.

    Uses a single fixed small scenario (oracle is only tractable there) and
    times each selector over ``n_repeat`` calls after a warm-up.  The output
    is one row per method with ``mean_s`` / ``mean_us``.  The greedy-vs-oracle
    gap in *runtime* is the practical payoff of the low-complexity selector,
    complementing the *quality* gap reported by ``oracle-gap``.
    """
    import time

    from .model import build_base_gains, compute_link_tables, generate_geometry
    from .oracle import oracle_exhaustive
    from .selection import (
        select_all_neighbor,
        select_c2f,
        select_lagrangian,
        select_topk_baseline,
    )

    small_cfg = apply_overrides(cfg, {
        "scale.M": int(small_M),
        "scale.Q": int(small_Q),
        "refine.enable": True,
        # Tighten the link budgets so the exhaustive oracle stays tractable
        # (and so the greedy-vs-oracle comparison matches ``oracle-gap``).
        "selector.max_links_per_target": 3,
        "selector.max_total_links": 6,
    })

    _banner(f"Running runtime benchmark (M={small_M}, Q={small_Q}, "
            f"n_repeat={n_repeat})")

    rng = np.random.default_rng([cfg.run.seed, 0])
    geom = generate_geometry(small_cfg, rng)
    base = build_base_gains(small_cfg, geom, rng)
    tables = compute_link_tables(small_cfg, base)

    ref_selected, _ = select_lagrangian(small_cfg, base, tables)
    ref_counts = {q: len(ref_selected.get(q, [])) for q in range(small_cfg.scale.Q)}

    rows: List[Dict[str, Any]] = []

    def bench(name: str, fn, repeats: int = n_repeat) -> None:
        fn()  # warm-up
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        dt = (time.perf_counter() - t0) / repeats
        rows.append({"experiment": "runtime", "method": name,
                     "mean_s": float(dt), "mean_us": float(dt * 1e6)})
        print(f"  {name:24s} {dt * 1e3:10.3f} ms/select")

    bench("proposed_lagrangian", lambda: select_lagrangian(small_cfg, base, tables))
    bench("proposed_c2f", lambda: select_c2f(small_cfg, base, tables)[0])
    bench("topk_deflection",
          lambda: select_topk_baseline(small_cfg, base, tables, ref_counts,
                                       "topk_deflection", np.random.default_rng(1)))
    bench("sense_sinr",
          lambda: select_topk_baseline(small_cfg, base, tables, ref_counts,
                                       "sense_sinr", np.random.default_rng(2)))
    bench("all_neighbor", lambda: select_all_neighbor(small_cfg, base, tables))
    if include_oracle:
        bench("oracle_exhaustive", lambda: oracle_exhaustive(small_cfg, base, tables),
              repeats=min(n_repeat, 3))

    return rows


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
EXPERIMENTS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "lambda-sweep": lambda_sweep,
    "ablation": ablation,
    "fair-ablation": fair_ablation,
    "dd-ablation": dd_ablation,
    "comm-sweep": comm_sweep,
    "robustness": robustness,
    "c2f": c2f,
    "prior-sweep": prior_sweep,
    "waveform-check": waveform_check,
    "oracle-gap": oracle_gap,
    "runtime": runtime,
}

EXPERIMENT_NAMES: List[str] = sorted(EXPERIMENTS)

# Experiments whose sweep grid can be overridden from the command line with
# ``--values``.  The ablation-style modes have fixed variant lists instead.
VALUE_MODES: set = {"lambda-sweep", "comm-sweep", "prior-sweep"}
