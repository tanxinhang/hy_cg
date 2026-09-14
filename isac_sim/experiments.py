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
    "direct_cancellation": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
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
        s = run_simulation(variant, methods=["proposed_lagrangian"])["proposed_lagrangian"]
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
        summary = run_simulation(variant, methods=methods)
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
        summary = run_simulation(variant, methods=SWEEP_METHODS)
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
    (``detect.soft_error_sigma_scale``), ``residual_direct``
    (``radio.residual_direct_factor``, legacy model only) and
    ``direct_cancellation`` (``interference.direct_cancellation_db``; implies
    the coupled interference model).
    """
    if values is None:
        values = ROBUSTNESS_VALUES[axis]
    field_by_axis = {
        "comm_model": "detect.comm_error_model",
        "error_sigma": "detect.soft_error_sigma_scale",
        "residual_direct": "radio.residual_direct_factor",
        "direct_cancellation": "interference.direct_cancellation_db",
    }
    # Axes that only make sense inside a particular model switch on that model
    # themselves, so the sweep is a one-parameter family.
    extra_by_axis: Dict[str, Dict[str, Any]] = {
        "direct_cancellation": {
            "interference.coupling": "shared_spectrum",
            "radio.eps_mode": "noise_relative",
        },
        # ``residual_direct_factor`` only exists in the decoupled model, so this
        # axis must pin it; otherwise the sweep would be a no-op now that the
        # coupled model is the default.
        "residual_direct": {
            "interference.coupling": "legacy",
            "radio.eps_mode": "legacy",
        },
    }
    if axis not in field_by_axis:
        raise ValueError(f"Unknown robustness axis {axis!r}. "
                         f"Choose from {sorted(field_by_axis)}.")

    rows: List[Dict[str, Any]] = []
    for value in values:
        variant = apply_overrides(cfg, {**extra_by_axis.get(axis, {}), field_by_axis[axis]: value})
        _banner(f"Running robustness sweep: {axis} = {value}")
        summary = run_simulation(variant, methods=SWEEP_METHODS)
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


def waveform_detection(
    cfg: Config,
    n_trials: int = 30_000,
) -> List[Dict[str, Any]]:
    """Waveform-derived detector validation under leakage and link failures."""
    from .waveform import waveform_llr_detection_check

    scenarios = (
        {"name": "noise_only", "interference_gamma": 0.0, "report_success": 1.0},
        {"name": "multi_target_leakage", "interference_gamma": 1.0,
         "report_success": 1.0, "num_interferers": 3},
        {"name": "leakage_comm_representative", "interference_gamma": 1.0,
         "report_success": 0.99, "num_interferers": 3},
        {"name": "leakage_comm_stress", "interference_gamma": 1.0,
         "report_success": 0.90, "num_interferers": 3},
        {"name": "full_waveform_impairments", "interference_gamma": 1.0,
         "report_success": 0.99, "num_interferers": 3, "clutter_gamma": 1.0,
         "multipath_ratio": 0.25, "sync_error": (0.15, -0.12)},
    )
    seed_sequence = np.random.SeedSequence(cfg.run.seed)
    rows: List[Dict[str, Any]] = []
    for scenario, child in zip(
        scenarios, seed_sequence.spawn(len(scenarios))
    ):
        kwargs = dict(scenario)
        name = str(kwargs.pop("name"))
        row = waveform_llr_detection_check(
            cfg,
            raw_gamma=0.5,
            n_trials=n_trials,
            rng=np.random.default_rng(child),
            **kwargs,
        )
        rows.append({"experiment": "waveform_detection", "scenario": name, **row})

    _banner(f"Waveform-derived LLR detection (n={n_trials} per scenario)")
    print("scenario                         gamma_eff  legacy_PD  exact_PD  calibrated_PD  calibrated_PFA")
    for row in rows:
        print(
            f"{row['scenario']:<32s} {row['gamma_effective']:.4f}"
            f"      {row['empirical_pd']:.4f}    {row['exact_mixture_pd']:.4f}"
            f"       {row['calibrated_empirical_pd']:.4f}"
            f"         {row['calibrated_empirical_pfa']:.4f}"
        )
    return rows


def waveform_detection_grid(
    cfg: Config,
    n_offsets: int = 24,
    n_trials: int = 5_000,
) -> List[Dict[str, Any]]:
    """Random fractional-DD stress grid for the calibrated detector."""
    from .waveform import waveform_llr_detection_check

    rng = np.random.default_rng(cfg.run.seed)
    raw_gammas = (0.25, 0.5, 1.0)
    interference_gammas = (0.5, 1.0, 2.0)
    report_successes = (1.0, 0.99, 0.90)
    rows: List[Dict[str, Any]] = []
    for idx in range(max(int(n_offsets), 1)):
        target = tuple(rng.uniform(-0.5, 0.5, size=2))
        # Keep the second target within about one DD bin so both resolved and
        # strongly overlapping PSFs occur in the same validation grid.
        interferer = tuple(np.asarray(target) + rng.uniform(-0.9, 0.9, size=2))
        sync_error = tuple(rng.normal(0.0, 0.08, size=2))
        row = waveform_llr_detection_check(
            cfg,
            raw_gamma=raw_gammas[idx % len(raw_gammas)],
            interference_gamma=interference_gammas[(idx // 3) % 3],
            report_success=report_successes[(idx // 9) % 3],
            target_offset=target,
            interferer_offset=interferer,
            num_interferers=1 + (idx % 3),
            clutter_gamma=(0.0, 0.5, 1.0)[idx % 3],
            multipath_ratio=(0.0, 0.15, 0.30)[(idx // 3) % 3],
            sync_error=sync_error,
            n_trials=n_trials,
            rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
        )
        rows.append({"experiment": "waveform_detection_grid", "case": idx, **row})

    pd_error = np.asarray([
        row["calibrated_empirical_pd"] - row["calibrated_exact_pd"] for row in rows
    ])
    pfa_error = np.asarray([
        row["calibrated_empirical_pfa"] - cfg.detect.Pfa_target for row in rows
    ])
    _banner(f"Waveform detection random DD grid ({len(rows)} x {n_trials})")
    print(f"P_D empirical-exact: mean={pd_error.mean():+.4e}, "
          f"RMSE={np.sqrt(np.mean(pd_error**2)):.4e}, max_abs={np.max(np.abs(pd_error)):.4e}")
    print(f"P_FA target error:   mean={pfa_error.mean():+.4e}, "
          f"RMSE={np.sqrt(np.mean(pfa_error**2)):.4e}, max_abs={np.max(np.abs(pfa_error)):.4e}")
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
    from .reporting import assign_fusion_nodes
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
        plan = (
            assign_fusion_nodes(small_cfg, base, tables, geom)
            if small_cfg.fusion.mode.lower() == "explicit" else None
        )
        selected_greedy, _ = select_lagrangian(small_cfg, base, tables, plan)
        selected_oracle, obj_oracle = oracle_exhaustive(small_cfg, base, tables, plan)
        obj_greedy = greedy_objective(small_cfg, tables, selected_greedy, plan, base)

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
def belief_mismatch(cfg: Config, grid: List[Dict[str, float]] | None = None) -> List[Dict[str, Any]]:
    """Truth-vs-belief sweep: P_D degradation under tracker belief mismatch.

    With ``prior.belief_mode`` the scheduler only sees a noisy *belief* of the
    target state while detection runs on the *truth*.  Sweeps the belief error
    amplitude and reports, for the proposed selector and the all-neighbor
    reference, the detection probability and the fraction of selected links
    whose belief-guided search window actually captured the true bin.
    """
    grid = grid if grid is not None else [
        {"belief_sigma_pos_m": 0.0, "belief_sigma_vel_mps": 0.0},
        {"belief_sigma_pos_m": 50.0, "belief_sigma_vel_mps": 5.0},
        {"belief_sigma_pos_m": 150.0, "belief_sigma_vel_mps": 15.0},
        {"belief_sigma_pos_m": 300.0, "belief_sigma_vel_mps": 25.0},
    ]
    methods = ["proposed_lagrangian", "all_neighbor"]
    rows: List[Dict[str, Any]] = []
    for entry in grid:
        sp = float(entry["belief_sigma_pos_m"])
        sv = float(entry["belief_sigma_vel_mps"])
        variant = apply_overrides(cfg, {
            "prior.belief_mode": True,
            "prior.belief_sigma_pos_m": sp,
            "prior.belief_sigma_vel_mps": sv,
        })
        _banner(f"belief mismatch: sigma_pos={sp:g} m, sigma_vel={sv:g} m/s")
        summary = run_simulation(variant, methods=methods)
        for method in methods:
            s = summary.get(method)
            if not s:
                continue
            rows.append({
                "experiment": "belief_mismatch",
                "method": method,
                "belief_sigma_pos_m": sp,
                "belief_sigma_vel_mps": sv,
                "P_D": s["P_D"],
                "P_D_ci95_low": s["P_D_ci95"][0],
                "P_D_ci95_high": s["P_D_ci95"][1],
                "D_mean": s["D_mean"],
                "belief_capture_rate_mean": s["belief_capture_rate_mean"],
                "selected_links_mean": s["selected_links_mean"],
                "T_mean_ms": s["T_mean_ms"],
            })
        p = summary["proposed_lagrangian"]
        print(f"  proposed P_D={p['P_D']:.4f}  capture={p['belief_capture_rate_mean']:.3f}  "
              f"D={p['D_mean']:.2f}")
    return rows


FBL_N_BLOCK_VALUES: List[int] = [256, 512, 1024, 2048, 4096]


def fbl_sweep(cfg: Config, n_block_values: List[int] | None = None) -> List[Dict[str, Any]]:
    """Finite-blocklength sweep: reliability vs latency of the reporting links.

    Varies the report blocklength ``n`` at fixed payload ``k``.  As ``n`` grows
    the per-packet error probability ``eps = Q((C - k/n)/sqrt(V/n))`` falls (so
    ``chi = 1 - eps`` rises and the fusion sees less pollution) but the report
    latency ``n / B`` grows.  Reports the selected-link mean ``chi`` and the
    proposed method's ``P_D`` and latency, exposing the reliability-latency
    trade-off the legacy heuristic model could not express.
    """
    values = n_block_values if n_block_values is not None else FBL_N_BLOCK_VALUES
    rows: List[Dict[str, Any]] = []
    for n_raw in values:
        n = int(n_raw)
        variant = apply_overrides(cfg, {
            "comm.reliability_model": "fbl",
            "comm.n_block": n,
            "comm.latency_model": "blocklength",
        })
        _banner(f"FBL sweep: n_block={n}")
        summary = run_simulation(variant, methods=["proposed_lagrangian"])
        s = summary["proposed_lagrangian"]
        rows.append({
            "experiment": "fbl_sweep",
            "method": "proposed_lagrangian",
            "n_block": int(n),
            "k_bits": float(cfg.comm.K_candidates) * cfg.comm.b_d,
            "P_D": s["P_D"],
            "P_D_ci95_low": s["P_D_ci95"][0],
            "P_D_ci95_high": s["P_D_ci95"][1],
            "selected_chi_mean": s["selected_chi_mean"],
            "selected_chi_min_mean": s["selected_chi_min_mean"],
            "T_mean_ms": s["T_mean_ms"],
            "D_mean": s["D_mean"],
        })
        print(f"  n={n:5d}  chi_mean={s['selected_chi_mean']:.4f}  "
              f"P_D={s['P_D']:.4f}  T={s['T_mean_ms']:.3f} ms")
    return rows


def correlation_ablation(cfg: Config) -> List[Dict[str, Any]]:
    """Correlation-aware vs independence-assuming fusion.

    Compares the proposed selector with the observation-correlation model on
    and off, and records the mean redundancy index (how much of the naive
    deflection was double-counted) plus the resulting ``P_D`` / ``D``.
    """
    rows: List[Dict[str, Any]] = []
    for enable in (False, True):
        variant = apply_overrides(cfg, {"corr.enable": bool(enable)})
        _banner(f"correlation ablation: corr.enable={enable}")
        summary = run_simulation(variant, methods=["proposed_lagrangian"])
        s = summary["proposed_lagrangian"]
        rows.append({
            "experiment": "correlation_ablation",
            "method": "proposed_lagrangian",
            "corr_enable": bool(enable),
            "P_D": s["P_D"],
            "P_D_ci95_low": s["P_D_ci95"][0],
            "P_D_ci95_high": s["P_D_ci95"][1],
            "D_mean": s["D_mean"],
            "selected_links_mean": s["selected_links_mean"],
            "T_mean_ms": s["T_mean_ms"],
        })
        print(f"  corr={enable}: P_D={s['P_D']:.4f}  D_mean={s['D_mean']:.3f}  "
              f"links={s['selected_links_mean']:.1f}")
    return rows


# --------------------------------------------------------------------------
# Communication / sensing interference coupling
# --------------------------------------------------------------------------
INTERFERENCE_CANCEL_DB: List[float] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0]


def _interference_diagnostics(cfg: Config, n_trials: int = 20) -> Dict[str, float]:
    """Measure the interference bookkeeping on shared geometry.

    All quantities are medians over the DD-valid links of ``n_trials`` trials,
    averaged over trials.

    * ``near_far_db``     -- direct-path field over target echo at the sensing
      receiver.  This is the *classical* near-far ratio of bistatic sensing and
      it is exactly the direct-path suppression the receiver must supply for
      the echo to survive.
    * ``sense_inr_db``    -- interference-to-noise ratio at the sensing
      receiver, i.e. ``kappa_dc * field / n0``.
    * ``comm_inr_db``     -- interference-to-noise ratio at the communication
      receiver receiving ``i -> j``, over the same transmitter set.
    * ``sense_sinr_now``  -- SINR the sensing receiver actually gets.
    * ``sense_sinr_no_cancel_db`` -- SINR it would get with *no* direct-path
      cancellation at all (``kappa_dc = 1``).  This is the number that decides
      whether an ISAC design can exist.
    """
    from .model import (build_base_gains, compute_link_tables, denominator_guard,
                        generate_geometry, noise_power)

    M = cfg.scale.M
    R = cfg.radio
    c = cfg.comm
    n0 = noise_power(cfg)
    guard = denominator_guard(cfg, n0)
    kappa = 10.0 ** (-cfg.interference.direct_cancellation_db / 10.0)
    G_proc = cfg.waveform.N * cfg.waveform.L

    keys = ("near_far_db", "sense_inr_db", "comm_inr_db", "sense_sinr_db",
            "raw_sinr_db", "comm_sinr_db", "sense_sinr_no_cancel_db",
            "guard_over_n0_db", "sense_inr_table_db")
    acc: Dict[str, List[float]] = {k: [] for k in keys}

    for t in range(max(1, int(n_trials))):
        rng = np.random.default_rng([cfg.run.seed, t])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tb = compute_link_tables(cfg, base)

        P = np.full(M, R.P_default)
        P_sense, P_comm = R.rho * P, (1.0 - R.rho) * P
        # Every UAV radiates the joint ISAC waveform -> one field serves both.
        field = (P_sense + P_comm) @ base.direct_gain
        pay = P_comm + c.comm_leakage_from_sensing * P_sense
        pay_field = pay @ base.direct_gain

        valid = base.valid_dd & base.edge_mask[..., None]
        i, j, q = np.argwhere(valid).T
        if len(i) == 0:
            continue
        echo = P_sense[i] * base.target_gain[i, j, q] * G_proc
        field_j = field[j]
        comm_i = pay_field[j] - pay[i] * base.direct_gain[i, j]

        acc["near_far_db"].append(float(10 * np.log10(np.median(field_j / echo))))
        acc["sense_inr_db"].append(float(10 * np.log10(kappa * np.median(field_j) / n0)))
        acc["comm_inr_db"].append(float(10 * np.log10(np.median(comm_i) / n0)))
        acc["sense_sinr_no_cancel_db"].append(
            float(10 * np.log10(np.median(echo / (n0 + field_j)))))
        acc["guard_over_n0_db"].append(float(10 * np.log10(guard / n0)))
        # What the table actually used, i.e. the left-hand side of the model's
        # own arithmetic, as a cross-check on the formula above.
        rinr_used = tb.rinr[i, j]
        rinr_used = rinr_used[np.isfinite(rinr_used) & (rinr_used > 0)]
        if len(rinr_used):
            acc["sense_inr_table_db"].append(float(10 * np.log10(np.median(rinr_used))))
        for key, values in (("sense_sinr_db", tb.gamma_sense[i, j, q]),
                            ("raw_sinr_db", tb.raw_gamma_sense[i, j, q]),
                            ("comm_sinr_db", tb.gamma_comm[i, j])):
            vals = np.asarray(values)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                acc[key].append(float(10 * np.log10(np.median(vals))))

    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}
    out["inr_ratio_db"] = out["comm_inr_db"] - out["sense_inr_db"]
    return out


def interference_consistency(
    cfg: Config,
    cancel_db_values: List[float] | None = None,
) -> List[Dict[str, Any]]:
    """Audit whether communication and sensing see the same interference.

    Part 1 compares four bookkeeping variants at one operating point: the legacy
    decoupled residual floors, the legacy floors with the SINR guard moved below
    the noise floor, the coupled shared-spectrum field, and the coupled field
    with the corrected guard.  For each, the measured communication and sensing
    interference-to-noise ratios are recorded next to ``P_D``.

    Part 2 sweeps the direct-path cancellation ``kappa_dc`` of the coupled model.
    Since the near-far ratio is 30-40 dB, this sweep locates the suppression the
    sensing receiver must achieve for the ISAC task to survive at all -- the
    resource-sharing cost that the decoupled model made disappear.
    """
    rows: List[Dict[str, Any]] = []
    values = list(cancel_db_values) if cancel_db_values else list(INTERFERENCE_CANCEL_DB)

    def _row_extra(variant: Config, group: str, tag: str, diag: Dict[str, float]) -> Dict[str, Any]:
        """Bookkeeping columns, including the operating point.

        The interference model, the coupling and the guard mode are recorded on
        every row: two runs of this experiment under different operating points
        must never be silently comparable.
        """
        return {
            "experiment": "interference_consistency",
            "group": group,
            "variant": tag,
            "interference_model": variant.comm.interference_model,
            "coupling": variant.interference.coupling,
            "eps_mode": variant.radio.eps_mode,
            "sense_gate_by_active_tx": variant.interference.sense_gate_by_active_tx,
            "direct_cancellation_db": variant.interference.direct_cancellation_db,
            **diag,
        }

    # ---- Part 1: bookkeeping variants -----------------------------------
    # Every variant sets BOTH switches explicitly.  Inheriting them from the
    # caller's config would make the labels lie whenever the run is launched
    # with, say, ``--set radio.eps_mode=noise_relative``: the row labelled
    # "legacy" would then not be legacy at all.
    variants = [
        ("legacy", {"interference.coupling": "legacy",
                    "radio.eps_mode": "legacy"}),
        ("legacy+guard", {"interference.coupling": "legacy",
                          "radio.eps_mode": "noise_relative"}),
        ("coupled", {"interference.coupling": "shared_spectrum",
                     "radio.eps_mode": "legacy"}),
        ("coupled+guard", {"interference.coupling": "shared_spectrum",
                           "radio.eps_mode": "noise_relative"}),
    ]
    _banner("interference bookkeeping variants")
    for tag, overrides in variants:
        variant = apply_overrides(cfg, overrides)
        diag = _interference_diagnostics(variant)
        summary = run_simulation(variant, methods=["proposed_lagrangian", "all_neighbor"])
        for method in ("proposed_lagrangian", "all_neighbor"):
            rows.append(scalar_summary_row(
                summary, method, _row_extra(variant, "bookkeeping", tag, diag)))
        sp = summary["proposed_lagrangian"]
        print(f"  {tag:14s} near-far={diag['near_far_db']:6.1f} dB | "
              f"I_comm/N0={diag['comm_inr_db']:6.1f} dB  I_sense/N0={diag['sense_inr_db']:6.1f} dB | "
              f"gamma^s={diag['sense_sinr_db']:6.1f} dB | P_D={sp['P_D']:.4f}")

    # ---- Part 2: direct-path cancellation sweep -------------------------
    base_cfg = apply_overrides(cfg, {
        "interference.coupling": "shared_spectrum",
        "radio.eps_mode": "noise_relative",
    })
    _banner("direct-path cancellation sweep (coupled model)")
    for db in values:
        variant = apply_overrides(base_cfg, {"interference.direct_cancellation_db": float(db)})
        diag = _interference_diagnostics(variant)
        summary = run_simulation(variant, methods=["proposed_lagrangian", "all_neighbor"])
        for method in ("proposed_lagrangian", "all_neighbor"):
            rows.append(scalar_summary_row(
                summary, method,
                _row_extra(variant, "cancellation_sweep", f"kappa_dc_{db:g}dB", diag)))
        sp = summary["proposed_lagrangian"]
        print(f"  cancel={db:5.1f} dB  I_sense/N0={diag['sense_inr_db']:6.1f} dB  "
              f"gamma^s={diag['sense_sinr_db']:7.1f} dB  P_D={sp['P_D']:.4f}  "
              f"D={sp['D_mean']:8.2f}  T={sp['T_mean_ms']:7.2f} ms")
    return rows


def submodularity(
    cfg: Config,
    small_M: int = 4,
    small_Q: int = 3,
    n_samples: int = 800,
) -> List[Dict[str, Any]]:
    """Finite-instance diminishing-returns and curvature audit.

    The audit evaluates the implemented fair utility and reporting cost.  It is
    evidence about the tested instances, not a universal submodularity proof.
    """
    from .model import build_base_gains, compute_link_tables, generate_geometry
    from .reporting import assign_fusion_nodes
    from .theory import curvature_reference_bound, submodularity_audit

    small_cfg = apply_overrides(cfg, {
        "scale.M": int(small_M),
        "scale.Q": int(small_Q),
        "selector.max_links_per_target": 3,
        "selector.max_total_links": 8,
        "selector.candidate_topk_per_target": 30,
    })
    _banner(f"submodularity audit (M={small_M}, Q={small_Q}, n_samples={n_samples})")
    rows: List[Dict[str, Any]] = []
    agg = {
        "monotone_violation_rate": 0.0,
        "submodularity_violation_rate": 0.0,
        "min_marginal_ratio": 1.0,
        "curvature": 0.0,
    }
    n_trials = max(1, min(cfg.run.num_mc, 10))
    for t in range(n_trials):
        rng = np.random.default_rng([cfg.run.seed, t])
        geom = generate_geometry(small_cfg, rng)
        base = build_base_gains(small_cfg, geom, rng)
        tables = compute_link_tables(small_cfg, base)
        plan = (
            assign_fusion_nodes(small_cfg, base, tables, geom)
            if small_cfg.fusion.mode.lower() == "explicit" else None
        )
        audit = submodularity_audit(
            small_cfg, base, tables, plan=plan, n_samples=n_samples, seed=t
        )
        for k in agg:
            agg[k] += float(audit.get(k, 0.0)) / n_trials
    c = float(np.clip(agg["curvature"], 0.0, 1.0))

    rows.append({
        "experiment": "submodularity",
        "method": "proposed_lagrangian",
        "M": int(small_M),
        "Q": int(small_Q),
        "n_samples": int(n_samples),
        "monotone_violation_rate": agg["monotone_violation_rate"],
        "submodularity_violation_rate": agg["submodularity_violation_rate"],
        "min_marginal_ratio": agg["min_marginal_ratio"],
        "curvature": c,
        "matroid_reference_bound": float(curvature_reference_bound(c)),
        "guarantee_applicable": False,
    })
    print(f"  monotone violations={agg['monotone_violation_rate']:.4f}  "
          f"submodularity violations={agg['submodularity_violation_rate']:.4f}")
    print(f"  empirical curvature={c:.4f}  "
          f"matroid reference={curvature_reference_bound(c):.4f} (not a guarantee)")
    return rows


def same_objective_gap(
    cfg: Config,
    small_M: int = 3,
    small_Q: int = 3,
    max_per_target: int = 3,
    max_total: int = 6,
) -> List[Dict[str, Any]]:
    """Greedy-vs-oracle gap on the *same* objective the greedy optimises.

    Unlike ``oracle-gap`` (which maximised ``sum_q D_q``), the oracle here
    maximises the fair sensing potential minus reporting cost, exactly matching
    the paper-canonical marginal rule.
    """
    from .model import build_base_gains, compute_link_tables, generate_geometry
    from .selection import select_lagrangian
    from .theory import same_objective_oracle, task_objective
    from .reporting import assign_fusion_nodes

    small_cfg = apply_overrides(cfg, {
        "scale.M": int(small_M),
        "scale.Q": int(small_Q),
        "selector.max_links_per_target": int(max_per_target),
        "selector.max_total_links": int(max_total),
        "selector.candidate_topk_per_target": 30,
    })
    _banner(f"same-objective gap (M={small_M}, Q={small_Q})")
    rows: List[Dict[str, Any]] = []
    gaps: List[float] = []
    for t in range(cfg.run.num_mc):
        rng = np.random.default_rng([cfg.run.seed, t])
        geom = generate_geometry(small_cfg, rng)
        base = build_base_gains(small_cfg, geom, rng)
        tables = compute_link_tables(small_cfg, base)
        plan = (
            assign_fusion_nodes(small_cfg, base, tables, geom)
            if small_cfg.fusion.mode.lower() == "explicit" else None
        )
        selected_greedy, _ = select_lagrangian(small_cfg, base, tables, plan)
        selected_oracle, obj_oracle = same_objective_oracle(small_cfg, base, tables, plan)
        obj_greedy = task_objective(small_cfg, tables, selected_greedy, plan, base)
        gap = float((obj_oracle - obj_greedy) / max(obj_oracle, 1e-9))
        gaps.append(gap)
        rows.append({
            "experiment": "same_objective_gap",
            "trial": t,
            "greedy_obj": obj_greedy,
            "oracle_obj": obj_oracle,
            "gap": gap,
            "greedy_links": int(sum(len(v) for v in selected_greedy.values())),
            "oracle_links": int(sum(len(v) for v in selected_oracle.values())),
        })
    gaps_arr = np.asarray(gaps, dtype=float)
    print(f"same-objective gap: mean={gaps_arr.mean():.4f}, median={np.median(gaps_arr):.4f}, "
          f"max={gaps_arr.max():.4f}")
    return rows


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
    "waveform-detection": waveform_detection,
    "waveform-detection-grid": waveform_detection_grid,
    "oracle-gap": oracle_gap,
    "runtime": runtime,
    "belief-mismatch": belief_mismatch,
    "fbl-sweep": fbl_sweep,
    "correlation-ablation": correlation_ablation,
    "submodularity": submodularity,
    "same-objective-gap": same_objective_gap,
    "interference-consistency": interference_consistency,
}

EXPERIMENT_NAMES: List[str] = sorted(EXPERIMENTS)

# Experiments whose sweep grid can be overridden from the command line with
# ``--values``.  The ablation-style modes have fixed variant lists instead.
VALUE_MODES: set = {"lambda-sweep", "comm-sweep", "prior-sweep", "fbl-sweep",
                    "interference-consistency"}
