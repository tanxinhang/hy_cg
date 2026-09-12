from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import EPS, SimConfig, parse_float_list
from .scenario import make_rng
from .assignment import (
    assignment_collision_aware_greedy,
    assignment_global_utility,
    assignment_nearest,
    assignment_random,
    assignment_strongest,
    build_collision_edges,
    build_visibility,
    task_edge_list,
)
from .kernels import (
    _component_from_power_and_template,
    angle_overlap,
    bistatic_gain,
    dd_overlap_full,
    observation_signal,
    path_dd_center,
    stable_method_offset,
    support_signal_vector_full,
)
from .detection import (
    _coherent_signal_vector,
    _detect_coherent_matched_components,
    _detect_energy_components,
    _detect_noncoherent_matched_components,
    _expected_noncoherent_signal_power,
)
from .experiments import (
    run_gate1,
    run_gate2,
    run_tracking_experiment,
    _gate1_method_records_for_scenario,
    _make_gate1_jittered_scenario,
)


def _copy_config(cfg: SimConfig) -> SimConfig:
    return SimConfig(**asdict(cfg))


def _write_runtime_row(records: List[Dict[str, float]], name: str, start_s: float, status: str = "ok") -> None:
    records.append(
        dict(
            step=name,
            status=status,
            elapsed_s=float(perf_counter() - start_s),
        )
    )


def _primary_pd(cfg: SimConfig, rng: np.random.Generator, components: list, sigma2_total: float) -> float:
    mode = str(cfg.detector_mode).lower()
    if mode == "energy":
        return _detect_energy_components(cfg, rng, components, sigma2_total)
    if mode in ("matched", "coherent", "coherent_matched"):
        return _detect_coherent_matched_components(cfg, rng, components, sigma2_total)
    return _detect_noncoherent_matched_components(cfg, rng, components, sigma2_total)


def _task_pd_loss_records(
    cfg: SimConfig,
    sc,
    z: np.ndarray,
    method: str,
    mc: int,
    seed_base: int,
) -> List[Dict[str, float]]:
    tasks = task_edge_list(z)
    if not tasks:
        return []
    proxy_edges = build_collision_edges(cfg, sc, z, use_full_kernel=False)
    full_edges = build_collision_edges(cfg, sc, z, use_full_kernel=True)
    rows: List[Dict[str, float]] = []
    for task_idx, (j, q) in enumerate(tasks):
        k_prot, l_prot, *_ = path_dd_center(cfg, sc, j, j, q)
        components = []
        s_self = observation_signal(cfg, sc, j, q)
        tmpl_self = support_signal_vector_full(cfg, k_prot, l_prot, k_prot, l_prot)
        components.append(_component_from_power_and_template(s_self, tmpl_self, True))

        for i, qi in tasks:
            if qi == q and i != j:
                k_src, l_src, *_ = path_dd_center(cfg, sc, i, j, q)
                tmpl = support_signal_vector_full(cfg, k_src, l_src, k_prot, l_prot)
                power = 0.8 * bistatic_gain(cfg, sc, i, j, q)
                components.append(_component_from_power_and_template(power, tmpl, False))

        if str(cfg.coop_phase_mode).lower() == "coherent":
            s_eff = float(np.sum(np.abs(_coherent_signal_vector(components)) ** 2))
        else:
            s_eff = _expected_noncoherent_signal_power(components)

        i_full = 0.0
        for i, qp in tasks:
            if qp == q:
                continue
            k_src, l_src, *_ = path_dd_center(cfg, sc, i, j, qp)
            ov = dd_overlap_full(cfg, k_src, l_src, k_prot, l_prot)
            ang = angle_overlap(cfg, sc, j, qp, q)
            i_full += bistatic_gain(cfg, sc, i, j, qp) * ov * ang

        if len(proxy_edges):
            mask = (proxy_edges["rx_uav"] == j) & (proxy_edges["protected_target"] == q)
            cbar_proxy = float(proxy_edges.loc[mask, "C_norm"].sum())
            c_proxy = float(proxy_edges.loc[mask, "C"].sum())
        else:
            cbar_proxy = 0.0
            c_proxy = 0.0
        if len(full_edges):
            mask = (full_edges["rx_uav"] == j) & (full_edges["protected_target"] == q)
            cbar_full = float(full_edges.loc[mask, "C_norm"].sum())
        else:
            cbar_full = 0.0

        # Common random numbers reduce Monte-Carlo noise in the paired loss.
        seed = int(seed_base + 1000 * mc + 31 * task_idx + 7 * j + q)
        pd_clean = _primary_pd(cfg, make_rng(seed), components, cfg.snr_floor)
        pd_collision = _primary_pd(cfg, make_rng(seed), components, i_full + cfg.snr_floor)
        rows.append(
            dict(
                mc=mc,
                method=method,
                rx_uav=int(j),
                target=int(q),
                cbar_proxy=cbar_proxy,
                c_proxy=c_proxy,
                cbar_full=cbar_full,
                full_interference=i_full,
                s_eff=float(s_eff),
                pd_clean=float(pd_clean),
                pd_collision=float(pd_collision),
                delta_pd=float(pd_clean - pd_collision),
                scnr_clean_db=float(10.0 * np.log10(s_eff / (cfg.snr_floor + EPS) + EPS)),
                scnr_collision_db=float(10.0 * np.log10(s_eff / (i_full + cfg.snr_floor + EPS) + EPS)),
            )
        )
    return rows


def _rankdata(x: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy(dtype=float)


def _corr_pair(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    if kind == "spearman":
        x = _rankdata(x)
        y = _rankdata(y)
    if float(np.std(x)) <= EPS or float(np.std(y)) <= EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _partial_corr_pair(x: np.ndarray, y: np.ndarray, z: np.ndarray, kind: str) -> float:
    """Correlation between x and y after regressing out one confounder z."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    if x.size < 4 or float(np.std(z)) <= EPS:
        return float("nan")
    if kind == "spearman":
        x = _rankdata(x)
        y = _rankdata(y)
        z = _rankdata(z)
    z_design = np.column_stack([np.ones_like(z), z])
    bx = np.linalg.lstsq(z_design, x, rcond=None)[0]
    by = np.linalg.lstsq(z_design, y, rcond=None)[0]
    rx = x - z_design @ bx
    ry = y - z_design @ by
    return _corr_pair(rx, ry, "pearson")


def _bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, kind: str, rng: np.random.Generator, n_boot: int) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    point = _corr_pair(x, y, kind)
    if n < 3 or not np.isfinite(point):
        return dict(n_points=n, corr=point, ci95_low=np.nan, ci95_high=np.nan)
    vals = []
    for _ in range(max(int(n_boot), 1)):
        idx = rng.integers(0, n, size=n)
        val = _corr_pair(x[idx], y[idx], kind)
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return dict(n_points=n, corr=point, ci95_low=np.nan, ci95_high=np.nan)
    arr = np.asarray(vals, dtype=float)
    return dict(
        n_points=n,
        corr=point,
        ci95_low=float(np.percentile(arr, 2.5)),
        ci95_high=float(np.percentile(arr, 97.5)),
    )


def _bootstrap_partial_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    kind: str,
    rng: np.random.Generator,
    n_boot: int,
) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    n = int(x.size)
    point = _partial_corr_pair(x, y, z, kind)
    if n < 4 or not np.isfinite(point):
        return dict(n_points=n, corr=point, ci95_low=np.nan, ci95_high=np.nan)
    vals = []
    for _ in range(max(int(n_boot), 1)):
        idx = rng.integers(0, n, size=n)
        val = _partial_corr_pair(x[idx], y[idx], z[idx], kind)
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return dict(n_points=n, corr=point, ci95_low=np.nan, ci95_high=np.nan)
    arr = np.asarray(vals, dtype=float)
    return dict(
        n_points=n,
        corr=point,
        ci95_low=float(np.percentile(arr, 2.5)),
        ci95_high=float(np.percentile(arr, 97.5)),
    )


def _write_cbar_binned_diagnostics(df: pd.DataFrame, out: Path, xcol: str = "cbar_proxy", bins: int = 10) -> None:
    if df.empty or xcol not in df.columns:
        pd.DataFrame().to_csv(out / "cbar_delta_pd_binned.csv", index=False)
        return
    x = np.log10(df[xcol].to_numpy(dtype=float) + EPS)
    finite = np.isfinite(x) & np.isfinite(df["delta_pd"].to_numpy(dtype=float))
    if int(np.sum(finite)) < 3:
        pd.DataFrame().to_csv(out / "cbar_delta_pd_binned.csv", index=False)
        return
    work = df.loc[finite].copy()
    work["log_cbar"] = x[finite]
    n_bins = max(2, min(int(bins), int(work.shape[0])))
    try:
        work["bin"] = pd.qcut(work["log_cbar"], q=n_bins, duplicates="drop")
    except ValueError:
        work["bin"] = pd.cut(work["log_cbar"], bins=n_bins, duplicates="drop")
    binned = work.groupby("bin", observed=True).agg(
        n=("delta_pd", "size"),
        log_cbar_mean=("log_cbar", "mean"),
        log_cbar_min=("log_cbar", "min"),
        log_cbar_max=("log_cbar", "max"),
        delta_pd_mean=("delta_pd", "mean"),
        delta_pd_median=("delta_pd", "median"),
        delta_pd_p10=("delta_pd", lambda s: float(np.percentile(s, 10))),
        delta_pd_p90=("delta_pd", lambda s: float(np.percentile(s, 90))),
        scnr_clean_db_mean=("scnr_clean_db", "mean"),
    ).reset_index()
    means = binned["delta_pd_mean"].to_numpy(dtype=float)
    binned["monotone_non_decreasing_delta_mean"] = bool(np.all(np.diff(means) >= -1e-12)) if means.size > 1 else True
    binned.to_csv(out / "cbar_delta_pd_binned.csv", index=False)


def run_cbar_pd_correlation(cfg: SimConfig, out: Path) -> None:
    """Validate that the optimization proxy predicts actual detection loss."""
    methods = ["random", "nearest", "strongest", "collision_aware"]
    records: List[Dict[str, float]] = []
    for mc in range(int(cfg.correlation_mc)):
        rng = make_rng(cfg.seed + 92000 + mc)
        sc = _make_gate1_jittered_scenario(cfg, rng)
        _, u, v = build_visibility(cfg, sc)
        assigners = {
            "random": lambda: assignment_random(cfg, u, v, rng),
            "nearest": lambda: assignment_nearest(cfg, sc, v),
            "strongest": lambda: assignment_strongest(cfg, u, v),
            "collision_aware": lambda: assignment_collision_aware_greedy(cfg, sc, u, v),
        }
        for method in methods:
            z = assigners[method]()
            records.extend(_task_pd_loss_records(cfg, sc, z, method, mc, cfg.seed + 92500))

    df = pd.DataFrame(records)
    df.to_csv(out / "cbar_delta_pd_trials.csv", index=False)
    if df.empty:
        pd.DataFrame().to_csv(out / "cbar_delta_pd_correlation.csv", index=False)
        return

    rng = make_rng(cfg.seed + 92900)
    rows = []
    for xcol in ["cbar_proxy", "cbar_full"]:
        for ycol in ["delta_pd"]:
            x = np.log10(df[xcol].to_numpy(dtype=float) + EPS)
            y = df[ycol].to_numpy(dtype=float)
            z_snr = df["scnr_clean_db"].to_numpy(dtype=float)
            for kind in ["pearson", "spearman"]:
                stats = _bootstrap_corr_ci(x, y, kind, rng, int(cfg.bootstrap_iters))
                rows.append(dict(x=f"log10({xcol})", y=ycol, control="", corr_type=kind, **stats))
                pstats = _bootstrap_partial_corr_ci(x, y, z_snr, kind, rng, int(cfg.bootstrap_iters))
                rows.append(dict(x=f"log10({xcol})", y=ycol, control="scnr_clean_db", corr_type=f"partial_{kind}", **pstats))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "cbar_delta_pd_correlation.csv", index=False)

    near_zero = df["cbar_proxy"].to_numpy(dtype=float) <= 1e-10
    positive_loss = df["delta_pd"].to_numpy(dtype=float) > 0.02
    diagnostics = pd.DataFrame([dict(
        n_points=int(len(df)),
        n_proxy_near_zero=int(np.sum(near_zero)),
        n_proxy_near_zero_positive_loss=int(np.sum(near_zero & positive_loss)),
        frac_proxy_near_zero_positive_loss=float(np.mean(near_zero & positive_loss)) if len(df) else 0.0,
        proxy_near_zero_threshold=1e-10,
        positive_loss_threshold=0.02,
    )])
    diagnostics.to_csv(out / "cbar_delta_pd_diagnostics.csv", index=False)
    _write_cbar_binned_diagnostics(df, out, xcol="cbar_proxy", bins=10)

    plt.figure(figsize=(6.2, 4.6))
    for method in methods:
        sub = df[df.method == method]
        if len(sub):
            plt.scatter(np.log10(sub["cbar_proxy"].to_numpy(dtype=float) + EPS), sub["delta_pd"], s=14, alpha=0.55, label=method)
    plt.xlabel(r"$\log_{10}(\bar C_{\mathrm{proxy}})$")
    plt.ylabel(r"$\Delta P_D = P_D(\mathrm{clean}) - P_D(\mathrm{collision})$")
    plt.title("Proxy collision predicts full-link detection loss")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "cbar_delta_pd_scatter.png", dpi=cfg.dpi)
    plt.close()

    print("\n[Paper] Cbar-vs-Delta-Pd correlation")
    print(summary.to_string(index=False))
    print(f"Saved: {out / 'cbar_delta_pd_correlation.csv'}")


def _paper_baseline_assignments(cfg: SimConfig, sc, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Main-paper baseline set aligned with related-work comparisons."""
    _, u, v = build_visibility(cfg, sc)
    out: Dict[str, np.ndarray] = {}
    out["random"] = assignment_random(cfg, u, v, rng)
    out["global_utility"] = assignment_global_utility(cfg, u, v)

    # Box-DD: collision-aware assignment with a hard in-window DD overlap proxy.
    box_cfg = _copy_config(cfg)
    box_cfg.proxy_kernel_mode = "no_spread"
    out["box_dd"] = assignment_collision_aware_greedy(box_cfg, sc, u, v)

    # DD-only: keep Dirichlet DD leakage, but disable angular discrimination in
    # the assignment proxy.  Evaluation still uses the full cfg.
    dd_only_cfg = _copy_config(cfg)
    dd_only_cfg.sigma_theta_deg = float("inf")
    out["dd_only"] = assignment_collision_aware_greedy(dd_only_cfg, sc, u, v)

    out["proposed"] = assignment_collision_aware_greedy(cfg, sc, u, v)
    return out


def run_paper_baseline_suite(cfg: SimConfig, out: Path) -> None:
    """Run the compact main-paper baseline suite.

    Baselines:
      random         : no geometry/resource optimization.
      global_utility : utility-only assignment, no collision coupling.
      box_dd         : collision-aware, but with hard box DD overlap.
      dd_only        : Dirichlet DD collision-aware, no angle overlap.
      proposed       : Dirichlet DD-angle collision-aware assignment.
    """
    records: List[Dict[str, float]] = []
    for mc in range(int(cfg.gate1_mc)):
        rng = make_rng(cfg.seed + 95000 + mc)
        sc = _make_gate1_jittered_scenario(cfg, rng)
        _, u, v = build_visibility(cfg, sc)
        z_by_method = _paper_baseline_assignments(cfg, sc, rng)
        for method, z in z_by_method.items():
            # Inline the Gate-1 row construction so assignment can be custom.
            from .assignment import collision_summary, is_feasible, objective
            from .detection import evaluate_assignment_full_link

            eval_rng = make_rng(cfg.seed + 95500 + 1000 * mc + stable_method_offset(method))
            full = evaluate_assignment_full_link(cfg, sc, z, eval_rng)
            edges_proxy = build_collision_edges(cfg, sc, z, use_full_kernel=False)
            summ = collision_summary(cfg, edges_proxy, len(task_edge_list(z)))
            row = dict(
                mc=mc,
                method=method,
                feasible=is_feasible(cfg, z, v),
                objective_proxy=objective(cfg, sc, z, u, use_full_kernel=False),
                visibility_edges=int(v.sum()),
                coverage_ratio=float(np.mean(z.sum(axis=0) >= cfg.k_tgt_min)),
                avg_tasks_per_uav=float(z.sum(axis=1).mean()),
                num_task_edges=int(len(task_edge_list(z))),
            )
            row.update(summ)
            row.update(full)
            records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(out / "paper_baseline_trials.csv", index=False)
    if df.empty:
        pd.DataFrame().to_csv(out / "paper_baseline_summary.csv", index=False)
        return
    summary = df.groupby("method").agg(
        pd_mean=("pd_mean", "mean"),
        pd_p05=("pd_p05", "mean"),
        pd_min=("pd_min", "mean"),
        full_cross_norm_mean=("full_cross_norm_mean", "mean"),
        proxy_cross_norm=("proxy_cross_norm", "mean"),
        collision_cost_norm=("collision_cost_norm", "mean"),
        full_scnr_mean=("full_scnr_mean", "mean"),
        coverage_ratio=("coverage_ratio", "mean"),
        num_task_edges=("num_task_edges", "mean"),
        objective_proxy=("objective_proxy", "mean"),
    ).reset_index()
    order = ["random", "global_utility", "box_dd", "dd_only", "proposed"]
    summary["method_order"] = summary["method"].map({m: i for i, m in enumerate(order)})
    summary = summary.sort_values("method_order").drop(columns=["method_order"])
    summary.to_csv(out / "paper_baseline_summary.csv", index=False)
    print("\n[Paper] Main baseline suite")
    print(summary.to_string(index=False))
    print(f"Saved: {out / 'paper_baseline_summary.csv'}")


def run_lambda_sweep(cfg: SimConfig, out: Path) -> None:
    """Paper sensitivity sweep over the collision penalty lambda_col.

    The sweep keeps the Gate-1 scenario, cardinality constraints, detector, and
    OTFS evaluation kernel unchanged while varying only the collision penalty in
    the collision-aware assignment objective.
    """
    records: List[Dict[str, float]] = []
    lambdas = parse_float_list(cfg.lambda_col_list)
    methods = ["nearest", "strongest", "collision_aware"]
    for lam in lambdas:
        local = _copy_config(cfg)
        local.lambda_col = float(lam)
        for mc in range(int(cfg.lambda_mc)):
            rng = make_rng(cfg.seed + 93000 + 1000 * int(round(1000 * lam)) + mc)
            sc = _make_gate1_jittered_scenario(local, rng)
            rows = _gate1_method_records_for_scenario(local, sc, rng, mc, methods=methods)
            for row in rows:
                row["lambda_col"] = float(lam)
                records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(out / "lambda_sweep_trials.csv", index=False)
    if df.empty:
        pd.DataFrame().to_csv(out / "lambda_sweep_summary.csv", index=False)
        return

    summary = df.groupby(["lambda_col", "method"]).agg(
        pd_mean=("pd_mean", "mean"),
        pd_p05=("pd_p05", "mean"),
        full_cross_norm_mean=("full_cross_norm_mean", "mean"),
        proxy_cross_norm=("proxy_cross_norm", "mean"),
        collision_cost_norm=("collision_cost_norm", "mean"),
        objective_proxy=("objective_proxy", "mean"),
        coverage_ratio=("coverage_ratio", "mean"),
        num_task_edges=("num_task_edges", "mean"),
    ).reset_index()
    summary.to_csv(out / "lambda_sweep_summary.csv", index=False)
    print("\n[Paper] Lambda sensitivity summary")
    print(summary.to_string(index=False))
    print(f"Saved: {out / 'lambda_sweep_summary.csv'}")


def _bootstrap_ci(delta: np.ndarray, rng: np.random.Generator, n_boot: int) -> Dict[str, float]:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    n = int(delta.size)
    if n == 0:
        return dict(n_pairs=0, delta_mean=np.nan, ci95_low=np.nan, ci95_high=np.nan, prob_gt0=np.nan)
    if n == 1:
        val = float(delta[0])
        return dict(n_pairs=1, delta_mean=val, ci95_low=val, ci95_high=val, prob_gt0=float(val > 0))
    idx = rng.integers(0, n, size=(max(int(n_boot), 1), n))
    boot = delta[idx].mean(axis=1)
    return dict(
        n_pairs=n,
        delta_mean=float(np.mean(delta)),
        ci95_low=float(np.percentile(boot, 2.5)),
        ci95_high=float(np.percentile(boot, 97.5)),
        prob_gt0=float(np.mean(boot > 0.0)),
    )


def run_gate1_paired_bootstrap(cfg: SimConfig, out: Path, trials_path: Optional[Path] = None) -> None:
    """Paired bootstrap CIs for Gate-1 method differences.

    Pairing is by MC scenario index, so the confidence intervals quantify the
    within-scenario improvement of collision-aware assignment over each baseline.
    """
    path = trials_path or (out / "gate1_trials.csv")
    if not path.exists():
        print(f"[Paper] Skip paired bootstrap; missing {path}")
        return
    df = pd.read_csv(path)
    if df.empty or "collision_aware" not in set(df.get("method", [])):
        print("[Paper] Skip paired bootstrap; gate1_trials.csv has no collision_aware rows")
        return

    metrics = [
        "pd_mean",
        "pd_p05",
        "full_scnr_mean",
        "full_cross_norm_mean",
        "proxy_cross_norm",
        "collision_cost_norm",
    ]
    metrics = [m for m in metrics if m in df.columns]
    baselines = [m for m in sorted(df["method"].dropna().unique()) if m != "collision_aware"]
    rng = make_rng(cfg.seed + 94000)
    rows: List[Dict[str, float]] = []
    for metric in metrics:
        pivot = df.pivot_table(index="mc", columns="method", values=metric, aggfunc="mean")
        if "collision_aware" not in pivot.columns:
            continue
        for baseline in baselines:
            if baseline not in pivot.columns:
                continue
            paired = pivot[["collision_aware", baseline]].dropna()
            delta = paired["collision_aware"].to_numpy() - paired[baseline].to_numpy()
            stats = _bootstrap_ci(delta, rng, int(cfg.bootstrap_iters))
            rows.append(dict(metric=metric, baseline=baseline, comparison="collision_aware_minus_baseline", **stats))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out / "paper_gate1_paired_ci.csv", index=False)
    print("\n[Paper] Gate-1 paired bootstrap CIs")
    print(out_df.to_string(index=False))
    print(f"Saved: {out / 'paper_gate1_paired_ci.csv'}")


def write_paper_tables(cfg: SimConfig, out: Path) -> None:
    """Create compact CSV tables and an output index for manuscript assembly."""
    table_rows: List[Dict[str, object]] = []

    def add_index(name: str, path: Path, role: str) -> None:
        table_rows.append(dict(name=name, path=str(path.resolve()), exists=path.exists(), role=role))

    gate1 = out / "gate1_summary.csv"
    if gate1.exists():
        df = pd.read_csv(gate1)
        cols = [
            c for c in [
                "method",
                "pd_mean_mean",
                "pd_mean_std",
                "pd_p05_mean",
                "full_cross_norm_mean_mean",
                "proxy_cross_norm_mean",
                "collision_cost_norm_mean",
                "coverage_ratio_mean",
                "num_task_edges_mean",
            ] if c in df.columns
        ]
        df[cols].to_csv(out / "paper_table_gate1.csv", index=False)
        add_index("paper_table_gate1", out / "paper_table_gate1.csv", "Main detection/collision comparison")

    gate2 = out / "gate2_regime_summary.csv"
    if gate2.exists():
        df = pd.read_csv(gate2)
        compact = df.agg(
            {
                "rho_col_mean": "mean",
                "eta_top_mean": "mean",
                "coverage_mean": "mean",
                "mean_C_norm": "mean",
                "p95_C_norm": "mean",
                "sparse_collision_frac": "mean",
                "angle_removed_frac_mean": "mean",
            }
        ).to_frame().T
        compact.to_csv(out / "paper_table_gate2_overall.csv", index=False)
        add_index("paper_table_gate2_overall", out / "paper_table_gate2_overall.csv", "Overall collision-regime statistics")
    gate2_strength = out / "gate2_strength_tail_summary.csv"
    if gate2_strength.exists():
        add_index("gate2_strength_tail_summary", gate2_strength, "Collision-strength skew and angle pseudo-collision diagnostics")
    gate2_hill = out / "gate2_pooled_hill_summary.csv"
    if gate2_hill.exists():
        add_index("gate2_pooled_hill_summary", gate2_hill, "Pooled per-edge Hill tail-index diagnostics across cutoffs")

    lambda_summary = out / "lambda_sweep_summary.csv"
    if lambda_summary.exists():
        add_index("lambda_sweep_summary", lambda_summary, "Collision-penalty sensitivity")
    baseline_summary = out / "paper_baseline_summary.csv"
    if baseline_summary.exists():
        add_index("paper_baseline_summary", baseline_summary, "Main manuscript baseline comparison")
    baseline_trials = out / "paper_baseline_trials.csv"
    if baseline_trials.exists():
        add_index("paper_baseline_trials", baseline_trials, "Per-trial main baseline comparison")

    paired_ci = out / "paper_gate1_paired_ci.csv"
    if paired_ci.exists():
        add_index("paper_gate1_paired_ci", paired_ci, "Paired bootstrap confidence intervals")

    cbar_corr = out / "cbar_delta_pd_correlation.csv"
    if cbar_corr.exists():
        add_index("cbar_delta_pd_correlation", cbar_corr, "Proxy collision versus detection-loss correlation")
    cbar_trials = out / "cbar_delta_pd_trials.csv"
    if cbar_trials.exists():
        add_index("cbar_delta_pd_trials", cbar_trials, "Per-task proxy collision and detection loss")
    cbar_binned = out / "cbar_delta_pd_binned.csv"
    if cbar_binned.exists():
        add_index("cbar_delta_pd_binned", cbar_binned, "Binned monotonicity diagnostic for proxy validation")
    cbar_diag = out / "cbar_delta_pd_diagnostics.csv"
    if cbar_diag.exists():
        add_index("cbar_delta_pd_diagnostics", cbar_diag, "Proxy-near-zero positive-loss diagnostic")

    tracking_candidates = sorted(out.glob("*tracking*summary*.csv"))
    for path in tracking_candidates:
        add_index(path.stem, path, "Closed-loop tracking summary")

    pd.DataFrame(table_rows).to_csv(out / "paper_output_index.csv", index=False)
    print(f"[Paper] Saved output index: {out / 'paper_output_index.csv'}")


def run_paper_experiments(cfg: SimConfig, out: Path) -> None:
    """Run the optimized paper simulation bundle.

    Default bundle:
      1. Gate 1 full-link collision/detection validation.
      2. Gate 2 collision-regime statistics.
      3. Lambda sensitivity sweep.
      4. Paired bootstrap CIs and compact manuscript tables.

    Closed-loop tracking is included only when cfg.paper_include_tracking is set,
    because it is substantially more expensive than the main mechanism tests.
    """
    runtime_rows: List[Dict[str, float]] = []

    def timed(name: str, fn: Callable[[], None]) -> None:
        start = perf_counter()
        try:
            fn()
            _write_runtime_row(runtime_rows, name, start)
        except Exception:
            _write_runtime_row(runtime_rows, name, start, status="failed")
            raise
        finally:
            pd.DataFrame(runtime_rows).to_csv(out / "paper_runtime_summary.csv", index=False)

    timed("gate1", lambda: run_gate1(cfg, out))
    timed("gate2", lambda: run_gate2(cfg, out))
    timed("paper_baseline_suite", lambda: run_paper_baseline_suite(cfg, out))
    timed("cbar_delta_pd_correlation", lambda: run_cbar_pd_correlation(cfg, out))
    if cfg.paper_include_tracking:
        timed("tracking", lambda: run_tracking_experiment(cfg, out))
    timed("lambda_sweep", lambda: run_lambda_sweep(cfg, out))
    timed("gate1_paired_bootstrap", lambda: run_gate1_paired_bootstrap(cfg, out))
    timed("paper_tables", lambda: write_paper_tables(cfg, out))

    print(f"\n[Paper] Runtime summary saved: {out / 'paper_runtime_summary.csv'}")
