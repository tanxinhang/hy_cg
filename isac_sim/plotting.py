"""Figures.

Every experiment writes a small set of single-metric figures plus one combined
``_paper.png`` panel figure, into a dedicated figure directory.  Layout mirrors
the v10 prototype so existing paper drafts keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .naming import method_order, paper_label


_PLT = None


def _pyplot():
    """Import pyplot once, on the headless Agg backend."""
    global _PLT
    if _PLT is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _PLT = plt
    return _PLT


def _save(fig, out_dir: Path, name: str) -> None:
    """Save ``fig`` (or the current figure when ``fig`` is None) and close it."""
    plt = _pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    target = fig if fig is not None else plt.gcf()
    target.tight_layout()
    target.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(target)


# ==========================================================================
# Main comparison (single run)
# ==========================================================================
def plot_main_comparison(summary: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    plt = _pyplot()
    methods = method_order(list(summary.keys()))
    labels = [paper_label(m) for m in methods]
    x = np.arange(len(methods))

    def values(metric: str) -> np.ndarray:
        return np.array([float(summary[m].get(metric, np.nan)) for m in methods], dtype=float)

    def bar(metric: str, ylabel: str, name: str, yerr_metric: Optional[str] = None) -> None:
        plt.figure(figsize=(max(7.0, 1.15 * len(methods)), 4.8))
        if yerr_metric is None:
            plt.bar(x, values(metric))
        else:
            plt.bar(x, values(metric), yerr=values(yerr_metric), capsize=3)
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.grid(True, axis="y", alpha=0.3)
        _save(None, out_dir, name)

    bar("P_D", "Detection probability $P_D$", "main_pd.png", "P_D_ci95_half_width")
    bar("T_mean_ms", "Mean exchange delay (ms)", "main_delay.png")
    bar("P_D_per_ms", "$P_D$ per ms", "main_pd_per_ms.png")
    bar("selected_links_mean", "Mean selected links", "main_selected_links.png")
    bar("actual_worst_target_P_D", "Actual worst-target $P_D$", "main_actual_worst_pd.png")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    panels = [("P_D", "Detection probability $P_D$", "P_D_ci95_half_width"),
              ("T_mean_ms", "Mean exchange delay (ms)", None),
              ("P_D_per_ms", "$P_D$ per ms", None)]
    for ax, (metric, ylabel, err_metric) in zip(axes, panels):
        if err_metric is None:
            ax.bar(x, values(metric))
        else:
            ax.bar(x, values(metric), yerr=values(err_metric), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_dir, "main_paper.png")


# ==========================================================================
# Lambda sweep
# ==========================================================================
def plot_lambda_sweep(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    plt = _pyplot()
    lam = np.array([r["lambda_cost"] for r in rows], dtype=float)
    p_d = np.array([r["P_D"] for r in rows], dtype=float)
    t_ms = np.array([r["T_mean_ms"] for r in rows], dtype=float)
    links = np.array([r["selected_links_mean"] for r in rows], dtype=float)
    pd_per_ms = np.array([r["P_D_per_ms"] for r in rows], dtype=float)
    worst_sat = np.array([r["worst_target_satisfied_prob"] for r in rows], dtype=float)
    yerr = np.array([r.get("P_D_ci95_half_width", 0.0) for r in rows], dtype=float)

    plt.figure()
    plt.plot(t_ms, p_d, marker="o")
    for xv, yv, lv in zip(t_ms, p_d, lam):
        plt.annotate(f"{lv:g}", (xv, yv), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Mean soft-information exchange delay (ms)")
    plt.ylabel("Detection probability $P_D$")
    plt.title("$P_D$ vs exchange delay under different $\\lambda_c$")
    plt.grid(True, alpha=0.3)
    _save(None, out_dir, "lambda_pd_vs_delay.png")

    plt.figure()
    plt.plot(lam, links, marker="o")
    plt.xlabel("Lagrangian communication price $\\lambda_c$")
    plt.ylabel("Mean selected links")
    plt.grid(True, alpha=0.3)
    _save(None, out_dir, "lambda_links_vs_lambda.png")

    plt.figure()
    plt.plot(lam, pd_per_ms, marker="o")
    plt.xlabel("Lagrangian communication price $\\lambda_c$")
    plt.ylabel("$P_D$ per ms")
    plt.grid(True, alpha=0.3)
    _save(None, out_dir, "lambda_pd_per_ms_vs_lambda.png")

    plt.figure()
    plt.plot(lam, worst_sat, marker="o")
    plt.xlabel("Lagrangian communication price $\\lambda_c$")
    plt.ylabel("Worst-target satisfied probability")
    plt.grid(True, alpha=0.3)
    _save(None, out_dir, "lambda_worst_sat_vs_lambda.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    axes[0].plot(lam, links, marker="o")
    axes[0].set_xlabel("Communication price $\\lambda_c$")
    axes[0].set_ylabel("Mean selected links")
    axes[0].grid(True, alpha=0.3)
    axes[1].errorbar(t_ms, p_d, yerr=yerr, marker="o", capsize=3)
    for xv, yv, lv in zip(t_ms, p_d, lam):
        axes[1].annotate(f"{lv:g}", (xv, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[1].set_xlabel("Mean exchange delay (ms)")
    axes[1].set_ylabel("Detection probability $P_D$")
    axes[1].grid(True, alpha=0.3)
    _save(fig, out_dir, "lambda_paper.png")


# ==========================================================================
# Ablation (plain and budget-capped)
# ==========================================================================
def _ablation_plots(rows: List[Dict[str, Any]], out_dir: Path, stem: str, with_worst: bool = True) -> None:
    if not rows:
        return
    plt = _pyplot()
    labels = [paper_label(str(r["variant"])) for r in rows]
    x = np.arange(len(rows))

    def barplot(metric: str, ylabel: str, name: str) -> None:
        plt.figure(figsize=(max(7, 1.4 * len(labels)), 4.8))
        plt.bar(x, [float(r[metric]) for r in rows])
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.grid(True, axis="y", alpha=0.3)
        _save(None, out_dir, name)

    barplot("P_D", "Detection probability $P_D$", f"{stem}_pd.png")
    barplot("T_mean_ms", "Mean exchange delay (ms)", f"{stem}_delay.png")
    barplot("P_D_per_ms", "$P_D$ per ms", f"{stem}_pd_per_ms.png")
    if with_worst:
        barplot("worst_target_satisfied_prob", "Worst-target satisfied probability", f"{stem}_worst_sat.png")

    yerr = np.array([float(r.get("P_D_ci95_half_width", 0.0)) for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for ax, metric, ylabel in zip(axes,
                                  ["P_D", "T_mean_ms", "P_D_per_ms"],
                                  ["Detection probability $P_D$", "Mean exchange delay (ms)", "$P_D$ per ms"]):
        vals = np.array([float(r[metric]) for r in rows], dtype=float)
        if metric == "P_D":
            ax.bar(x, vals, yerr=yerr, capsize=3)
        else:
            ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_dir, f"{stem}_paper.png")

    if "actual_worst_target_P_D" in rows[0]:
        plt.figure(figsize=(max(7, 1.4 * len(labels)), 4.8))
        plt.bar(x, [float(r["actual_worst_target_P_D"]) for r in rows])
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel("Actual worst-target $P_D$")
        plt.grid(True, axis="y", alpha=0.3)
        _save(None, out_dir, f"{stem}_actual_worst_pd.png")


def plot_ablation(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    _ablation_plots(rows, out_dir, "ablation")


def plot_fair_ablation(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    _ablation_plots(rows, out_dir, "fair_ablation")


# ==========================================================================
# Communication-constraint sweep
# ==========================================================================
def plot_comm_sweep(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    plt = _pyplot()
    methods = method_order(list(dict.fromkeys(str(r["method"]) for r in rows)))
    R_vals = list(dict.fromkeys(float(r["R_min_mbps"]) for r in rows))
    x = np.array(R_vals, dtype=float)

    def get_vals(method: str, metric: str) -> List[float]:
        out = []
        for rv in R_vals:
            match = [r for r in rows
                     if str(r["method"]) == method and np.isclose(float(r["R_min_mbps"]), rv, rtol=0.0, atol=1e-9)]
            out.append(float(match[0][metric]) if match else np.nan)
        return out

    feasible = []
    for rv in R_vals:
        match = [r for r in rows if np.isclose(float(r["R_min_mbps"]), rv, rtol=0.0, atol=1e-9)]
        feasible.append(float(match[0]["comm_feasible_edge_ratio_mean"]) if match else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    for method in methods:
        axes[0].plot(x, get_vals(method, "P_D"), marker="o", label=paper_label(method))
        axes[2].plot(x, get_vals(method, "selected_rate_mean_mbps"), marker="o", label=paper_label(method))
        axes[0].set_xlabel("$R_{\\min}$ (Mbit/s)")
        axes[0].set_ylabel("Detection probability $P_D$")
        axes[2].set_xlabel("$R_{\\min}$ (Mbit/s)")
        axes[2].set_ylabel("Selected-link mean rate (Mbit/s)")
    axes[0].grid(True, alpha=0.3)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)
    axes[1].plot(x, feasible, marker="o")
    axes[1].set_xlabel("$R_{\\min}$ (Mbit/s)")
    axes[1].set_ylabel("Feasible edge ratio")
    axes[1].grid(True, alpha=0.3)
    _save(fig, out_dir, "comm_sweep_paper.png")

    # Zoomed delay plot without the all-neighbor upper-resource baseline.
    plt.figure(figsize=(7.2, 4.8))
    for method in [m for m in methods if m != "all_neighbor"]:
        plt.plot(x, get_vals(method, "T_mean_ms"), marker="o", label=paper_label(method))
    plt.xlabel("$R_{\\min}$ (Mbit/s)")
    plt.ylabel("Mean exchange delay (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save(None, out_dir, "comm_sweep_delay_zoom.png")


# ==========================================================================
# Robustness
# ==========================================================================
def plot_robustness(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    plt = _pyplot()
    axis = str(rows[0]["robustness_type"])
    methods = method_order(list(dict.fromkeys(str(r["method"]) for r in rows)))
    conditions = list(dict.fromkeys(r["condition_value"] for r in rows))

    def cond_x(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(conditions.index(v))

    x = np.array([cond_x(v) for v in conditions], dtype=float)

    def get_vals(method: str, metric: str) -> List[float]:
        out = []
        for c in conditions:
            match = [r for r in rows if str(r["method"]) == method and r["condition_value"] == c]
            out.append(float(match[0][metric]) if match else np.nan)
        return out

    def lineplot(metric: str, ylabel: str, name: str) -> None:
        plt.figure(figsize=(7.2, 4.8))
        for method in methods:
            plt.plot(x, get_vals(method, metric), marker="o", label=paper_label(method))
        if axis == "residual_direct":
            plt.xscale("log")
        elif axis == "comm_model":
            plt.xticks(x, [str(c) for c in conditions])
        plt.xlabel(axis)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        _save(None, out_dir, name)

    lineplot("P_D", "Detection probability $P_D$", "robustness_pd.png")
    lineplot("T_mean_ms", "Mean exchange delay (ms)", "robustness_delay.png")
    lineplot("worst_target_satisfied_prob", "Worst-target satisfied probability", "robustness_worst_sat.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    for method in methods:
        axes[0].plot(x, get_vals(method, "P_D"), marker="o", label=paper_label(method))
        axes[1].plot(x, get_vals(method, "P_D_per_ms"), marker="o", label=paper_label(method))
    for ax, ylabel in zip(axes, ["Detection probability $P_D$", "$P_D$ per ms"]):
        if axis == "residual_direct":
            ax.set_xscale("log")
        elif axis == "comm_model":
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in conditions])
        ax.set_xlabel(axis)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    _save(fig, out_dir, "robustness_paper.png")


# ==========================================================================
# OTFS delay-Doppler ablation
# ==========================================================================
def plot_dd_ablation(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    plt = _pyplot()
    variants = list(dict.fromkeys(str(r["variant"]) for r in rows))
    methods = list(dict.fromkeys(str(r["method"]) for r in rows))
    x = np.arange(len(variants))

    def grouped_bar(metric: str, ylabel: str, name: str) -> None:
        width = 0.8 / max(len(methods), 1)
        plt.figure(figsize=(max(8, 1.5 * len(variants)), 4.8))
        for idx, method in enumerate(methods):
            vals = []
            for variant in variants:
                match = [r for r in rows if str(r["method"]) == method and str(r["variant"]) == variant]
                vals.append(float(match[0][metric]) if match else np.nan)
            plt.bar(x + (idx - (len(methods) - 1) / 2.0) * width, vals, width=width, label=paper_label(method))
        plt.xticks(x, [paper_label(v) for v in variants], rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        _save(None, out_dir, name)

    grouped_bar("P_D", "Detection probability $P_D$", "dd_ablation_pd.png")
    grouped_bar("T_mean_ms", "Mean exchange delay (ms)", "dd_ablation_delay.png")
    grouped_bar("selected_links_mean", "Mean selected links", "dd_ablation_links.png")
    grouped_bar("feasible_links_mean", "Feasible links per trial", "dd_ablation_feasible_links.png")

    # Proposed-only compact panel.
    prop = [r for r in rows if str(r.get("method", "")) == "proposed_lagrangian"]
    if not prop:
        return
    labels = [paper_label(str(r["variant"])) for r in prop]
    xi = np.arange(len(prop))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for ax, metric, ylabel in zip(axes,
                                  ["P_D", "feasible_links_mean", "P_D_per_ms"],
                                  ["Detection probability $P_D$", "Feasible links/trial", "$P_D$ per ms"]):
        ax.bar(xi, [float(r[metric]) for r in prop])
        ax.set_xticks(xi)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_dir, "dd_ablation_proposed_paper.png")


def plot_c2f(rows, out_dir: Path) -> None:
    """Bar chart of P_D across coarse / C2F / full / all-neighbor."""
    if not rows:
        return
    plt = _pyplot()
    by_method: Dict[str, Dict[str, Any]] = {r["method"]: r for r in rows}
    methods = ["proposed_lagrangian", "proposed_c2f", "proposed_c2f_full", "all_neighbor"]
    pds = [by_method[m]["P_D"] for m in methods if m in by_method]
    labels = [m.replace("proposed_", "").replace("_", " ") for m in methods if m in by_method]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, pds, color=["#888", "#3a7", "#5c9", "#dd5"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("P_D")
    ax.set_title("C2F ablation: P_D vs refinement scheme")
    ax.grid(True, axis="y", alpha=0.3)
    for xi, v in zip(x, pds):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    _save(fig, out_dir, "c2f_pd.png")


def plot_prior_sweep(rows, out_dir: Path) -> None:
    """P_D vs target-state prior uncertainty."""
    if not rows:
        return
    plt = _pyplot()
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = {"proposed_lagrangian": "#3a7", "all_neighbor": "#dd5"}
    for method, mrows in by_method.items():
        ordered = sorted(mrows, key=lambda r: r["sigma_vel_mps"])
        sigmas = [r["sigma_vel_mps"] for r in ordered]
        pds = [r["P_D"] for r in ordered]
        ax.plot(sigmas, pds, "-o", label=method, color=colors.get(method, None))
        for xi, v, row in zip(sigmas, pds, ordered):
            ax.text(xi, v + 0.005, f"{v:.2f}", fontsize=7, ha="center",
                    color=colors.get(method, "black"))
    ax.set_xlabel("sigma_vel_mps (paired with sigma_pos_m)")
    ax.set_ylabel("P_D")
    ax.set_title("Target-state prior sensitivity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, out_dir, "prior_sweep_pd.png")


def plot_waveform_check(rows, out_dir: Path) -> None:
    """Scatter analytic eta^c / eta^loc vs physical OTFS PSF capture."""
    if not rows:
        return
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, key_a, key_p, title in (
        (axes[0], "eta_c_analytic", "eta_c_psf", r"$\eta^c$: analytic vs OTFS PSF"),
        (axes[1], "eta_loc_analytic", "eta_loc_psf", r"$\eta^{\mathrm{loc}}$: analytic vs OTFS PSF"),
    ):
        ax.scatter([r[key_a] for r in rows], [r[key_p] for r in rows], s=14, alpha=0.6)
        ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(1.0, max(r[key_p] for r in rows) * 1.1))
        ax.set_xlabel("analytic")
        ax.set_ylabel("OTFS PSF")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "waveform_check_scatter.png")


def plot_oracle_gap(rows, out_dir: Path) -> None:
    """Greedy vs oracle objective per trial, with the gap annotation."""
    if not rows:
        return
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    trials = [r["trial"] for r in rows]
    greedy = [r["greedy_obj"] for r in rows]
    oracle = [r["oracle_obj"] for r in rows]
    ax.plot(trials, oracle, "-o", label="oracle (exact)", color="#d55")
    ax.plot(trials, greedy, "-s", label="greedy (proposed)", color="#3a7")
    mean_gap = float(np.mean([r["gap"] for r in rows]))
    ax.set_xlabel("trial")
    ax.set_ylabel(r"$\sum_q D_q$")
    ax.set_title(f"Greedy vs exact optimum (mean gap = {mean_gap:.1%})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    _save(fig, out_dir, "oracle_gap.png")


def plot_runtime(rows, out_dir: Path) -> None:
    """Per-selector wall-clock bar chart (log scale, ms)."""
    if not rows:
        return
    plt = _pyplot()
    rows = sorted(rows, key=lambda r: r["mean_s"])
    labels = [r["method"].replace("_", " ") for r in rows]
    vals_ms = [r["mean_s"] * 1e3 for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(labels))
    ax.bar(x, vals_ms, color="#3a7")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("mean selection time (ms, log scale)")
    ax.set_title("Selector runtime")
    for xi, v in zip(x, vals_ms):
        ax.text(xi, v * 1.2, f"{v:.3f}", ha="center", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_dir, "runtime.png")


# ==========================================================================
# Registry
# ==========================================================================
PLOTTERS = {
    "lambda-sweep": plot_lambda_sweep,
    "ablation": plot_ablation,
    "fair-ablation": plot_fair_ablation,
    "dd-ablation": plot_dd_ablation,
    "comm-sweep": plot_comm_sweep,
    "robustness": plot_robustness,
    "c2f": plot_c2f,
    "prior-sweep": plot_prior_sweep,
    "waveform-check": plot_waveform_check,
    "oracle-gap": plot_oracle_gap,
    "runtime": plot_runtime,
}
