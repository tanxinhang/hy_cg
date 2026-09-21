"""Reporting: console summary, CSV export and LaTeX tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from experiments.app.naming import latex_escape, method_order, paper_label

# Scalar metrics exported by the single-run CSV.  Order is kept stable so that
# downstream comparison scripts keep working.
SCALAR_KEYS: List[str] = [
    "P_D", "P_D_ci95_low", "P_D_ci95_high", "P_D_ci95_half_width",
    "P_D_cluster_ci95_low", "P_D_cluster_ci95_high",
    "P_D_cluster_ci95_half_width",
    "P_D_weak", "P_D_weak_cluster_ci95_low", "P_D_weak_cluster_ci95_high",
    "P_FA", "P_FA_ci95_low", "P_FA_ci95_high", "P_FA_ci95_half_width",
    "P_FA_cluster_ci95_low", "P_FA_cluster_ci95_high",
    "P_FA_cluster_ci95_half_width",
    "P_FA_overall", "P_FA_overall_ci95_low", "P_FA_overall_ci95_high",
    "P_FA_overall_ci95_half_width", "B_mean_bits", "B_std_bits",
    "T_mean_ms", "T_std_ms", "selected_links_mean", "selected_links_std",
    "selected_observations_mean", "selected_observations_std",
    "active_target_ratio_mean", "feasible_target_ratio_mean", "feasible_links_mean",
    "comm_feasible_edge_ratio_mean",
    "selected_rate_mean_mbps", "selected_rate_min_mbps_mean", "selected_rate_p10_mbps_mean",
    "selected_chi_mean", "selected_chi_min_mean", "selected_chi_p10_mean",
    "selected_gamma_comm_mean_db", "selected_rate_satisfaction_ratio_mean",
    "selected_chi_ge_min_ratio_mean",
    "D_mean", "D_median", "D_p10", "D_p90",
    "fine_eval_full_mean", "fine_eval_c2f_mean", "belief_capture_rate_mean",
    "receiver_capacity_max_utilization_mean", "receiver_capacity_binding_rate",
    "fusion_capacity_max_utilization_mean", "fusion_capacity_binding_rate",
    "report_capacity_utilization_mean", "report_capacity_binding_rate",
    "assignment_capacity_max_utilization_mean", "assignment_capacity_binding_rate",
    "cpu_capacity_max_utilization_mean", "cpu_capacity_binding_rate",
    "selector_score_evaluations_mean", "coordination_messages_mean", "bid_rounds_mean",
    # Coordination fixed-point diagnostics: a reported coordination number must
    # carry whether the fixed point was actually reached (0 when it is off).
    "coordination_rounds_mean", "coordination_converged_rate", "coordination_n_tx_mean",
    "bundle_column_count_mean", "bundle_pricing_iterations_mean",
    "bundle_lp_worst_deficit_bound_mean",
    "detection_runtime_mean_ms", "detection_runtime_p90_ms",
    "all_targets_satisfied_prob", "worst_target_D_mean", "worst_target_satisfied_prob",
    "actual_mean_target_P_D", "actual_worst_target_P_D", "actual_best_target_P_D",
    "paired_proposed_delta_P_D", "paired_proposed_delta_ci95_low",
    "paired_proposed_delta_ci95_high",
    "paired_reference_method", "paired_reference_delta_P_D",
    "paired_reference_delta_ci95_low", "paired_reference_delta_ci95_high",
    "P_D_per_kbit", "P_D_per_ms", "D_per_kbit", "D_per_ms",
]


def scalar_summary_row(summary: Dict[str, Dict[str, Any]], method: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact scalar row for one method from a simulation summary."""
    s = summary[method]
    row: Dict[str, Any] = dict(extra)
    row.update({
        "method": method,
        "P_D": s["P_D"],
        "P_D_ci95_low": s["P_D_ci95"][0],
        "P_D_ci95_high": s["P_D_ci95"][1],
        "P_D_ci95_half_width": s["P_D_ci95_half_width"],
        "P_D_cluster_ci95_low": s["P_D_cluster_ci95_low"],
        "P_D_cluster_ci95_high": s["P_D_cluster_ci95_high"],
        "P_D_cluster_ci95_half_width": s["P_D_cluster_ci95_half_width"],
        "P_D_weak": s["P_D_weak"],
        "P_D_weak_cluster_ci95_low": s["P_D_weak_cluster_ci95_low"],
        "P_D_weak_cluster_ci95_high": s["P_D_weak_cluster_ci95_high"],
        "P_FA_active": s["P_FA"],
        "P_FA_cluster_ci95_low": s["P_FA_cluster_ci95_low"],
        "P_FA_cluster_ci95_high": s["P_FA_cluster_ci95_high"],
        "P_FA_cluster_ci95_half_width": s["P_FA_cluster_ci95_half_width"],
        "P_FA_overall": s["P_FA_overall"],
        "B_mean_bits": s["B_mean_bits"],
        "T_mean_ms": s["T_mean_ms"],
        "selected_links_mean": s["selected_links_mean"],
        "selected_observations_mean": s["selected_observations_mean"],
        "active_target_ratio_mean": s["active_target_ratio_mean"],
        "feasible_target_ratio_mean": s["feasible_target_ratio_mean"],
        "feasible_links_mean": s["feasible_links_mean"],
        "comm_feasible_edge_ratio_mean": s["comm_feasible_edge_ratio_mean"],
        "selected_rate_mean_mbps": s["selected_rate_mean_mbps"],
        "selected_rate_min_mbps_mean": s["selected_rate_min_mbps_mean"],
        "selected_rate_p10_mbps_mean": s["selected_rate_p10_mbps_mean"],
        "selected_chi_mean": s["selected_chi_mean"],
        "selected_chi_min_mean": s["selected_chi_min_mean"],
        "selected_chi_p10_mean": s["selected_chi_p10_mean"],
        "selected_gamma_comm_mean_db": s["selected_gamma_comm_mean_db"],
        "selected_rate_satisfaction_ratio_mean": s["selected_rate_satisfaction_ratio_mean"],
        "selected_chi_ge_min_ratio_mean": s["selected_chi_ge_min_ratio_mean"],
        "D_mean": s["D_mean"],
        "D_median": s["D_median"],
        "D_p10": s["D_p10"],
        "D_p90": s["D_p90"],
        "fine_eval_full_mean": s["fine_eval_full_mean"],
        "fine_eval_c2f_mean": s["fine_eval_c2f_mean"],
        "belief_capture_rate_mean": s["belief_capture_rate_mean"],
        "receiver_capacity_max_utilization_mean": s["receiver_capacity_max_utilization_mean"],
        "receiver_capacity_binding_rate": s["receiver_capacity_binding_rate"],
        "fusion_capacity_max_utilization_mean": s["fusion_capacity_max_utilization_mean"],
        "fusion_capacity_binding_rate": s["fusion_capacity_binding_rate"],
        "report_capacity_utilization_mean": s["report_capacity_utilization_mean"],
        "report_capacity_binding_rate": s["report_capacity_binding_rate"],
        "assignment_capacity_max_utilization_mean": s["assignment_capacity_max_utilization_mean"],
        "assignment_capacity_binding_rate": s["assignment_capacity_binding_rate"],
        "cpu_capacity_max_utilization_mean": s["cpu_capacity_max_utilization_mean"],
        "cpu_capacity_binding_rate": s["cpu_capacity_binding_rate"],
        "selector_score_evaluations_mean": s["selector_score_evaluations_mean"],
        "coordination_messages_mean": s["coordination_messages_mean"],
        "detection_runtime_mean_ms": s["detection_runtime_mean_ms"],
        "detection_runtime_p90_ms": s["detection_runtime_p90_ms"],
        "bid_rounds_mean": s["bid_rounds_mean"],
        "bundle_column_count_mean": s["bundle_column_count_mean"],
        "bundle_pricing_iterations_mean": s["bundle_pricing_iterations_mean"],
        "bundle_lp_worst_deficit_bound_mean": s["bundle_lp_worst_deficit_bound_mean"],
        "all_targets_satisfied_prob": s["all_targets_satisfied_prob"],
        "worst_target_D_mean": s["worst_target_D_mean"],
        "worst_target_satisfied_prob": s["worst_target_satisfied_prob"],
        "actual_mean_target_P_D": s["actual_mean_target_P_D"],
        "actual_worst_target_P_D": s["actual_worst_target_P_D"],
        "actual_best_target_P_D": s["actual_best_target_P_D"],
        "P_D_per_kbit": s["P_D_per_kbit"],
        "P_D_per_ms": s["P_D_per_ms"],
        "D_per_kbit": s["D_per_kbit"],
        "D_per_ms": s["D_per_ms"],
    })
    for key in (
        "paired_proposed_delta_P_D",
        "paired_proposed_delta_ci95_low",
        "paired_proposed_delta_ci95_high",
        "paired_reference_method",
        "paired_reference_delta_P_D",
        "paired_reference_delta_ci95_low",
        "paired_reference_delta_ci95_high",
    ):
        if key in s:
            row[key] = s[key]
    return row


def print_summary(summary: Dict[str, Dict[str, Any]]) -> None:
    print("\n========== Lagrangian DOTFS-ISAC Simplified Simulation Summary ==========")
    for method, s in summary.items():
        print(f"\n[{method}]")
        print(f"  P_D                    : {s['P_D']:.4f} "
              f"(95% CI [{s['P_D_ci95'][0]:.4f}, {s['P_D_ci95'][1]:.4f}], +/- {s['P_D_ci95_half_width']:.4f})")
        print(f"  P_FA active            : {s['P_FA']:.4f} "
              f"(95% CI [{s['P_FA_ci95'][0]:.4f}, {s['P_FA_ci95'][1]:.4f}], +/- {s['P_FA_ci95_half_width']:.4f})")
        print(f"  P_FA trial-cluster CI  : [{s['P_FA_cluster_ci95_low']:.4f}, "
              f"{s['P_FA_cluster_ci95_high']:.4f}]")
        print(f"  P_FA system-level      : {s['P_FA_overall']:.4f} "
              f"(95% CI [{s['P_FA_overall_ci95'][0]:.4f}, {s['P_FA_overall_ci95'][1]:.4f}], "
              f"+/- {s['P_FA_overall_ci95_half_width']:.4f})")
        print(f"  Overhead B             : {s['B_mean_bits']:.2f} bit (std {s['B_std_bits']:.2f})")
        print(f"  Overhead T             : {s['T_mean_ms']:.4f} ms (std {s['T_std_ms']:.4f})")
        print(f"  Selected observations  : {s['selected_observations_mean']:.2f} "
              f"(std {s['selected_observations_std']:.2f})")
        print(f"  Remote reports         : {s['selected_links_mean']:.2f} "
              f"(std {s['selected_links_std']:.2f})")
        print(f"  Active target ratio    : {s['active_target_ratio_mean']:.4f}")
        print(f"  Feasible target ratio  : {s['feasible_target_ratio_mean']:.4f}")
        print(f"  Feasible links/trial   : {s['feasible_links_mean']:.2f}")
        print(f"  Comm feasible edge ratio: {s['comm_feasible_edge_ratio_mean']:.4f}")
        print(f"  Selected rate mean/min : {s['selected_rate_mean_mbps']:.4f} / "
              f"{s['selected_rate_min_mbps_mean']:.4f} Mbps")
        print(f"  Selected chi mean/min  : {s['selected_chi_mean']:.4f} / {s['selected_chi_min_mean']:.4f}")
        print(f"  Rate/chi sat. ratio    : {s['selected_rate_satisfaction_ratio_mean']:.4f} / "
              f"{s['selected_chi_ge_min_ratio_mean']:.4f}")
        print(f"  D mean/median          : {s['D_mean']:.4f} / {s['D_median']:.4f}")
        print(f"  D p10/p90              : {s['D_p10']:.4f} / {s['D_p90']:.4f}")
        print(f"  All targets satisfy    : {s['all_targets_satisfied_prob']:.4f}")
        print(f"  Worst target D mean    : {s['worst_target_D_mean']:.4f}")
        print(f"  Worst target sat. prob : {s['worst_target_satisfied_prob']:.4f}")
        print(f"  Actual worst-target P_D: {s['actual_worst_target_P_D']:.4f}")
        if s["selector_score_evaluations_mean"] > 0:
            print(f"  Selector score evals    : {s['selector_score_evaluations_mean']:.2f}")
            print(f"  Coordination messages  : {s['coordination_messages_mean']:.2f}")
        print(f"  Detection runtime mean : {s['detection_runtime_mean_ms']:.3f} ms")
        if "paired_proposed_delta_P_D" in s:
            print(f"  Paired proposed - method: {s['paired_proposed_delta_P_D']:+.4f} "
                  f"(95% CI [{s['paired_proposed_delta_ci95_low']:+.4f}, "
                  f"{s['paired_proposed_delta_ci95_high']:+.4f}])")
        elif "paired_reference_delta_P_D" in s:
            print(f"  Paired {s['paired_reference_method']} - method: "
                  f"{s['paired_reference_delta_P_D']:+.4f} "
                  f"(95% CI [{s['paired_reference_delta_ci95_low']:+.4f}, "
                  f"{s['paired_reference_delta_ci95_high']:+.4f}])")
        print(f"  Efficiency P_D/kbit    : {s['P_D_per_kbit']:.4f}")
        print(f"  Efficiency P_D/ms      : {s['P_D_per_ms']:.4f}")


def write_csv(summary: Dict[str, Dict[str, Any]], path: Path) -> None:
    """Write the single-run summary, one row per method."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + SCALAR_KEYS)
        writer.writeheader()
        for method, s in summary.items():
            row = {"method": method}
            for k in SCALAR_KEYS:
                row[k] = s.get(k, "")
            writer.writerow(row)


def write_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write a list of scalar metric rows to CSV, preserving first-row key order."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_main_summary_latex(summary: Dict[str, Dict[str, Any]], path: Path) -> None:
    """Write a compact IEEE-style LaTeX table for the main comparison."""
    methods = method_order(list(summary.keys()))
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Main numerical comparison of different link selection methods.}",
        r"    \label{tab:main_summary}",
        r"    \begin{tabular}{l c c c c c}",
        r"        \hline",
        r"        Method & $P_D$ & $P_{\mathrm{FA}}$ & Delay/ms & Reports & $P_D$/ms \\",
        r"        \hline",
    ]
    for m in methods:
        s = summary[m]
        lines.append(
            "        "
            + f"{latex_escape(paper_label(m))} & {float(s['P_D']):.4f} & {float(s['P_FA']):.4f} & "
            + f"{float(s['T_mean_ms']):.2f} & {float(s['selected_links_mean']):.2f} & {float(s['P_D_per_ms']):.4f} \\"
        )
    lines.extend([
        r"        \hline",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
