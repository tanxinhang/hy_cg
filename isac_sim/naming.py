"""Presentation helpers shared by the tables and the figures.

Keeping the display order and the human-readable labels in one place means a
new method or ablation variant gets a consistent name everywhere.
"""

from __future__ import annotations

from typing import Dict, List

# Canonical display order: proposed first, then resource-constrained baselines,
# then the upper-resource reference last.
METHOD_ORDER: List[str] = [
    "rcs_robust_bundle_cg",
    "joint_bundle_cg",
    "joint_bundle_cg_exact_llr",
    "fixed_fusion_bundle",
    "local_only_bundle",
    "proposed_c2f",
    "proposed_c2f_adaptive",
    "proposed_c2f_adaptive_pd",
    "proposed_c2f_adaptive_pd_distributed",
    "proposed_c2f_adaptive_pd_robust",
    "proposed_c2f_adaptive_pd_calibrated",
    "proposed_c2f_pd",
    "proposed_c2f_full",
    "proposed_c2f_full_pd",
    "proposed_lagrangian",
    "sense_sinr",
    "raw_sense_sinr",
    "single_best",
    "topk_deflection",
    "global_topk_deflection",
    "cost_aware_greedy",
    "exact_marginal_greedy",
    "nearest",
    "shortest_bistatic",
    "random",
    "all_neighbor",
]

LABELS: Dict[str, str] = {
    # Methods
    "rcs_robust_bundle_cg": "RCS-robust joint bundle CG",
    "joint_bundle_cg": "Joint bundle column generation",
    "joint_bundle_cg_exact_llr": "Joint bundle CG + exact LLR sum",
    "fixed_fusion_bundle": "Fixed fusion + joint bundle",
    "local_only_bundle": "Local-only joint bundle",
    "proposed_lagrangian": "Proposed",
    "proposed_c2f": "Proposed C2F",
    "proposed_c2f_adaptive": "Proposed adaptive C2F",
    "proposed_c2f_adaptive_pd": "Proposed adaptive C2F (detector-aligned)",
    "proposed_c2f_adaptive_pd_distributed": "Distributed-bid detector-aligned C2F",
    "proposed_c2f_adaptive_pd_robust": "Geometry-robust detector-aligned C2F",
    "proposed_c2f_adaptive_pd_calibrated": "Calibrated-replay detector-aligned C2F",
    "proposed_c2f_pd": "Proposed C2F (detector-aligned)",
    "proposed_c2f_full": "Proposed full DD refinement",
    "proposed_c2f_full_pd": "Full refinement + detector-PD greedy",
    "sense_sinr": "Sensing-SINR",
    "raw_sense_sinr": "Raw sensing-SINR",
    "single_best": "Single-best",
    "topk_deflection": "Top-K Deflection",
    "global_topk_deflection": "Global Top-K Deflection",
    "cost_aware_greedy": "Cost-Aware Greedy",
    "exact_marginal_greedy": "Exact-Marginal Greedy",
    "nearest": "Nearest",
    "shortest_bistatic": "Shortest-bistatic",
    "random": "Random",
    "all_neighbor": "All-neighbor",
    # Ablation variants
    "full": "Full",
    "w/o_alpha": r"w/o $\alpha_q$",
    "w/o_delay_price": "w/o delay price",
    "w/o_comm_error_calib": "w/o comm. calib.",
    "w/o_softmin_alpha": "w/o soft-min",
    "full_dd": "Full DD",
    "w/o_dd_validity": "w/o DD validity",
    "w/o_dd_fractional_loss": "w/o frac. loss",
    "w/o_dd_collision_penalty": "w/o collision",
    "w/o_all_dd_effects": "w/o all DD",
}


def method_order(methods: List[str]) -> List[str]:
    """Sort ``methods`` into the canonical display order, keeping extras at the end."""
    return [m for m in METHOD_ORDER if m in methods] + [m for m in methods if m not in METHOD_ORDER]


def paper_label(name: str) -> str:
    """Human-readable label for a method or ablation variant."""
    return LABELS.get(name, name.replace("_", "-"))


def latex_escape(text: str) -> str:
    """Escape the characters that would otherwise break a LaTeX cell."""
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )
