"""policy（自 ``experiments/methods.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List, Literal


C2F_METHODS: Dict[str, bool] = {
    "rcs_robust_bundle_cg": False,
    "joint_bundle_cg": False,
    "joint_bundle_cg_exact_llr": False,
    "fixed_fusion_bundle": False,
    "local_only_bundle": False,
    "proposed_c2f": False,
    "proposed_c2f_adaptive": False,
    "proposed_c2f_adaptive_pd": False,
    "proposed_c2f_adaptive_pd_distributed": False,
    "proposed_c2f_adaptive_pd_robust": False,
    "proposed_c2f_adaptive_pd_calibrated": False,
    "proposed_c2f_adaptive_pd_fusion_polish": False,
    "proposed_c2f_pd": False,
    "proposed_c2f_full": True,
    "proposed_c2f_full_pd": True,
}


METHOD_RNG_OFFSETS: Dict[str, int] = {
    "rcs_robust_bundle_cg": 96,
    "joint_bundle_cg": 97,
    "joint_bundle_cg_exact_llr": 95,
    "fixed_fusion_bundle": 98,
    "local_only_bundle": 99,
    "proposed_lagrangian": 101,
    "proposed_c2f": 131,
    "proposed_c2f_adaptive": 133,
    "proposed_c2f_adaptive_pd": 135,
    "proposed_c2f_adaptive_pd_distributed": 136,
    "proposed_c2f_adaptive_pd_robust": 138,
    "proposed_c2f_adaptive_pd_calibrated": 140,
    "proposed_c2f_adaptive_pd_fusion_polish": 142,
    "proposed_c2f_pd": 139,
    "proposed_c2f_full": 137,
    "proposed_c2f_full_pd": 141,
    "all_neighbor": 211,
    "random": 307,
    "nearest": 401,
    "shortest_bistatic": 457,
    "raw_sense_sinr": 461,
    "sense_sinr": 503,
    "sense_sinr_budgeted": 509,
    "single_best": 601,
    "topk_deflection": 641,
    "global_topk_deflection": 647,
    "cost_aware_greedy": 653,
    "exact_marginal_greedy": 659,
}


def c2f_method_name(apply_to_all: bool) -> str:
    """Return the canonical method name for a C2F variant."""
    return "proposed_c2f_full" if apply_to_all else "proposed_c2f"


def fusion_weight_mode_for_method(method: str) -> str:
    """Self-consistent fusion weights for each method.

    Lives with the roster, not in the library: this is a *method → policy* table
    (experiment bookkeeping), while ``deflection`` / ``equal`` / ``exact_llr_sum``
    themselves stay in :mod:`isac_sim.detection.fusion`.
    """
    if method == "joint_bundle_cg_exact_llr":
        return "exact_llr_sum"
    return "equal" if method == "raw_sense_sinr" else "deflection"
