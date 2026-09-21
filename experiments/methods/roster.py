"""roster（自 ``experiments/methods.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List, Literal


MethodName = Literal[
    "rcs_robust_bundle_cg",
    "joint_bundle_cg",
    "joint_bundle_cg_exact_llr",
    "fixed_fusion_bundle",
    "local_only_bundle",
    "proposed_lagrangian",
    "all_neighbor",
    "random",
    "nearest",
    "shortest_bistatic",
    "raw_sense_sinr",
    "sense_sinr",
    "sense_sinr_budgeted",
    "single_best",
    "topk_deflection",
    "global_topk_deflection",
    "cost_aware_greedy",
    "exact_marginal_greedy",
    "proposed_c2f",
    "proposed_c2f_adaptive",
    "proposed_c2f_adaptive_pd",
    "proposed_c2f_adaptive_pd_distributed",
    "proposed_c2f_adaptive_pd_robust",
    "proposed_c2f_adaptive_pd_calibrated",
    "proposed_c2f_adaptive_pd_fusion_polish",
    "proposed_c2f_pd",
    "proposed_c2f_full",
    "proposed_c2f_full_pd",
]


METHODS: List[MethodName] = [
    "rcs_robust_bundle_cg",
    "joint_bundle_cg",
    "joint_bundle_cg_exact_llr",
    "fixed_fusion_bundle",
    "local_only_bundle",
    "proposed_lagrangian",
    "proposed_c2f",
    "proposed_c2f_adaptive",
    "proposed_c2f_adaptive_pd",
    "proposed_c2f_adaptive_pd_distributed",
    "proposed_c2f_adaptive_pd_robust",
    "proposed_c2f_adaptive_pd_calibrated",
    "proposed_c2f_adaptive_pd_fusion_polish",
    "proposed_c2f_pd",
    "proposed_c2f_full",
    "proposed_c2f_full_pd",
    "all_neighbor",
    "random",
    "nearest",
    "shortest_bistatic",
    "raw_sense_sinr",
    "sense_sinr",
    "sense_sinr_budgeted",
    "single_best",
    "topk_deflection",
    "global_topk_deflection",
    "cost_aware_greedy",
    "exact_marginal_greedy",
]


EXPERIMENTAL_METHODS: set[str] = {
    "rcs_robust_bundle_cg",
    "joint_bundle_cg",
    "joint_bundle_cg_exact_llr",
    "fixed_fusion_bundle",
    "local_only_bundle",
    "proposed_c2f_adaptive_pd_distributed",
    "proposed_c2f_adaptive_pd_robust",
    "proposed_c2f_adaptive_pd_calibrated",
    "proposed_c2f_adaptive_pd_fusion_polish",
}


DEFAULT_METHODS: List[MethodName] = [
    method for method in METHODS if method not in EXPERIMENTAL_METHODS
]
