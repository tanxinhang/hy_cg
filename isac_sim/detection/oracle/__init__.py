"""Exact small-scale oracle for the proposed link-selection problem."""

from isac_sim.detection.oracle.exhaustive import (
    oracle_exhaustive,
    greedy_objective,
)
from isac_sim.detection.oracle.helpers import (
    _comb,
    _selection_respects_hard_budgets,
)
from isac_sim.detection.oracle.joint import (
    joint_fusion_selection_oracle,
)
from isac_sim.detection.oracle.lexicographic import (
    lexicographic_objective_vector,
    lexicographic_gap_components,
)

__all__ = [
    "oracle_exhaustive",
    "_comb",
    "greedy_objective",
    "lexicographic_objective_vector",
    "lexicographic_gap_components",
    "joint_fusion_selection_oracle",
    "_selection_respects_hard_budgets",
]
