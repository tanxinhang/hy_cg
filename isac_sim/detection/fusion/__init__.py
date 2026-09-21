"""Soft-information fusion: weights, deflection and the target priority alpha_q."""

from isac_sim.detection.fusion.h0 import (
    fused_h0_variance,
    fused_h0_skewness,
)
from isac_sim.detection.fusion.link_stats import (
    effective_h1_mean_for_link,
    _base_std_for_link,
    h0_variance_for_link,
    deflection_variance_for_link,
    _per_link_arrays,
)
from isac_sim.detection.fusion.threshold import (
    calibrated_fused_threshold,
)
from isac_sim.detection.fusion.maxmin import (
    maxmin_support,
    worst_case_pd,
)
from isac_sim.detection.fusion.leximin import (
    leximin_gradient,
    leximin_potential,
)
from isac_sim.detection.fusion.utility import (
    selection_utility_from_pd,
    selection_utility,
    target_alpha,
)
from isac_sim.detection.fusion.weights import (
    compute_weights,
    deflection_for_links,
)
from isac_sim.detection.fusion.pd_prediction import (
    predicted_pd_for_links,
)

__all__ = [
    "effective_h1_mean_for_link",
    "_base_std_for_link",
    "h0_variance_for_link",
    "deflection_variance_for_link",
    "_per_link_arrays",
    "compute_weights",
    "deflection_for_links",
    "predicted_pd_for_links",
    "fused_h0_variance",
    "fused_h0_skewness",
    "calibrated_fused_threshold",
    "selection_utility_from_pd",
    "selection_utility",
    "target_alpha",
    "worst_case_pd",
    "maxmin_support",
    "leximin_potential",
    "leximin_gradient",
]
