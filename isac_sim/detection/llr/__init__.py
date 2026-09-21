"""Waveform-derived local log-likelihood ratio (LLR) soft statistic."""

from isac_sim.detection.llr.divergences import (
    llr_kld,
    llr_reverse_kld,
    llr_jeffreys,
    llr_h0_offset,
    optimal_fusion_weight,
)
from isac_sim.detection.llr.moments import (
    soft_mean,
    soft_var0,
    soft_var1,
)
from isac_sim.detection.llr.sampling import (
    draw_llr,
    draw_llr_erased,
)
from isac_sim.detection.llr.statistics import (
    EPS,
    llr_delta,
    llr_var0,
    llr_var1,
    llr_deflection,
)

__all__ = [
    "EPS",
    "llr_delta",
    "llr_var0",
    "llr_var1",
    "llr_kld",
    "llr_reverse_kld",
    "llr_jeffreys",
    "llr_h0_offset",
    "llr_deflection",
    "optimal_fusion_weight",
    "draw_llr",
    "draw_llr_erased",
    "soft_mean",
    "soft_var0",
    "soft_var1",
]
