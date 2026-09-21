"""Correlation-aware soft-information fusion."""

from isac_sim.detection.corr.kernels import (
    EPS,
    _rhos,
    dd_laplace_kernel,
)
from isac_sim.detection.corr.matrices import (
    correlation_matrix,
    link_dd_bins,
    covariance_matrix,
)
from isac_sim.detection.corr.weights import (
    solve_psd,
    correlation_aware_weights,
    correlated_deflection,
    redundancy_index,
)

__all__ = [
    "EPS",
    "_rhos",
    "dd_laplace_kernel",
    "correlation_matrix",
    "link_dd_bins",
    "covariance_matrix",
    "solve_psd",
    "correlation_aware_weights",
    "correlated_deflection",
    "redundancy_index",
]
