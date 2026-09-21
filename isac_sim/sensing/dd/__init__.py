"""Delay-Doppler domain kernels and the C2F (coarse-to-fine) gain model."""

from isac_sim.sensing.dd.array import (
    eta_local_array,
    eta_fine_array,
)
from isac_sim.sensing.dd.gain import (
    eta_coarse,
    eta_local_dirichlet,
    eta_local_sinc,
    eta_local,
    eta_refine,
)
from isac_sim.sensing.dd.kernels import (
    EPS,
    dirichlet_kernel,
    _quantise_centre,
    leakage_1d,
    window_bins,
)

__all__ = [
    "EPS",
    "dirichlet_kernel",
    "_quantise_centre",
    "leakage_1d",
    "window_bins",
    "eta_coarse",
    "eta_local_dirichlet",
    "eta_local_sinc",
    "eta_local",
    "eta_refine",
    "eta_local_array",
    "eta_fine_array",
]
