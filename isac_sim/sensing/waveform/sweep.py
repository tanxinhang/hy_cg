"""waveform-check 用的扫描驱动：解析 eta 与物理 PSF 逐点对照。"""

from __future__ import annotations

from typing import Dict

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.waveform.psf import psf_local_capture, psf_main_bin


def sweep_compare_analytic_vs_psf(
    cfg: Config,
    n_samples: int = 64,
    rng: np.random.Generator | None = None,
) -> Dict[str, np.ndarray]:
    """Compare analytic eta-c / eta-loc against the physical OTFS PSF.

    Returns a dict with arrays:

    * ``frac_l``, ``frac_k``      -- the sampled fractional offsets.
    * ``eta_c_analytic``          -- main-bin energy from sinc^2.
    * ``eta_c_psf``               -- main-bin energy from the OTFS PSF.
    * ``eta_loc_analytic``        -- windowed energy from Dirichlet leakage.
    * ``eta_loc_psf``             -- windowed energy from the OTFS PSF.
    """
    from isac_sim.sensing.dd import eta_coarse, eta_local_dirichlet

    if rng is None:
        rng = np.random.default_rng(0)
    # Fractional residual after nearest-bin rounding lies in [-0.5, 0.5].
    fracs = rng.uniform(-0.5, 0.5, size=(n_samples, 2))  # frac_l, frac_k

    eta_c_a = np.empty(n_samples)
    eta_c_p = np.empty(n_samples)
    eta_loc_a = np.empty(n_samples)
    eta_loc_p = np.empty(n_samples)
    half_w = cfg.refine.half_width
    for i, (lf, kf) in enumerate(fracs):
        eta_c_a[i] = eta_coarse(lf, kf)
        eta_c_p[i] = psf_main_bin(cfg, kf, lf)
        eta_loc_a[i] = eta_local_dirichlet(cfg.waveform.N, cfg.waveform.L, kf, lf, half_w)
        eta_loc_p[i] = psf_local_capture(cfg, kf, lf, half_w)

    return {
        "frac_l": fracs[:, 0],
        "frac_k": fracs[:, 1],
        "eta_c_analytic": eta_c_a,
        "eta_c_psf": eta_c_p,
        "eta_loc_analytic": eta_loc_a,
        "eta_loc_psf": eta_loc_p,
    }
