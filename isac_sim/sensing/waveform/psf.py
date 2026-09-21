"""从物理 PSF 直接量出主瓣能量与局部窗口捕获能量（解析 eta 的物理对照）。"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.waveform.dd_transform import full_otfs_kernel


def psf_main_bin(
    cfg: Config,
    frac_k: float,
    frac_l: float,
) -> float:
    """Fraction of the OTFS PSF energy that lands in the main DD bin.

    The full OTFS kernel is shifted so that an impulse with fractional
    offset ``(frac_k, frac_l)`` peaks at ``(round(frac_k), round(frac_l))``
    in the returned grid.  We therefore report ``|K[round(frac_k),
    round(frac_l)]|^2`` -- this is the *physical* counterpart of the
    analytic ``eta^c``.
    """
    K = full_otfs_kernel(
        cfg.waveform.N,
        cfg.waveform.L,
        round(frac_k, 3),
        round(frac_l, 3),
        float(cfg.waveform.delta_f),
        int(round(cfg.waveform.fc / 1e6)),
    )
    n_nu, n_tau = K.shape
    mk = int(round(frac_k)) % n_nu
    ml = int(round(frac_l)) % n_tau
    return float(np.abs(K[mk, ml]) ** 2)


def psf_local_capture(
    cfg: Config,
    frac_k: float,
    frac_l: float,
    half_w: int,
) -> float:
    """Fraction of the OTFS PSF energy inside the local ``(2W+1)^2`` window.

    The window is centred on ``(round(frac_k), round(frac_l))`` to mirror
    :func:`isac_sim.sensing.dd.eta_local_dirichlet` and the paper's ``(2W+1) x
    (2W+1)`` DD neighbourhood definition.
    """
    K = full_otfs_kernel(
        cfg.waveform.N,
        cfg.waveform.L,
        round(frac_k, 3),
        round(frac_l, 3),
        float(cfg.waveform.delta_f),
        int(round(cfg.waveform.fc / 1e6)),
    )
    n_nu, n_tau = K.shape
    mk = int(round(frac_k)) % n_nu
    ml = int(round(frac_l)) % n_tau
    e = 0.0
    for dk in range(-half_w, half_w + 1):
        for dl in range(-half_w, half_w + 1):
            e += float(np.abs(K[(mk + dk) % n_nu, (ml + dl) % n_tau]) ** 2)
    return float(min(max(e, 0.0), 1.0))
