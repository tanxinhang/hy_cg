"""array（自 ``isac_sim/sensing/dd.py`` 拆出）。"""

from __future__ import annotations

import numpy as np
from isac_sim.core.config import Config

from isac_sim.sensing.dd.gain import eta_local_dirichlet, eta_local_sinc


def eta_local_array(cfg: Config, l_float: np.ndarray, k_float: np.ndarray) -> np.ndarray:
    """Vectorised :func:`eta_local` over the ``(M, M, Q)`` DD-centre array.

    Skips invalid DD centres (returns the coarse gain, which is also the
    default for disabled ``cfg.refine``).
    """
    half_w = cfg.refine.half_width
    kernel = cfg.refine.window_kernel.lower()
    out = np.empty_like(l_float, dtype=float)

    if kernel == "sinc":
        # Signed fractional offsets (kept in the same sign convention as
        # ``np.sinc``: np.sinc(x) = sin(pi x)/(pi x) and x = bin - center.)
        for idx in np.ndindex(*l_float.shape):
            lf = float(l_float[idx])
            kf = float(k_float[idx])
            if not np.isfinite(lf) or not np.isfinite(kf):
                out[idx] = 1.0
                continue
            out[idx] = eta_local_sinc(lf - round(lf), kf - round(kf), half_w)
        return out

    if kernel == "dirichlet":
        for idx in np.ndindex(*l_float.shape):
            lf = float(l_float[idx])
            kf = float(k_float[idx])
            if not np.isfinite(lf) or not np.isfinite(kf):
                out[idx] = 1.0
                continue
            out[idx] = eta_local_dirichlet(
                cfg.waveform.N,
                cfg.waveform.L,
                kf,
                lf,
                half_w,
            )
        return out

    raise ValueError(f"Unknown refine.window_kernel={kernel!r}")


def eta_fine_array(cfg: Config, eta_c: np.ndarray, eta_loc: np.ndarray) -> np.ndarray:
    """Apply :func:`eta_refine` elementwise.  When refinement is disabled the
    output is identical to the coarse gain so existing trials stay bit-exact.
    """
    if not (cfg.refine.enable or cfg.refine.apply_to_all):
        return eta_c.copy()
    if getattr(cfg.refine, "mode", "interp").lower() == "window":
        return np.clip(eta_loc, 0.0, 1.0)
    f = eta_c + cfg.refine.kappa_dd * (eta_loc - eta_c)
    return np.clip(f, cfg.refine.eta_min, 1.0)
