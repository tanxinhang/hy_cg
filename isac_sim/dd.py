"""Delay-Doppler domain kernels and the C2F (coarse-to-fine) gain model.

These routines power the proposed method's DD-aware selection.  They are
ported/adapted from the sibling ``gate_otfs_collision`` package
(``CodeCg/gate_otfs_collision/kernels.py`` and ``Code/kernels.py``), where the
Dirichlet leakage kernel and the protected-window energy sum act as the
optimisation-layer proxy for OTFS inter-Doppler / inter-delay interference.

Paper mapping
-------------
* :func:`eta_coarse`         -- eq. (coarse_dd_gain), the main-bin fractional
                                 gain ``sinc^2(eps_tau) * sinc^2(eps_nu)``.
* :func:`eta_local`          -- the recoverable energy from a local
                                 ``(2W+1) x (2W+1)`` DD neighbourhood (the paper
                                 only states the definition; this module makes
                                 it explicit using either the Dirichlet
                                 leakage distribution or the analytic sinc
                                 window).
* :func:`eta_refine`         -- eq. (fine_dd_gain):
                                 ``min{1, max[eta_min, eta_c + kappa_dd *
                                 (eta_loc - eta_c)]}``.

Design choice: the window sum uses the *periodic* Dirichlet kernel that lives
in the OTFS frame (``dirichlet_kernel(N, x)``).  As ``N`` grows the periodic
sinc approaches the continuous ``sinc`` used in the existing coarse gain, so
the two agree asymptotically; the Dirichlet version is more faithful to the
finite-N OTFS grid used by the simulator (``Waveform.N`` / ``Waveform.L``).

All entry points are pure functions on numpy arrays.  No random draws, no
mutable module state apart from an LRU cache on ``leakage_1d`` which is keyed
on a coarse quantisation of the floating-point centre.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np

from .config import Config


# Local numeric guard. Kept identical to ``isac_sim.model.EPS`` (1e-12) so the
# two modules stay interchangeable; the value is large enough to avoid 0/0
# inside the Dirichlet limiter but small enough not to perturb the gain.
EPS = 1e-12


# ==========================================================================
# 1. Dirichlet leakage kernel
# ==========================================================================
def dirichlet_kernel(N: int, x) -> np.ndarray:
    """Periodic Dirichlet kernel ``D_N(x) = sin(pi x) / (N sin(pi x / N))``.

    Normalised to 1 at ``x == 0 mod N``.  Matches the convention used by the
    sibling ``gate_otfs_collision/kernels.py`` package so the two DD models
    agree to numerical precision.
    """
    x = np.asarray(x, dtype=float)
    denom = N * np.sin(np.pi * x / N)
    numer = np.sin(np.pi * x)
    out = np.empty_like(x, dtype=float)
    small = np.abs(denom) < 1e-10
    out[small] = 1.0
    out[~small] = numer[~small] / denom[~small]
    return out


def _quantise_centre(x: float, step: float = 1e-3) -> float:
    """Quantise a fractional DD centre for cache reuse.

    The window-summed leakage depends smoothly on the centre, so a 1e-3
    quantisation removes spurious recomputation without visible numerical
    drift.
    """
    return float(round(x / step) * step)


@lru_cache(maxsize=65536)
def leakage_1d(N: int, centre: float) -> np.ndarray:
    """Energy distribution of a fractional-DD source over a periodic N-bin axis.

    Returns a length-``N`` vector that sums to one.  ``p[b]`` is the share of
    the source's energy that leaks into physical bin ``b``.  Integer centres
    collapse to a near-Dirac one-bin distribution; fractional centres spread
    the energy over a few neighbours, modelling OTFS inter-Doppler /
    inter-delay interference.
    """
    centre = _quantise_centre(centre)
    idx = np.arange(N, dtype=float)
    # Signed circular offset from ``centre`` to integer bin ``idx``.
    dx = ((idx - centre + N / 2.0) % N) - N / 2.0
    p = np.abs(dirichlet_kernel(N, dx)) ** 2
    return p / (float(np.sum(p)) + EPS)


def window_bins(
    centre_k: float,
    centre_l: float,
    half_w: int,
    N_k: int,
    N_l: int,
) -> List[Tuple[int, int]]:
    """Bin indices inside a local ``(2*half_w+1) x (2*half_w+1)`` DD window.

    The window is centred on the nearest bin to ``(centre_k, centre_l)``.
    Bin indices wrap modulo ``N_k`` / ``N_l`` so the window stays inside the
    OTFS frame for every centre.
    """
    ck = int(np.round(centre_k)) % N_k
    cl = int(np.round(centre_l)) % N_l
    out: List[Tuple[int, int]] = []
    for dk in range(-half_w, half_w + 1):
        for dl in range(-half_w, half_w + 1):
            out.append(((ck + dk) % N_k, (cl + dl) % N_l))
    return out


# ==========================================================================
# 2. eta^c (main-bin fractional loss)
# ==========================================================================
def eta_coarse(delay_frac: float, doppler_frac: float) -> float:
    """Coarse main-bin DD gain.

    Matches ``dd_frac_loss`` already used by ``isac_sim.model.build_base_gains``
    so the two definitions are interchangeable.
    """
    return float((np.sinc(delay_frac) ** 2) * (np.sinc(doppler_frac) ** 2))


# ==========================================================================
# 3. eta^loc (local-window recoverable energy)
# ==========================================================================
def eta_local_dirichlet(
    N_k: int,
    N_l: int,
    centre_k: float,
    centre_l: float,
    half_w: int,
) -> float:
    """Recoverable DD energy inside the local window using Dirichlet leakage.

    The source centre stays fractional until this function.  Only the window
    is rounded to physical bin indices, mirroring how the coarse gain rounds
    its own offset (and the protected-window convention in the sibling
    ``omega_dd_dirichlet``).
    """
    p_k = leakage_1d(N_k, centre_k)
    p_l = leakage_1d(N_l, centre_l)
    inds = window_bins(centre_k, centre_l, half_w, N_k, N_l)
    e = 0.0
    for k, l in inds:
        e += float(p_k[k] * p_l[l])
    return float(min(max(e, 0.0), 1.0))


def eta_local_sinc(
    delay_frac: float,
    doppler_frac: float,
    half_w: int,
) -> float:
    """Recoverable DD energy using a continuous-sinc window.

    ``delay_frac`` / ``doppler_frac`` are the signed fractional offsets
    (``l_float - round(l_float)``); the window sum is over integer offsets
    ``Delta in [-half_w, +half_w]``.  Returns a value in ``[0, 1]``.
    """
    total = 0.0
    for dk in range(-half_w, half_w + 1):
        for dl in range(-half_w, half_w + 1):
            total += float((np.sinc(dl - delay_frac) ** 2) * (np.sinc(dk - doppler_frac) ** 2))
    return float(min(max(total, 0.0), 1.0))


def eta_local(
    cfg: Config,
    delay_frac: float,
    doppler_frac: float,
    l_float: float,
    k_float: float,
) -> float:
    """Recoverable DD energy inside the local ``(2W+1) x (2W+1)`` window.

    Selects the kernel via ``cfg.refine.window_kernel``:

    * ``"dirichlet"`` (default) uses the periodic Dirichlet leakage
      :func:`eta_local_dirichlet` and the *fractional* centres
      ``l_float`` / ``k_float``.
    * ``"sinc"`` uses the continuous-sinc window
      :func:`eta_local_sinc` and the *signed* fractional offsets.

    Both definitions reduce to the coarse gain when ``half_w == 0``.
    """
    r = cfg.refine
    if r.window_kernel.lower() == "sinc":
        return eta_local_sinc(delay_frac, doppler_frac, r.half_width)
    if r.window_kernel.lower() == "dirichlet":
        return eta_local_dirichlet(
            cfg.waveform.N,
            cfg.waveform.L,
            k_float,
            l_float,
            r.half_width,
        )
    raise ValueError(f"Unknown refine.window_kernel={r.window_kernel!r}")


# ==========================================================================
# 4. eta^f (refined gain with floor and clamp)
# ==========================================================================
def eta_refine(
    cfg: Config,
    eta_c: float,
    eta_loc: float,
) -> float:
    """Fine-grain DD gain.

    * ``cfg.refine.mode = "interp"`` (legacy): eq. (fine_dd_gain)
      ``min{1, max[eta_min, eta_c + kappa_dd * (eta_loc - eta_c)]}``.
    * ``cfg.refine.mode = "window"``: the fine estimator resolves the
      fractional delay-Doppler offset and therefore recovers the local-window
      energy exactly, so ``eta^f = eta^loc``.  No free parameter; the
      ``kappa_dd`` / ``eta_min`` heuristics disappear.
    """
    r = cfg.refine
    if getattr(r, "mode", "interp").lower() == "window":
        return float(min(1.0, max(0.0, eta_loc)))
    f = eta_c + r.kappa_dd * (eta_loc - eta_c)
    return float(min(1.0, max(r.eta_min, f)))


# ==========================================================================
# 5. Vectorised helpers used by build_base_gains
# ==========================================================================
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