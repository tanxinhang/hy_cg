"""OTFS waveform-level kernel and physical-validation helpers.

Ported from ``CodeCg/gate_otfs_collision/kernels.py`` (functions
``otfs_modulate`` / ``otfs_demodulate`` / ``apply_fractional_delay_doppler``
/ ``full_otfs_kernel``).  Used by the ``waveform-check`` experiment to give
the *analytic* eta-c / eta-loc estimates a physical reference: how much
energy does an actual OTFS impulse response capture in the main bin and in
the local ``(2W+1) x (2W+1)`` window?

The kernel follows the compact no-CP OTFS model used by the sibling
package.  This is not a standards-compliant OTFS receiver -- it is a
deterministic end-to-end modulation / channel / demodulation chain that
gives a physically meaningful impulse response for each fractional
``(frac_k, frac_l)`` DD offset.  Caching is keyed on a coarse
quantisation of the offset so a Monte-Carlo trial sees only a handful of
fresh kernel evaluations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from .config import Config


EPS = 1e-12


# ==========================================================================
# 1. OTFS modulation / demodulation (compact no-CP model)
# ==========================================================================
def otfs_modulate(Xdd: np.ndarray) -> np.ndarray:
    """Compact ISFFT + OFDM: ``Xdd`` (Doppler x delay) -> time-domain samples.

    Convention matches ``gate_otfs_collision/kernels.py``: unitary FFTs with
    ``norm='ortho'``.  The output length is ``N_nu * N_tau``.
    """
    Xtf = np.fft.fft(np.fft.ifft(Xdd, axis=1, norm="ortho"), axis=0, norm="ortho")
    time_mat = np.fft.ifft(Xtf, axis=1, norm="ortho")
    return time_mat.reshape(-1)


def otfs_demodulate(y: np.ndarray, n_nu: int, n_tau: int) -> np.ndarray:
    """Inverse of :func:`otfs_modulate`.  Returns the DD-domain grid."""
    Ymat = y.reshape(n_nu, n_tau)
    Ytf = np.fft.fft(Ymat, axis=1, norm="ortho")
    Ydd = np.fft.ifft(np.fft.fft(Ytf, axis=1, norm="ortho"), axis=0, norm="ortho")
    return Ydd


def apply_fractional_delay_doppler(
    x: np.ndarray,
    delay_samp: float,
    doppler_hz: float,
    ts: float,
) -> np.ndarray:
    """Apply a circular fractional delay and a Doppler phase rotation."""
    L = x.size
    freqs = np.fft.fftfreq(L)  # cycles per sample
    X = np.fft.fft(x)
    y = np.fft.ifft(X * np.exp(-1j * 2.0 * np.pi * freqs * delay_samp))
    t = np.arange(L) * ts
    y = y * np.exp(1j * 2.0 * np.pi * doppler_hz * t)
    return y


# ==========================================================================
# 2. Full OTFS PSF kernel (cached)
# ==========================================================================
@lru_cache(maxsize=4096)
def full_otfs_kernel(
    n_nu: int,
    n_tau: int,
    frac_k: float,
    frac_l: float,
    delta_f: float,
    fc_MHz: int,
) -> np.ndarray:
    """OTFS-like impulse response for a fractional DD offset.

    Returns an ``(N_nu, N_tau)`` complex grid whose values are the spread
    energy that lands in each physical DD bin.  The kernel is L2-normalised.
    """
    Xdd = np.zeros((n_nu, n_tau), dtype=complex)
    ck, cl = n_nu // 2, n_tau // 2
    Xdd[ck, cl] = 1.0
    ts = 1.0 / delta_f
    x = otfs_modulate(Xdd)
    doppler_bin_hz = delta_f / float(n_nu)
    y = apply_fractional_delay_doppler(x, frac_l, frac_k * doppler_bin_hz, ts)
    Ydd = otfs_demodulate(y, n_nu, n_tau)
    K = np.roll(np.roll(Ydd, -ck, axis=0), -cl, axis=1)
    norm = np.sqrt(np.sum(np.abs(K) ** 2))
    return K / (norm + EPS)


# ==========================================================================
# 3. Physical metrics (main-bin / window-capture energy)
# ==========================================================================
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
    :func:`isac_sim.dd.eta_local_dirichlet` and the paper's ``(2W+1) x
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


# ==========================================================================
# 4. Sweep driver used by the ``waveform-check`` experiment
# ==========================================================================
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
    from .dd import eta_coarse, eta_local_dirichlet

    if rng is None:
        rng = np.random.default_rng(0)
    fracs = rng.uniform(0.0, 1.0, size=(n_samples, 2))  # frac_l, frac_k

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