"""OTFS 的时频/Delay-Doppler 变换原语与全 PSF 核（带缓存）。"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from isac_sim.sensing.model import EPS


def otfs_modulate(Xdd: np.ndarray) -> np.ndarray:
    """Compact ISFFT + OFDM: ``Xdd`` (Doppler x delay) -> time-domain samples.

    Convention matches ``gate_otfs_collision/kernels.py``: unitary FFTs with
    ``norm='ortho'``.  The output length is ``N_nu * N_tau``.
    """
    # Axes are (Doppler, delay).  The ISFFT is inverse along Doppler and
    # forward along delay; OFDM then applies the inverse transform along the
    # delay/subcarrier axis.
    Xtf = np.fft.fft(np.fft.ifft(Xdd, axis=0, norm="ortho"), axis=1, norm="ortho")
    time_mat = np.fft.ifft(Xtf, axis=1, norm="ortho")
    return time_mat.reshape(-1)


def otfs_demodulate(y: np.ndarray, n_nu: int, n_tau: int) -> np.ndarray:
    """Inverse of :func:`otfs_modulate`.  Returns the DD-domain grid."""
    Ymat = y.reshape(n_nu, n_tau)
    Ytf = np.fft.fft(Ymat, axis=1, norm="ortho")
    # SFFT: forward along Doppler and inverse along delay.
    Ydd = np.fft.ifft(np.fft.fft(Ytf, axis=0, norm="ortho"), axis=1, norm="ortho")
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
    # One OFDM symbol lasts 1/delta_f and contains n_tau time samples.
    ts = 1.0 / (n_tau * delta_f)
    x = otfs_modulate(Xdd)
    doppler_bin_hz = delta_f / float(n_nu)
    y = apply_fractional_delay_doppler(x, frac_l, frac_k * doppler_bin_hz, ts)
    Ydd = otfs_demodulate(y, n_nu, n_tau)
    K = np.roll(np.roll(Ydd, -ck, axis=0), -cl, axis=1)
    norm = np.sqrt(np.sum(np.abs(K) ** 2))
    return K / (norm + EPS)
