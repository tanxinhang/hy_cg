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
    # One OFDM symbol lasts 1/delta_f and contains n_tau time samples.
    ts = 1.0 / (n_tau * delta_f)
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


def waveform_llr_detection_check(
    cfg: Config,
    *,
    raw_gamma: float = 0.5,
    interference_gamma: float = 0.0,
    report_success: float = 1.0,
    target_offset: tuple[float, float] = (0.31, -0.27),
    interferer_offset: tuple[float, float] = (-0.18, 0.22),
    num_interferers: int = 1,
    clutter_gamma: float = 0.0,
    multipath_ratio: float = 0.0,
    sync_error: tuple[float, float] = (0.0, 0.0),
    n_trials: int = 30_000,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    r"""Validate the finite-look LLR from an OTFS matched-filter output.

    A desired fractional-DD PSF and a displaced interfering PSF are sliced by
    the configured local window.  Their complex overlap determines the actual
    post-matched-filter interference; complex Gaussian target coefficients and
    receiver noise are then drawn for every look.  Finally, a report succeeds
    with probability ``report_success`` or is replaced by the same zero-mean
    Gaussian erasure surrogate used by the system simulator.

    The returned analytic prediction uses the current moment-matched detector,
    while ``empirical_*`` comes from the waveform-derived Monte Carlo samples.
    """
    from .model import qfunc, threshold_from_pfa

    if rng is None:
        rng = np.random.default_rng(0)
    n_trials = max(int(n_trials), 1)
    n_looks = max(int(cfg.detect.n_looks), 1)
    chi = float(np.clip(report_success, 0.0, 1.0))
    g_raw = max(float(raw_gamma), 0.0)
    g_int = max(float(interference_gamma), 0.0)
    g_clutter = max(float(clutter_gamma), 0.0)
    multipath_ratio = max(float(multipath_ratio), 0.0)

    # Non-integer target and a nearby unresolved interferer.  Their local PSFs
    # overlap substantially, so this is a genuine multi-target leakage test
    # rather than an effectively orthogonal second target.
    target_k, target_l = map(float, target_offset)
    interferer_k, interferer_l = map(float, interferer_offset)
    target_kernel = full_otfs_kernel(
        cfg.waveform.N, cfg.waveform.L, target_k, target_l,
        float(cfg.waveform.delta_f), int(round(cfg.waveform.fc / 1e6)),
    )
    sync_k, sync_l = map(float, sync_error)
    template_kernel = full_otfs_kernel(
        cfg.waveform.N, cfg.waveform.L, target_k + sync_k, target_l + sync_l,
        float(cfg.waveform.delta_f), int(round(cfg.waveform.fc / 1e6)),
    )
    W = max(int(cfg.refine.half_width), 0)
    indices = [
        (dk % cfg.waveform.N, dl % cfg.waveform.L)
        for dk in range(-W, W + 1)
        for dl in range(-W, W + 1)
    ]
    h_true = np.asarray([target_kernel[k, l] for k, l in indices], dtype=complex)
    h_template = np.asarray([template_kernel[k, l] for k, l in indices], dtype=complex)
    template_energy = float(np.vdot(h_template, h_template).real)
    h_unit = h_template / np.sqrt(max(template_energy, EPS))
    local_window_energy = float(np.vdot(h_true, h_true).real)
    captured_energy = float(np.abs(np.vdot(h_unit, h_true)) ** 2)

    interferer_shifts = ((0.0, 0.0), (0.43, -0.31), (-0.37, 0.52), (0.61, 0.44))
    n_interferers = min(max(int(num_interferers), 1), len(interferer_shifts))
    leakage_terms = []
    for shift_k, shift_l in interferer_shifts[:n_interferers]:
        kernel = full_otfs_kernel(
            cfg.waveform.N, cfg.waveform.L,
            interferer_k + shift_k, interferer_l + shift_l,
            float(cfg.waveform.delta_f), int(round(cfg.waveform.fc / 1e6)),
        )
        vector = np.asarray([kernel[k, l] for k, l in indices], dtype=complex)
        leakage_terms.append(float(np.abs(np.vdot(h_unit, vector)) ** 2))
    leakage_projection = float(np.mean(leakage_terms))

    clutter_terms = []
    for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        kernel = full_otfs_kernel(
            cfg.waveform.N, cfg.waveform.L,
            target_k + 0.75 * np.cos(angle), target_l + 0.75 * np.sin(angle),
            float(cfg.waveform.delta_f), int(round(cfg.waveform.fc / 1e6)),
        )
        vector = np.asarray([kernel[k, l] for k, l in indices], dtype=complex)
        clutter_terms.append(float(np.abs(np.vdot(h_unit, vector)) ** 2))
    clutter_projection = float(np.mean(clutter_terms))

    multipath_projection = 0.0
    for power, (shift_k, shift_l) in zip((0.65, 0.35), ((0.22, 0.35), (-0.41, 0.18))):
        kernel = full_otfs_kernel(
            cfg.waveform.N, cfg.waveform.L, target_k + shift_k, target_l + shift_l,
            float(cfg.waveform.delta_f), int(round(cfg.waveform.fc / 1e6)),
        )
        vector = np.asarray([kernel[k, l] for k, l in indices], dtype=complex)
        multipath_projection += power * float(np.abs(np.vdot(h_unit, vector)) ** 2)

    background_variance = (
        1.0 + g_int * leakage_projection + g_clutter * clutter_projection
    )
    desired_variance = g_raw * (
        captured_energy + multipath_ratio * multipath_projection
    )
    gamma_effective = desired_variance / max(background_variance, EPS)

    shape = (n_trials, n_looks)
    cn = lambda: (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ) / np.sqrt(2.0)
    # Exact matched-filter projection.  The full PSFs above determine both the
    # captured desired energy and the interfering complex overlap.
    noise0, noise1 = cn(), cn()
    interference_variance = (
        g_int * leakage_projection + g_clutter * clutter_projection
    )
    interference0 = np.sqrt(interference_variance) * cn()
    interference1 = np.sqrt(interference_variance) * cn()
    target = np.sqrt(desired_variance) * cn()
    z0 = (noise0 + interference0) / np.sqrt(background_variance)
    z1 = (noise1 + interference1 + target) / np.sqrt(background_variance)
    x0 = np.sum(np.abs(z0) ** 2, axis=1)
    x1 = np.sum(np.abs(z1) ** 2, axis=1)

    a = gamma_effective / (1.0 + gamma_effective)
    llr0 = a * (x0 - n_looks)
    llr1 = a * (x1 - n_looks)
    delta = n_looks * gamma_effective ** 2 / (1.0 + gamma_effective)
    local_v0 = n_looks * gamma_effective ** 2 / (1.0 + gamma_effective) ** 2
    local_v1 = n_looks * gamma_effective ** 2
    failure_v = cfg.detect.soft_error_sigma_scale ** 2 * local_v0

    if chi < 1.0:
        success0 = rng.random(n_trials) < chi
        success1 = rng.random(n_trials) < chi
        failure0 = rng.normal(0.0, np.sqrt(max(failure_v, 0.0)), n_trials)
        failure1 = rng.normal(0.0, np.sqrt(max(failure_v, 0.0)), n_trials)
        received0 = np.where(success0, llr0, failure0)
        received1 = np.where(success1, llr1, failure1)
    else:
        received0, received1 = llr0, llr1

    mean1 = chi * delta
    var0 = chi * local_v0 + (1.0 - chi) * failure_v
    var1 = (
        chi * (local_v1 + (delta - mean1) ** 2)
        + (1.0 - chi) * (failure_v + mean1 ** 2)
    )
    mu3_h0 = chi * 2.0 * n_looks * a ** 3
    skew0 = mu3_h0 / max(var0, EPS) ** 1.5
    z = threshold_from_pfa(cfg)
    z_cf = z + (skew0 / 6.0) * (z * z - 1.0)
    threshold = z_cf * np.sqrt(max(var0, EPS))
    predicted_pd = float(qfunc((threshold - mean1) / np.sqrt(max(var1, EPS))))

    # With a successful singleton report, the finite-look energy is exactly
    # Gamma(n_looks, scale=1) under H0 and Gamma(n_looks, 1+gamma) under H1.
    # The Erlang survival function gives a dependency-free exact reference.
    def erlang_sf(x: float, order: int) -> float:
        if x <= 0.0:
            return 1.0
        term = 1.0
        series = 1.0
        for k in range(1, order):
            term *= x / k
            series += term
        return float(np.exp(-x) * series)

    if a > EPS:
        energy_threshold = n_looks + threshold / a
        exact_success_pfa = erlang_sf(energy_threshold, n_looks)
        exact_success_pd = erlang_sf(
            energy_threshold / (1.0 + gamma_effective), n_looks
        )
    else:
        energy_threshold = float("inf")
        exact_success_pfa = 0.0
        exact_success_pd = 0.0

    if failure_v > EPS:
        failure_tail = float(qfunc(threshold / np.sqrt(failure_v)))
    else:
        failure_tail = float(threshold < 0.0)
    exact_mixture_pfa = chi * exact_success_pfa + (1.0 - chi) * failure_tail
    exact_mixture_pd = chi * exact_success_pd + (1.0 - chi) * failure_tail

    # Calibrate the threshold against the actual singleton H0 mixture rather
    # than its first three moments.  This is exact for the waveform validation
    # model and exposes how much performance the Cornish--Fisher approximation
    # leaves on the table when packet failures are non-negligible.
    def mixture_tails(candidate: float) -> tuple[float, float]:
        energy_cut = n_looks + candidate / max(a, EPS)
        success_pfa = erlang_sf(energy_cut, n_looks)
        success_pd = erlang_sf(
            energy_cut / (1.0 + gamma_effective), n_looks
        )
        if failure_v > EPS:
            fail_tail = float(qfunc(candidate / np.sqrt(failure_v)))
        else:
            fail_tail = float(candidate < 0.0)
        return (
            chi * success_pfa + (1.0 - chi) * fail_tail,
            chi * success_pd + (1.0 - chi) * fail_tail,
        )

    scale = np.sqrt(max(var0, failure_v, EPS))
    lo, hi = -20.0 * scale, 20.0 * scale
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mixture_tails(mid)[0] > cfg.detect.Pfa_target:
            lo = mid
        else:
            hi = mid
    calibrated_threshold = 0.5 * (lo + hi)
    calibrated_exact_pfa, calibrated_exact_pd = mixture_tails(calibrated_threshold)

    return {
        "raw_gamma": g_raw,
        "interference_gamma": g_int,
        "num_interferers": float(n_interferers),
        "clutter_gamma": g_clutter,
        "multipath_ratio": multipath_ratio,
        "sync_error_k": sync_k,
        "sync_error_l": sync_l,
        "report_success": chi,
        "target_k": target_k,
        "target_l": target_l,
        "interferer_k": interferer_k,
        "interferer_l": interferer_l,
        "captured_energy": captured_energy,
        "local_window_energy": local_window_energy,
        "leakage_projection": leakage_projection,
        "clutter_projection": clutter_projection,
        "multipath_projection": multipath_projection,
        "gamma_effective": gamma_effective,
        "predicted_pd": predicted_pd,
        "exact_success_pd": exact_success_pd,
        "exact_success_pfa": exact_success_pfa,
        "exact_mixture_pd": exact_mixture_pd,
        "exact_mixture_pfa": exact_mixture_pfa,
        "empirical_pd": float(np.mean(received1 > threshold)),
        "empirical_pfa": float(np.mean(received0 > threshold)),
        "threshold": float(threshold),
        "calibrated_threshold": float(calibrated_threshold),
        "calibrated_exact_pd": calibrated_exact_pd,
        "calibrated_exact_pfa": calibrated_exact_pfa,
        "calibrated_empirical_pd": float(np.mean(received1 > calibrated_threshold)),
        "calibrated_empirical_pfa": float(np.mean(received0 > calibrated_threshold)),
        "n_trials": float(n_trials),
    }
