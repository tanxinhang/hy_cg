from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import json
import shutil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import *
from .scenario import *

# -----------------------------
# Signal geometry and kernels
# -----------------------------

def bistatic_delay_doppler(cfg: SimConfig, sc: Scenario, tx: int, rx: int, q: int) -> Tuple[float, float]:
    pi = sc.uav_pos[tx]
    pj = sc.uav_pos[rx]
    uq = sc.tgt_pos[q]
    vi = sc.uav_vel[tx]
    vj = sc.uav_vel[rx]
    vq = sc.tgt_vel[q]
    riq = np.linalg.norm(uq - pi) + EPS
    rqj = np.linalg.norm(uq - pj) + EPS
    tau = (riq + rqj) / C0
    eiq = (uq - pi) / riq
    ejq = (uq - pj) / rqj
    # Bistatic Doppler: sum of radial rates on Tx-target and target-Rx legs.
    nu = ((vq - vi).dot(eiq) + (vq - vj).dot(ejq)) / cfg.wavelength
    return float(tau), float(nu)


def dd_center_bins(cfg: SimConfig, tau: float, nu: float) -> Tuple[float, float, bool]:
    # Delay bin in [0, N_tau); Doppler bin centered in [-N_nu/2, N_nu/2) then mapped to [0,N_nu)
    l = (tau / cfg.delay_bin_s) % cfg.n_tau
    nu_unamb = cfg.doppler_max_hz
    alias = abs(nu) > nu_unamb
    # periodic Doppler index
    k_raw = nu / cfg.doppler_bin_hz
    k = (k_raw + cfg.n_nu / 2) % cfg.n_nu
    return float(k), float(l), bool(alias)


def dirichlet_kernel(N: int, x: np.ndarray) -> np.ndarray:
    # Periodic sinc-like kernel normalized to 1 at x=0.
    x = np.asarray(x, dtype=float)
    denom = N * np.sin(np.pi * x / N)
    numer = np.sin(np.pi * x)
    out = np.empty_like(x, dtype=float)
    small = np.abs(denom) < 1e-10
    out[small] = 1.0
    out[~small] = numer[~small] / denom[~small]
    return out


def circ_delta(a: float, b: float, N: int) -> float:
    return float(((a - b + N/2) % N) - N/2)


def dirichlet_leakage_1d(N: int, center: float) -> np.ndarray:
    """Normalized Dirichlet leakage energy over an N-bin circular axis.

    ``center`` is intentionally fractional.  Integer centers collapse to a
    Kronecker-like one-bin spectrum, while fractional centers produce OTFS
    inter-Doppler/inter-delay leakage.  Normalizing the finite-precision spectrum
    makes the returned vector an energy distribution with sum one.
    """
    idx = np.arange(N, dtype=float)
    dx = ((idx - float(center) + N / 2) % N) - N / 2
    p = np.abs(dirichlet_kernel(N, dx)) ** 2
    return p / (float(np.sum(p)) + EPS)


def support_indices(center_k: float, center_l: float, w_k: int, w_l: int, N_k: int, N_l: int) -> List[Tuple[int, int]]:
    hk = w_k // 2
    hl = w_l // 2
    ck = int(np.round(center_k)) % N_k
    cl = int(np.round(center_l)) % N_l
    inds = []
    for dk in range(-hk, hk + 1):
        for dl in range(-hl, hl + 1):
            inds.append(((ck + dk) % N_k, (cl + dl) % N_l))
    return inds


def omega_dd_dirichlet(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> float:
    """OTFS Dirichlet leakage fraction into the protected DD support.

    The source and protected centers remain fractional until this function.
    Only the protected support window is rounded to physical DD bin indices.
    This is the optimization-layer proxy for fractional Doppler/inter-Doppler
    interference rather than a box-overlap test.
    """
    p_k = dirichlet_leakage_1d(cfg.n_nu, src_k)
    p_l = dirichlet_leakage_1d(cfg.n_tau, src_l)
    inds = support_indices(prot_k, prot_l, cfg.support_doppler_bins, cfg.support_delay_bins, cfg.n_nu, cfg.n_tau)
    e = 0.0
    for k, l in inds:
        e += float(p_k[k] * p_l[l])
    return float(min(max(e, 0.0), 1.0))


def dd_overlap_dirichlet(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> float:
    """Backward-compatible name for the OTFS Dirichlet leakage proxy."""
    return omega_dd_dirichlet(cfg, src_k, src_l, prot_k, prot_l)


def dd_overlap_no_spread(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> float:
    """Idealized no-spreading DD overlap: a point contributes only if its center falls inside the protected support."""
    dk = abs(circ_delta(src_k, prot_k, cfg.n_nu))
    dl = abs(circ_delta(src_l, prot_l, cfg.n_tau))
    return float((dk <= cfg.support_doppler_bins // 2 + 1e-12) and (dl <= cfg.support_delay_bins // 2 + 1e-12))


def dd_overlap_gaussian(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> float:
    """Gaussian DD-spread ablation kernel. Used only as a simple non-OTFS proxy."""
    inds = support_indices(prot_k, prot_l, cfg.support_doppler_bins, cfg.support_delay_bins, cfg.n_nu, cfg.n_tau)
    e = 0.0
    for k, l in inds:
        dk = circ_delta(k, src_k, cfg.n_nu)
        dl = circ_delta(l, src_l, cfg.n_tau)
        e += float(np.exp(-0.5 * ((dk / (cfg.gaussian_sigma_k + EPS))**2 + (dl / (cfg.gaussian_sigma_l + EPS))**2)))
    # Normalize approximately so that the peak total support energy is no larger than one.
    norm = 2.0 * np.pi * cfg.gaussian_sigma_k * cfg.gaussian_sigma_l
    return float(min(max(e / (norm + EPS), 0.0), 1.0))


def dd_overlap_by_mode(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float, mode: str) -> float:
    """DD overlap dispatcher used for kernel ablations."""
    mode = str(mode).lower()
    if mode in ("none", "no", "no_spread", "point"):
        return dd_overlap_no_spread(cfg, src_k, src_l, prot_k, prot_l)
    if mode in ("gaussian", "gauss"):
        return dd_overlap_gaussian(cfg, src_k, src_l, prot_k, prot_l)
    if mode in ("dirichlet", "sinc", "proxy"):
        return dd_overlap_dirichlet(cfg, src_k, src_l, prot_k, prot_l)
    if mode in ("actual", "actual_otfs", "otfs", "full"):
        return dd_overlap_actual_otfs(cfg, src_k, src_l, prot_k, prot_l)
    if mode in ("broadened", "broadened_dirichlet", "legacy"):
        return dd_overlap_broadened_dirichlet(cfg, src_k, src_l, prot_k, prot_l)
    raise ValueError(f"Unknown DD-kernel mode={mode!r}")


def dd_overlap_proxy(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> float:
    """Optimization-layer DD overlap. Default is Dirichlet; use actual_otfs only for diagnostics."""
    return dd_overlap_by_mode(cfg, src_k, src_l, prot_k, prot_l, cfg.proxy_kernel_mode)


_FULL_KERNEL_CACHE: Dict[Tuple[int, int, float, float, int, int], np.ndarray] = {}
_FULL_OVERLAP_CACHE: Dict[Tuple[int, int, float, float, int, int, int], float] = {}
_FULL_TEMPLATE_CACHE: Dict[Tuple[int, int, float, float, int, int, int], np.ndarray] = {}


def clear_full_otfs_caches() -> None:
    """Clear all actual-OTFS kernel/template/overlap caches.

    Long closed-loop tracking runs continuously change target positions and
    velocities, which can create many unique fractional delay-Doppler offsets.
    Clearing caches at worker-job boundaries prevents unbounded memory growth.
    """
    _FULL_KERNEL_CACHE.clear()
    _FULL_OVERLAP_CACHE.clear()
    _FULL_TEMPLATE_CACHE.clear()


def stable_method_offset(method: str) -> int:
    """Small deterministic integer used for method-specific RNG streams.

    Do not use Python's hash() here because it is randomized per process on many
    Python installations, which breaks reproducibility under multiprocessing.
    """
    mapping = {
        "random": 101,
        "nearest": 211,
        "strongest": 307,
        "collision_aware": 401,
        "collision_aware_switch": 503,
        "oracle": 601,
        "dd_only": 701,
        "dd_angle": 709,
        "global_utility": 801,
        "box_dd": 809,
        "proposed": 811,
    }
    return mapping.get(str(method), 997)


def _quantize_offset(x: float, step: float) -> float:
    step = max(float(step), 1e-6)
    return float(round(x / step) * step)


def otfs_modulate(cfg: SimConfig, Xdd: np.ndarray) -> np.ndarray:
    """Compact OTFS modulation: DD -> TF via ISFFT, then OFDM modulation."""
    # Xdd shape: N_nu x N_tau (Doppler x delay)
    # ISFFT convention: TF = F_Nnu * Xdd * F_Ntau^H (unitary via norm='ortho')
    Xtf = np.fft.fft(np.fft.ifft(Xdd, axis=1, norm='ortho'), axis=0, norm='ortho')
    # OFDM IFFT over subcarriers for each time slot.
    time_mat = np.fft.ifft(Xtf, axis=1, norm='ortho')
    return time_mat.reshape(-1)


def otfs_demodulate(cfg: SimConfig, y: np.ndarray) -> np.ndarray:
    Ymat = y.reshape(cfg.n_nu, cfg.n_tau)
    Ytf = np.fft.fft(Ymat, axis=1, norm='ortho')
    # SFFT inverse of above.
    Ydd = np.fft.ifft(np.fft.fft(Ytf, axis=1, norm='ortho'), axis=0, norm='ortho')
    return Ydd


def apply_fractional_delay_doppler(cfg: SimConfig, x: np.ndarray, delay_samp: float, doppler_hz: float) -> np.ndarray:
    L = x.size
    freqs = np.fft.fftfreq(L)  # cycles/sample
    X = np.fft.fft(x)
    # circular fractional delay
    y = np.fft.ifft(X * np.exp(-1j * 2 * np.pi * freqs * delay_samp))
    t = np.arange(L) * cfg.ts
    y = y * np.exp(1j * 2 * np.pi * doppler_hz * t)
    return y


def full_otfs_kernel(cfg: SimConfig, frac_k: float, frac_l: float) -> np.ndarray:
    """Calibrated OTFS-like DD impulse spread for a fractional DD offset.

    frac_k / frac_l are offsets in Doppler/delay bins relative to the impulse center.
    The implementation uses actual ISFFT/OFDM modulation, circular fractional delay,
    Doppler phase rotation, and demodulation. It is independent from the fast proxy
    used by the assignment algorithm.
    """
    # Quantize only for cache reuse. The caller already quantizes in overlap/template functions.
    key = (cfg.n_nu, cfg.n_tau, round(frac_k, 3), round(frac_l, 3), int(cfg.delta_f), int(cfg.fc/1e6))
    if key in _FULL_KERNEL_CACHE:
        return _FULL_KERNEL_CACHE[key]
    Xdd = np.zeros((cfg.n_nu, cfg.n_tau), dtype=complex)
    ck, cl = cfg.n_nu // 2, cfg.n_tau // 2
    Xdd[ck, cl] = 1.0
    x = otfs_modulate(cfg, Xdd)
    delay_samp = frac_l  # because one delay bin = one sample in this no-CP compact model
    doppler_hz = frac_k * cfg.doppler_bin_hz
    y = apply_fractional_delay_doppler(cfg, x, delay_samp, doppler_hz)
    Ydd = otfs_demodulate(cfg, y)
    # Shift center to zero-offset map. A source with offset (frac_k, frac_l)
    # appears at that offset relative to the protected support centered at (0,0).
    K = np.roll(np.roll(Ydd, -ck, axis=0), -cl, axis=1)
    K = K / (np.sqrt(np.sum(np.abs(K)**2)) + EPS)
    _FULL_KERNEL_CACHE[key] = K
    return K


def _support_inds_zero(cfg: SimConfig) -> List[Tuple[int, int]]:
    return support_indices(0.0, 0.0, cfg.support_doppler_bins, cfg.support_delay_bins, cfg.n_nu, cfg.n_tau)


def support_signal_vector_actual_otfs(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> np.ndarray:
    """Return the complex OTFS spread vector inside the protected support.

    This is used by the matched-filter detector. It is an actual kernel slice,
    not the one-bin signal approximation used in earlier versions.
    """
    off_k = circ_delta(src_k, prot_k, cfg.n_nu)
    off_l = circ_delta(src_l, prot_l, cfg.n_tau)
    qk = _quantize_offset(off_k, cfg.overlap_round_bins)
    ql = _quantize_offset(off_l, cfg.overlap_round_bins)
    key = (cfg.n_nu, cfg.n_tau, qk, ql, cfg.support_doppler_bins, cfg.support_delay_bins, int(cfg.fc/1e6))
    if key in _FULL_TEMPLATE_CACHE:
        return _FULL_TEMPLATE_CACHE[key]
    K = full_otfs_kernel(cfg, qk, ql)
    vec = np.array([K[k, l] for k, l in _support_inds_zero(cfg)], dtype=complex)
    _FULL_TEMPLATE_CACHE[key] = vec
    return vec


def support_signal_vector_no_spread(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> np.ndarray:
    off_k = circ_delta(src_k, prot_k, cfg.n_nu)
    off_l = circ_delta(src_l, prot_l, cfg.n_tau)
    vec = np.zeros(cfg.support_delay_bins * cfg.support_doppler_bins, dtype=complex)
    inds = _support_inds_zero(cfg)
    # Put all energy in the nearest protected support cell if the point lies inside the support.
    best_idx = None
    best_dist = 1e18
    for idx, (k, l) in enumerate(inds):
        dk = circ_delta(k, off_k, cfg.n_nu)
        dl = circ_delta(l, off_l, cfg.n_tau)
        d2 = dk*dk + dl*dl
        if d2 < best_dist:
            best_dist = d2
            best_idx = idx
    if dd_overlap_no_spread(cfg, src_k, src_l, prot_k, prot_l) > 0 and best_idx is not None:
        vec[best_idx] = 1.0 + 0.0j
    return vec


def support_signal_vector_gaussian(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> np.ndarray:
    off_k = circ_delta(src_k, prot_k, cfg.n_nu)
    off_l = circ_delta(src_l, prot_l, cfg.n_tau)
    vals = []
    for k, l in _support_inds_zero(cfg):
        dk = circ_delta(k, off_k, cfg.n_nu)
        dl = circ_delta(l, off_l, cfg.n_tau)
        vals.append(np.exp(-0.25 * ((dk / (cfg.gaussian_sigma_k + EPS))**2 + (dl / (cfg.gaussian_sigma_l + EPS))**2)))
    return np.asarray(vals, dtype=complex)


def support_signal_vector_dirichlet(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> np.ndarray:
    off_k = circ_delta(src_k, prot_k, cfg.n_nu)
    off_l = circ_delta(src_l, prot_l, cfg.n_tau)
    vals = []
    for k, l in _support_inds_zero(cfg):
        dk = circ_delta(k, off_k, cfg.n_nu)
        dl = circ_delta(l, off_l, cfg.n_tau)
        vals.append(dirichlet_kernel(cfg.n_nu, np.array([dk]))[0] * dirichlet_kernel(cfg.n_tau, np.array([dl]))[0])
    return np.asarray(vals, dtype=complex)


def support_signal_vector_full(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> np.ndarray:
    """Support template consistent with the evaluation kernel mode."""
    mode = str(cfg.eval_kernel_mode).lower()
    if mode in ("none", "no", "no_spread", "point"):
        return support_signal_vector_no_spread(cfg, src_k, src_l, prot_k, prot_l)
    if mode in ("gaussian", "gauss"):
        return support_signal_vector_gaussian(cfg, src_k, src_l, prot_k, prot_l)
    if mode in ("dirichlet", "sinc", "proxy"):
        return support_signal_vector_dirichlet(cfg, src_k, src_l, prot_k, prot_l)
    if mode in ("actual", "actual_otfs", "otfs", "full"):
        return support_signal_vector_actual_otfs(cfg, src_k, src_l, prot_k, prot_l)
    if mode in ("broadened", "broadened_dirichlet", "legacy"):
        # Broadening only affects overlap; use the Dirichlet template as a simple shape.
        return support_signal_vector_dirichlet(cfg, src_k, src_l, prot_k, prot_l)
    raise ValueError(f"Unknown eval_kernel_mode={cfg.eval_kernel_mode!r}")


def _component_from_power_and_template(P: float, tmpl_raw: np.ndarray, is_self: bool) -> SignalComponent:
    """Turn a path power and a raw support template into a detection component.

    The path contributes only the fraction of its energy captured in the protected
    support.  Earlier versions normalized the template without reducing amplitude,
    which was optimistic for paths whose spread lies partially outside the support.
    """
    cap = float(np.linalg.norm(tmpl_raw))
    return (float(np.sqrt(max(P, 0.0)) * cap), tmpl_raw, is_self)


def dd_overlap_actual_otfs(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> float:
    """Actual OTFS-like DD overlap used in independent Gate-1 evaluation.

    It calls full_otfs_kernel() and sums the physical PSF energy that falls into
    the protected support centered at the origin.
    """
    off_k = circ_delta(src_k, prot_k, cfg.n_nu)
    off_l = circ_delta(src_l, prot_l, cfg.n_tau)
    qk = _quantize_offset(off_k, cfg.overlap_round_bins)
    ql = _quantize_offset(off_l, cfg.overlap_round_bins)
    key = (cfg.n_nu, cfg.n_tau, qk, ql, cfg.support_doppler_bins, cfg.support_delay_bins, int(cfg.fc/1e6))
    if key in _FULL_OVERLAP_CACHE:
        return _FULL_OVERLAP_CACHE[key]
    vec = support_signal_vector_actual_otfs(cfg, qk, ql, 0.0, 0.0)
    e = float(np.sum(np.abs(vec)**2))
    e = float(min(max(e, 0.0), 1.0))
    _FULL_OVERLAP_CACHE[key] = e
    return e


def dd_overlap_broadened_dirichlet(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> float:
    """Legacy independent evaluation overlap: broadened analytic Dirichlet kernel.

    Kept as an ablation/debug option. It should not be the default Gate-1 full-link
    evaluation mode because it is still an analytic proxy.
    """
    off_k = circ_delta(src_k, prot_k, cfg.n_nu)
    off_l = circ_delta(src_l, prot_l, cfg.n_tau)
    inds = _support_inds_zero(cfg)
    e = 0.0
    for k, l in inds:
        dk = circ_delta(k, off_k, cfg.n_nu)
        dl = circ_delta(l, off_l, cfg.n_tau)
        main = (dirichlet_kernel(cfg.n_nu, np.array([dk]))[0] ** 2) * (dirichlet_kernel(cfg.n_tau, np.array([dl]))[0] ** 2)
        side1 = (dirichlet_kernel(cfg.n_nu, np.array([dk - 0.35]))[0] ** 2) * (dirichlet_kernel(cfg.n_tau, np.array([dl + 0.20]))[0] ** 2)
        side2 = (dirichlet_kernel(cfg.n_nu, np.array([dk + 0.35]))[0] ** 2) * (dirichlet_kernel(cfg.n_tau, np.array([dl - 0.20]))[0] ** 2)
        e += float(0.82 * main + 0.09 * side1 + 0.09 * side2)
    return float(min(max(e, 0.0), 1.0))


def dd_overlap_full(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> float:
    """Independent full-link DD overlap.

    Default is the actual OTFS-like kernel. Simpler modes are exposed only for
    kernel-ablation experiments that test whether OTFS spreading physics matters.
    """
    return dd_overlap_by_mode(cfg, src_k, src_l, prot_k, prot_l, cfg.eval_kernel_mode)

def tx_rx_angles(sc: Scenario, tx: int, rx: int, q: int) -> Tuple[float, float, float]:
    theta_t = angle_xy(sc.uav_pos[tx], sc.tgt_pos[q])
    theta_r = angle_xy(sc.uav_pos[rx], sc.tgt_pos[q])
    # bistatic angle at target
    a = sc.uav_pos[tx] - sc.tgt_pos[q]
    b = sc.uav_pos[rx] - sc.tgt_pos[q]
    beta = math.acos(float(np.clip(a.dot(b) / ((np.linalg.norm(a)+EPS)*(np.linalg.norm(b)+EPS)), -1, 1)))
    return theta_t, theta_r, beta


def antenna_gain(theta: float, bore: float, sigma_deg: float) -> float:
    sig = np.deg2rad(sigma_deg)
    return float(np.exp(-0.5 * (wrap_angle(theta - bore) / (sig + EPS))**2))


def bistatic_gain(cfg: SimConfig, sc: Scenario, tx: int, rx: int, q: int) -> float:
    theta_t, theta_r, beta = tx_rx_angles(sc, tx, rx, q)
    gt = antenna_gain(theta_t, sc.uav_bore[tx], cfg.fov_sigma_deg)
    gr = antenna_gain(theta_r, sc.uav_bore[rx], cfg.fov_sigma_deg)
    # simple bistatic RCS aspect model; keep bounded.
    rcs = sc.tgt_rcs[q] * (0.35 + 0.65 * np.cos(beta/2)**2)
    riq = np.linalg.norm(sc.tgt_pos[q] - sc.uav_pos[tx]) + EPS
    rqj = np.linalg.norm(sc.tgt_pos[q] - sc.uav_pos[rx]) + EPS
    return float(gt * gr * rcs / (riq**2 * rqj**2 + EPS))


def observation_signal(cfg: SimConfig, sc: Scenario, j: int, q: int) -> float:
    # Monostatic/self echo proxy.
    theta = angle_xy(sc.uav_pos[j], sc.tgt_pos[q])
    g = antenna_gain(theta, sc.uav_bore[j], cfg.fov_sigma_deg)
    r = np.linalg.norm(sc.tgt_pos[q] - sc.uav_pos[j]) + EPS
    return float(g**2 * sc.tgt_rcs[q] / (r**4 + EPS))


def angle_overlap(cfg: SimConfig, sc: Scenario, rx: int, q_src: int, q_prot: int) -> float:
    if math.isinf(cfg.sigma_theta_deg):
        return 1.0
    sig = np.deg2rad(cfg.sigma_theta_deg)
    th_src = angle_xy(sc.uav_pos[rx], sc.tgt_pos[q_src])
    th_prot = angle_xy(sc.uav_pos[rx], sc.tgt_pos[q_prot])
    d = wrap_angle(th_src - th_prot)
    return float(np.exp(-0.5 * (d / (sig + EPS))**2))


def path_dd_center(cfg: SimConfig, sc: Scenario, tx: int, rx: int, q: int) -> Tuple[float, float, bool, float, float]:
    tau, nu = bistatic_delay_doppler(cfg, sc, tx, rx, q)
    k, l, alias = dd_center_bins(cfg, tau, nu)
    return k, l, alias, tau, nu
