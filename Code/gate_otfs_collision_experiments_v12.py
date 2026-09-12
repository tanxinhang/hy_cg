#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate experiments for OTFS task-edge DD collision modeling.

This script is intentionally focused on two go/no-go questions:

Gate 1: Does task-edge DD collision translate into independent full-link
        OTFS detection degradation, under fixed task-cardinality assignment?
Gate 2: Do realistic multi-UAV/multi-target deployments naturally produce a
        sparse-heavy-tail collision regime where collision-aware assignment is useful?

The script separates:
  - Optimization proxy: fast Dirichlet/sinc DD collision kernel.
  - Evaluation kernel: actual calibrated OTFS-like ISFFT/OFDM/channel/SFFT impulse response.

It is not a full standards-grade OTFS modem. It is a compact research simulator
for validating whether the proposed collision-modeling direction is worth pursuing.

V12 updates:
  - Keeps V11 clean-output, multiprocessing, and CRN-tracking fixes.
  - Adds filter-ablation controls for the tracking gate: alpha, standard KF,
    and DD reliability-weighted information filter (DD-RWIF).
  - Adds optional DD-collision structured measurement pollution. This gate is
    intended to test whether DD-collision information carries value beyond
    ordinary Gaussian measurement covariance adaptation.
  - Keeps DD-RWIF intentionally conservative: in the unbiased Gaussian model it
    should be compared against standard KF to test redundancy; its value should
    emerge mainly under biased/cluttered DD-peak pollution.

Author: generated for research prototyping
"""

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

EPS = 1e-15
C0 = 299792458.0


# -----------------------------
# Configurations
# -----------------------------

@dataclass
class SimConfig:
    seed: int = 7
    out_dir: str = "./gate_otfs_collision_outputs"

    # OTFS grid
    n_tau: int = 64      # delay bins / subcarriers
    n_nu: int = 32       # Doppler bins / OFDM symbols
    delta_f: float = 30e3
    fc: float = 28e9

    # Scenario
    area_size_m: float = 1000.0
    uav_alt_m: float = 120.0
    target_alt_m: float = 0.0
    uav_speed_mps: float = 15.0
    target_speed_mps: float = 18.0

    # Visibility / task constraints
    k_uav_max: int = 2
    k_tgt_min: int = 2
    k_tgt_max: int = 4
    k_cand: int = 5
    b_min: float = 0.03
    range_r0_m: float = 650.0
    fov_sigma_deg: float = 60.0
    snr_floor: float = 1e-24

    # Collision model
    support_delay_bins: int = 3      # protected support half-ish total width (odd recommended)
    support_doppler_bins: int = 3
    dd_min: float = 2e-3
    ang_min: float = 5e-3
    g_min: float = 1e-32
    c_norm_min: float = 0.1          # effective collision if C / protected_self_signal > this threshold (-10 dB)
    sigma_theta_deg: float = 20.0
    sigma_theta_list: str = "10,20,40,inf"

    # Assignment objective weights
    mu_div: float = 0.15
    lambda_col: float = 1.0
    xi_switch: float = 0.0
    use_normalized_collision_cost: bool = True
    fixed_task_count_gate1: bool = True
    include_oracle_gate1: bool = False

    # Full-link evaluation controls
    eval_kernel_mode: str = "actual_otfs"  # no_spread | gaussian | dirichlet | actual_otfs | broadened_dirichlet
    proxy_kernel_mode: str = "dirichlet"   # no_spread | gaussian | dirichlet | actual_otfs (slow, diagnostics only)
    gaussian_sigma_k: float = 0.65          # Doppler-bin width for Gaussian DD-kernel ablation
    gaussian_sigma_l: float = 0.65          # Delay-bin width for Gaussian DD-kernel ablation
    detector_mode: str = "noncoherent_matched"  # noncoherent_matched | matched | energy | both
    coop_phase_mode: str = "random_phase"        # random_phase | coherent | noncoherent_power
    overlap_round_bins: float = 0.01       # cache quantization in DD bins for actual OTFS overlap
    candidate_pool_size: int = 10          # bounded coalition enumeration candidate count

    # Full-link detection
    pfa: float = 1e-2
    inner_mc: int = 400
    gate1_mc: int = 100
    gate2_mc: int = 30

    # Supplemental Gate-1 sweeps
    difficulty_mc: int = 30
    phase_mc: int = 30
    kernel_mc: int = 20
    velocity_mc: int = 20
    angle_mc: int = 20
    pred_mc: int = 20
    tracking_mc: int = 10
    difficulty_noise_scales: str = "0.3,1,3,10,30"
    phase_mode_list: str = "coherent,random_phase,noncoherent_power"
    kernel_eval_modes: str = "no_spread,gaussian,dirichlet,actual_otfs"
    velocity_list_mps: str = "0,20,50,100,150"

    # Prediction-robustness and multi-frame tracking gates
    pred_pos_sigma_list_m: str = "0,5,10,20,50"
    pred_vel_sigma_list_mps: str = "0,1,3,5,10"
    tracking_M: int = 15
    tracking_Q: int = 10
    num_frames: int = 30
    dt_frame: float = 0.2
    track_init_pos_std_m: float = 8.0
    track_init_vel_std_mps: float = 1.5
    track_pred_pos_std_m: float = 2.0
    track_pred_vel_std_mps: float = 0.5
    target_process_pos_std_m: float = 0.3
    target_process_vel_std_mps: float = 0.15
    track_meas_pos_floor_m: float = 3.0
    track_meas_vel_floor_mps: float = 0.5
    track_loss_pos_thr_m: float = 80.0
    track_loss_miss_thr: int = 3
    lambda_switch: float = 0.0
    tracking_crn: bool = True
    target_pd_fusion: str = "mean"  # mean | or
    tracking_reinit_lost: bool = False
    tracking_alpha_pos: float = 0.65
    tracking_alpha_vel: float = 0.55

    # V12 tracking-filter / pollution ablation controls
    tracker_mode: str = "alpha"           # alpha | kf | dd_rwif
    meas_pollution: str = "none"          # none | biased_peak | clutter_peak
    pollution_pmax: float = 0.35           # max probability of structured DD pollution
    pollution_c0: float = 3.0              # C/(C+C0) scale for pollution probability
    pollution_pos_bias_m: float = 80.0     # structured peak pull in meters
    pollution_vel_bias_mps: float = 8.0    # structured peak pull in m/s
    dd_weight_alpha: float = 1.0           # Pd exponent in DD-RWIF reliability weight
    dd_weight_beta: float = 0.15           # exp(-beta*Ccross) factor in DD-RWIF
    dd_weight_gamma_min: float = 0.05      # minimum information weight for detected measurements
    kf_process_pos_std_m: float = 1.0      # CV-KF process position std per frame
    kf_process_vel_std_mps: float = 0.4    # CV-KF process velocity std per frame
    kf_init_pos_std_m: float = 8.0
    kf_init_vel_std_mps: float = 1.5

    # Gate-2 sweeps
    m_list: str = "9,12,15,18"
    q_list: str = "6,8,10,12"
    ktmin_list: str = "2,3"
    kumax_list: str = "2,3"

    # Collision regime thresholds
    regime_i_max: float = 0.05
    regime_ii_max: float = 0.30
    regime_iii_min: float = 0.50
    top_frac: float = 0.10

    # Runtime / parallel execution
    num_workers: int = 1
    clear_worker_caches: bool = True
    save_all_detectors: bool = False
    save_diagnostics: bool = False
    fresh_out_dir: bool = False
    run_id: str = ""

    # Plotting
    dpi: int = 180

    @property
    def wavelength(self) -> float:
        return C0 / self.fc

    @property
    def sample_rate(self) -> float:
        return self.n_tau * self.delta_f

    @property
    def ts(self) -> float:
        return 1.0 / self.sample_rate

    @property
    def ofdm_symbol_time(self) -> float:
        # No CP in this compact simulator.
        return 1.0 / self.delta_f

    @property
    def delay_bin_s(self) -> float:
        return 1.0 / (self.n_tau * self.delta_f)

    @property
    def doppler_bin_hz(self) -> float:
        return 1.0 / (self.n_nu * self.ofdm_symbol_time)

    @property
    def doppler_max_hz(self) -> float:
        return 0.5 * self.n_nu * self.doppler_bin_hz


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def parse_float_list(s: str) -> List[float]:
    out = []
    for x in s.split(','):
        x = x.strip().lower()
        if not x:
            continue
        out.append(float('inf') if x in ('inf', 'infty', 'infinite') else float(x))
    return out


# -----------------------------
# Geometry helpers
# -----------------------------

def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def unit_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    n = np.linalg.norm(d) + EPS
    return d / n


def angle_xy(src: np.ndarray, dst: np.ndarray) -> float:
    d = dst[:2] - src[:2]
    return float(np.arctan2(d[1], d[0]))


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class Scenario:
    uav_pos: np.ndarray      # M x 3
    uav_vel: np.ndarray      # M x 3
    uav_bore: np.ndarray     # M, radians
    tgt_pos: np.ndarray      # Q x 3
    tgt_vel: np.ndarray      # Q x 3
    tgt_rcs: np.ndarray      # Q

    @property
    def M(self) -> int:
        return self.uav_pos.shape[0]

    @property
    def Q(self) -> int:
        return self.tgt_pos.shape[0]


def generate_gate1_scenario(cfg: SimConfig, rng: np.random.Generator) -> Scenario:
    """Controlled scenario: T1/T2 DD-angle close; T3/T4 DD close but angle-separated."""
    M, Q = 6, 4
    # UAVs on two arcs facing center.
    center = np.array([500.0, 500.0, cfg.target_alt_m])
    angles = np.deg2rad(np.array([-150, -105, -60, 40, 85, 130]))
    radius = 430.0
    uav_pos = np.zeros((M, 3))
    for m, a in enumerate(angles):
        uav_pos[m] = center + np.array([radius * np.cos(a), radius * np.sin(a), cfg.uav_alt_m])
    # Boresight to center.
    uav_bore = np.array([angle_xy(uav_pos[m], center) for m in range(M)])
    # Tangential-ish velocities.
    uav_vel = np.zeros((M, 3))
    for m, a in enumerate(angles):
        tangent = np.array([-np.sin(a), np.cos(a), 0.0])
        uav_vel[m] = cfg.uav_speed_mps * tangent

    # Targets: 1/2 are close in both space and direction for many UAVs.
    # 3/4 are chosen to have similar range/Doppler but wider angular separation.
    tgt_pos = np.array([
        [510.0, 490.0, cfg.target_alt_m],
        [540.0, 515.0, cfg.target_alt_m],
        [420.0, 610.0, cfg.target_alt_m],
        [620.0, 380.0, cfg.target_alt_m],
    ], dtype=float)
    tgt_vel = np.array([
        [14.0, -5.0, 0.0],
        [11.0, -3.0, 0.0],
        [12.0, 8.0, 0.0],
        [-7.0, 12.0, 0.0],
    ], dtype=float)
    # Mild RCS variation.
    tgt_rcs = np.array([1.0, 0.85, 0.9, 0.95])
    return Scenario(uav_pos, uav_vel, uav_bore, tgt_pos, tgt_vel, tgt_rcs)


def generate_clustered_scenario(cfg: SimConfig, M: int, Q: int, rng: np.random.Generator) -> Scenario:
    """Clustered UAV/target geometry for Gate 2 regime sweep."""
    G = 3
    centers = np.array([
        [250.0, 260.0, cfg.target_alt_m],
        [740.0, 260.0, cfg.target_alt_m],
        [500.0, 760.0, cfg.target_alt_m],
    ])
    # UAVs around cluster centers, elevated.
    uav_pos = np.zeros((M, 3))
    uav_vel = np.zeros((M, 3))
    uav_bore = np.zeros(M)
    for m in range(M):
        g = m % G
        # Offset around the cluster, enough spread for angular diversity.
        rad = rng.uniform(120, 260)
        ang = rng.uniform(0, 2*np.pi)
        xy = centers[g, :2] + rad * np.array([np.cos(ang), np.sin(ang)])
        xy = np.clip(xy, 20, cfg.area_size_m - 20)
        uav_pos[m] = [xy[0], xy[1], cfg.uav_alt_m + rng.normal(0, 8)]
        uav_bore[m] = angle_xy(uav_pos[m], centers[g])
        heading = uav_bore[m] + rng.normal(0, 0.4)
        uav_vel[m] = cfg.uav_speed_mps * np.array([np.cos(heading), np.sin(heading), 0.0])

    # Targets in local groups; each group has a few closely spaced targets.
    tgt_pos = np.zeros((Q, 3))
    tgt_vel = np.zeros((Q, 3))
    tgt_rcs = np.zeros(Q)
    for q in range(Q):
        g = q % G
        group_center = centers[g, :2] + rng.normal(0, 80, size=2)
        # Local grouping radius controls natural DD-angle collisions.
        xy = group_center + rng.normal(0, 35, size=2)
        xy = np.clip(xy, 30, cfg.area_size_m - 30)
        tgt_pos[q] = [xy[0], xy[1], cfg.target_alt_m]
        vel_ang = rng.uniform(0, 2*np.pi)
        speed = rng.uniform(5, cfg.target_speed_mps)
        tgt_vel[q] = speed * np.array([np.cos(vel_ang), np.sin(vel_ang), 0.0])
        tgt_rcs[q] = 10 ** rng.normal(0.0, 0.15)  # log-normal, mild
    return Scenario(uav_pos, uav_vel, uav_bore, tgt_pos, tgt_vel, tgt_rcs)


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


def dd_overlap_dirichlet(cfg: SimConfig, src_k: float, src_l: float, prot_k: float, prot_l: float) -> float:
    """Dirichlet/sinc proxy energy from a source DD point falling inside protected support."""
    inds = support_indices(prot_k, prot_l, cfg.support_doppler_bins, cfg.support_delay_bins, cfg.n_nu, cfg.n_tau)
    e = 0.0
    for k, l in inds:
        dk = circ_delta(k, src_k, cfg.n_nu)
        dl = circ_delta(l, src_l, cfg.n_tau)
        amp = dirichlet_kernel(cfg.n_nu, np.array([dk]))[0] * dirichlet_kernel(cfg.n_tau, np.array([dl]))[0]
        e += float(np.abs(amp)**2)
    return float(min(max(e, 0.0), 1.0))


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


# -----------------------------
# Visibility and assignment
# -----------------------------

def build_visibility(cfg: SimConfig, sc: Scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M, Q = sc.M, sc.Q
    b = np.zeros((M, Q), dtype=float)
    U = np.zeros((M, Q), dtype=float)
    snr_obs = np.zeros((M, Q), dtype=float)
    for j in range(M):
        for q in range(Q):
            r = np.linalg.norm(sc.tgt_pos[q] - sc.uav_pos[j]) + EPS
            chi_range = np.exp(-r**2 / (2 * cfg.range_r0_m**2))
            th = angle_xy(sc.uav_pos[j], sc.tgt_pos[q])
            chi_fov = np.exp(-0.5 * (wrap_angle(th - sc.uav_bore[j]) / np.deg2rad(cfg.fov_sigma_deg))**2)
            s = observation_signal(cfg, sc, j, q)
            snr_obs[j, q] = s / (cfg.snr_floor + EPS)
            chi_snr = s / (s + cfg.snr_floor + EPS)
            # For now, DD observability: 1 unless Doppler alias severe.
            k, l, alias, tau, nu = path_dd_center(cfg, sc, j, j, q)
            chi_dd = 0.75 if alias else 1.0
            b[j, q] = chi_range * chi_fov * chi_snr * chi_dd
            U[j, q] = math.log1p(snr_obs[j, q]) + 0.4 * b[j, q]
    v = (b >= cfg.b_min)
    # Enforce top-k candidates per UAV.
    for j in range(M):
        order = np.argsort(-b[j])
        keep = set(order[:min(cfg.k_cand, Q)].tolist())
        for q in range(Q):
            if q not in keep:
                v[j, q] = False
    return b, U, v


def angle_diversity(sc: Scenario, z: np.ndarray) -> float:
    M, Q = z.shape
    div = 0.0
    for q in range(Q):
        js = np.where(z[:, q] > 0.5)[0]
        for a, b in itertools.combinations(js, 2):
            th_a = angle_xy(sc.uav_pos[a], sc.tgt_pos[q])
            th_b = angle_xy(sc.uav_pos[b], sc.tgt_pos[q])
            div += np.sin(wrap_angle(th_a - th_b))**2
    return float(div)


def task_edge_list(z: np.ndarray) -> List[Tuple[int, int]]:
    js, qs = np.where(z > 0.5)
    return list(zip(js.tolist(), qs.tolist()))


def build_collision_edges(cfg: SimConfig, sc: Scenario, z: np.ndarray, use_full_kernel: bool = False) -> pd.DataFrame:
    edges = []
    tasks = task_edge_list(z)
    # Precompute protected self centers.
    prot_center: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for j, q in tasks:
        k, l, alias, tau, nu = path_dd_center(cfg, sc, j, j, q)
        prot_center[(j, q)] = (k, l)

    for i, qp in tasks:
        for j, q in tasks:
            if qp == q:
                continue
            G = bistatic_gain(cfg, sc, i, j, qp)
            src_k, src_l, alias, tau, nu = path_dd_center(cfg, sc, i, j, qp)
            prot_k, prot_l = prot_center[(j, q)]
            omega_dd = dd_overlap_full(cfg, src_k, src_l, prot_k, prot_l) if use_full_kernel else dd_overlap_proxy(cfg, src_k, src_l, prot_k, prot_l)
            omega_ang = angle_overlap(cfg, sc, j, qp, q)
            C = G * omega_dd * omega_ang
            edges.append({
                "tx_uav": i, "src_target": qp, "rx_uav": j, "protected_target": q,
                "G": G, "omega_dd": omega_dd, "omega_ang": omega_ang,
                "C": C, "doppler_alias": alias,
                "src_k": src_k, "src_l": src_l, "prot_k": prot_k, "prot_l": prot_l,
            })
    df = pd.DataFrame(edges)
    if len(df) == 0:
        return pd.DataFrame(columns=["tx_uav","src_target","rx_uav","protected_target","G","omega_dd","omega_ang","C","S_protected","C_norm","doppler_alias","is_collision"])
    # Normalize collision energy by the protected task's self observation energy.
    # This prevents the collision density from being an artifact of an arbitrary
    # global power scale or a fixed top-quantile threshold.
    s_prot = []
    for _, rr in df.iterrows():
        s_prot.append(observation_signal(cfg, sc, int(rr["rx_uav"]), int(rr["protected_target"])))
    df["S_protected"] = np.asarray(s_prot, dtype=float)
    df["C_norm"] = df["C"] / (df["S_protected"] + cfg.snr_floor + EPS)
    df["is_collision"] = (
        (df["G"] > cfg.g_min) &
        (df["omega_dd"] > cfg.dd_min) &
        (df["omega_ang"] > cfg.ang_min) &
        (df["C_norm"] > cfg.c_norm_min)
    )
    return df


def collision_cost(edges: pd.DataFrame, normalized: bool = True) -> float:
    """Return collision cost used by the assignment objective.

    Normalized collision C_norm = C / (S_protected + noise) is better aligned
    with detection degradation than raw received interference energy. Raw C is
    still reported as a diagnostic.
    """
    if len(edges) == 0:
        return 0.0
    col = "C_norm" if normalized and "C_norm" in edges.columns else "C"
    return float(edges[col].sum())


def top_energy_fraction(x: np.ndarray, frac: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    total = float(np.sum(x))
    if total <= EPS:
        return 0.0
    n_top = max(1, int(math.ceil(frac * x.size)))
    return float(np.sum(np.sort(x)[-n_top:]) / (total + EPS))


def collision_summary(cfg: SimConfig, edges: pd.DataFrame, num_task_edges: int) -> Dict[str, float]:
    if len(edges) == 0:
        return dict(num_task_edges=num_task_edges, num_pairs=0, num_collisions=0, rho_col=0.0,
                    eta_top=0.0, eta_top_raw=0.0, eta_top_norm=0.0, N_DD=0, N_DDA=0,
                    mean_C=0.0, p95_C=0.0, max_C=0.0,
                    mean_C_norm=0.0, p95_C_norm=0.0, max_C_norm=0.0,
                    collision_cost_raw=0.0, collision_cost_norm=0.0,
                    doppler_alias_ratio=0.0)
    C = edges["C"].to_numpy(dtype=float)
    Cn = edges["C_norm"].to_numpy(dtype=float) if "C_norm" in edges.columns else C
    N_DD = int(np.sum(edges["omega_dd"].to_numpy() > cfg.dd_min))
    N_DDA = int(np.sum((edges["omega_dd"].to_numpy() > cfg.dd_min) & (edges["omega_ang"].to_numpy() > cfg.ang_min)))
    eta_raw = top_energy_fraction(C, cfg.top_frac)
    eta_norm = top_energy_fraction(Cn, cfg.top_frac)
    return dict(
        num_task_edges=num_task_edges,
        num_pairs=len(edges),
        num_collisions=int(edges["is_collision"].sum()),
        rho_col=float(edges["is_collision"].mean()),
        # Default eta_top now follows normalized collision because regime detection
        # is based on task-relative pollution, not absolute path power.
        eta_top=float(eta_norm),
        eta_top_raw=float(eta_raw),
        eta_top_norm=float(eta_norm),
        N_DD=N_DD,
        N_DDA=N_DDA,
        mean_C=float(np.mean(C)),
        p95_C=float(np.percentile(C, 95)),
        max_C=float(np.max(C)),
        mean_C_norm=float(np.mean(Cn)),
        p95_C_norm=float(np.percentile(Cn, 95)),
        max_C_norm=float(np.max(Cn)),
        collision_cost_raw=float(np.sum(C)),
        collision_cost_norm=float(np.sum(Cn)),
        doppler_alias_ratio=float(edges["doppler_alias"].mean()) if "doppler_alias" in edges else 0.0,
    )

def switching_cost(z: np.ndarray, z_prev: Optional[np.ndarray]) -> float:
    if z_prev is None:
        return 0.0
    return float(np.sum(np.abs(z.astype(float) - z_prev.astype(float))))


def objective(cfg: SimConfig, sc: Scenario, z: np.ndarray, U: np.ndarray, use_full_kernel: bool = False, z_prev: Optional[np.ndarray] = None) -> float:
    obs = float(np.sum(z * U))
    div = angle_diversity(sc, z)
    edges = build_collision_edges(cfg, sc, z, use_full_kernel=use_full_kernel)
    csum = collision_cost(edges, normalized=cfg.use_normalized_collision_cost)
    sw = switching_cost(z, z_prev)
    return obs + cfg.mu_div * div - cfg.lambda_col * csum - cfg.lambda_switch * sw

def is_feasible(cfg: SimConfig, z: np.ndarray, v: np.ndarray) -> bool:
    if np.any(z > v.astype(float) + 1e-9):
        return False
    if np.any(z.sum(axis=1) > cfg.k_uav_max + 1e-9):
        return False
    if np.any(z.sum(axis=0) < cfg.k_tgt_min - 1e-9):
        return False
    if np.any(z.sum(axis=0) > cfg.k_tgt_max + 1e-9):
        return False
    return True


def assignment_strongest(cfg: SimConfig, U: np.ndarray, v: np.ndarray) -> np.ndarray:
    M, Q = U.shape
    z = np.zeros((M, Q), dtype=int)
    load = np.zeros(M, dtype=int)
    # Coverage first target by target.
    for q in range(Q):
        cand = [j for j in np.argsort(-U[:, q]) if v[j, q] and load[j] < cfg.k_uav_max]
        for j in cand[:cfg.k_tgt_min]:
            z[j, q] = 1
            load[j] += 1
    # If some target not covered due capacity, relax by trying any candidate.
    for q in range(Q):
        while z[:, q].sum() < cfg.k_tgt_min:
            cand = [j for j in np.argsort(-U[:, q]) if v[j, q] and z[j, q] == 0]
            if not cand:
                break
            j = cand[0]
            z[j, q] = 1
            load[j] += 1
    return z


def assignment_nearest(cfg: SimConfig, sc: Scenario, v: np.ndarray) -> np.ndarray:
    M, Q = v.shape
    z = np.zeros((M, Q), dtype=int)
    load = np.zeros(M, dtype=int)
    dist = np.linalg.norm(sc.uav_pos[:, None, :] - sc.tgt_pos[None, :, :], axis=2)
    for q in range(Q):
        cand = [j for j in np.argsort(dist[:, q]) if v[j, q] and load[j] < cfg.k_uav_max]
        for j in cand[:cfg.k_tgt_min]:
            z[j, q] = 1
            load[j] += 1
    return z


def assignment_random(cfg: SimConfig, U: np.ndarray, v: np.ndarray, rng: np.random.Generator, max_tries: int = 200) -> np.ndarray:
    M, Q = U.shape
    best = None
    best_cov = -1
    for _ in range(max_tries):
        z = np.zeros((M, Q), dtype=int)
        load = np.zeros(M, dtype=int)
        order_q = rng.permutation(Q)
        for q in order_q:
            cand = [j for j in np.where(v[:, q])[0].tolist() if load[j] < cfg.k_uav_max]
            rng.shuffle(cand)
            for j in cand[:cfg.k_tgt_min]:
                z[j, q] = 1
                load[j] += 1
        cov = int(np.sum(z.sum(axis=0) >= cfg.k_tgt_min))
        if cov > best_cov:
            best, best_cov = z, cov
        if is_feasible(cfg, z, v):
            return z
    return best if best is not None else np.zeros((M, Q), dtype=int)


def assignment_collision_aware_greedy(cfg: SimConfig, sc: Scenario, U: np.ndarray, v: np.ndarray, z_prev: Optional[np.ndarray] = None) -> np.ndarray:
    """Fixed-cardinality collision-aware assignment.

    Earlier versions allowed the collision-aware method to add extra task edges up
    to K_tgt_max, which made Gate-1 unfair: more task edges naturally create more
    collision pairs. This version keeps exactly K_tgt_min UAVs per target and
    improves the assignment only by replacement / coalition swaps under the same
    task-cardinality budget as the baselines.
    """
    M, Q = U.shape

    # Start from strongest feasible assignment with exactly K_tgt_min per target.
    z = assignment_strongest(cfg, U, v)

    def force_exact_min(zcur: np.ndarray) -> np.ndarray:
        zcur = zcur.copy()
        # Remove surplus per target if any, keeping high-observation edges.
        for q in range(Q):
            js = np.where(zcur[:, q] > 0)[0].tolist()
            if len(js) > cfg.k_tgt_min:
                js_sorted = sorted(js, key=lambda j: U[j, q], reverse=True)
                for j in js_sorted[cfg.k_tgt_min:]:
                    zcur[j, q] = 0
        # Fill shortages if possible.
        load = zcur.sum(axis=1).astype(int)
        for q in range(Q):
            while zcur[:, q].sum() < cfg.k_tgt_min:
                cand = [j for j in np.argsort(-U[:, q]) if v[j, q] and zcur[j, q] == 0 and load[j] < cfg.k_uav_max]
                if not cand:
                    # Last-resort fallback: choose visible even if it violates load;
                    # diagnostics will mark infeasible. Avoid silently changing edge count.
                    cand = [j for j in np.argsort(-U[:, q]) if v[j, q] and zcur[j, q] == 0]
                if not cand:
                    break
                j = int(cand[0])
                zcur[j, q] = 1
                load[j] += 1
        return zcur

    z = force_exact_min(z)

    def target_combo_candidates(q: int, zbase: np.ndarray) -> List[Tuple[int, ...]]:
        # Current load excluding target q.
        load_excl = zbase.sum(axis=1).astype(int) - zbase[:, q].astype(int)
        cand = [j for j in range(M) if v[j, q] and load_excl[j] < cfg.k_uav_max]
        if len(cand) < cfg.k_tgt_min:
            cand = [j for j in range(M) if v[j, q]]
        combos = list(itertools.combinations(cand, cfg.k_tgt_min))
        # Keep enumeration bounded in larger cases. Unlike v2, the truncation is
        # collision-aware: a UAV with slightly lower observation gain but much lower
        # marginal collision cost must remain searchable.
        if len(combos) > 80:
            current = set(np.where(zbase[:, q] > 0)[0].tolist())
            z_without_q = zbase.copy()
            z_without_q[:, q] = 0
            base_cost = collision_cost(build_collision_edges(cfg, sc, z_without_q, use_full_kernel=False),
                                       normalized=cfg.use_normalized_collision_cost)
            scored = []
            for j in cand:
                z_tmp = z_without_q.copy()
                z_tmp[j, q] = 1
                cost_j = collision_cost(build_collision_edges(cfg, sc, z_tmp, use_full_kernel=False),
                                        normalized=cfg.use_normalized_collision_cost)
                # Observation-minus-collision score for candidate preselection.
                score = float(U[j, q] - cfg.lambda_col * (cost_j - base_cost))
                scored.append((score, j))
            pool = [j for _, j in sorted(scored, reverse=True)[:max(cfg.candidate_pool_size, cfg.k_tgt_min)]]
            cand2 = sorted(set(pool).union(current))
            cand2 = [j for j in cand2 if v[j, q] and load_excl[j] < cfg.k_uav_max]
            if len(cand2) < cfg.k_tgt_min:
                cand2 = sorted(set(pool).union(current))
                cand2 = [j for j in cand2 if v[j, q]]
            combos = list(itertools.combinations(cand2, cfg.k_tgt_min))
        return combos

    # Coordinate descent over target coalitions: replace the UAV set assigned to
    # one target at a time while preserving exactly K_tgt_min edges per target.
    cur_obj = objective(cfg, sc, z, U, use_full_kernel=False, z_prev=z_prev)
    improved = True
    outer = 0
    while improved and outer < 12:
        improved = False
        outer += 1
        for q in range(Q):
            best_obj = cur_obj
            best_z = None
            zbase = z.copy()
            zbase[:, q] = 0
            for combo in target_combo_candidates(q, z):
                z2 = zbase.copy()
                for j in combo:
                    z2[j, q] = 1
                if np.any(z2.sum(axis=1) > cfg.k_uav_max):
                    continue
                if np.any(z2.sum(axis=0) != cfg.k_tgt_min):
                    continue
                if np.any(z2 > v.astype(int)):
                    continue
                obj2 = objective(cfg, sc, z2, U, use_full_kernel=False, z_prev=z_prev)
                if obj2 > best_obj + 1e-12:
                    best_obj = obj2
                    best_z = z2
            if best_z is not None:
                z = best_z
                cur_obj = best_obj
                improved = True

    return z

def assignment_oracle_small(cfg: SimConfig, sc: Scenario, U: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exact target-wise enumeration for small Gate-1 cases, using exactly K_tgt_min per target."""
    M, Q = U.shape
    choices_per_q = []
    for q in range(Q):
        cand = [j for j in range(M) if v[j, q]]
        if len(cand) < cfg.k_tgt_min:
            cand = list(range(M))  # fallback for diagnostics only
        choices = list(itertools.combinations(cand, cfg.k_tgt_min))
        choices_per_q.append(choices)
    best_obj = -np.inf
    best_z = None
    for combo in itertools.product(*choices_per_q):
        z = np.zeros((M, Q), dtype=int)
        for q, js in enumerate(combo):
            for j in js:
                z[j, q] = 1
        if np.any(z.sum(axis=1) > cfg.k_uav_max):
            continue
        if not np.all(z <= v.astype(int)):
            continue
        obj = objective(cfg, sc, z, U, use_full_kernel=False)
        if obj > best_obj:
            best_obj, best_z = obj, z
    return best_z if best_z is not None else assignment_collision_aware_greedy(cfg, sc, U, v)


# -----------------------------
# Independent detection evaluation
# -----------------------------

def _noise_matrix(rng: np.random.Generator, n_mc: int, dim: int, total_power: float) -> np.ndarray:
    """Complex Gaussian noise/interference with expected total support power."""
    total_power = max(float(total_power), 0.0)
    return np.sqrt(total_power / (2 * dim + EPS)) * (
        rng.standard_normal((n_mc, dim)) + 1j * rng.standard_normal((n_mc, dim))
    )


# A signal component is (amplitude, normalized_template, is_self_path).
SignalComponent = Tuple[float, np.ndarray, bool]


def _compose_signal_matrix(
    cfg: SimConfig,
    rng: np.random.Generator,
    components: List[SignalComponent],
    n_mc: int,
    phase_mode: Optional[str] = None,
) -> np.ndarray:
    """Create Monte-Carlo signal realizations under a cooperative phase model.

    - coherent: all same-target components are phase aligned; this is an upper bound.
    - random_phase: cooperative bistatic paths have unknown independent phases.
      The self echo is kept as the receiver's phase reference.
    - noncoherent_power: same random-phase waveform generation, but downstream
      metrics use power-level accumulation rather than coherent norm.
    """
    if not components:
        return np.zeros((n_mc, cfg.support_delay_bins * cfg.support_doppler_bins), dtype=complex)
    mode = str(phase_mode or cfg.coop_phase_mode).lower()
    D = components[0][1].size
    sig = np.zeros((n_mc, D), dtype=complex)
    for amp, tmpl, is_self in components:
        if amp <= 0:
            continue
        tmpl = tmpl / (np.linalg.norm(tmpl) + EPS)
        if mode == "coherent" or is_self:
            phase = np.ones(n_mc, dtype=complex)
        elif mode in ("random_phase", "random", "noncoherent_power", "noncoherent"):
            phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=n_mc))
        else:
            raise ValueError(f"Unknown coop_phase_mode={cfg.coop_phase_mode!r}")
        sig += (amp * phase)[:, None] * tmpl[None, :]
    return sig


def _coherent_signal_vector(components: List[SignalComponent]) -> np.ndarray:
    """Phase-aligned desired signal vector; only an upper-bound model."""
    if not components:
        return np.zeros(0, dtype=complex)
    D = components[0][1].size
    sig = np.zeros(D, dtype=complex)
    for amp, tmpl, _is_self in components:
        if amp <= 0:
            continue
        tmpl = tmpl / (np.linalg.norm(tmpl) + EPS)
        sig += amp * tmpl
    return sig


def _expected_noncoherent_signal_power(components: List[SignalComponent]) -> float:
    """Expected signal power under independent unknown cooperative phases."""
    return float(sum(max(amp, 0.0) ** 2 for amp, _tmpl, _is_self in components))


def _detect_energy_components(
    cfg: SimConfig,
    rng: np.random.Generator,
    components: List[SignalComponent],
    sigma2_total: float,
) -> float:
    D = components[0][1].size if components else cfg.support_delay_bins * cfg.support_doppler_bins
    y0 = _noise_matrix(rng, cfg.inner_mc, D, sigma2_total)
    sig = _compose_signal_matrix(cfg, rng, components, cfg.inner_mc)
    y1 = sig + _noise_matrix(rng, cfg.inner_mc, D, sigma2_total)
    e0 = np.sum(np.abs(y0) ** 2, axis=1)
    e1 = np.sum(np.abs(y1) ** 2, axis=1)
    th = np.quantile(e0, 1.0 - cfg.pfa)
    return float(np.mean(e1 > th))


def _detect_coherent_matched_components(
    cfg: SimConfig,
    rng: np.random.Generator,
    components: List[SignalComponent],
    sigma2_total: float,
) -> float:
    """Coherent matched filter upper bound with known relative phases."""
    sig_vec = _coherent_signal_vector(components)
    D = sig_vec.size
    norm = float(np.linalg.norm(sig_vec))
    if norm <= EPS:
        return 0.0
    h = sig_vec / norm
    y0 = _noise_matrix(rng, cfg.inner_mc, D, sigma2_total)
    # For the coherent upper bound, H1 also uses phase-aligned signal.
    y1 = sig_vec[None, :] + _noise_matrix(rng, cfg.inner_mc, D, sigma2_total)
    t0 = np.abs(y0 @ np.conj(h)) ** 2
    t1 = np.abs(y1 @ np.conj(h)) ** 2
    th = np.quantile(t0, 1.0 - cfg.pfa)
    return float(np.mean(t1 > th))


def _detect_noncoherent_matched_components(
    cfg: SimConfig,
    rng: np.random.Generator,
    components: List[SignalComponent],
    sigma2_total: float,
) -> float:
    """Noncoherent matched-filter bank for unknown cooperative phases.

    The receiver knows the predicted DD support templates of the self/cooperative
    paths, but not their relative carrier phases. It therefore sums weighted
    projection energies instead of matching to a coherent vector sum.
    """
    if not components:
        return 0.0
    D = components[0][1].size
    T = []
    weights = []
    for amp, tmpl, _is_self in components:
        n = np.linalg.norm(tmpl)
        if amp <= 0 or n <= EPS:
            continue
        T.append(tmpl / n)
        weights.append(amp)
    if not T:
        return 0.0
    T = np.stack(T, axis=1)  # D x K
    weights = np.asarray(weights, dtype=float)
    # Normalize weights to avoid making the statistic depend on arbitrary bank size.
    weights = weights / (np.linalg.norm(weights) + EPS)

    y0 = _noise_matrix(rng, cfg.inner_mc, D, sigma2_total)
    sig = _compose_signal_matrix(cfg, rng, components, cfg.inner_mc, phase_mode="random_phase")
    y1 = sig + _noise_matrix(rng, cfg.inner_mc, D, sigma2_total)

    proj0 = y0 @ np.conj(T)
    proj1 = y1 @ np.conj(T)
    stat0 = np.sum((weights[None, :] ** 2) * np.abs(proj0) ** 2, axis=1)
    stat1 = np.sum((weights[None, :] ** 2) * np.abs(proj1) ** 2, axis=1)
    th = np.quantile(stat0, 1.0 - cfg.pfa)
    return float(np.mean(stat1 > th))


def evaluate_assignment_full_link(cfg: SimConfig, sc: Scenario, z: np.ndarray, rng: np.random.Generator) -> Dict[str, float]:
    """Independent full-link detection evaluation using actual OTFS-like kernels.

    Assignment still uses the fast proxy. This evaluator uses dd_overlap_full(),
    which defaults to full_otfs_kernel(), and builds desired support templates
    from OTFS PSF slices. Same-target cooperative returns are no longer assumed
    phase-aligned by default: cfg.coop_phase_mode controls coherent upper-bound,
    random-phase, or noncoherent-power evaluation.
    """
    tasks = task_edge_list(z)
    if not tasks:
        return dict(pd_mean=0, pd_p05=0, pd_min=0, pd_energy_mean=0, pd_energy_p05=0,
                    pd_matched_mean=0, pd_matched_p05=0,
                    pd_coherent_matched_mean=0, pd_coherent_matched_p05=0,
                    pd_noncoh_matched_mean=0, pd_noncoh_matched_p05=0,
                    rmse_tau_mean=np.inf, rmse_nu_mean=np.inf,
                    full_cross=0.0, full_cross_norm_sum=0.0, full_cross_norm_mean=0.0,
                    proxy_cross=0.0, proxy_cross_norm=0.0, full_scnr_mean=-300.0, full_scnr_p05=-300.0,
                    mean_num_signal_components=0.0, mean_coop_signal_power_frac=0.0)

    pd_energy_list = []
    pd_coherent_list = []
    pd_noncoh_list = []
    rmse_tau_list = []
    rmse_nu_list = []
    scnr_list = []
    full_cross_total = 0.0
    full_cross_norm_sum = 0.0
    full_cross_norm_list = []
    num_comp_list = []
    coop_frac_list = []

    for j, q in tasks:
        k_prot, l_prot, _, _, _ = path_dd_center(cfg, sc, j, j, q)

        components: List[SignalComponent] = []
        S_self = observation_signal(cfg, sc, j, q)
        tmpl_self = support_signal_vector_full(cfg, k_prot, l_prot, k_prot, l_prot)
        components.append(_component_from_power_and_template(S_self, tmpl_self, True))

        coop_power_sum = 0.0
        for i, qi in tasks:
            if qi == q and i != j:
                k_src, l_src, alias, tau, nu = path_dd_center(cfg, sc, i, j, q)
                tmpl = support_signal_vector_full(cfg, k_src, l_src, k_prot, l_prot)
                P = 0.8 * bistatic_gain(cfg, sc, i, j, q)
                comp = _component_from_power_and_template(P, tmpl, False)
                coop_power_sum += max(comp[0] ** 2, 0.0)
                components.append(comp)

        mode = str(cfg.coop_phase_mode).lower()
        if mode == "coherent":
            S_eff = float(np.sum(np.abs(_coherent_signal_vector(components)) ** 2))
        else:
            # Under random unknown cooperative phases, cross terms average out.
            S_eff = _expected_noncoherent_signal_power(components)

        num_comp_list.append(len(components))
        coop_frac_list.append(float(coop_power_sum / (S_eff + EPS)))

        # Cross-target interference power inside protected support.
        I = 0.0
        for i, qp in tasks:
            if qp == q:
                continue
            k_src, l_src, alias, tau, nu = path_dd_center(cfg, sc, i, j, qp)
            ov = dd_overlap_full(cfg, k_src, l_src, k_prot, l_prot)
            ang = angle_overlap(cfg, sc, j, qp, q)
            I += bistatic_gain(cfg, sc, i, j, qp) * ov * ang

        full_cross_total += I
        noise = cfg.snr_floor
        full_cross_norm = I / (S_eff + noise + EPS)
        full_cross_norm_sum += full_cross_norm
        full_cross_norm_list.append(full_cross_norm)

        sigma2_total = I + noise
        mode_det = str(cfg.detector_mode).lower()
        compute_all_detectors = bool(cfg.save_all_detectors or mode_det == "both")
        pd_ed = np.nan
        pd_coh = np.nan
        pd_ncmf = np.nan
        if compute_all_detectors or mode_det == "energy":
            pd_ed = _detect_energy_components(cfg, rng, components, sigma2_total)
        if compute_all_detectors or mode_det in ("matched", "coherent", "coherent_matched"):
            pd_coh = _detect_coherent_matched_components(cfg, rng, components, sigma2_total)
        if compute_all_detectors or mode_det not in ("energy", "matched", "coherent", "coherent_matched"):
            # Default practical detector and the primary detector for mode_det=="both".
            pd_ncmf = _detect_noncoherent_matched_components(cfg, rng, components, sigma2_total)
        pd_energy_list.append(pd_ed)
        pd_coherent_list.append(pd_coh)
        pd_noncoh_list.append(pd_ncmf)

        scnr = S_eff / (I + noise + EPS)
        scnr_list.append(10 * np.log10(scnr + EPS))
        rmse_tau_list.append(cfg.delay_bin_s / np.sqrt(scnr + EPS))
        rmse_nu_list.append(cfg.doppler_bin_hz / np.sqrt(scnr + EPS))

    proxy_edges = build_collision_edges(cfg, sc, z, use_full_kernel=False)
    proxy_cross = float(proxy_edges["C"].sum()) if len(proxy_edges) else 0.0
    pd_energy_arr = np.asarray(pd_energy_list, dtype=float)
    pd_coherent_arr = np.asarray(pd_coherent_list, dtype=float)
    pd_noncoh_arr = np.asarray(pd_noncoh_list, dtype=float)
    mode_det = str(cfg.detector_mode).lower()
    if mode_det == "energy":
        primary = pd_energy_arr
    elif mode_det in ("matched", "coherent", "coherent_matched"):
        primary = pd_coherent_arr
    else:
        # Default practical detector: noncoherent matched-filter bank.
        primary = pd_noncoh_arr

    def _nanmean(a: np.ndarray) -> float:
        return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")

    def _nanp(a: np.ndarray, q: float) -> float:
        return float(np.nanpercentile(a, q)) if np.isfinite(a).any() else float("nan")

    return dict(
        pd_mean=_nanmean(primary),
        pd_p05=_nanp(primary, 5),
        pd_min=float(np.nanmin(primary)) if np.isfinite(primary).any() else float("nan"),
        pd_energy_mean=_nanmean(pd_energy_arr),
        pd_energy_p05=_nanp(pd_energy_arr, 5),
        pd_matched_mean=_nanmean(primary if mode_det != "energy" else pd_noncoh_arr),
        pd_matched_p05=_nanp(primary if mode_det != "energy" else pd_noncoh_arr, 5),
        pd_coherent_matched_mean=_nanmean(pd_coherent_arr),
        pd_coherent_matched_p05=_nanp(pd_coherent_arr, 5),
        pd_noncoh_matched_mean=_nanmean(pd_noncoh_arr),
        pd_noncoh_matched_p05=_nanp(pd_noncoh_arr, 5),
        rmse_tau_mean=float(np.mean(rmse_tau_list)),
        rmse_nu_mean=float(np.mean(rmse_nu_list)),
        full_cross=float(full_cross_total),
        full_cross_norm_sum=float(full_cross_norm_sum),
        full_cross_norm_mean=float(np.mean(full_cross_norm_list)) if full_cross_norm_list else 0.0,
        proxy_cross=float(proxy_cross),
        proxy_cross_norm=float(collision_cost(proxy_edges, normalized=True)),
        full_scnr_mean=float(np.mean(scnr_list)),
        full_scnr_p05=float(np.percentile(scnr_list, 5)),
        mean_num_signal_components=float(np.mean(num_comp_list)) if num_comp_list else 0.0,
        mean_coop_signal_power_frac=float(np.mean(coop_frac_list)) if coop_frac_list else 0.0,
    )

# -----------------------------
# Gate experiments
# -----------------------------

def run_gate1(cfg: SimConfig, out: Path) -> None:
    rng = make_rng(cfg.seed)
    records = []
    scatter_records = []
    methods = ["random", "nearest", "strongest", "collision_aware"]
    if cfg.include_oracle_gate1:
        methods.append("oracle")
    for mc in range(cfg.gate1_mc):
        # Slightly jitter controlled scenario across MC.
        sc = generate_gate1_scenario(cfg, rng)
        sc.tgt_pos[:, :2] += rng.normal(0, 8.0, size=(sc.Q, 2))
        sc.tgt_vel[:, :2] += rng.normal(0, 1.2, size=(sc.Q, 2))
        b, U, v = build_visibility(cfg, sc)
        assigners = {
            "random": lambda: assignment_random(cfg, U, v, rng),
            "nearest": lambda: assignment_nearest(cfg, sc, v),
            "strongest": lambda: assignment_strongest(cfg, U, v),
            "collision_aware": lambda: assignment_collision_aware_greedy(cfg, sc, U, v),
            "oracle": lambda: assignment_oracle_small(cfg, sc, U, v),
        }
        for method in methods:
            z = assigners[method]()
            ev = evaluate_assignment_full_link(cfg, sc, z, rng)
            edges_proxy = build_collision_edges(cfg, sc, z, use_full_kernel=False)
            summ = collision_summary(cfg, edges_proxy, len(task_edge_list(z)))
            row = dict(mc=mc, method=method, feasible=is_feasible(cfg, z, v),
                       objective_proxy=objective(cfg, sc, z, U, use_full_kernel=False),
                       visibility_edges=int(v.sum()), coverage_ratio=float(np.mean(z.sum(axis=0) >= cfg.k_tgt_min)),
                       avg_tasks_per_uav=float(z.sum(axis=1).mean()))
            row.update(summ)
            row.update(ev)
            records.append(row)
            scatter_records.append(dict(mc=mc, method=method, proxy_cross=ev["proxy_cross"], proxy_cross_norm=ev["proxy_cross_norm"], full_cross=ev["full_cross"], full_cross_norm_mean=ev["full_cross_norm_mean"], pd_mean=ev["pd_mean"], rmse_tau_mean=ev["rmse_tau_mean"]))
    df = pd.DataFrame(records)
    df.to_csv(out / "gate1_trials.csv", index=False)
    summary = df.groupby("method").agg({
        "pd_mean": ["mean", "std"], "pd_p05": "mean",
        "pd_energy_mean": "mean", "pd_matched_mean": "mean",
        "pd_energy_p05": "mean", "pd_matched_p05": "mean",
        "rmse_tau_mean": "mean", "rmse_nu_mean": "mean",
        "proxy_cross": "mean", "proxy_cross_norm": "mean",
        "full_cross": "mean", "full_cross_norm_mean": "mean", "full_scnr_mean": "mean", "rho_col": "mean",
        "eta_top": "mean", "coverage_ratio": "mean", "objective_proxy": "mean",
        "num_task_edges": "mean", "num_pairs": "mean", "collision_cost_norm": "mean"
    })
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "gate1_summary.csv", index=False)

    # Bar plot: Pd and collisions.
    order = [m for m in methods if m in summary.method.values]
    x = np.arange(len(order))
    pd_means = [summary.loc[summary.method == m, "pd_mean_mean"].iloc[0] for m in order]
    pd_stds = [summary.loc[summary.method == m, "pd_mean_std"].iloc[0] for m in order]
    plt.figure(figsize=(8, 4.5))
    plt.bar(x, pd_means, yerr=pd_stds, capsize=3)
    plt.xticks(x, order, rotation=20)
    plt.ylabel(f"Mean $P_D$ @ $P_{{FA}}={cfg.pfa:g}$")
    plt.title("Gate 1: Independent OTFS-link detection performance")
    plt.tight_layout()
    plt.savefig(out / "gate1_pd_bar.png", dpi=cfg.dpi)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    sca = pd.DataFrame(scatter_records)
    for method in order:
        sub = sca[sca.method == method]
        plt.scatter(sub["full_cross_norm_mean"], sub["pd_mean"], s=15, alpha=0.65, label=method)
    plt.xscale("log")
    plt.xlabel("Full-link normalized cross interference")
    plt.ylabel(f"$P_D$ @ $P_{{FA}}={cfg.pfa:g}$")
    plt.title("Gate 1: Does collision reduce detection?")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "gate1_collision_vs_pd.png", dpi=cfg.dpi)
    plt.close()

    print("\n[Gate 1] Summary")
    print(summary.to_string(index=False))
    print(f"Saved: {out / 'gate1_summary.csv'}")


def _gate1_method_records_for_scenario(cfg: SimConfig, sc: Scenario, rng: np.random.Generator, mc: int, methods: Optional[List[str]] = None) -> List[Dict[str, float]]:
    """Evaluate Gate-1 methods for one scenario.

    This helper is used by supplemental sweeps so that the comparison remains
    identical to run_gate1(): fixed task edge count, proxy assignment, and actual
    OTFS-like full-link evaluation.
    """
    b, U, v = build_visibility(cfg, sc)
    if methods is None:
        methods = ["random", "nearest", "strongest", "collision_aware"]
        if cfg.include_oracle_gate1:
            methods.append("oracle")
    assigners = {
        "random": lambda: assignment_random(cfg, U, v, rng),
        "nearest": lambda: assignment_nearest(cfg, sc, v),
        "strongest": lambda: assignment_strongest(cfg, U, v),
        "collision_aware": lambda: assignment_collision_aware_greedy(cfg, sc, U, v),
        "oracle": lambda: assignment_oracle_small(cfg, sc, U, v),
    }
    rows: List[Dict[str, float]] = []
    for method in methods:
        z = assigners[method]()
        ev = evaluate_assignment_full_link(cfg, sc, z, rng)
        edges_proxy = build_collision_edges(cfg, sc, z, use_full_kernel=False)
        summ = collision_summary(cfg, edges_proxy, len(task_edge_list(z)))
        row = dict(
            mc=mc,
            method=method,
            feasible=is_feasible(cfg, z, v),
            objective_proxy=objective(cfg, sc, z, U, use_full_kernel=False),
            visibility_edges=int(v.sum()),
            coverage_ratio=float(np.mean(z.sum(axis=0) >= cfg.k_tgt_min)),
            avg_tasks_per_uav=float(z.sum(axis=1).mean()),
        )
        row.update(summ)
        row.update(ev)
        rows.append(row)
    return rows


def _make_gate1_jittered_scenario(cfg: SimConfig, rng: np.random.Generator) -> Scenario:
    sc = generate_gate1_scenario(cfg, rng)
    sc.tgt_pos[:, :2] += rng.normal(0, 8.0, size=(sc.Q, 2))
    sc.tgt_vel[:, :2] += rng.normal(0, 1.2, size=(sc.Q, 2))
    return sc



def _run_parallel_jobs(worker_fn, jobs: List[Tuple], num_workers: int, label: str) -> List:
    """Run independent jobs serially or with ProcessPoolExecutor.

    Results are returned in the same order as the input jobs.  The helper keeps
    the rest of the code deterministic and Windows-safe because it is called
    only under the main() guard.
    """
    if num_workers is None or int(num_workers) <= 1 or len(jobs) <= 1:
        out = []
        for idx, job in enumerate(jobs, 1):
            out.append(worker_fn(job))
            if len(jobs) > 1 and (idx == len(jobs) or idx % max(1, len(jobs)//10) == 0):
                print(f"[{label}] {idx}/{len(jobs)} jobs finished (serial)")
        return out

    max_workers = max(1, min(int(num_workers), len(jobs)))
    print(f"[{label}] Running {len(jobs)} independent jobs with {max_workers} workers...")
    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(worker_fn, job): idx for idx, job in enumerate(jobs)}
        done = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            done += 1
            if done == len(jobs) or done % max(1, len(jobs)//10) == 0:
                print(f"[{label}] {done}/{len(jobs)} jobs finished")
    return results



# -----------------------------
# Prediction-robustness and tracking helpers (V8)
# -----------------------------

def copy_scene_with_targets(sc: Scenario, tgt_pos: np.ndarray, tgt_vel: np.ndarray) -> Scenario:
    """Return a scene with the same UAV state and RCS but replaced target states."""
    return Scenario(
        uav_pos=sc.uav_pos.copy(),
        uav_vel=sc.uav_vel.copy(),
        uav_bore=sc.uav_bore.copy(),
        tgt_pos=np.asarray(tgt_pos, dtype=float).copy(),
        tgt_vel=np.asarray(tgt_vel, dtype=float).copy(),
        tgt_rcs=sc.tgt_rcs.copy(),
    )


def perturb_scene_prediction(sc_true: Scenario, pos_sigma_m: float, vel_sigma_mps: float, rng: np.random.Generator) -> Scenario:
    """Create a predicted scene by perturbing target position/velocity only."""
    pred_pos = sc_true.tgt_pos + rng.normal(0.0, pos_sigma_m, size=sc_true.tgt_pos.shape)
    pred_vel = sc_true.tgt_vel + rng.normal(0.0, vel_sigma_mps, size=sc_true.tgt_vel.shape)
    pred_pos[:, 2] = sc_true.tgt_pos[:, 2]
    pred_vel[:, 2] = 0.0
    return copy_scene_with_targets(sc_true, pred_pos, pred_vel)


def propagate_scene(cfg: SimConfig, sc: Scenario, rng: np.random.Generator) -> Scenario:
    """Simple constant-velocity target propagation for closed-loop tracking."""
    tgt_vel = sc.tgt_vel + rng.normal(0.0, cfg.target_process_vel_std_mps, size=sc.tgt_vel.shape)
    tgt_vel[:, 2] = 0.0
    tgt_pos = sc.tgt_pos + tgt_vel * cfg.dt_frame + rng.normal(0.0, cfg.target_process_pos_std_m, size=sc.tgt_pos.shape)
    tgt_pos[:, 0] = np.clip(tgt_pos[:, 0], 20.0, cfg.area_size_m - 20.0)
    tgt_pos[:, 1] = np.clip(tgt_pos[:, 1], 20.0, cfg.area_size_m - 20.0)
    tgt_pos[:, 2] = cfg.target_alt_m
    return copy_scene_with_targets(sc, tgt_pos, tgt_vel)


def predict_tracks(cfg: SimConfig, track_pos: np.ndarray, track_vel: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    pred_pos = track_pos + track_vel * cfg.dt_frame + rng.normal(0.0, cfg.track_pred_pos_std_m, size=track_pos.shape)
    pred_vel = track_vel + rng.normal(0.0, cfg.track_pred_vel_std_mps, size=track_vel.shape)
    pred_pos[:, 2] = cfg.target_alt_m
    pred_vel[:, 2] = 0.0
    return pred_pos, pred_vel


def predict_tracks_with_crn_noise(cfg: SimConfig, track_pos: np.ndarray, track_vel: np.ndarray,
                                  pred_pos_eps: np.ndarray, pred_vel_eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Predict tracks with method-independent unit-normal noise primitives.

    This is used by the tracking gate to implement common random numbers (CRN).
    Different methods still have different track states, but they see the same
    prediction perturbation realization in each frame/target.
    """
    pred_pos = track_pos + track_vel * cfg.dt_frame + cfg.track_pred_pos_std_m * pred_pos_eps
    pred_vel = track_vel + cfg.track_pred_vel_std_mps * pred_vel_eps
    pred_pos[:, 2] = cfg.target_alt_m
    pred_vel[:, 2] = 0.0
    return pred_pos, pred_vel


def fuse_target_pd(pds: List[float], mode: str = "mean") -> float:
    """Fuse per-receiver detection probabilities into a target-level Pd.

    mean: conservative average used in earlier gates.
    or:   independent noncoherent OR fusion, 1-prod(1-pd_i).
    """
    if not pds:
        return 0.0
    arr = np.clip(np.asarray(pds, dtype=float), 0.0, 1.0)
    if str(mode).lower() == "or":
        return float(1.0 - np.prod(1.0 - arr))
    return float(np.mean(arr))




def _kf_cv_matrices(cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant-velocity matrices for the 6-D state [p(3), v(3)]."""
    dt = float(cfg.dt_frame)
    F = np.eye(6)
    F[0:3, 3:6] = dt * np.eye(3)
    H = np.eye(6)
    q_pos = max(float(cfg.kf_process_pos_std_m), EPS) ** 2
    q_vel = max(float(cfg.kf_process_vel_std_mps), EPS) ** 2
    Q = np.diag([q_pos, q_pos, 1e-12, q_vel, q_vel, 1e-12])
    return F, H, Q


def _states_from_pos_vel(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    x = np.zeros((pos.shape[0], 6), dtype=float)
    x[:, 0:3] = pos
    x[:, 3:6] = vel
    return x


def _pos_vel_from_states(x: np.ndarray, cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    pos = x[:, 0:3].copy()
    vel = x[:, 3:6].copy()
    pos[:, 2] = cfg.target_alt_m
    vel[:, 2] = 0.0
    return pos, vel


def _ensure_spd(P: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    P = 0.5 * (P + np.swapaxes(P, -1, -2))
    if P.ndim == 2:
        return P + floor * np.eye(P.shape[0])
    eye = np.eye(P.shape[-1])[None, :, :]
    return P + floor * eye


def _tracking_predict_filter(cfg: SimConfig, st: Dict[str, np.ndarray],
                             pred_pos_eps: np.ndarray, pred_vel_eps: np.ndarray,
                             rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return predicted pos/vel plus optional KF state/covariance."""
    mode = str(cfg.tracker_mode).lower()
    if mode == "alpha":
        if cfg.tracking_crn:
            pred_pos, pred_vel = predict_tracks_with_crn_noise(cfg, st["track_pos"], st["track_vel"], pred_pos_eps, pred_vel_eps)
        else:
            pred_pos, pred_vel = predict_tracks(cfg, st["track_pos"], st["track_vel"], rng)
        return pred_pos, pred_vel, None, None

    F, _, Q = _kf_cv_matrices(cfg)
    x_prev = st.get("x")
    P_prev = st.get("P")
    if x_prev is None:
        x_prev = _states_from_pos_vel(st["track_pos"], st["track_vel"])
    if P_prev is None:
        q = st["track_pos"].shape[0]
        P0 = np.diag([
            cfg.kf_init_pos_std_m**2, cfg.kf_init_pos_std_m**2, 1e-12,
            cfg.kf_init_vel_std_mps**2, cfg.kf_init_vel_std_mps**2, 1e-12,
        ])
        P_prev = np.repeat(P0[None, :, :], q, axis=0)
    x_pred = (F @ x_prev[:, :, None])[:, :, 0]
    P_pred = np.empty_like(P_prev)
    for q in range(P_prev.shape[0]):
        P_pred[q] = _ensure_spd(F @ P_prev[q] @ F.T + Q)
    pred_pos, pred_vel = _pos_vel_from_states(x_pred, cfg)
    return pred_pos, pred_vel, x_pred, P_pred


def _tracking_kf_update_one(cfg: SimConfig, x_pred: np.ndarray, P_pred: np.ndarray,
                            meas_pos: np.ndarray, meas_vel: np.ndarray,
                            pos_std: float, vel_std: float,
                            pd_val: float, cross_val: float,
                            mode: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Standard KF or DD reliability-weighted information-form update.

    The returned gamma equals 1 for standard KF and equals the DD reliability
    information weight for DD-RWIF. In the unbiased Gaussian measurement model,
    this function is intended to test whether the DD weight is redundant with R.
    """
    _, H, _ = _kf_cv_matrices(cfg)
    z = np.r_[meas_pos, meas_vel].astype(float)
    pos_var = max(float(pos_std), EPS) ** 2
    vel_var = max(float(vel_std), EPS) ** 2
    R = np.diag([pos_var, pos_var, 1e-12, vel_var, vel_var, 1e-12])
    mode = str(mode).lower()
    gamma = 1.0
    if mode == "dd_rwif":
        c = float(cross_val) if np.isfinite(cross_val) else 1e3
        pd_c = float(np.clip(pd_val, 0.0, 1.0))
        gamma = (pd_c ** float(cfg.dd_weight_alpha)) * math.exp(-float(cfg.dd_weight_beta) * max(c, 0.0))
        gamma = float(np.clip(gamma, float(cfg.dd_weight_gamma_min), 1.0))
        # Information-form update: Lambda = P^-1 + gamma H^T R^-1 H.
        Pinv = np.linalg.pinv(_ensure_spd(P_pred))
        Rinv = np.linalg.pinv(_ensure_spd(R))
        Lambda = _ensure_spd(Pinv + gamma * (H.T @ Rinv @ H))
        eta = Pinv @ x_pred + gamma * (H.T @ Rinv @ z)
        P_new = _ensure_spd(np.linalg.pinv(Lambda))
        x_new = P_new @ eta
    else:
        S = _ensure_spd(H @ P_pred @ H.T + R)
        K = P_pred @ H.T @ np.linalg.pinv(S)
        innov = z - H @ x_pred
        x_new = x_pred + K @ innov
        P_new = _ensure_spd((np.eye(6) - K @ H) @ P_pred @ (np.eye(6) - K @ H).T + K @ R @ K.T)
    x_new[2] = cfg.target_alt_m
    x_new[5] = 0.0
    return x_new, P_new, gamma


def _apply_structured_dd_measurement_pollution(cfg: SimConfig,
                                               meas_pos: np.ndarray, meas_vel: np.ndarray,
                                               sc_true: Scenario, q: int, cross_val: float,
                                               u_pollute: float, dir_pos_unit: np.ndarray, dir_vel_unit: np.ndarray,
                                               rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    """Optional structured measurement pollution driven by target-level Ccross.

    none: unbiased Gaussian measurement model.
    biased_peak/clutter_peak: with probability p(C), pull the measurement toward
    a false/biased DD peak. This gives DD-collision information a non-redundant
    role beyond measurement variance and is the central V12 falsification test.
    """
    mode = str(cfg.meas_pollution).lower()
    c = float(cross_val) if np.isfinite(cross_val) else 0.0
    p_pol = float(cfg.pollution_pmax) * max(c, 0.0) / (max(c, 0.0) + float(cfg.pollution_c0) + EPS)
    p_pol = float(np.clip(p_pol, 0.0, 1.0))
    if mode in ("none", "off", "false") or u_pollute >= p_pol:
        return meas_pos, meas_vel, False, p_pol
    mp = meas_pos.copy()
    mv = meas_vel.copy()
    if mode in ("biased_peak", "bias", "biased"):
        mp[:2] += float(cfg.pollution_pos_bias_m) * dir_pos_unit[:2]
        mv[:2] += float(cfg.pollution_vel_bias_mps) * dir_vel_unit[:2]
    elif mode in ("clutter_peak", "clutter", "false_peak"):
        # Pull toward a randomly selected other target's current position/velocity,
        # mimicking a DD false-peak or association ambiguity under strong collision.
        candidates = [idx for idx in range(sc_true.Q) if idx != q]
        if candidates:
            q2 = int(candidates[int(rng.integers(0, len(candidates)))])
            frac = min(0.85, 0.35 + 0.50 * max(c, 0.0) / (max(c, 0.0) + cfg.pollution_c0 + EPS))
            mp = (1.0 - frac) * mp + frac * sc_true.tgt_pos[q2]
            mv = (1.0 - frac) * mv + frac * sc_true.tgt_vel[q2]
    else:
        raise ValueError(f"Unknown meas_pollution={cfg.meas_pollution!r}")
    mp[2] = cfg.target_alt_m
    mv[2] = 0.0
    return mp, mv, True, p_pol

def assign_by_method(cfg: SimConfig, method: str, sc_pred: Scenario, U: np.ndarray, v: np.ndarray, rng: np.random.Generator, z_prev: Optional[np.ndarray] = None) -> np.ndarray:
    if method == "random":
        return assignment_random(cfg, U, v, rng)
    if method == "nearest":
        return assignment_nearest(cfg, sc_pred, v)
    if method == "strongest":
        return assignment_strongest(cfg, U, v)
    if method in ("collision_aware", "collision_aware_switch"):
        return assignment_collision_aware_greedy(cfg, sc_pred, U, v, z_prev=z_prev)
    raise ValueError(f"Unknown assignment method={method!r}")


def evaluate_assignment_targets(cfg: SimConfig, sc: Scenario, z: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Per-target full-link proxy used by the tracking loop.

    V10 fixes:
      - Uncovered targets are marked as covered=False and cross-norm=NaN, not inf.
      - target Pd supports conservative mean fusion or noncoherent OR fusion.
      - Cross-norm aggregation is only defined over covered task targets.
    """
    rows = []
    tasks = task_edge_list(z)
    for q in range(sc.Q):
        task_js = [j for j, qq in tasks if qq == q]
        if not task_js:
            rows.append(dict(
                target=q, covered=False, pd_target=0.0, pd_target_p05=0.0,
                rmse_pos_proxy_m=np.nan, rmse_vel_proxy_mps=np.nan,
                full_cross_norm_mean=np.nan, full_cross_norm_median=np.nan, full_cross_norm_p90=np.nan,
                num_task_receivers=0,
            ))
            continue
        pds: List[float] = []
        pos_rmses: List[float] = []
        vel_rmses: List[float] = []
        cross_norms: List[float] = []
        for j in task_js:
            k_prot, l_prot, _, _, _ = path_dd_center(cfg, sc, j, j, q)
            components: List[SignalComponent] = []
            S_self = observation_signal(cfg, sc, j, q)
            tmpl_self = support_signal_vector_full(cfg, k_prot, l_prot, k_prot, l_prot)
            components.append(_component_from_power_and_template(S_self, tmpl_self, True))
            for i, qi in tasks:
                if qi == q and i != j:
                    k_src, l_src, *_ = path_dd_center(cfg, sc, i, j, q)
                    tmpl = support_signal_vector_full(cfg, k_src, l_src, k_prot, l_prot)
                    P = 0.8 * bistatic_gain(cfg, sc, i, j, q)
                    components.append(_component_from_power_and_template(P, tmpl, False))
            mode = str(cfg.coop_phase_mode).lower()
            if mode == "coherent":
                S_eff = float(np.sum(np.abs(_coherent_signal_vector(components)) ** 2))
            else:
                S_eff = _expected_noncoherent_signal_power(components)
            I = 0.0
            for i, qp in tasks:
                if qp == q:
                    continue
                k_src, l_src, *_ = path_dd_center(cfg, sc, i, j, qp)
                ov = dd_overlap_full(cfg, k_src, l_src, k_prot, l_prot)
                I += bistatic_gain(cfg, sc, i, j, qp) * ov
            sigma2_total = I + cfg.snr_floor
            mode_det = str(cfg.detector_mode).lower()
            if mode_det == "energy":
                pd_val = _detect_energy_components(cfg, rng, components, sigma2_total)
            elif mode_det in ("matched", "coherent", "coherent_matched"):
                pd_val = _detect_coherent_matched_components(cfg, rng, components, sigma2_total)
            else:
                pd_val = _detect_noncoherent_matched_components(cfg, rng, components, sigma2_total)
            scnr = S_eff / (I + cfg.snr_floor + EPS)
            tau_rmse = cfg.delay_bin_s / np.sqrt(scnr + EPS)
            nu_rmse = cfg.doppler_bin_hz / np.sqrt(scnr + EPS)
            pos_rmses.append(max(cfg.track_meas_pos_floor_m, 0.5 * C0 * tau_rmse))
            vel_rmses.append(max(cfg.track_meas_vel_floor_mps, cfg.wavelength * nu_rmse))
            pds.append(float(pd_val))
            cross_norm = I / (S_eff + cfg.snr_floor + EPS)
            cross_norms.append(float(cross_norm) if np.isfinite(cross_norm) else np.nan)
        finite_cross = np.asarray(cross_norms, dtype=float)
        rows.append(dict(
            target=q,
            covered=True,
            pd_target=fuse_target_pd(pds, cfg.target_pd_fusion),
            pd_target_p05=float(np.percentile(pds, 5)),
            rmse_pos_proxy_m=float(np.nanmean(pos_rmses)),
            rmse_vel_proxy_mps=float(np.nanmean(vel_rmses)),
            full_cross_norm_mean=float(np.nanmean(finite_cross)) if np.isfinite(finite_cross).any() else np.nan,
            full_cross_norm_median=float(np.nanmedian(finite_cross)) if np.isfinite(finite_cross).any() else np.nan,
            full_cross_norm_p90=float(np.nanpercentile(finite_cross, 90)) if np.isfinite(finite_cross).any() else np.nan,
            num_task_receivers=len(task_js),
        ))
    return pd.DataFrame(rows)


def _prediction_robustness_worker(job: Tuple[SimConfig, float, float, int]) -> List[Dict[str, float]]:
    """Worker for one (position-error, velocity-error, MC) prediction job."""
    cfg, pos_sig, vel_sig, mc = job
    methods = ["nearest", "strongest", "collision_aware"]
    rng = make_rng(cfg.seed + 60000 + 1000 * int(pos_sig * 10 + vel_sig) + mc)
    sc_true = generate_clustered_scenario(cfg, cfg.tracking_M, cfg.tracking_Q, rng)
    sc_pred = perturb_scene_prediction(sc_true, pos_sig, vel_sig, rng)
    b, U, v = build_visibility(cfg, sc_pred)
    rows: List[Dict[str, float]] = []
    for method in methods:
        z = assign_by_method(cfg, method, sc_pred, U, v, rng)
        ev = evaluate_assignment_full_link(cfg, sc_true, z, rng)
        edges = build_collision_edges(cfg, sc_true, z, use_full_kernel=True)
        summ = collision_summary(cfg, edges, len(task_edge_list(z)))
        row = dict(
            mc=mc,
            method=method,
            pred_pos_sigma_m=pos_sig,
            pred_vel_sigma_mps=vel_sig,
            feasible_pred=is_feasible(cfg, z, v),
            coverage_ratio=float(np.mean(z.sum(axis=0) >= cfg.k_tgt_min)),
        )
        row.update(summ)
        row.update(ev)
        rows.append(row)
    if cfg.clear_worker_caches:
        clear_full_otfs_caches()
    return rows


def run_prediction_robustness(cfg: SimConfig, out: Path) -> None:
    methods = ["nearest", "strongest", "collision_aware"]
    pos_list = parse_float_list(cfg.pred_pos_sigma_list_m)
    vel_list = parse_float_list(cfg.pred_vel_sigma_list_mps)
    jobs: List[Tuple[SimConfig, float, float, int]] = []
    for pos_sig in pos_list:
        for vel_sig in vel_list:
            for mc in range(cfg.pred_mc):
                jobs.append((cfg, pos_sig, vel_sig, mc))
    job_results = _run_parallel_jobs(_prediction_robustness_worker, jobs, cfg.num_workers, "Prediction robustness")
    records: List[Dict[str, float]] = []
    for rows in job_results:
        records.extend(rows)
    df = pd.DataFrame(records)
    df.to_csv(out / "pred_robust_trials.csv", index=False)
    summary = df.groupby(["pred_pos_sigma_m", "pred_vel_sigma_mps", "method"]).agg({
        "pd_mean": "mean", "pd_p05": "mean", "rmse_tau_mean": "mean", "rmse_nu_mean": "mean",
        "full_cross_norm_mean": "mean", "coverage_ratio": "mean", "rho_col": "mean", "eta_top": "mean",
    }).reset_index()
    summary.to_csv(out / "pred_robust_summary.csv", index=False)
    # Plots versus position error, averaged over velocity errors for readability.
    avg = summary.groupby(["pred_pos_sigma_m", "method"]).mean(numeric_only=True).reset_index()
    for metric, ylabel, fname in [
        ("pd_mean", "Mean $P_D$", "pred_pd_vs_pos_error.png"),
        ("pd_p05", "p05 $P_D$", "pred_p05_pd_vs_pos_error.png"),
        ("full_cross_norm_mean", "Full-link cross norm", "pred_cross_vs_pos_error.png"),
    ]:
        plt.figure(figsize=(7, 4.5))
        for method in methods:
            sub = avg[avg.method == method].sort_values("pred_pos_sigma_m")
            plt.plot(sub["pred_pos_sigma_m"], sub[metric], marker="o", label=method)
        plt.xlabel("Prediction position error std (m)")
        plt.ylabel(ylabel)
        plt.title("Prediction robustness")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=cfg.dpi)
        plt.close()
    print("\n[Prediction robustness] Summary preview")
    print(summary.head(20).to_string(index=False))
    print(f"Saved: {out / 'pred_robust_summary.csv'}")


def _tracking_worker(job: Tuple[SimConfig, int]) -> List[Dict[str, float]]:
    """Worker for one closed-loop tracking MC trial.

    V10 uses common random numbers (CRN) across methods for the tracking layer:
    initial track errors, prediction noises, detection uniforms, and measurement
    unit noises are method-independent.  Evaluation RNGs remain method-specific
    and are isolated from tracking randomness.
    """
    cfg, mc = job
    base_methods = ["nearest", "strongest", "collision_aware"]
    methods = base_methods + (["collision_aware_switch"] if cfg.lambda_switch > 0 else [])
    rows: List[Dict[str, float]] = []
    rng0 = make_rng(cfg.seed + 70000 + mc)
    sc_true = generate_clustered_scenario(cfg, cfg.tracking_M, cfg.tracking_Q, rng0)
    true0_pos = sc_true.tgt_pos.copy()
    true0_vel = sc_true.tgt_vel.copy()

    # Common initial track perturbations across all methods.
    if cfg.tracking_crn:
        rng_init = make_rng(cfg.seed + 71000 + 1000 * mc)
        init_pos_eps = rng_init.normal(0.0, 1.0, size=true0_pos.shape)
        init_vel_eps = rng_init.normal(0.0, 1.0, size=true0_vel.shape)
    states = {}
    for method in methods:
        if cfg.tracking_crn:
            init_pos = true0_pos + cfg.track_init_pos_std_m * init_pos_eps
            init_vel = true0_vel + cfg.track_init_vel_std_mps * init_vel_eps
        else:
            rng = make_rng(cfg.seed + 71000 + 1000 * mc + stable_method_offset(method))
            init_pos = true0_pos + rng.normal(0.0, cfg.track_init_pos_std_m, size=true0_pos.shape)
            init_vel = true0_vel + rng.normal(0.0, cfg.track_init_vel_std_mps, size=true0_vel.shape)
        state_dict = dict(
            track_pos=init_pos.copy(),
            track_vel=init_vel.copy(),
            miss=np.zeros(cfg.tracking_Q, dtype=int),
            z_prev=None,
        )
        state_dict["track_pos"][:, 2] = cfg.target_alt_m
        state_dict["track_vel"][:, 2] = 0.0
        if str(cfg.tracker_mode).lower() in ("kf", "kalman", "dd_rwif", "rwif", "information"):
            x0 = _states_from_pos_vel(state_dict["track_pos"], state_dict["track_vel"])
            P0 = np.diag([
                cfg.kf_init_pos_std_m**2, cfg.kf_init_pos_std_m**2, 1e-12,
                cfg.kf_init_vel_std_mps**2, cfg.kf_init_vel_std_mps**2, 1e-12,
            ])
            state_dict["x"] = x0
            state_dict["P"] = np.repeat(P0[None, :, :], cfg.tracking_Q, axis=0)
        states[method] = state_dict

    for frame in range(cfg.num_frames):
        rng_frame = make_rng(cfg.seed + 72000 + 10000 * mc + frame)
        if frame > 0:
            sc_true = propagate_scene(cfg, sc_true, rng_frame)

        # Common tracking primitives for this frame. These are independent of method.
        rng_crn = make_rng(cfg.seed + 72500 + 10000 * mc + frame)
        pred_pos_eps = rng_crn.normal(0.0, 1.0, size=sc_true.tgt_pos.shape)
        pred_vel_eps = rng_crn.normal(0.0, 1.0, size=sc_true.tgt_vel.shape)
        det_uniform = rng_crn.random(cfg.tracking_Q)
        meas_pos_eps = rng_crn.normal(0.0, 1.0, size=sc_true.tgt_pos.shape)
        meas_vel_eps = rng_crn.normal(0.0, 1.0, size=sc_true.tgt_vel.shape)
        pollution_uniform = rng_crn.random(cfg.tracking_Q)
        pollution_pos_dir = rng_crn.normal(0.0, 1.0, size=sc_true.tgt_pos.shape)
        pollution_vel_dir = rng_crn.normal(0.0, 1.0, size=sc_true.tgt_vel.shape)
        pollution_pos_dir[:, 2] = 0.0
        pollution_vel_dir[:, 2] = 0.0
        pollution_pos_dir = pollution_pos_dir / (np.linalg.norm(pollution_pos_dir, axis=1, keepdims=True) + EPS)
        pollution_vel_dir = pollution_vel_dir / (np.linalg.norm(pollution_vel_dir, axis=1, keepdims=True) + EPS)

        for method in methods:
            st = states[method]
            rng_assign = make_rng(cfg.seed + 73000 + 10000 * mc + 100 * frame + stable_method_offset(method))
            pred_pos, pred_vel, x_pred, P_pred = _tracking_predict_filter(cfg, st, pred_pos_eps, pred_vel_eps, rng_assign)
            sc_pred = copy_scene_with_targets(sc_true, pred_pos, pred_vel)
            b, U, v = build_visibility(cfg, sc_pred)
            z_prev = st["z_prev"] if method == "collision_aware_switch" else None
            method_assign = "collision_aware" if method == "collision_aware_switch" else method
            z = assign_by_method(cfg, method_assign, sc_pred, U, v, rng_assign, z_prev=z_prev)

            # Evaluation RNG is method-specific and isolated from tracking CRN.
            rng_eval = make_rng(cfg.seed + 73500 + 10000 * mc + 100 * frame + stable_method_offset(method))
            tgt = evaluate_assignment_targets(cfg, sc_true, z, rng_eval)
            tgt_sorted = tgt.sort_values("target")
            pd_vec = np.nan_to_num(tgt_sorted["pd_target"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            pos_std_vec = tgt_sorted["rmse_pos_proxy_m"].to_numpy(dtype=float)
            vel_std_vec = tgt_sorted["rmse_vel_proxy_mps"].to_numpy(dtype=float)
            pos_std_vec = np.where(np.isfinite(pos_std_vec), pos_std_vec, cfg.track_loss_pos_thr_m)
            vel_std_vec = np.where(np.isfinite(vel_std_vec), vel_std_vec, 10.0 * cfg.track_meas_vel_floor_mps)

            detected = det_uniform < pd_vec if cfg.tracking_crn else rng_assign.random(cfg.tracking_Q) < pd_vec
            cross_vec = tgt_sorted["full_cross_norm_mean"].to_numpy(dtype=float)
            polluted_flags = np.zeros(cfg.tracking_Q, dtype=bool)
            pollution_probs = np.zeros(cfg.tracking_Q, dtype=float)
            gamma_vals = np.ones(cfg.tracking_Q, dtype=float)
            tracker_mode = str(cfg.tracker_mode).lower()
            for q in range(cfg.tracking_Q):
                pos_err_before = np.linalg.norm(st["track_pos"][q, :2] - sc_true.tgt_pos[q, :2])
                was_lost = (pos_err_before > cfg.track_loss_pos_thr_m) or (st["miss"][q] >= cfg.track_loss_miss_thr)
                if detected[q]:
                    if cfg.tracking_crn:
                        meas_pos = sc_true.tgt_pos[q] + pos_std_vec[q] * meas_pos_eps[q]
                        meas_vel = sc_true.tgt_vel[q] + vel_std_vec[q] * meas_vel_eps[q]
                    else:
                        meas_pos = sc_true.tgt_pos[q] + rng_assign.normal(0.0, pos_std_vec[q], size=3)
                        meas_vel = sc_true.tgt_vel[q] + rng_assign.normal(0.0, vel_std_vec[q], size=3)
                    meas_pos[2] = cfg.target_alt_m
                    meas_vel[2] = 0.0
                    meas_pos, meas_vel, polluted, p_pol = _apply_structured_dd_measurement_pollution(
                        cfg, meas_pos, meas_vel, sc_true, q, cross_vec[q],
                        pollution_uniform[q] if cfg.tracking_crn else float(rng_assign.random()),
                        pollution_pos_dir[q] if cfg.tracking_crn else rng_assign.normal(0.0, 1.0, size=3),
                        pollution_vel_dir[q] if cfg.tracking_crn else rng_assign.normal(0.0, 1.0, size=3),
                        rng_assign,
                    )
                    polluted_flags[q] = polluted
                    pollution_probs[q] = p_pol

                    if tracker_mode == "alpha":
                        if cfg.tracking_reinit_lost and was_lost:
                            st["track_pos"][q] = meas_pos
                            st["track_vel"][q] = meas_vel
                        else:
                            st["track_pos"][q] = cfg.tracking_alpha_pos * meas_pos + (1 - cfg.tracking_alpha_pos) * pred_pos[q]
                            st["track_vel"][q] = cfg.tracking_alpha_vel * meas_vel + (1 - cfg.tracking_alpha_vel) * pred_vel[q]
                    else:
                        if x_pred is None or P_pred is None:
                            raise RuntimeError("KF/DD-RWIF tracker requires predicted state/covariance")
                        mode_update = "dd_rwif" if tracker_mode in ("dd_rwif", "rwif", "information") else "kf"
                        if cfg.tracking_reinit_lost and was_lost:
                            x_new = np.r_[meas_pos, meas_vel].astype(float)
                            P_new = np.diag([
                                pos_std_vec[q]**2, pos_std_vec[q]**2, 1e-12,
                                vel_std_vec[q]**2, vel_std_vec[q]**2, 1e-12,
                            ])
                            gamma = 1.0
                        else:
                            x_new, P_new, gamma = _tracking_kf_update_one(
                                cfg, x_pred[q], P_pred[q], meas_pos, meas_vel,
                                pos_std_vec[q], vel_std_vec[q], pd_vec[q], cross_vec[q], mode_update
                            )
                        st["x"][q] = x_new
                        st["P"][q] = P_new
                        st["track_pos"][q] = x_new[0:3]
                        st["track_vel"][q] = x_new[3:6]
                        gamma_vals[q] = gamma
                    st["miss"][q] = 0
                else:
                    if tracker_mode == "alpha":
                        st["track_pos"][q] = pred_pos[q]
                        st["track_vel"][q] = pred_vel[q]
                    else:
                        if x_pred is None or P_pred is None:
                            raise RuntimeError("KF/DD-RWIF tracker requires predicted state/covariance")
                        st["x"][q] = x_pred[q]
                        st["P"][q] = P_pred[q]
                        st["track_pos"][q] = pred_pos[q]
                        st["track_vel"][q] = pred_vel[q]
                    st["miss"][q] += 1
            pos_err = np.linalg.norm(st["track_pos"][:, :2] - sc_true.tgt_pos[:, :2], axis=1)
            vel_err = np.linalg.norm(st["track_vel"][:, :2] - sc_true.tgt_vel[:, :2], axis=1)
            track_lost = (pos_err > cfg.track_loss_pos_thr_m) | (st["miss"] >= cfg.track_loss_miss_thr)
            sw = switching_cost(z, st["z_prev"])
            st["z_prev"] = z.copy()
            covered = tgt_sorted["covered"].to_numpy(dtype=bool) if "covered" in tgt_sorted else (z.sum(axis=0) > 0)
            finite_cross = tgt_sorted["full_cross_norm_mean"].to_numpy(dtype=float)
            rows.append(dict(
                mc=mc, frame=frame, method=method,
                mean_track_pos_rmse_m=float(np.sqrt(np.mean(pos_err**2))),
                median_track_pos_error_m=float(np.median(pos_err)),
                p90_track_pos_error_m=float(np.percentile(pos_err, 90)),
                mean_track_vel_rmse_mps=float(np.sqrt(np.mean(vel_err**2))),
                median_track_vel_error_mps=float(np.median(vel_err)),
                p90_track_vel_error_mps=float(np.percentile(vel_err, 90)),
                track_loss_rate=float(np.mean(track_lost)),
                mean_pd_target=float(np.mean(pd_vec)),
                p05_pd_target=float(np.percentile(pd_vec, 5)),
                polluted_measurement_rate=float(np.mean(polluted_flags[detected])) if np.any(detected) else 0.0,
                pollution_prob_mean=float(np.mean(pollution_probs)),
                dd_info_weight_mean=float(np.mean(gamma_vals[detected])) if np.any(detected) else np.nan,
                tracker_mode=str(cfg.tracker_mode),
                meas_pollution=str(cfg.meas_pollution),
                covered_target_count=float(np.sum(covered)),
                uncovered_target_count=float(cfg.tracking_Q - np.sum(covered)),
                full_cross_norm_mean=float(np.nanmean(finite_cross)) if np.isfinite(finite_cross).any() else np.nan,
                full_cross_norm_median=float(np.nanmedian(finite_cross)) if np.isfinite(finite_cross).any() else np.nan,
                full_cross_norm_p90=float(np.nanpercentile(finite_cross, 90)) if np.isfinite(finite_cross).any() else np.nan,
                switch_count=float(sw),
                switch_rate=float(sw / (np.sum(z) + EPS)),
                coverage_ratio=float(np.mean(z.sum(axis=0) >= cfg.k_tgt_min)),
                one_receiver_coverage_ratio=float(np.mean(z.sum(axis=0) >= 1)),
            ))
    if cfg.clear_worker_caches:
        clear_full_otfs_caches()
    return rows


def run_tracking_experiment(cfg: SimConfig, out: Path) -> None:
    base_methods = ["nearest", "strongest", "collision_aware"]
    methods = base_methods + (["collision_aware_switch"] if cfg.lambda_switch > 0 else [])
    jobs: List[Tuple[SimConfig, int]] = [(cfg, mc) for mc in range(cfg.tracking_mc)]
    job_results = _run_parallel_jobs(_tracking_worker, jobs, cfg.num_workers, "Tracking")
    records: List[Dict[str, float]] = []
    for rows in job_results:
        records.extend(rows)
    df = pd.DataFrame(records)
    df.to_csv(out / "tracking_trials.csv", index=False)

    # Compact paper-facing summary. The raw tracking_trials.csv remains the full source table.
    agg_cols = {
        "mean_track_pos_rmse_m": "mean",
        "median_track_pos_error_m": "mean",
        "p90_track_pos_error_m": "mean",
        "track_loss_rate": "mean",
        "mean_pd_target": "mean",
        "p05_pd_target": "mean",
        "full_cross_norm_mean": "mean",
        "full_cross_norm_p90": "mean",
        "uncovered_target_count": "mean",
        "polluted_measurement_rate": "mean",
        "pollution_prob_mean": "mean",
        "dd_info_weight_mean": "mean",
        "switch_rate": "mean",
        "coverage_ratio": "mean",
    }
    if cfg.save_diagnostics:
        agg_cols.update({
            "mean_track_vel_rmse_mps": "mean",
            "median_track_vel_error_mps": "mean",
            "p90_track_vel_error_mps": "mean",
            "full_cross_norm_median": "mean",
            "covered_target_count": "mean",
            "one_receiver_coverage_ratio": "mean",
        })
    summary = df.groupby(["frame", "method"]).agg(agg_cols).reset_index()
    summary.to_csv(out / "tracking_summary.csv", index=False)

    # Trial-level paired diagnostics for the main comparison.
    trial_metrics = df.groupby(["mc", "method"]).agg({
        "mean_track_pos_rmse_m": "mean",
        "median_track_pos_error_m": "mean",
        "p90_track_pos_error_m": "mean",
        "track_loss_rate": "mean",
        "mean_pd_target": "mean",
        "p05_pd_target": "mean",
        "full_cross_norm_mean": "mean",
        "coverage_ratio": "mean",
        "switch_rate": "mean",
    }).reset_index()
    if cfg.save_diagnostics:
        trial_metrics.to_csv(out / "tracking_trial_level_metrics.csv", index=False)
    paired_df = pd.DataFrame()
    if {"collision_aware", "strongest"}.issubset(set(trial_metrics["method"])):
        ca = trial_metrics[trial_metrics.method == "collision_aware"].set_index("mc")
        st = trial_metrics[trial_metrics.method == "strongest"].set_index("mc")
        common = sorted(set(ca.index).intersection(st.index))
        paired_rows = []
        for metric in ["mean_pd_target", "p05_pd_target", "mean_track_pos_rmse_m", "median_track_pos_error_m", "p90_track_pos_error_m", "track_loss_rate"]:
            delta = ca.loc[common, metric].to_numpy() - st.loc[common, metric].to_numpy()
            lower_is_better = metric not in ("mean_pd_target", "p05_pd_target")
            if lower_is_better:
                win = np.mean(delta < 0)
            else:
                win = np.mean(delta > 0)
            paired_rows.append(dict(
                metric=metric,
                delta_mean=float(np.nanmean(delta)),
                delta_median=float(np.nanmedian(delta)),
                win_rate_collision_aware=float(win),
                n_pairs=len(common),
            ))
        paired_df = pd.DataFrame(paired_rows)
        if cfg.save_diagnostics:
            paired_df.to_csv(out / "tracking_paired_diagnostics.csv", index=False)

    # Finite/cross diagnostics: NaN cross means uncovered target, not numerical explosion.
    cross_diag = df.groupby("method").agg(
        finite_cross_rate=("full_cross_norm_mean", lambda x: float(np.isfinite(np.asarray(x, dtype=float)).mean())),
        mean_uncovered_targets=("uncovered_target_count", "mean"),
        mean_coverage_ratio=("coverage_ratio", "mean"),
        mean_one_receiver_coverage_ratio=("one_receiver_coverage_ratio", "mean"),
    ).reset_index()
    if cfg.save_diagnostics:
        cross_diag.to_csv(out / "tracking_cross_finite_diagnostics.csv", index=False)
        if not paired_df.empty:
            print("\n[Tracking] Paired diagnostics: collision_aware vs strongest")
            print(paired_df.to_string(index=False))

    for metric, ylabel, fname in [
        ("mean_track_pos_rmse_m", "Mean track position RMSE (m)", "tracking_pos_rmse_time.png"),
        ("p90_track_pos_error_m", "p90 track position error (m)", "tracking_p90_pos_error_time.png"),
        ("track_loss_rate", "Track loss rate", "tracking_loss_rate_time.png"),
        ("switch_rate", "Assignment switch rate", "tracking_switch_rate_time.png"),
        ("p05_pd_target", "p05 target $P_D$", "tracking_p05_pd_time.png"),
        ("uncovered_target_count", "Uncovered targets", "tracking_uncovered_targets_time.png"),
    ]:
        plt.figure(figsize=(7.5, 4.5))
        for method in methods:
            sub = summary[summary.method == method].sort_values("frame")
            plt.plot(sub["frame"], sub[metric], marker="o", markersize=3, label=method)
        plt.xlabel("Frame")
        plt.ylabel(ylabel)
        plt.title("Closed-loop tracking gate")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=cfg.dpi)
        plt.close()
    final = summary[summary.frame == summary.frame.max()].sort_values("method")
    print("\n[Tracking] Final-frame summary")
    print(final.to_string(index=False))
    print("\n[Tracking] Cross/coverage diagnostics")
    print(cross_diag.to_string(index=False))
    print(f"Saved: {out / 'tracking_summary.csv'}")


def run_difficulty_sweep(cfg: SimConfig, out: Path) -> None:
    """Supplemental experiment: noise/difficulty sweep.

    Purpose: avoid making the main Gate-1 conclusion only at a saturated-Pd point.
    The assignment and evaluation are re-run for each noise scale. Increasing the
    noise floor makes detection harder and also changes visibility consistently,
    reflecting a harder sensing frame rather than only post-hoc detector noise.
    """
    rng = make_rng(cfg.seed + 3000)
    scales = parse_float_list(cfg.difficulty_noise_scales)
    records: List[Dict[str, float]] = []
    methods = ["random", "nearest", "strongest", "collision_aware"]
    if cfg.include_oracle_gate1:
        methods.append("oracle")

    for scale in scales:
        for mc in range(cfg.difficulty_mc):
            local = SimConfig(**asdict(cfg))
            local.snr_floor = cfg.snr_floor * float(scale)
            # Keep practical default unless user explicitly changed it.
            sc = _make_gate1_jittered_scenario(local, rng)
            rows = _gate1_method_records_for_scenario(local, sc, rng, mc, methods=methods)
            for row in rows:
                row["noise_scale"] = float(scale)
                row["snr_floor_effective"] = local.snr_floor
                records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(out / "difficulty_trials.csv", index=False)
    summary = df.groupby(["noise_scale", "method"]).agg(
        pd_mean=("pd_mean", "mean"),
        pd_std=("pd_mean", "std"),
        pd_p05=("pd_p05", "mean"),
        pd_energy_mean=("pd_energy_mean", "mean"),
        pd_noncoh_matched_mean=("pd_noncoh_matched_mean", "mean"),
        pd_coherent_matched_mean=("pd_coherent_matched_mean", "mean"),
        rmse_tau_mean=("rmse_tau_mean", "mean"),
        rmse_nu_mean=("rmse_nu_mean", "mean"),
        full_cross_norm_mean=("full_cross_norm_mean", "mean"),
        proxy_cross_norm=("proxy_cross_norm", "mean"),
        collision_cost_norm=("collision_cost_norm", "mean"),
        coverage_ratio=("coverage_ratio", "mean"),
        num_task_edges=("num_task_edges", "mean"),
    ).reset_index()
    summary.to_csv(out / "difficulty_summary.csv", index=False)

    # Plots: Pd, lower-tail Pd, and RMSE vs noise scale.
    order = ["random", "nearest", "strongest", "collision_aware"]
    plt.figure(figsize=(7, 4.8))
    for method in order:
        sub = summary[summary.method == method].sort_values("noise_scale")
        if len(sub):
            plt.plot(sub["noise_scale"], sub["pd_mean"], marker="o", label=method)
    plt.xscale("log")
    plt.xlabel("Noise-floor scale")
    plt.ylabel(f"Mean $P_D$ @ $P_{{FA}}={cfg.pfa:g}$")
    plt.title("Supplement: detection difficulty sweep")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "difficulty_pd_vs_noise.png", dpi=cfg.dpi)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    for method in order:
        sub = summary[summary.method == method].sort_values("noise_scale")
        if len(sub):
            plt.plot(sub["noise_scale"], sub["pd_p05"], marker="o", label=method)
    plt.xscale("log")
    plt.xlabel("Noise-floor scale")
    plt.ylabel(f"5th-percentile $P_D$ @ $P_{{FA}}={cfg.pfa:g}$")
    plt.title("Supplement: lower-tail detection reliability")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "difficulty_p05_pd_vs_noise.png", dpi=cfg.dpi)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    for method in order:
        sub = summary[summary.method == method].sort_values("noise_scale")
        if len(sub):
            plt.plot(sub["noise_scale"], sub["rmse_tau_mean"], marker="o", label=method)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Noise-floor scale")
    plt.ylabel(r"Mean delay RMSE proxy")
    plt.title("Supplement: delay-estimation degradation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "difficulty_rmse_tau_vs_noise.png", dpi=cfg.dpi)
    plt.close()

    print("\n[Supplement] Difficulty sweep summary preview")
    print(summary.head(20).to_string(index=False))
    print(f"Saved: {out / 'difficulty_summary.csv'}")


def _detector_for_phase_mode(mode: str) -> str:
    mode = mode.lower()
    if mode == "coherent":
        return "coherent_matched"
    return "noncoherent_matched"


def run_phase_sweep(cfg: SimConfig, out: Path) -> None:
    """Supplemental experiment: coherent vs random-phase vs noncoherent-power.

    Purpose: show whether the collision-aware gain depends on an unrealistic
    distributed coherent-combining assumption. Main paper should emphasize the
    random-phase/noncoherent result, while coherent is only an upper bound.
    """
    rng = make_rng(cfg.seed + 4000)
    phase_modes = [x.strip() for x in cfg.phase_mode_list.split(',') if x.strip()]
    records: List[Dict[str, float]] = []
    methods = ["random", "nearest", "strongest", "collision_aware"]
    if cfg.include_oracle_gate1:
        methods.append("oracle")

    for phase_mode in phase_modes:
        for mc in range(cfg.phase_mc):
            local = SimConfig(**asdict(cfg))
            local.coop_phase_mode = phase_mode
            local.detector_mode = _detector_for_phase_mode(phase_mode)
            sc = _make_gate1_jittered_scenario(local, rng)
            rows = _gate1_method_records_for_scenario(local, sc, rng, mc, methods=methods)
            for row in rows:
                row["phase_mode"] = phase_mode
                row["detector_mode_used"] = local.detector_mode
                records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(out / "phase_trials.csv", index=False)
    summary = df.groupby(["phase_mode", "method"]).agg(
        pd_mean=("pd_mean", "mean"),
        pd_std=("pd_mean", "std"),
        pd_p05=("pd_p05", "mean"),
        pd_energy_mean=("pd_energy_mean", "mean"),
        pd_noncoh_matched_mean=("pd_noncoh_matched_mean", "mean"),
        pd_coherent_matched_mean=("pd_coherent_matched_mean", "mean"),
        rmse_tau_mean=("rmse_tau_mean", "mean"),
        rmse_nu_mean=("rmse_nu_mean", "mean"),
        full_cross_norm_mean=("full_cross_norm_mean", "mean"),
        proxy_cross_norm=("proxy_cross_norm", "mean"),
        collision_cost_norm=("collision_cost_norm", "mean"),
        mean_num_signal_components=("mean_num_signal_components", "mean"),
        mean_coop_signal_power_frac=("mean_coop_signal_power_frac", "mean"),
        num_task_edges=("num_task_edges", "mean"),
    ).reset_index()
    summary.to_csv(out / "phase_summary.csv", index=False)

    order = ["random", "nearest", "strongest", "collision_aware"]
    # Bar grouped by phase mode.
    for metric, fname, ylabel in [
        ("pd_mean", "phase_pd_bar.png", f"Mean $P_D$ @ $P_{{FA}}={cfg.pfa:g}$"),
        ("pd_p05", "phase_p05_pd_bar.png", f"5th-percentile $P_D$ @ $P_{{FA}}={cfg.pfa:g}$"),
        ("rmse_tau_mean", "phase_rmse_tau_bar.png", r"Mean delay RMSE proxy"),
    ]:
        phases = [p for p in phase_modes if p in summary.phase_mode.unique()]
        x = np.arange(len(phases))
        width = 0.18
        plt.figure(figsize=(8, 4.8))
        for idx, method in enumerate(order):
            vals = []
            for phase in phases:
                sub = summary[(summary.phase_mode == phase) & (summary.method == method)]
                vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
            plt.bar(x + (idx - 1.5) * width, vals, width=width, label=method)
        plt.xticks(x, phases, rotation=15)
        plt.ylabel(ylabel)
        plt.title("Supplement: cooperative phase model comparison")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=cfg.dpi)
        plt.close()

    print("\n[Supplement] Phase-mode sweep summary preview")
    print(summary.head(20).to_string(index=False))
    print(f"Saved: {out / 'phase_summary.csv'}")



def _proxy_mode_for_eval_kernel(mode: str) -> str:
    """Use the same simple kernel for assignment when feasible; keep actual_otfs for evaluation only."""
    m = str(mode).lower()
    if m in ("actual", "actual_otfs", "otfs", "full", "broadened", "broadened_dirichlet", "legacy"):
        return "dirichlet"
    if m in ("no_spread", "none", "point"):
        return "no_spread"
    if m in ("gaussian", "gauss"):
        return "gaussian"
    return "dirichlet"



def _collision_aware_assignment_with_angle_mode(
    cfg: SimConfig,
    sc: Scenario,
    U: np.ndarray,
    v: np.ndarray,
    angle_mode: str,
) -> np.ndarray:
    """Collision-aware assignment with a controlled angular term.

    angle_mode="dd_only" disables the angular overlap in the *assignment proxy*
    by setting sigma_theta=inf.  The returned assignment can still be evaluated
    under the physical cfg.sigma_theta_deg in the full-link evaluator.  This is
    the key test for whether the angular term changes decisions and improves
    full-link performance, rather than being a decorative factor.
    """
    local = SimConfig(**asdict(cfg))
    if str(angle_mode).lower() in ("dd_only", "ddonly", "dd"):
        local.sigma_theta_deg = float("inf")
    elif str(angle_mode).lower() in ("dd_angle", "dda", "angle"):
        local.sigma_theta_deg = cfg.sigma_theta_deg
    else:
        raise ValueError(f"Unknown angle_mode={angle_mode!r}")
    return assignment_collision_aware_greedy(local, sc, U, v)


def _dd_angle_diagnostics(cfg: SimConfig, sc: Scenario, z: np.ndarray) -> Dict[str, float]:
    """Compute DD-only and DD-angle collision diagnostics for a fixed assignment.

    N_DD uses the DD-overlap threshold only.  N_DDA additionally uses the angular
    threshold.  The ratio N_DDA/N_DD directly tells whether the angular dimension
    is removing pseudo-collisions for this assignment and scenario.
    """
    edges = build_collision_edges(cfg, sc, z, use_full_kernel=False)
    if len(edges) == 0:
        return dict(N_DD=0, N_DDA=0, dda_over_dd=0.0, mean_omega_ang=0.0,
                    p50_omega_ang=0.0, p90_omega_ang=0.0, angle_removed_frac=0.0,
                    rho_col=0.0, eta_top=0.0, collision_cost_norm=0.0)
    omega_dd = edges["omega_dd"].to_numpy(dtype=float)
    omega_ang = edges["omega_ang"].to_numpy(dtype=float)
    dd_mask = omega_dd > cfg.dd_min
    dda_mask = dd_mask & (omega_ang > cfg.ang_min)
    N_DD = int(np.sum(dd_mask))
    N_DDA = int(np.sum(dda_mask))
    ratio = float(N_DDA / (N_DD + EPS)) if N_DD > 0 else 0.0
    summ = collision_summary(cfg, edges, len(task_edge_list(z)))
    return dict(
        N_DD=N_DD,
        N_DDA=N_DDA,
        dda_over_dd=ratio,
        angle_removed_frac=float(1.0 - ratio) if N_DD > 0 else 0.0,
        mean_omega_ang=float(np.mean(omega_ang[dd_mask])) if N_DD > 0 else 0.0,
        p50_omega_ang=float(np.percentile(omega_ang[dd_mask], 50)) if N_DD > 0 else 0.0,
        p90_omega_ang=float(np.percentile(omega_ang[dd_mask], 90)) if N_DD > 0 else 0.0,
        rho_col=float(summ.get("rho_col", 0.0)),
        eta_top=float(summ.get("eta_top", 0.0)),
        collision_cost_norm=float(summ.get("collision_cost_norm", 0.0)),
    )


def run_angle_usefulness_sweep(cfg: SimConfig, out: Path) -> None:
    """Gate: does the angular term actually matter?

    For each sigma_theta, compare:
      - strongest baseline,
      - DD-only collision-aware assignment (assignment proxy ignores angle),
      - DD-angle collision-aware assignment (assignment proxy includes angle).

    All assignments are evaluated with the same physical full-link settings at
    that sigma_theta.  This prevents the common failure mode where N_DDA==N_DD
    and the advertised angle dimension is in fact dead.
    """
    rng = make_rng(cfg.seed + 7000)
    sigmas = parse_float_list(cfg.sigma_theta_list)
    records: List[Dict[str, float]] = []

    for sigma in sigmas:
        for mc in range(cfg.angle_mc):
            local = SimConfig(**asdict(cfg))
            local.sigma_theta_deg = float(sigma)
            local.eval_kernel_mode = cfg.eval_kernel_mode
            local.proxy_kernel_mode = cfg.proxy_kernel_mode
            sc = _make_gate1_jittered_scenario(local, rng)
            b, U, v = build_visibility(local, sc)

            assigners = {
                "strongest": lambda: assignment_strongest(local, U, v),
                "collision_aware_dd_only": lambda: _collision_aware_assignment_with_angle_mode(local, sc, U, v, "dd_only"),
                "collision_aware_dd_angle": lambda: _collision_aware_assignment_with_angle_mode(local, sc, U, v, "dd_angle"),
            }
            for method, fn in assigners.items():
                z = fn()
                ev = evaluate_assignment_full_link(local, sc, z, rng)
                diag = _dd_angle_diagnostics(local, sc, z)
                # Also compute the assignment-proxy cost actually used by DD-only if relevant.
                if method == "collision_aware_dd_only":
                    assign_cfg = SimConfig(**asdict(local)); assign_cfg.sigma_theta_deg = float("inf")
                else:
                    assign_cfg = local
                edges_assign = build_collision_edges(assign_cfg, sc, z, use_full_kernel=False)
                assign_summ = collision_summary(assign_cfg, edges_assign, len(task_edge_list(z)))
                row = dict(
                    mc=mc,
                    sigma_theta_deg=float(999.0 if math.isinf(sigma) else sigma),
                    sigma_theta_label=("inf" if math.isinf(sigma) else f"{sigma:g}"),
                    method=method,
                    feasible=is_feasible(local, z, v),
                    visibility_edges=int(v.sum()),
                    coverage_ratio=float(np.mean(z.sum(axis=0) >= local.k_tgt_min)),
                    avg_tasks_per_uav=float(z.sum(axis=1).mean()),
                    num_task_edges=len(task_edge_list(z)),
                    objective_eval_proxy=objective(local, sc, z, U, use_full_kernel=False),
                    assignment_collision_cost_norm=float(assign_summ.get("collision_cost_norm", 0.0)),
                    assignment_N_DD=float(assign_summ.get("N_DD", 0.0)),
                    assignment_N_DDA=float(assign_summ.get("N_DDA", 0.0)),
                )
                row.update({f"angle_{k}": val for k, val in diag.items()})
                row.update(ev)
                records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(out / "angle_usefulness_trials.csv", index=False)
    summary = df.groupby(["sigma_theta_label", "sigma_theta_deg", "method"]).agg(
        pd_mean=("pd_mean", "mean"),
        pd_std=("pd_mean", "std"),
        pd_p05=("pd_p05", "mean"),
        pd_energy_mean=("pd_energy_mean", "mean"),
        pd_noncoh_matched_mean=("pd_noncoh_matched_mean", "mean"),
        rmse_tau_mean=("rmse_tau_mean", "mean"),
        rmse_nu_mean=("rmse_nu_mean", "mean"),
        full_cross_norm_mean=("full_cross_norm_mean", "mean"),
        proxy_cross_norm=("proxy_cross_norm", "mean"),
        angle_N_DD=("angle_N_DD", "mean"),
        angle_N_DDA=("angle_N_DDA", "mean"),
        dda_over_dd=("angle_dda_over_dd", "mean"),
        angle_removed_frac=("angle_angle_removed_frac", "mean"),
        mean_omega_ang=("angle_mean_omega_ang", "mean"),
        p50_omega_ang=("angle_p50_omega_ang", "mean"),
        p90_omega_ang=("angle_p90_omega_ang", "mean"),
        collision_cost_norm=("angle_collision_cost_norm", "mean"),
        assignment_collision_cost_norm=("assignment_collision_cost_norm", "mean"),
        coverage_ratio=("coverage_ratio", "mean"),
        num_task_edges=("num_task_edges", "mean"),
    ).reset_index().sort_values(["sigma_theta_deg", "method"])
    summary.to_csv(out / "angle_usefulness_summary.csv", index=False)

    # Pairwise DD-angle advantage over DD-only and strongest.
    piv = summary.pivot_table(index=["sigma_theta_label", "sigma_theta_deg"], columns="method", values=["pd_mean", "pd_p05", "full_cross_norm_mean", "dda_over_dd", "angle_removed_frac"]).reset_index()
    # Flatten columns robustly.
    piv.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip("_") for col in piv.columns]
    def col(metric, method): return f"{metric}_{method}"
    for metric in ["pd_mean", "pd_p05", "full_cross_norm_mean"]:
        ca = col(metric, "collision_aware_dd_angle")
        dd = col(metric, "collision_aware_dd_only")
        st = col(metric, "strongest")
        if ca in piv.columns and dd in piv.columns:
            piv[f"delta_{metric}_DDA_minus_DDonly"] = piv[ca] - piv[dd]
        if ca in piv.columns and st in piv.columns:
            piv[f"delta_{metric}_DDA_minus_strongest"] = piv[ca] - piv[st]
    piv.to_csv(out / "angle_usefulness_pairwise.csv", index=False)

    # Plots.
    method_order = ["strongest", "collision_aware_dd_only", "collision_aware_dd_angle"]
    labels_order = sorted(summary[["sigma_theta_label", "sigma_theta_deg"]].drop_duplicates().values.tolist(), key=lambda x: x[1])
    labels = [x[0] for x in labels_order]
    xs = np.arange(len(labels))

    def _plot_metric(metric: str, fname: str, ylabel: str):
        plt.figure(figsize=(8.2, 4.8))
        for method in method_order:
            vals = []
            for lab, _sg in labels_order:
                sub = summary[(summary.sigma_theta_label == lab) & (summary.method == method)]
                vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
            plt.plot(xs, vals, marker="o", label=method)
        plt.xticks(xs, labels)
        plt.xlabel(r"Angle resolution $\sigma_\theta$ (deg)")
        plt.ylabel(ylabel)
        plt.title("Angle-usefulness gate: DD-only vs DD-angle assignment")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=cfg.dpi)
        plt.close()

    _plot_metric("pd_mean", "angle_usefulness_pd.png", f"Mean $P_D$ @ $P_{{FA}}={cfg.pfa:g}$")
    _plot_metric("pd_p05", "angle_usefulness_p05_pd.png", f"5th-percentile $P_D$ @ $P_{{FA}}={cfg.pfa:g}$")
    _plot_metric("full_cross_norm_mean", "angle_usefulness_cross.png", "Full-link normalized cross interference")

    plt.figure(figsize=(7.5, 4.8))
    sub = summary[summary.method == "collision_aware_dd_angle"].sort_values("sigma_theta_deg")
    plt.plot(np.arange(len(sub)), sub["dda_over_dd"], marker="o", label=r"$N_{DDA}/N_{DD}$")
    plt.plot(np.arange(len(sub)), sub["angle_removed_frac"], marker="s", label="Removed fraction")
    plt.xticks(np.arange(len(sub)), sub["sigma_theta_label"])
    plt.xlabel(r"Angle resolution $\sigma_\theta$ (deg)")
    plt.ylabel("Ratio")
    plt.ylim(-0.05, 1.05)
    plt.title("Angle-usefulness gate: does angle remove DD-only pseudo-collisions?")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "angle_usefulness_dda_ratio.png", dpi=cfg.dpi)
    plt.close()

    print("\n[Angle-usefulness gate] Summary preview")
    print(summary.head(30).to_string(index=False))
    print(f"Saved: {out / 'angle_usefulness_summary.csv'}")


def run_kernel_ablation(cfg: SimConfig, out: Path) -> None:
    """Supplemental experiment: does OTFS/DD-spreading physics matter?

    Compare no-spread, Gaussian, Dirichlet, and actual-OTFS-like evaluation kernels.
    For no-spread/Gaussian/Dirichlet, the assignment proxy uses the same kernel.
    For actual_otfs, assignment keeps the fast Dirichlet proxy, while evaluation uses
    the independent actual OTFS-like kernel.
    """
    rng = make_rng(cfg.seed + 5000)
    modes = [x.strip() for x in cfg.kernel_eval_modes.split(',') if x.strip()]
    records: List[Dict[str, float]] = []
    methods = ["random", "nearest", "strongest", "collision_aware"]
    if cfg.include_oracle_gate1:
        methods.append("oracle")

    for mode in modes:
        for mc in range(cfg.kernel_mc):
            local = SimConfig(**asdict(cfg))
            local.eval_kernel_mode = mode
            local.proxy_kernel_mode = _proxy_mode_for_eval_kernel(mode)
            sc = _make_gate1_jittered_scenario(local, rng)
            rows = _gate1_method_records_for_scenario(local, sc, rng, mc, methods=methods)
            for row in rows:
                row["eval_kernel_mode"] = mode
                row["proxy_kernel_mode_used"] = local.proxy_kernel_mode
                records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(out / "kernel_ablation_trials.csv", index=False)
    summary = df.groupby(["eval_kernel_mode", "method"]).agg(
        pd_mean=("pd_mean", "mean"),
        pd_p05=("pd_p05", "mean"),
        pd_energy_mean=("pd_energy_mean", "mean"),
        pd_noncoh_matched_mean=("pd_noncoh_matched_mean", "mean"),
        rmse_tau_mean=("rmse_tau_mean", "mean"),
        rmse_nu_mean=("rmse_nu_mean", "mean"),
        full_cross_norm_mean=("full_cross_norm_mean", "mean"),
        proxy_cross_norm=("proxy_cross_norm", "mean"),
        collision_cost_norm=("collision_cost_norm", "mean"),
        coverage_ratio=("coverage_ratio", "mean"),
        num_task_edges=("num_task_edges", "mean"),
    ).reset_index()
    summary.to_csv(out / "kernel_ablation_summary.csv", index=False)

    order = ["random", "nearest", "strongest", "collision_aware"]
    kernels = [m for m in modes if m in summary.eval_kernel_mode.unique()]
    for metric, fname, ylabel in [
        ("pd_mean", "kernel_ablation_pd.png", f"Mean $P_D$ @ $P_{{FA}}={cfg.pfa:g}$"),
        ("pd_p05", "kernel_ablation_p05_pd.png", f"5th-percentile $P_D$ @ $P_{{FA}}={cfg.pfa:g}$"),
        ("full_cross_norm_mean", "kernel_ablation_cross.png", "Full-link normalized cross interference"),
    ]:
        x = np.arange(len(kernels))
        width = 0.18
        plt.figure(figsize=(8.5, 4.8))
        for idx, method in enumerate(order):
            vals = []
            for mode in kernels:
                sub = summary[(summary.eval_kernel_mode == mode) & (summary.method == method)]
                vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
            plt.bar(x + (idx - 1.5) * width, vals, width=width, label=method)
        plt.xticks(x, kernels, rotation=15)
        plt.ylabel(ylabel)
        plt.title("Supplement: DD-kernel / OTFS-spreading ablation")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=cfg.dpi)
        plt.close()

    print("\n[Supplement] Kernel ablation summary preview")
    print(summary.head(24).to_string(index=False))
    print(f"Saved: {out / 'kernel_ablation_summary.csv'}")


def _set_target_speed(sc: Scenario, speed_mps: float, rng: np.random.Generator) -> Scenario:
    """Return a shallow-copied scenario with all targets rescaled to the requested speed."""
    out = Scenario(sc.uav_pos.copy(), sc.uav_vel.copy(), sc.uav_bore.copy(), sc.tgt_pos.copy(), sc.tgt_vel.copy(), sc.tgt_rcs.copy())
    speed = float(speed_mps)
    if speed <= 1e-12:
        out.tgt_vel[:] = 0.0
        return out
    for q in range(out.Q):
        v = out.tgt_vel[q, :2]
        n = np.linalg.norm(v)
        if n <= EPS:
            ang = rng.uniform(0, 2*np.pi)
            direction = np.array([np.cos(ang), np.sin(ang)])
        else:
            direction = v / n
        out.tgt_vel[q, :2] = speed * direction
        out.tgt_vel[q, 2] = 0.0
    return out


def run_velocity_sweep(cfg: SimConfig, out: Path) -> None:
    """Supplemental experiment: target-velocity / Doppler-spreading sweep.

    This tests whether the OTFS DD-collision mechanism becomes more important in
    high-mobility conditions.  The geometry is regenerated per MC trial, then the
    target velocities are rescaled to each requested speed to isolate Doppler effects.
    """
    rng = make_rng(cfg.seed + 6000)
    speeds = parse_float_list(cfg.velocity_list_mps)
    records: List[Dict[str, float]] = []
    methods = ["random", "nearest", "strongest", "collision_aware"]
    if cfg.include_oracle_gate1:
        methods.append("oracle")

    for speed in speeds:
        for mc in range(cfg.velocity_mc):
            local = SimConfig(**asdict(cfg))
            local.target_speed_mps = max(float(speed), 0.0)
            # Keep actual OTFS-like evaluation to expose fractional-Doppler effects.
            local.eval_kernel_mode = cfg.eval_kernel_mode
            base = _make_gate1_jittered_scenario(local, rng)
            sc = _set_target_speed(base, speed, rng)
            rows = _gate1_method_records_for_scenario(local, sc, rng, mc, methods=methods)
            # Doppler diagnostics for the generated geometry.
            vals = []
            aliases = []
            for i in range(sc.M):
                for j in range(sc.M):
                    for q in range(sc.Q):
                        _, nu = bistatic_delay_doppler(local, sc, i, j, q)
                        vals.append(abs(nu))
                        aliases.append(float(abs(nu) > local.doppler_max_hz))
            for row in rows:
                row["target_speed_mps"] = float(speed)
                row["mean_abs_bistatic_doppler_hz"] = float(np.mean(vals)) if vals else 0.0
                row["max_abs_bistatic_doppler_hz"] = float(np.max(vals)) if vals else 0.0
                row["doppler_alias_ratio_geom"] = float(np.mean(aliases)) if aliases else 0.0
                records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(out / "velocity_trials.csv", index=False)
    summary = df.groupby(["target_speed_mps", "method"]).agg(
        pd_mean=("pd_mean", "mean"),
        pd_p05=("pd_p05", "mean"),
        pd_energy_mean=("pd_energy_mean", "mean"),
        pd_noncoh_matched_mean=("pd_noncoh_matched_mean", "mean"),
        rmse_tau_mean=("rmse_tau_mean", "mean"),
        rmse_nu_mean=("rmse_nu_mean", "mean"),
        full_cross_norm_mean=("full_cross_norm_mean", "mean"),
        proxy_cross_norm=("proxy_cross_norm", "mean"),
        collision_cost_norm=("collision_cost_norm", "mean"),
        mean_abs_bistatic_doppler_hz=("mean_abs_bistatic_doppler_hz", "mean"),
        max_abs_bistatic_doppler_hz=("max_abs_bistatic_doppler_hz", "mean"),
        doppler_alias_ratio=("doppler_alias_ratio_geom", "mean"),
        coverage_ratio=("coverage_ratio", "mean"),
        num_task_edges=("num_task_edges", "mean"),
    ).reset_index()
    summary.to_csv(out / "velocity_summary.csv", index=False)

    order = ["random", "nearest", "strongest", "collision_aware"]
    for metric, fname, ylabel in [
        ("pd_mean", "velocity_pd.png", f"Mean $P_D$ @ $P_{{FA}}={cfg.pfa:g}$"),
        ("pd_p05", "velocity_p05_pd.png", f"5th-percentile $P_D$ @ $P_{{FA}}={cfg.pfa:g}$"),
        ("full_cross_norm_mean", "velocity_cross.png", "Full-link normalized cross interference"),
        ("rmse_nu_mean", "velocity_rmse_nu.png", "Mean Doppler RMSE proxy"),
    ]:
        plt.figure(figsize=(7.5, 4.8))
        for method in order:
            sub = summary[summary.method == method].sort_values("target_speed_mps")
            if len(sub):
                plt.plot(sub["target_speed_mps"], sub[metric], marker="o", label=method)
        plt.xlabel("Target speed (m/s)")
        plt.ylabel(ylabel)
        plt.title("Supplement: target-velocity / Doppler sweep")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=cfg.dpi)
        plt.close()

    print("\n[Supplement] Velocity sweep summary preview")
    print(summary.head(24).to_string(index=False))
    print(f"Saved: {out / 'velocity_summary.csv'}")


def run_supplement(cfg: SimConfig, out: Path) -> None:
    run_difficulty_sweep(cfg, out)
    run_phase_sweep(cfg, out)
    run_kernel_ablation(cfg, out)
    run_velocity_sweep(cfg, out)


def classify_regime(cfg: SimConfig, rho: float) -> str:
    if rho < cfg.regime_i_max:
        return "I_low_collision"
    if rho <= cfg.regime_ii_max:
        return "II_sparse_heavy_tail"
    if rho > cfg.regime_iii_min:
        return "III_dense_collision"
    return "transition"


def run_gate2(cfg: SimConfig, out: Path) -> None:
    base_rng = make_rng(cfg.seed + 1000)
    records = []
    Ms = parse_int_list(cfg.m_list)
    Qs = parse_int_list(cfg.q_list)
    ktmins = parse_int_list(cfg.ktmin_list)
    kumaxs = parse_int_list(cfg.kumax_list)
    sigmas = parse_float_list(cfg.sigma_theta_list)

    for M in Ms:
        for Q in Qs:
            for kt in ktmins:
                for ku in kumaxs:
                    for sig in sigmas:
                        for mc in range(cfg.gate2_mc):
                            seed = int(base_rng.integers(0, 2**31 - 1))
                            rng = make_rng(seed)
                            local = SimConfig(**asdict(cfg))
                            local.k_tgt_min = kt
                            local.k_uav_max = ku
                            local.sigma_theta_deg = sig
                            sc = generate_clustered_scenario(local, M, Q, rng)
                            b, U, v = build_visibility(local, sc)
                            z = assignment_strongest(local, U, v)
                            edges = build_collision_edges(local, sc, z, use_full_kernel=False)
                            summ = collision_summary(local, edges, len(task_edge_list(z)))
                            coverage = float(np.mean(z.sum(axis=0) >= local.k_tgt_min))
                            visibility_deg_uav = float(v.sum(axis=1).mean())
                            visibility_deg_tgt = float(v.sum(axis=0).mean())
                            regime = classify_regime(local, summ["rho_col"])
                            row = dict(M=M, Q=Q, K_tgt_min=kt, K_uav_max=ku,
                                       sigma_theta_deg=(999.0 if math.isinf(sig) else sig), mc=mc, seed=seed,
                                       coverage_ratio=coverage, visibility_deg_uav=visibility_deg_uav,
                                       visibility_deg_tgt=visibility_deg_tgt, regime=regime)
                            row.update(summ)
                            records.append(row)
    df = pd.DataFrame(records)
    df.to_csv(out / "gate2_trials.csv", index=False)

    group_cols = ["M", "Q", "K_tgt_min", "K_uav_max", "sigma_theta_deg"]
    summary = df.groupby(group_cols).agg(
        rho_col_mean=("rho_col", "mean"),
        rho_col_std=("rho_col", "std"),
        eta_top_mean=("eta_top", "mean"),
        eta_top_raw_mean=("eta_top_raw", "mean"),
        eta_top_norm_mean=("eta_top_norm", "mean"),
        coverage_mean=("coverage_ratio", "mean"),
        mean_C_norm=("mean_C_norm", "mean"),
        p95_C_norm=("p95_C_norm", "mean"),
        N_DD_mean=("N_DD", "mean"),
        N_DDA_mean=("N_DDA", "mean"),
        alias_ratio_mean=("doppler_alias_ratio", "mean"),
        regimeII_frac=("regime", lambda s: float(np.mean(s == "II_sparse_heavy_tail"))),
        regimeI_frac=("regime", lambda s: float(np.mean(s == "I_low_collision"))),
        regimeIII_frac=("regime", lambda s: float(np.mean(s == "III_dense_collision"))),
    ).reset_index()
    summary.to_csv(out / "gate2_regime_summary.csv", index=False)

    # Heatmaps for a representative sigma and K settings.
    # Use sigma closest to 20 deg if present.
    sig_vals = sorted(summary.sigma_theta_deg.unique())
    sig_sel = min(sig_vals, key=lambda x: abs(x - 20.0))
    sub = summary[(summary.K_tgt_min == min(ktmins)) & (summary.K_uav_max == max(kumaxs)) & (summary.sigma_theta_deg == sig_sel)]
    if len(sub):
        pivot = sub.pivot(index="M", columns="Q", values="rho_col_mean")
        plt.figure(figsize=(6, 4.8))
        im = plt.imshow(pivot.values, origin="lower", aspect="auto")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.xlabel("Number of targets Q")
        plt.ylabel("Number of UAVs M")
        plt.title(f"Gate 2: Mean collision density, sigma={sig_sel:g} deg")
        plt.colorbar(im, label=r"$\rho_{col}$")
        plt.tight_layout()
        plt.savefig(out / "gate2_rho_heatmap.png", dpi=cfg.dpi)
        plt.close()

    # Regime fractions vs sigma for largest M,Q.
    sub2 = summary[(summary.M == max(Ms)) & (summary.Q == max(Qs)) & (summary.K_tgt_min == max(ktmins)) & (summary.K_uav_max == max(kumaxs))]
    if len(sub2):
        sub2 = sub2.sort_values("sigma_theta_deg")
        labels = ["inf" if x >= 900 else f"{x:g}" for x in sub2.sigma_theta_deg]
        x = np.arange(len(sub2))
        plt.figure(figsize=(7, 4.5))
        b1 = sub2["regimeI_frac"].values
        b2 = sub2["regimeII_frac"].values
        b3 = sub2["regimeIII_frac"].values
        plt.bar(x, b1, label="Regime I")
        plt.bar(x, b2, bottom=b1, label="Regime II")
        plt.bar(x, b3, bottom=b1+b2, label="Regime III")
        plt.xticks(x, labels)
        plt.xlabel(r"Angle resolution $\sigma_\theta$ (deg)")
        plt.ylabel("Trial fraction")
        plt.title(f"Gate 2: Regime fractions, M={max(Ms)}, Q={max(Qs)}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "gate2_regime_fraction_vs_angle.png", dpi=cfg.dpi)
        plt.close()

    # N_DD vs N_DDA scatter.
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(df["N_DD"], df["N_DDA"], s=10, alpha=0.35)
    lim = max(float(df["N_DD"].max()), float(df["N_DDA"].max()), 1.0)
    plt.plot([0, lim], [0, lim], "--", linewidth=1)
    plt.xlabel("DD-only collision count")
    plt.ylabel("DD-angle collision count")
    plt.title("Gate 2: Angle dimension removes pseudo-collisions")
    plt.tight_layout()
    plt.savefig(out / "gate2_dd_vs_dda.png", dpi=cfg.dpi)
    plt.close()

    print("\n[Gate 2] Summary preview")
    print(summary.head(20).to_string(index=False))
    print(f"Saved: {out / 'gate2_regime_summary.csv'}")


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Gate experiments for OTFS DD task-edge collision modeling (V12 filter/pollution ablations)")
    parser.add_argument("--gate", choices=["gate1", "gate2", "both", "difficulty", "phase", "kernel", "velocity", "angle", "pred", "tracking", "otfs", "supplement", "all"], default="both")
    parser.add_argument("--out-dir", default="./gate_otfs_collision_outputs")
    parser.add_argument("--fresh-out-dir", action="store_true", help="Delete out-dir before running, preventing mixed old/new CSV and figure outputs")
    parser.add_argument("--save-diagnostics", action="store_true", help="Save verbose diagnostic CSV files in addition to trials/summary outputs")
    parser.add_argument("--save-all-detectors", action="store_true", help="Compute and save all detector Pd variants; otherwise only the selected primary detector is evaluated")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gate1-mc", type=int, default=10)
    parser.add_argument("--gate2-mc", type=int, default=5)
    parser.add_argument("--difficulty-mc", type=int, default=10, help="MC trials per noise-scale point in the difficulty sweep")
    parser.add_argument("--phase-mc", type=int, default=10, help="MC trials per phase mode in the phase-model sweep")
    parser.add_argument("--kernel-mc", type=int, default=10, help="MC trials per kernel mode in the OTFS-kernel ablation")
    parser.add_argument("--velocity-mc", type=int, default=10, help="MC trials per speed point in the velocity/Doppler sweep")
    parser.add_argument("--angle-mc", type=int, default=10, help="MC trials per angle-resolution point in the angle-usefulness gate")
    parser.add_argument("--pred-mc", type=int, default=10, help="MC trials per prediction-error point in the prediction robustness gate")
    parser.add_argument("--tracking-mc", type=int, default=5, help="MC trials in the multi-frame tracking gate")
    parser.add_argument("--inner-mc", type=int, default=200)
    parser.add_argument("--n-tau", type=int, default=64)
    parser.add_argument("--n-nu", type=int, default=32)
    parser.add_argument("--sigma-theta-deg", type=float, default=20.0)
    parser.add_argument("--sigma-theta-list", default="10,20,40,inf")
    parser.add_argument("--difficulty-noise-scales", default="0.3,1,3,10,30", help="Comma-separated multiplicative scales applied to snr_floor for difficulty sweep")
    parser.add_argument("--phase-mode-list", default="coherent,random_phase,noncoherent_power", help="Comma-separated cooperative phase modes for phase sweep")
    parser.add_argument("--kernel-eval-modes", default="no_spread,gaussian,dirichlet,actual_otfs", help="Comma-separated DD-kernel modes for OTFS role ablation")
    parser.add_argument("--velocity-list-mps", default="0,20,50,100,150", help="Comma-separated target speeds for velocity/Doppler sweep")
    parser.add_argument("--pred-pos-sigma-list", default="0,5,10,20,50", help="Comma-separated target position prediction-error std values in meters")
    parser.add_argument("--pred-vel-sigma-list", default="0,1,3,5,10", help="Comma-separated target velocity prediction-error std values in m/s")
    parser.add_argument("--tracking-M", type=int, default=15, help="Number of UAVs for the closed-loop tracking gate")
    parser.add_argument("--tracking-Q", type=int, default=10, help="Number of targets for the closed-loop tracking gate")
    parser.add_argument("--num-frames", type=int, default=30, help="Number of frames in the closed-loop tracking gate")
    parser.add_argument("--dt-frame", type=float, default=0.2, help="Frame interval in seconds for tracking gate")
    parser.add_argument("--lambda-switch", type=float, default=0.0, help="Assignment switching penalty used by collision_aware_switch in tracking gate")
    parser.add_argument("--no-tracking-crn", action="store_true", help="Disable common-random-number tracking primitives; mainly for ablation/debugging")
    parser.add_argument("--target-pd-fusion", choices=["mean", "or"], default="mean", help="Target-level Pd fusion in tracking gate: conservative mean or noncoherent OR fusion")
    parser.add_argument("--tracking-reinit-lost", action="store_true", help="Reinitialize a lost track when it is detected again")
    parser.add_argument("--tracking-alpha-pos", type=float, default=0.65, help="Alpha-filter position update gain in tracking gate")
    parser.add_argument("--tracking-alpha-vel", type=float, default=0.55, help="Alpha-filter velocity update gain in tracking gate")
    parser.add_argument("--tracker-mode", choices=["alpha", "kf", "dd_rwif"], default="alpha", help="Tracking update model: alpha filter, standard CV-KF, or DD reliability-weighted information filter")
    parser.add_argument("--meas-pollution", choices=["none", "biased_peak", "clutter_peak"], default="none", help="Structured DD-collision measurement pollution model for V12 falsification tests")
    parser.add_argument("--pollution-pmax", type=float, default=0.35, help="Max probability of structured DD measurement pollution")
    parser.add_argument("--pollution-c0", type=float, default=3.0, help="Ccross scale in p_pollute=pmax*C/(C+C0)")
    parser.add_argument("--pollution-pos-bias-m", type=float, default=80.0, help="Position bias magnitude for biased_peak pollution")
    parser.add_argument("--pollution-vel-bias-mps", type=float, default=8.0, help="Velocity bias magnitude for biased_peak pollution")
    parser.add_argument("--dd-weight-alpha", type=float, default=1.0, help="Pd exponent used by DD-RWIF reliability weight")
    parser.add_argument("--dd-weight-beta", type=float, default=0.15, help="Ccross exponent factor exp(-beta*C) used by DD-RWIF")
    parser.add_argument("--dd-weight-gamma-min", type=float, default=0.05, help="Minimum information weight for DD-RWIF detected measurements")
    parser.add_argument("--kf-process-pos-std-m", type=float, default=1.0, help="CV-KF process position std per frame")
    parser.add_argument("--kf-process-vel-std-mps", type=float, default=0.4, help="CV-KF process velocity std per frame")
    parser.add_argument("--proxy-kernel-mode", choices=["no_spread", "gaussian", "dirichlet", "actual_otfs"], default="dirichlet", help="Optimization-layer DD kernel used outside kernel ablations")
    parser.add_argument("--gaussian-sigma-k", type=float, default=0.65, help="Gaussian kernel Doppler-bin width for kernel ablation")
    parser.add_argument("--gaussian-sigma-l", type=float, default=0.65, help="Gaussian kernel delay-bin width for kernel ablation")
    parser.add_argument("--m-list", default="9,12,15,18")
    parser.add_argument("--q-list", default="6,8,10,12")
    parser.add_argument("--ktmin-list", default="2,3")
    parser.add_argument("--kumax-list", default="2,3")
    parser.add_argument("--lambda-col", type=float, default=1.0)
    parser.add_argument("--c-norm-min", type=float, default=0.1, help="Effective collision threshold C/protected_self; 0.1 = -10 dB")
    parser.add_argument("--mu-div", type=float, default=0.15)
    parser.add_argument("--include-oracle-gate1", action="store_true", help="Enable exact small-scale oracle in Gate 1; can be slow")
    parser.add_argument("--raw-collision-cost", action="store_true", help="Use raw collision energy instead of normalized collision in the assignment objective")
    parser.add_argument("--eval-kernel-mode", choices=["no_spread", "gaussian", "dirichlet", "actual_otfs", "broadened_dirichlet"], default="actual_otfs", help="Gate-1 full-link evaluation kernel")
    parser.add_argument("--detector-mode", choices=["noncoherent_matched", "matched", "coherent_matched", "energy", "both"], default="noncoherent_matched", help="Primary Pd: practical noncoherent matched-filter bank by default; matched/coherent_matched is coherent upper bound")
    parser.add_argument("--coop-phase-mode", choices=["random_phase", "coherent", "noncoherent_power"], default="random_phase", help="Phase model for same-target cooperative bistatic returns; coherent is an ideal upper bound")
    parser.add_argument("--overlap-round-bins", type=float, default=0.01, help="DD-bin quantization for actual-OTFS kernel cache")
    parser.add_argument("--candidate-pool-size", type=int, default=10, help="Collision-aware candidate pool used when coalition enumeration is truncated")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of multiprocessing workers for independent MC jobs; use 1 for serial execution")
    parser.add_argument("--no-clear-worker-caches", action="store_true", help="Do not clear actual-OTFS caches after each worker job (not recommended for long tracking runs)")
    parser.add_argument("--k-cand", type=int, default=5)
    parser.add_argument("--k-uav-max", type=int, default=2)
    parser.add_argument("--k-tgt-min", type=int, default=2)
    args = parser.parse_args()

    cfg = SimConfig(
        seed=args.seed,
        out_dir=args.out_dir,
        fresh_out_dir=args.fresh_out_dir,
        save_diagnostics=args.save_diagnostics,
        save_all_detectors=args.save_all_detectors,
        gate1_mc=args.gate1_mc,
        gate2_mc=args.gate2_mc,
        difficulty_mc=args.difficulty_mc,
        phase_mc=args.phase_mc,
        kernel_mc=args.kernel_mc,
        velocity_mc=args.velocity_mc,
        angle_mc=args.angle_mc,
        pred_mc=args.pred_mc,
        tracking_mc=args.tracking_mc,
        difficulty_noise_scales=args.difficulty_noise_scales,
        phase_mode_list=args.phase_mode_list,
        kernel_eval_modes=args.kernel_eval_modes,
        velocity_list_mps=args.velocity_list_mps,
        pred_pos_sigma_list_m=args.pred_pos_sigma_list,
        pred_vel_sigma_list_mps=args.pred_vel_sigma_list,
        tracking_M=args.tracking_M,
        tracking_Q=args.tracking_Q,
        num_frames=args.num_frames,
        dt_frame=args.dt_frame,
        lambda_switch=args.lambda_switch,
        tracking_crn=not args.no_tracking_crn,
        target_pd_fusion=args.target_pd_fusion,
        tracking_reinit_lost=args.tracking_reinit_lost,
        tracking_alpha_pos=args.tracking_alpha_pos,
        tracking_alpha_vel=args.tracking_alpha_vel,
        tracker_mode=args.tracker_mode,
        meas_pollution=args.meas_pollution,
        pollution_pmax=args.pollution_pmax,
        pollution_c0=args.pollution_c0,
        pollution_pos_bias_m=args.pollution_pos_bias_m,
        pollution_vel_bias_mps=args.pollution_vel_bias_mps,
        dd_weight_alpha=args.dd_weight_alpha,
        dd_weight_beta=args.dd_weight_beta,
        dd_weight_gamma_min=args.dd_weight_gamma_min,
        kf_process_pos_std_m=args.kf_process_pos_std_m,
        kf_process_vel_std_mps=args.kf_process_vel_std_mps,
        inner_mc=args.inner_mc,
        n_tau=args.n_tau,
        n_nu=args.n_nu,
        sigma_theta_deg=args.sigma_theta_deg,
        sigma_theta_list=args.sigma_theta_list,
        m_list=args.m_list,
        q_list=args.q_list,
        ktmin_list=args.ktmin_list,
        kumax_list=args.kumax_list,
        lambda_col=args.lambda_col,
        c_norm_min=args.c_norm_min,
        mu_div=args.mu_div,
        include_oracle_gate1=args.include_oracle_gate1,
        use_normalized_collision_cost=(not args.raw_collision_cost),
        eval_kernel_mode=args.eval_kernel_mode,
        proxy_kernel_mode=args.proxy_kernel_mode,
        gaussian_sigma_k=args.gaussian_sigma_k,
        gaussian_sigma_l=args.gaussian_sigma_l,
        detector_mode=args.detector_mode,
        coop_phase_mode=args.coop_phase_mode,
        overlap_round_bins=args.overlap_round_bins,
        candidate_pool_size=args.candidate_pool_size,
        num_workers=args.num_workers,
        clear_worker_caches=(not args.no_clear_worker_caches),
        k_cand=args.k_cand,
        k_uav_max=args.k_uav_max,
        k_tgt_min=args.k_tgt_min,
    )
    out = Path(cfg.out_dir)
    if cfg.fresh_out_dir and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.run_id:
        cfg.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_dict = asdict(cfg)
    pd.DataFrame([cfg_dict]).to_csv(out / "config.csv", index=False)
    manifest = dict(
        script_version="v12",
        run_id=cfg.run_id,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        argv=sys.argv,
        config=cfg_dict,
    )
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if args.gate in ("gate1", "both", "all"):
        run_gate1(cfg, out)
    if args.gate in ("gate2", "both", "all"):
        run_gate2(cfg, out)
    if args.gate in ("difficulty", "supplement", "all"):
        run_difficulty_sweep(cfg, out)
    if args.gate in ("pred", "supplement", "all"):
        run_prediction_robustness(cfg, out)
    if args.gate in ("tracking", "supplement", "all"):
        run_tracking_experiment(cfg, out)
    if args.gate in ("phase", "supplement", "all"):
        run_phase_sweep(cfg, out)
    if args.gate in ("angle", "otfs", "supplement", "all"):
        run_angle_usefulness_sweep(cfg, out)
    if args.gate in ("kernel", "otfs", "supplement", "all"):
        run_kernel_ablation(cfg, out)
    if args.gate in ("velocity", "otfs", "supplement", "all"):
        run_velocity_sweep(cfg, out)

    print(f"\nAll outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
