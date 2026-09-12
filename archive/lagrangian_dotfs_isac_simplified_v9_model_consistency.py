#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lagrangian-guided simplified DOTFS-ISAC cooperative sensing simulation (v9 model-consistency update).

Purpose
-------
This is a simplified, standalone simulation for comparing:

    1) proposed_lagrangian: Lagrangian multiplier based marginal link selection
    2) baseline methods:
        - all_neighbor
        - random
        - nearest
        - shortest_bistatic
        - raw_sense_sinr
        - sense_sinr
        - single_best

The code intentionally removes the previous heavy framework:
    - no power-grid search,
    - no soft-min outer power optimization,
    - no consensus module,
    - no exhaustive search.

The focus is:
    Lagrangian marginal decision:
        score_ijq = alpha_q * DeltaD_ijq - lambda_c * cost_ijq

where alpha_q is the current marginal value of improving target q, and lambda_c
is the communication-resource price.

This is still a soft-information-level abstraction, not a waveform-level OTFS
transceiver.

Additional experiment modes:
    --lambda-sweep: sweep the Lagrangian communication price.
    --ablation: compare full Lagrangian against ablated variants.
    --robustness: evaluate robustness under communication-error/residual-interference changes.
    --comm-sweep: sweep communication-rate constraints to expose the communication side of ISAC.

V9 updates:
    - explicit ISAC sensing-power model: sensing_only / joint_waveform / reliable_comm_assisted,
    - optional communication direct-leakage factor in communication SINR,
    - H0-side biased-error handling for robustness tests,
    - clearer system-level P_FA semantics,
    - safer CSV writing and float matching in communication-sweep plots.
    --fair-ablation: ablation under a fixed total selected-link budget.
    --dd-ablation: evaluate the contribution of OTFS delay-Doppler validity/loss/collision modeling.

The reported metrics include both sensing-side performance (P_D/P_FA) and
communication-side quantities (selected-link rate/reliability, exchange delay,
and rate-satisfaction ratios), so the script can support a sensing-centric ISAC
interpretation rather than a pure sensing-only evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

EPS = 1e-12
Link = Tuple[int, int]

MethodName = Literal[
    "proposed_lagrangian",
    "all_neighbor",
    "random",
    "nearest",
    "shortest_bistatic",
    "raw_sense_sinr",
    "sense_sinr",
    "single_best",
]

METHODS: List[MethodName] = [
    "proposed_lagrangian",
    "all_neighbor",
    "random",
    "nearest",
    "shortest_bistatic",
    "raw_sense_sinr",
    "sense_sinr",
    "single_best",
]

METHOD_RNG_OFFSETS: Dict[str, int] = {
    "proposed_lagrangian": 101,
    "all_neighbor": 211,
    "random": 307,
    "nearest": 401,
    "shortest_bistatic": 457,
    "raw_sense_sinr": 461,
    "sense_sinr": 503,
    "single_best": 601,
}


@dataclass
class SimConfig:
    # Scale
    M: int = 15
    Q: int = 10
    num_mc: int = 200

    # Geometry
    area_xy: float = 4000.0
    h_uav_min: float = 800.0
    h_uav_max: float = 1200.0
    h_target_min: float = 700.0
    h_target_max: float = 1500.0
    comm_range: float = 2500.0

    # Velocity
    uav_speed_min: float = 20.0
    uav_speed_max: float = 60.0
    target_speed_min: float = 30.0
    target_speed_max: float = 150.0

    # OTFS-style DD parameters
    N: int = 64
    L: int = 64
    delta_f: float = 30e3
    T: float = 1.0 / 30e3
    fc: float = 5.9e9
    c: float = 3e8

    # Power/noise
    P_default: float = 1.0
    rho: float = 0.80  # fixed sensing power ratio
    noise_psd_dbm_hz: float = -174.0
    noise_figure_db: float = 7.0
    # Sensing-power interpretation used by the simplified ISAC abstraction.
    #   sensing_only:            only rho*P contributes to sensing echoes (default, most conservative).
    #   joint_waveform:           the whole P contributes to sensing echoes.
    #   reliable_comm_assisted:   rho*P plus a reliability-weighted communication component contributes.
    isac_power_model: Literal["sensing_only", "joint_waveform", "reliable_comm_assisted"] = "sensing_only"

    # Communication constraints
    R_min: float = 2.0e5
    chi_min: float = 0.30
    enforce_chi_min: bool = False
    comm_leakage_from_sensing: float = 0.05
    # Optional leakage from the sensing component of transmitter i into the i->j communication receiver.
    # Default 0 keeps the previous communication SINR model; set >0 for robustness studies.
    comm_direct_leakage_factor: float = 0.0

    # Sensing/detection
    Pfa_target: float = 0.05
    D_min: float = 3.0
    soft_mu_scale: float = 8.0
    soft_sigma0: float = 1.0
    soft_sigma_floor: float = 0.25
    soft_error_sigma_scale: float = 3.0
    # Environment-level pollution used by the Monte Carlo detector.
    enable_comm_error_pollution: bool = True
    # Algorithm-side calibration used by the Lagrangian deflection model.
    # Set this False for ablation; the detection environment can still remain polluted.
    use_comm_error_calibration: bool = True
    comm_error_model: Literal["erasure", "flip", "biased"] = "erasure"
    soft_error_flip_scale: float = 1.0
    soft_error_bias_scale: float = 0.5
    # Optional H0-side bias when a failed communication packet produces biased soft information.
    # Default 0 keeps the nominal false-alarm calibration unchanged.
    h0_error_bias_scale: float = 0.0
    num_false_per_target: int = 30

    # Channel/radar abstraction
    path_loss_exp: float = 2.0
    shadow_std_db: float = 1.0
    rician_K_db: float = 15.0
    target_rcs: float = 50.0
    sensing_processing_gain: Optional[float] = None

    # Residual interference
    residual_self_factor: float = 1e-14
    residual_direct_factor: float = 1e-4
    residual_multi_uav_factor: float = 1e-10
    rinr_sigma_factor: float = 0.15

    # DD penalties
    use_otfs_bin_validity: bool = True
    enable_dd_fractional_penalty: bool = True
    enable_dd_collision_penalty: bool = True
    dd_collision_alpha: float = 1.0

    # Packet overhead
    K_candidates: int = 4
    b_d: float = 160.0

    # Link selection limits
    max_links_per_target: int = 6
    max_total_links: int = 60
    candidate_topk_per_target: int = 40
    min_marginal_D: float = 0.0

    # Lagrangian selection parameters
    lagrangian_lambda: float = 0.005  # price for 1 ms delay
    lagrangian_mu: float = 1.0  # target-deficit multiplier
    use_softmin_alpha: bool = True
    softmin_tau: float = 0.10
    alpha_floor: float = 0.0
    alpha_cap: float = 10.0
    # Ablation toggles. If use_target_priority=False, alpha_q=1 for all targets.
    # If use_delay_price=False, the -lambda_c*c_ijq term is removed.
    use_target_priority: bool = True
    use_delay_price: bool = True

    # Utility parameters
    beta_scale: float = 5.0
    beta_chi_power: float = 2.0
    beta_delay_exponent: float = 0.5

    # Randomness/printing
    seed: int = 2026
    verbose: bool = True


@dataclass
class Geometry:
    p_uav: np.ndarray
    v_uav: np.ndarray
    p_tgt: np.ndarray
    v_tgt: np.ndarray


@dataclass
class BaseGains:
    edge_mask: np.ndarray
    direct_gain: np.ndarray
    d_uu: np.ndarray
    d_uav_tgt: np.ndarray
    tau: np.ndarray
    doppler: np.ndarray
    delay_bin: np.ndarray
    doppler_bin: np.ndarray
    valid_dd: np.ndarray
    dd_frac_loss: np.ndarray
    dd_collision_count: np.ndarray
    target_gain: np.ndarray
    geom_factor: np.ndarray


@dataclass
class LinkTables:
    gamma_comm: np.ndarray
    rate: np.ndarray
    chi_comm: np.ndarray
    feasible_comm: np.ndarray
    raw_gamma_sense: np.ndarray
    gamma_sense: np.ndarray
    rinr: np.ndarray
    beta: np.ndarray
    mu_soft: np.ndarray
    sigma0: np.ndarray


@dataclass
class MethodResult:
    name: str
    detected: int
    total_targets: int
    # Main P_FA uses only active false-alarm trials, i.e., targets with selected links.
    false_alarm: int
    total_false: int
    # System-level P_FA keeps inactive targets in the potential denominator.
    false_alarm_overall: int
    total_false_overall: int
    overhead_bits: float
    overhead_delay_s: float
    selected_links: Dict[int, List[Link]]
    D_fuse_per_target: np.ndarray
    active_targets: int
    feasible_targets: int
    feasible_links: int
    # Communication-side metrics for selected soft-information links.
    selected_rate_mean_mbps: float
    selected_rate_min_mbps: float
    selected_rate_p10_mbps: float
    selected_chi_mean: float
    selected_chi_min: float
    selected_chi_p10: float
    selected_gamma_comm_mean_db: float
    selected_rate_satisfaction_ratio: float
    selected_chi_ge_min_ratio: float
    comm_feasible_edge_ratio: float


# ============================================================
# Basic math helpers
# ============================================================

def wavelength(cfg: SimConfig) -> float:
    return cfg.c / cfg.fc


def bandwidth(cfg: SimConfig) -> float:
    return cfg.N * cfg.delta_f


def noise_power(cfg: SimConfig) -> float:
    psd_w_hz = 10.0 ** ((cfg.noise_psd_dbm_hz - 30.0) / 10.0)
    nf_linear = 10.0 ** (cfg.noise_figure_db / 10.0)
    return psd_w_hz * bandwidth(cfg) * nf_linear


def path_gain(d: np.ndarray, cfg: SimConfig) -> np.ndarray:
    lam = wavelength(cfg)
    d = np.maximum(d, 1.0)
    return (lam / (4.0 * np.pi)) ** 2 / (d ** cfg.path_loss_exp)


def rician_power_gain(shape: Tuple[int, ...], K_db: float, rng: np.random.Generator) -> np.ndarray:
    K = 10.0 ** (K_db / 10.0)
    los = np.sqrt(K / (K + 1.0))
    nlos = np.sqrt(1.0 / (K + 1.0)) * (
            rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ) / np.sqrt(2.0)
    return np.abs(los + nlos) ** 2


def qfunc(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return 0.5 * math.erfc(float(arr.item()) / math.sqrt(2.0))
    vals = [0.5 * math.erfc(float(v) / math.sqrt(2.0)) for v in arr.ravel()]
    return np.array(vals, dtype=float).reshape(arr.shape)


def normal_pdf(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    out = np.exp(-0.5 * arr * arr) / math.sqrt(2.0 * math.pi)
    if out.ndim == 0:
        return float(out.item())
    return out


def threshold_from_pfa(cfg: SimConfig) -> float:
    table = {0.10: 1.2816, 0.05: 1.6449, 0.01: 2.3263, 0.001: 3.0902}
    for pfa, thr in table.items():
        if abs(cfg.Pfa_target - pfa) < 1e-12:
            return thr
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if qfunc(mid) > cfg.Pfa_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pd_from_deflection(cfg: SimConfig, D: np.ndarray | float) -> np.ndarray:
    eta = threshold_from_pfa(cfg)
    D_arr = np.asarray(D, dtype=float)
    return np.asarray(qfunc(eta - np.sqrt(np.maximum(D_arr, 0.0))), dtype=float)


def d_pd_d_D(cfg: SimConfig, D: np.ndarray | float) -> np.ndarray:
    """Derivative of Q(eta - sqrt(D)) w.r.t. D."""
    eta = threshold_from_pfa(cfg)
    D_arr = np.asarray(D, dtype=float)
    sqrtD = np.sqrt(np.maximum(D_arr, 1e-4))
    z = eta - sqrtD
    return np.asarray(normal_pdf(z), dtype=float) / (2.0 * sqrtD)


def binomial_ci95(success: int, total: int) -> Tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    z = 1.96
    n = float(total)
    p = success / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi, 0.5 * (hi - lo)


# ============================================================
# Geometry and link model
# ============================================================

def generate_geometry(cfg: SimConfig, rng: np.random.Generator) -> Geometry:
    p_uav = np.zeros((cfg.M, 3))
    p_uav[:, :2] = rng.uniform(0.0, cfg.area_xy, size=(cfg.M, 2))
    p_uav[:, 2] = rng.uniform(cfg.h_uav_min, cfg.h_uav_max, size=cfg.M)

    p_tgt = np.zeros((cfg.Q, 3))
    p_tgt[:, :2] = rng.uniform(0.0, cfg.area_xy, size=(cfg.Q, 2))
    p_tgt[:, 2] = rng.uniform(cfg.h_target_min, cfg.h_target_max, size=cfg.Q)

    v_uav = np.zeros((cfg.M, 3))
    sp_u = rng.uniform(cfg.uav_speed_min, cfg.uav_speed_max, size=cfg.M)
    ang_u = rng.uniform(0.0, 2.0 * np.pi, size=cfg.M)
    v_uav[:, 0] = sp_u * np.cos(ang_u)
    v_uav[:, 1] = sp_u * np.sin(ang_u)

    v_tgt = np.zeros((cfg.Q, 3))
    sp_t = rng.uniform(cfg.target_speed_min, cfg.target_speed_max, size=cfg.Q)
    ang_t = rng.uniform(0.0, 2.0 * np.pi, size=cfg.Q)
    v_tgt[:, 0] = sp_t * np.cos(ang_t)
    v_tgt[:, 1] = sp_t * np.sin(ang_t)

    return Geometry(p_uav=p_uav, v_uav=v_uav, p_tgt=p_tgt, v_tgt=v_tgt)


def build_base_gains(cfg: SimConfig, geom: Geometry, rng: np.random.Generator) -> BaseGains:
    M, Q = cfg.M, cfg.Q
    lam = wavelength(cfg)

    diff_uu = geom.p_uav[:, None, :] - geom.p_uav[None, :, :]
    d_uu = np.linalg.norm(diff_uu, axis=-1)
    edge_mask = (d_uu <= cfg.comm_range) & (~np.eye(M, dtype=bool))

    direct_gain = path_gain(d_uu, cfg)
    direct_gain *= 10.0 ** (rng.normal(0.0, cfg.shadow_std_db, size=(M, M)) / 10.0)
    direct_gain *= rician_power_gain((M, M), cfg.rician_K_db, rng)
    np.fill_diagonal(direct_gain, 0.0)

    diff_uav_tgt = geom.p_uav[:, None, :] - geom.p_tgt[None, :, :]
    d_uav_tgt = np.linalg.norm(diff_uav_tgt, axis=-1)

    tau = np.zeros((M, M, Q))
    doppler = np.zeros((M, M, Q))
    delay_bin = np.zeros((M, M, Q), dtype=int)
    doppler_bin = np.zeros((M, M, Q), dtype=int)
    valid_dd = np.zeros((M, M, Q), dtype=bool)
    dd_frac_loss = np.ones((M, M, Q), dtype=float)
    target_gain = np.zeros((M, M, Q))
    geom_factor = np.zeros((M, M, Q))

    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            for q in range(Q):
                p_i, p_j, p_q = geom.p_uav[i], geom.p_uav[j], geom.p_tgt[q]
                v_i, v_j, v_q = geom.v_uav[i], geom.v_uav[j], geom.v_tgt[q]

                vec_iq = p_q - p_i
                vec_jq = p_q - p_j
                d_iq = max(float(np.linalg.norm(vec_iq)), 1.0)
                d_jq = max(float(np.linalg.norm(vec_jq)), 1.0)
                u_iq = vec_iq / d_iq
                u_jq = vec_jq / d_jq

                tau_ijq = (d_iq + d_jq) / cfg.c
                rr_tx = float(np.dot(v_q - v_i, u_iq))
                rr_rx = float(np.dot(v_q - v_j, u_jq))
                nu_ijq = (rr_tx + rr_rx) / lam

                tau[i, j, q] = tau_ijq
                doppler[i, j, q] = nu_ijq

                l_float = tau_ijq * cfg.L * cfg.delta_f
                k_float = nu_ijq * cfg.N * cfg.T
                l_bin = int(np.round(l_float))
                k_bin = int(np.round(k_float))
                delay_bin[i, j, q] = l_bin
                doppler_bin[i, j, q] = k_bin
                valid_dd[i, j, q] = (0 <= l_bin < cfg.L) and (-(cfg.N // 2) <= k_bin < cfg.N // 2)

                delay_frac = abs(l_float - l_bin)
                doppler_frac = abs(k_float - k_bin)
                dd_frac_loss[i, j, q] = float((np.sinc(delay_frac) ** 2) * (np.sinc(doppler_frac) ** 2))

                rcs_fluct = rng.exponential(scale=cfg.target_rcs)
                target_gain[i, j, q] = lam ** 2 * rcs_fluct / ((4.0 * np.pi) ** 3 * d_iq ** 2 * d_jq ** 2)

                a = (p_i - p_q) / d_iq
                b = (p_j - p_q) / d_jq
                sin_angle = float(np.linalg.norm(np.cross(a, b)))
                geom_factor[i, j, q] = 0.1 + 0.9 * np.clip(sin_angle, 0.0, 1.0)

    dd_collision_count = np.ones((M, M, Q), dtype=float)
    if cfg.enable_dd_collision_penalty:
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                bins: Dict[Tuple[int, int], List[int]] = {}
                for q in range(Q):
                    if not valid_dd[i, j, q]:
                        continue
                    key = (int(delay_bin[i, j, q]), int(doppler_bin[i, j, q]))
                    bins.setdefault(key, []).append(q)
                for qs in bins.values():
                    for q in qs:
                        dd_collision_count[i, j, q] = float(len(qs))

    return BaseGains(
        edge_mask=edge_mask,
        direct_gain=direct_gain,
        d_uu=d_uu,
        d_uav_tgt=d_uav_tgt,
        tau=tau,
        doppler=doppler,
        delay_bin=delay_bin,
        doppler_bin=doppler_bin,
        valid_dd=valid_dd,
        dd_frac_loss=dd_frac_loss,
        dd_collision_count=dd_collision_count,
        target_gain=target_gain,
        geom_factor=geom_factor,
    )


def compute_link_tables(cfg: SimConfig, base: BaseGains) -> LinkTables:
    M, Q = cfg.M, cfg.Q
    P = np.full(M, cfg.P_default, dtype=float)
    P_sense = cfg.rho * P
    P_comm = (1.0 - cfg.rho) * P

    n0 = noise_power(cfg)
    B = bandwidth(cfg)
    gamma_req = max(2.0 ** (cfg.R_min / B) - 1.0, EPS)

    gamma_comm = np.zeros((M, M))
    rate = np.zeros((M, M))
    chi_comm = np.zeros((M, M))
    feasible_comm = np.zeros((M, M), dtype=bool)

    for i in range(M):
        for j in range(M):
            if i == j or not base.edge_mask[i, j]:
                continue

            signal = P_comm[i] * base.direct_gain[i, j]
            interf = 0.0
            for k in range(M):
                if k == i or k == j:
                    continue
                interf += (P_comm[k] + cfg.comm_leakage_from_sensing * P_sense[k]) * base.direct_gain[k, j]

            direct_leakage = cfg.comm_direct_leakage_factor * P_sense[i] * base.direct_gain[i, j]
            gamma = signal / (n0 + interf + direct_leakage + EPS)
            gamma_comm[i, j] = gamma
            rate[i, j] = B * np.log2(1.0 + gamma)
            chi_comm[i, j] = gamma / (gamma + gamma_req + EPS)

            if cfg.enforce_chi_min:
                feasible_comm[i, j] = (rate[i, j] >= cfg.R_min) and (chi_comm[i, j] >= cfg.chi_min)
            else:
                feasible_comm[i, j] = (rate[i, j] >= cfg.R_min)

    raw_gamma_sense = np.zeros((M, M, Q))
    gamma_sense = np.zeros((M, M, Q))
    rinr = np.zeros((M, M))
    beta = np.zeros((M, M, Q))
    mu_soft = np.zeros((M, M, Q))
    sigma0 = np.full((M, M), cfg.soft_sigma0)

    G_proc = cfg.N * cfg.L if cfg.sensing_processing_gain is None else cfg.sensing_processing_gain

    for i in range(M):
        for j in range(M):
            if i == j or not base.edge_mask[i, j]:
                continue

            residual_self = cfg.residual_self_factor * P[j]
            residual_direct = 0.0
            residual_multi = 0.0
            # Direct-link cancellation error and multi-UAV sensing leakage use
            # the same interferer set, so they are accumulated in one loop.
            for k in range(M):
                if k == i or k == j:
                    continue
                residual_direct += cfg.residual_direct_factor * P[k] * base.direct_gain[k, j]
                residual_multi += cfg.residual_multi_uav_factor * P_sense[k] * base.direct_gain[k, j]

            residual_total = residual_self + residual_direct + residual_multi
            rinr[i, j] = residual_total / (n0 + EPS)
            sigma0[i, j] = cfg.soft_sigma0 * math.sqrt(1.0 + cfg.rinr_sigma_factor * rinr[i, j])

            if cfg.isac_power_model == "sensing_only":
                effective_sensing_power = P_sense[i]
            elif cfg.isac_power_model == "joint_waveform":
                effective_sensing_power = P[i]
            elif cfg.isac_power_model == "reliable_comm_assisted":
                effective_sensing_power = P_sense[i] + chi_comm[i, j] * P_comm[i]
            else:
                raise ValueError(f"Unknown isac_power_model={cfg.isac_power_model!r}")

            for q in range(Q):
                if cfg.use_otfs_bin_validity and (not base.valid_dd[i, j, q]):
                    continue

                # Keep the raw sensing-SINR baseline consistent with the selected ISAC power model.
                raw_signal = effective_sensing_power * base.target_gain[i, j, q] * G_proc
                raw_gamma_sense[i, j, q] = raw_signal / (n0 + EPS)

                collision_penalty = 1.0 / (max(base.dd_collision_count[i, j, q], 1.0) ** cfg.dd_collision_alpha)
                dd_loss = base.dd_frac_loss[i, j, q] if cfg.enable_dd_fractional_penalty else 1.0

                signal = effective_sensing_power * base.target_gain[i, j, q] * G_proc * collision_penalty * dd_loss
                gamma = signal / (n0 + residual_total + EPS)
                gamma_sense[i, j, q] = gamma
                mu_soft[i, j, q] = cfg.soft_mu_scale * np.log1p(gamma)

                if feasible_comm[i, j]:
                    chi_eff = float(np.clip(chi_comm[i, j], 0.0, 1.0))
                    delay_ms = 1e3 * packet_bits_for_target(cfg, q) / max(rate[i, j], EPS)
                    beta[i, j, q] = (
                            cfg.beta_scale
                            * np.log1p(gamma)
                            * base.geom_factor[i, j, q]
                            * (chi_eff ** cfg.beta_chi_power)
                            / ((1.0 + rinr[i, j]) * ((delay_ms + EPS) ** cfg.beta_delay_exponent))
                    )

    return LinkTables(
        gamma_comm=gamma_comm,
        rate=rate,
        chi_comm=chi_comm,
        feasible_comm=feasible_comm,
        raw_gamma_sense=raw_gamma_sense,
        gamma_sense=gamma_sense,
        rinr=rinr,
        beta=beta,
        mu_soft=mu_soft,
        sigma0=sigma0,
    )


# ============================================================
# Fusion and selection
# ============================================================

def packet_bits_for_target(cfg: SimConfig, q: int) -> float:
    # Simplified version: all targets use the same number of candidate packets.
    # The q argument is kept for API consistency with link-delay helpers.
    return float(max(cfg.K_candidates, 0) * cfg.b_d)


def link_delay_s(cfg: SimConfig, tables: LinkTables, q: int, link: Link) -> float:
    i, j = link
    return packet_bits_for_target(cfg, q) / max(tables.rate[i, j], EPS)


def link_cost_ms(cfg: SimConfig, tables: LinkTables, q: int, link: Link) -> float:
    return 1e3 * link_delay_s(cfg, tables, q, link)


def feasible_links_for_target(cfg: SimConfig, base: BaseGains, tables: LinkTables, q: int) -> List[Link]:
    links: List[Link] = []
    for i in range(cfg.M):
        for j in range(cfg.M):
            if i == j:
                continue
            if not base.edge_mask[i, j]:
                continue
            if cfg.use_otfs_bin_validity and not base.valid_dd[i, j, q]:
                continue
            if not tables.feasible_comm[i, j]:
                continue
            if tables.beta[i, j, q] <= 0:
                continue
            links.append((i, j))
    return links


def sensing_only_links_for_target(cfg: SimConfig, base: BaseGains, q: int) -> List[Link]:
    links: List[Link] = []
    for i in range(cfg.M):
        for j in range(cfg.M):
            if i == j:
                continue
            if not base.edge_mask[i, j]:
                continue
            if cfg.use_otfs_bin_validity and not base.valid_dd[i, j, q]:
                continue
            links.append((i, j))
    return links


def effective_h1_mean_for_link(cfg: SimConfig, tables: LinkTables, link: Link, q: int) -> float:
    i, j = link
    mu = float(tables.mu_soft[i, j, q])

    if (not cfg.enable_comm_error_pollution) or (not cfg.use_comm_error_calibration):
        return mu

    chi = float(np.clip(tables.chi_comm[i, j], 0.0, 1.0))
    if cfg.comm_error_model == "erasure":
        return chi * mu
    if cfg.comm_error_model == "flip":
        return (chi - (1.0 - chi) * cfg.soft_error_flip_scale) * mu
    if cfg.comm_error_model == "biased":
        return (chi + (1.0 - chi) * cfg.soft_error_bias_scale) * mu
    raise ValueError(cfg.comm_error_model)


def h0_variance_for_link(cfg: SimConfig, tables: LinkTables, link: Link) -> float:
    i, j = link
    sigma0 = float(tables.sigma0[i, j])

    if not cfg.enable_comm_error_pollution:
        return sigma0 ** 2

    chi = float(np.clip(tables.chi_comm[i, j], 0.0, 1.0))
    sigma_err = cfg.soft_error_sigma_scale * sigma0
    return chi * sigma0 ** 2 + (1.0 - chi) * sigma_err ** 2


def deflection_variance_for_link(cfg: SimConfig, tables: LinkTables, link: Link) -> float:
    """Variance model used by the algorithm-side deflection estimate.

    Detection always uses h0_variance_for_link(). For the ablation
    w/o comm-error calibration, this function intentionally ignores
    communication-error variance inflation while the Monte Carlo detector
    remains polluted.
    """
    i, j = link
    sigma0 = float(tables.sigma0[i, j])
    if (not cfg.enable_comm_error_pollution) or (not cfg.use_comm_error_calibration):
        return sigma0 ** 2
    return h0_variance_for_link(cfg, tables, link)


def compute_weights(cfg: SimConfig, tables: LinkTables, q: int, links: List[Link], mode: str = "beta") -> Dict[
    Link, float]:
    if not links:
        return {}

    if mode == "equal":
        return {link: 1.0 / len(links) for link in links}

    vals: List[float] = []
    for link in links:
        i, j = link
        if mode == "deflection":
            mu_eff = effective_h1_mean_for_link(cfg, tables, link, q)
            vals.append(max(mu_eff, 0.0) / (deflection_variance_for_link(cfg, tables, link) + EPS))
        else:
            vals.append(max(float(tables.beta[i, j, q]), 0.0))

    total = float(np.sum(vals))
    if total <= EPS:
        return {link: 1.0 / len(links) for link in links}
    return {link: v / total for link, v in zip(links, vals)}


def fusion_weight_mode_for_method(method: str) -> str:
    """Use self-consistent fusion weights for detection/deflection.

    - raw_sense_sinr keeps equal weights because it is a pure sensing-only baseline.
    - all other methods use deflection weights proportional to calibrated mu_eff/sigma0^2.
    This avoids mixing beta-based selection utility with effective-H1 deflection estimation.
    """
    return "equal" if method == "raw_sense_sinr" else "deflection"


def deflection_for_links(
        cfg: SimConfig,
        tables: LinkTables,
        q: int,
        links: List[Link],
        weight_mode: str = "deflection",
) -> float:
    if not links:
        return 0.0

    weights = compute_weights(cfg, tables, q, links, mode=weight_mode)
    mean_gap = 0.0
    var0 = 0.0
    for link, w in weights.items():
        mean_gap += w * effective_h1_mean_for_link(cfg, tables, link, q)
        var0 += (w ** 2) * deflection_variance_for_link(cfg, tables, link)

    return float((max(mean_gap, 0.0) ** 2) / (var0 + EPS))


def target_alpha(cfg: SimConfig, D_fuse: np.ndarray) -> np.ndarray:
    """
    Target marginal value alpha_q.

    alpha_q = dR/dD_q + mu * normalized_deficit

    If use_softmin_alpha=True, dR/dD_q is weighted by soft-min over predicted Pd,
    so weak targets receive larger priority.
    """
    D = np.asarray(D_fuse, dtype=float)
    if not cfg.use_target_priority:
        return np.ones_like(D, dtype=float)

    pd = pd_from_deflection(cfg, D)
    dpd = d_pd_d_D(cfg, D)

    if cfg.use_softmin_alpha:
        tau = max(cfg.softmin_tau, EPS)
        logits = -(pd - np.min(pd)) / tau
        logits = logits - np.max(logits)  # stable softmax; max logit becomes 0
        w = np.exp(logits)
        w = w / max(float(np.sum(w)), EPS)
        # w sums to one. Multiplying by Q keeps the marginal scale comparable
        # to a sum-P_D objective instead of shrinking alpha by roughly 1/Q.
        marginal = w * dpd * cfg.Q
    else:
        marginal = dpd

    deficit = np.maximum(cfg.D_min - D, 0.0) / max(cfg.D_min, EPS)
    alpha = marginal + cfg.lagrangian_mu * deficit
    alpha = np.clip(alpha, cfg.alpha_floor, cfg.alpha_cap)
    return alpha


def topk_links_by_beta(
        cfg: SimConfig,
        tables: LinkTables,
        links: List[Link],
        q: int,
) -> List[Link]:
    """Prune candidate links by beta to reduce Lagrangian selection complexity."""
    if cfg.candidate_topk_per_target <= 0 or len(links) <= cfg.candidate_topk_per_target:
        return links
    scores = np.array([tables.beta[i, j, q] for i, j in links], dtype=float)
    order = np.argsort(scores)[::-1][: cfg.candidate_topk_per_target]
    return [links[int(idx)] for idx in order]


def select_lagrangian(
        cfg: SimConfig,
        base: BaseGains,
        tables: LinkTables,
) -> Tuple[Dict[int, List[Link]], np.ndarray]:
    """
    Lagrangian-guided link selection.

    For each candidate link (i,j,q), compute:
        score = alpha_q * DeltaD_ijq - lambda_c * delay_ms_ijq

    Select the best positive-score link iteratively.
    """
    selected: Dict[int, List[Link]] = {q: [] for q in range(cfg.Q)}
    selected_sets: Dict[int, set[Link]] = {q: set() for q in range(cfg.Q)}
    D_fuse = np.zeros(cfg.Q)

    candidates = {
        q: topk_links_by_beta(cfg, tables, feasible_links_for_target(cfg, base, tables, q), q)
        for q in range(cfg.Q)
    }
    active_candidate_targets = [q for q in range(cfg.Q) if len(candidates[q]) > 0]
    if not active_candidate_targets:
        return selected, D_fuse

    total_links = 0

    while total_links < cfg.max_total_links:
        alpha = target_alpha(cfg, D_fuse)
        best_tuple: Optional[Tuple[int, Link]] = None
        best_score = -np.inf
        best_D = 0.0
        best_marginal = 0.0

        for q in range(cfg.Q):
            if len(selected[q]) >= cfg.max_links_per_target:
                continue
            for link in candidates[q]:
                if link in selected_sets[q]:
                    continue
                new_links = selected[q] + [link]
                new_D = deflection_for_links(cfg, tables, q, new_links, weight_mode="deflection")
                marginal_D = new_D - D_fuse[q]
                if marginal_D <= cfg.min_marginal_D:
                    continue

                cost_ms = link_cost_ms(cfg, tables, q, link)
                delay_price = cfg.lagrangian_lambda * cost_ms if cfg.use_delay_price else 0.0
                score = alpha[q] * marginal_D - delay_price

                if score > best_score:
                    best_score = score
                    best_tuple = (q, link)
                    best_D = new_D
                    best_marginal = marginal_D

        if best_tuple is None or best_score <= 0.0:
            break

        q_best, link_best = best_tuple
        selected[q_best].append(link_best)
        selected_sets[q_best].add(link_best)
        D_fuse[q_best] = best_D
        total_links += 1

        # Stop early once all targets that actually have candidate links meet the target deflection.
        # Targets with no feasible candidates are excluded because their D_fuse cannot be improved.
        if np.all(D_fuse[active_candidate_targets] >= cfg.D_min):
            break

    return selected, D_fuse


def select_all_neighbor(cfg: SimConfig, base: BaseGains, tables: LinkTables) -> Tuple[
    Dict[int, List[Link]], np.ndarray]:
    """Upper-resource baseline using all feasible links.

    This method intentionally ignores max_links_per_target and max_total_links.
    It is an upper-resource / upper-bound baseline, not a fair resource-constrained competitor.
    """
    selected: Dict[int, List[Link]] = {}
    D = np.zeros(cfg.Q)
    for q in range(cfg.Q):
        links = feasible_links_for_target(cfg, base, tables, q)
        selected[q] = links
        D[q] = deflection_for_links(cfg, tables, q, links, weight_mode="deflection")
    return selected, D


def select_topk_baseline(
        cfg: SimConfig,
        base: BaseGains,
        tables: LinkTables,
        reference_counts: Dict[int, int],
        method: str,
        rng: np.random.Generator,
) -> Tuple[Dict[int, List[Link]], np.ndarray]:
    selected: Dict[int, List[Link]] = {}
    D = np.zeros(cfg.Q)

    for q in range(cfg.Q):
        if method == "raw_sense_sinr":
            links = sensing_only_links_for_target(cfg, base, q)
        else:
            links = feasible_links_for_target(cfg, base, tables, q)

        if not links:
            selected[q] = []
            continue

        if method == "single_best":
            K = 1
        else:
            K = min(reference_counts.get(q, 0), len(links))

        if K <= 0:
            selected[q] = []
            continue

        if method == "random":
            order = rng.permutation(len(links))
        elif method == "nearest":
            # Nearest baseline: choose the smallest UAV-to-UAV distance directly.
            distances = np.array([base.d_uu[i, j] for i, j in links])
            order = np.argsort(distances)
        elif method == "shortest_bistatic":
            scores = np.array([-base.tau[i, j, q] * cfg.c for i, j in links])
            order = np.argsort(scores)[::-1]
        elif method == "raw_sense_sinr":
            scores = np.array([tables.raw_gamma_sense[i, j, q] for i, j in links])
            order = np.argsort(scores)[::-1]
        elif method == "sense_sinr":
            scores = np.array([tables.gamma_sense[i, j, q] for i, j in links])
            order = np.argsort(scores)[::-1]
        elif method == "single_best":
            scores = np.array([tables.beta[i, j, q] for i, j in links])
            order = np.argsort(scores)[::-1]
        else:
            raise ValueError(method)

        chosen = [links[int(idx)] for idx in order[:K]]
        selected[q] = chosen
        weight_mode = fusion_weight_mode_for_method(method)
        D[q] = deflection_for_links(cfg, tables, q, chosen, weight_mode=weight_mode)

    return selected, D


# ============================================================
# Detection and metrics
# ============================================================

def draw_h1_soft_stat(cfg: SimConfig, tables: LinkTables, link: Link, q: int, rng: np.random.Generator) -> float:
    i, j = link
    mu = float(tables.mu_soft[i, j, q])
    gamma = float(tables.gamma_sense[i, j, q])
    sigma1 = max(cfg.soft_sigma_floor, float(tables.sigma0[i, j]) / math.sqrt(1.0 + gamma + EPS))

    if not cfg.enable_comm_error_pollution:
        return float(rng.normal(mu, sigma1))

    chi = float(np.clip(tables.chi_comm[i, j], 0.0, 1.0))
    if rng.random() < chi:
        return float(rng.normal(mu, sigma1))

    sigma_err = cfg.soft_error_sigma_scale * float(tables.sigma0[i, j])
    if cfg.comm_error_model == "erasure":
        return float(rng.normal(0.0, sigma_err))
    if cfg.comm_error_model == "flip":
        return float(rng.normal(-cfg.soft_error_flip_scale * mu, sigma_err))
    if cfg.comm_error_model == "biased":
        return float(rng.normal(cfg.soft_error_bias_scale * mu, sigma_err))
    raise ValueError(cfg.comm_error_model)


def draw_h0_soft_stat(cfg: SimConfig, tables: LinkTables, link: Link, rng: np.random.Generator) -> float:
    i, j = link
    sigma0 = float(tables.sigma0[i, j])

    if not cfg.enable_comm_error_pollution:
        return float(rng.normal(0.0, sigma0))

    chi = float(np.clip(tables.chi_comm[i, j], 0.0, 1.0))
    if rng.random() < chi:
        return float(rng.normal(0.0, sigma0))

    sigma_err = cfg.soft_error_sigma_scale * sigma0
    if cfg.comm_error_model == "erasure":
        return float(rng.normal(0.0, sigma_err))
    if cfg.comm_error_model == "flip":
        # Under H0 there is no target-dependent sign to flip; the failed packet only inflates noise.
        return float(rng.normal(0.0, sigma_err))
    if cfg.comm_error_model == "biased":
        # Optional H0-side bias makes biased-error robustness tests symmetric with the H1 model.
        return float(rng.normal(cfg.h0_error_bias_scale * sigma0, sigma_err))
    raise ValueError(cfg.comm_error_model)


def evaluate_detection(
        cfg: SimConfig,
        tables: LinkTables,
        selected: Dict[int, List[Link]],
        rng: np.random.Generator,
        method: str,
) -> Tuple[int, int, int, int, int, int]:
    """Evaluate detection and false alarms with two P_FA denominators.

    Main P_FA is the active-detector false alarm rate: only targets with at
    least one selected link contribute false-alarm trials. This is the proper
    denominator for checking whether the Gaussian threshold is calibrated.

    System-level P_FA keeps all target hypotheses in the potential denominator.
    Inactive targets generate no fused statistic and therefore contribute zero
    false alarms.  Hence the system-level numerator is the active false-alarm
    count, while the denominator is Q*num_false_per_target.  This is a system
    potential-false-alarm rate, not the active-detector calibration rate.
    """
    detected = 0
    false_alarm_active = 0
    total_targets = cfg.Q
    total_false_active = 0
    total_false_overall = cfg.Q * cfg.num_false_per_target

    base_thr = threshold_from_pfa(cfg)

    for q in range(cfg.Q):
        links = selected.get(q, [])
        if not links:
            # No soft information means no active detector for this target.
            # It contributes to the system-level potential denominator but not
            # to the active-detector denominator.
            continue

        total_false_active += cfg.num_false_per_target
        weight_mode = fusion_weight_mode_for_method(method)
        weights = compute_weights(cfg, tables, q, links, mode=weight_mode)
        var0 = sum((w ** 2) * h0_variance_for_link(cfg, tables, link) for link, w in weights.items())
        thr = base_thr * math.sqrt(max(var0, EPS))

        F = 0.0
        for link, w in weights.items():
            F += w * draw_h1_soft_stat(cfg, tables, link, q, rng)
        if F > thr:
            detected += 1

        for _ in range(cfg.num_false_per_target):
            F0 = 0.0
            for link, w in weights.items():
                F0 += w * draw_h0_soft_stat(cfg, tables, link, rng)
            if F0 > thr:
                false_alarm_active += 1

    # System-level numerator: inactive targets produce zero false alarms, so
    # the only observed false alarms are those from active detectors.
    false_alarm_overall = false_alarm_active
    return (
        detected,
        total_targets,
        false_alarm_active,
        total_false_active,
        false_alarm_overall,
        total_false_overall,
    )


def total_overhead_bits(cfg: SimConfig, selected: Dict[int, List[Link]]) -> float:
    return float(sum(packet_bits_for_target(cfg, q) * len(links) for q, links in selected.items()))


def total_overhead_delay_s(cfg: SimConfig, tables: LinkTables, selected: Dict[int, List[Link]]) -> float:
    total = 0.0
    for q, links in selected.items():
        for link in links:
            total += link_delay_s(cfg, tables, q, link)
    return float(total)


def active_target_count(selected: Dict[int, List[Link]]) -> int:
    return sum(1 for links in selected.values() if len(links) > 0)


def feasible_stats(cfg: SimConfig, base: BaseGains, tables: LinkTables) -> Tuple[int, int]:
    counts = [len(feasible_links_for_target(cfg, base, tables, q)) for q in range(cfg.Q)]
    return sum(1 for c in counts if c > 0), int(np.sum(counts))


def communication_metrics_for_selection(
        cfg: SimConfig,
        base: BaseGains,
        tables: LinkTables,
        selected: Dict[int, List[Link]],
) -> Dict[str, float]:
    """Communication-side statistics for the selected soft-information links.

    These metrics are not communication-throughput objectives. They document the
    communication role in the sensing-centric ISAC model: selected links must
    carry soft sensing information, so their rate, reliability, and delay matter.
    """
    selected_unique: List[Link] = []
    seen: set[Link] = set()
    for links in selected.values():
        for link in links:
            if link not in seen:
                seen.add(link)
                selected_unique.append(link)

    edge_count = int(np.sum(base.edge_mask))
    feasible_edge_count = int(np.sum(tables.feasible_comm & base.edge_mask))
    comm_feasible_edge_ratio = feasible_edge_count / max(edge_count, 1)

    if not selected_unique:
        return {
            "selected_rate_mean_mbps": 0.0,
            "selected_rate_min_mbps": 0.0,
            "selected_rate_p10_mbps": 0.0,
            "selected_chi_mean": 0.0,
            "selected_chi_min": 0.0,
            "selected_chi_p10": 0.0,
            "selected_gamma_comm_mean_db": 0.0,
            "selected_rate_satisfaction_ratio": 0.0,
            "selected_chi_ge_min_ratio": 0.0,
            "comm_feasible_edge_ratio": float(comm_feasible_edge_ratio),
        }

    rates = np.array([tables.rate[i, j] for i, j in selected_unique], dtype=float)
    chis = np.array([tables.chi_comm[i, j] for i, j in selected_unique], dtype=float)
    gammas = np.array([tables.gamma_comm[i, j] for i, j in selected_unique], dtype=float)
    gamma_db = 10.0 * np.log10(np.maximum(gammas, EPS))

    return {
        "selected_rate_mean_mbps": float(np.mean(rates) / 1e6),
        "selected_rate_min_mbps": float(np.min(rates) / 1e6),
        "selected_rate_p10_mbps": float(np.percentile(rates, 10) / 1e6),
        "selected_chi_mean": float(np.mean(chis)),
        "selected_chi_min": float(np.min(chis)),
        "selected_chi_p10": float(np.percentile(chis, 10)),
        "selected_gamma_comm_mean_db": float(np.mean(gamma_db)),
        "selected_rate_satisfaction_ratio": float(np.mean(rates >= cfg.R_min)),
        "selected_chi_ge_min_ratio": float(np.mean(chis >= cfg.chi_min)),
        "comm_feasible_edge_ratio": float(comm_feasible_edge_ratio),
    }


def rng_for_method(cfg: SimConfig, trial_index: int, method: str) -> np.random.Generator:
    offset = METHOD_RNG_OFFSETS[method]
    return np.random.default_rng([cfg.seed, 12345, trial_index, offset])


def run_method_on_trial(
        cfg: SimConfig,
        base: BaseGains,
        tables: LinkTables,
        method: MethodName,
        trial_index: int,
        reference_counts: Optional[Dict[int, int]] = None,
        cached_lagrangian: Optional[Tuple[Dict[int, List[Link]], np.ndarray]] = None,
) -> MethodResult:
    rng = rng_for_method(cfg, trial_index, method)

    if method == "proposed_lagrangian":
        if cached_lagrangian is None:
            selected, D = select_lagrangian(cfg, base, tables)
        else:
            selected, D = cached_lagrangian
    elif method == "all_neighbor":
        selected, D = select_all_neighbor(cfg, base, tables)
    else:
        if reference_counts is None:
            ref_selected, _ = select_lagrangian(cfg, base, tables)
            reference_counts = {q: len(ref_selected.get(q, [])) for q in range(cfg.Q)}
        selected, D = select_topk_baseline(cfg, base, tables, reference_counts, method, rng)

    detected, total_targets, fa, total_false, fa_overall, total_false_overall = evaluate_detection(
        cfg, tables, selected, rng, method
    )
    feasible_targets, feasible_links = feasible_stats(cfg, base, tables)
    comm_metrics = communication_metrics_for_selection(cfg, base, tables, selected)

    return MethodResult(
        name=method,
        detected=detected,
        total_targets=total_targets,
        false_alarm=fa,
        total_false=total_false,
        false_alarm_overall=fa_overall,
        total_false_overall=total_false_overall,
        overhead_bits=total_overhead_bits(cfg, selected),
        overhead_delay_s=total_overhead_delay_s(cfg, tables, selected),
        selected_links=selected,
        D_fuse_per_target=D,
        active_targets=active_target_count(selected),
        feasible_targets=feasible_targets,
        feasible_links=feasible_links,
        **comm_metrics,
    )


def run_one_trial(cfg: SimConfig, trial_index: int) -> Dict[str, MethodResult]:
    # Use SeedSequence-style multi-integer seeding to avoid correlations from consecutive seeds.
    rng = np.random.default_rng([cfg.seed, trial_index])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    tables = compute_link_tables(cfg, base)

    lag_selected, lag_D = select_lagrangian(cfg, base, tables)
    cached_lagrangian = (lag_selected, lag_D)
    reference_counts = {q: len(lag_selected.get(q, [])) for q in range(cfg.Q)}

    results: Dict[str, MethodResult] = {}
    for method in METHODS:
        results[method] = run_method_on_trial(
            cfg, base, tables, method, trial_index, reference_counts, cached_lagrangian
        )
    return results


def summarize(results: List[MethodResult], cfg: SimConfig) -> Dict[str, Any]:
    total_detected = sum(r.detected for r in results)
    total_targets = sum(r.total_targets for r in results)

    # Main P_FA is active-detector P_FA.
    total_fa = sum(r.false_alarm for r in results)
    total_false = sum(r.total_false for r in results)

    # Overall/system P_FA includes inactive target hypotheses in the denominator.
    total_fa_overall = sum(r.false_alarm_overall for r in results)
    total_false_overall = sum(r.total_false_overall for r in results)

    pd_ci = binomial_ci95(total_detected, total_targets)
    pfa_ci = binomial_ci95(total_fa, total_false)
    pfa_overall_ci = binomial_ci95(total_fa_overall, total_false_overall)

    D_arr = np.vstack([r.D_fuse_per_target for r in results])
    overhead_bits = np.array([r.overhead_bits for r in results], dtype=float)
    overhead_delay_s = np.array([r.overhead_delay_s for r in results], dtype=float)
    active = np.array([r.active_targets for r in results], dtype=float)
    feasible_targets = np.array([r.feasible_targets for r in results], dtype=float)
    feasible_links = np.array([r.feasible_links for r in results], dtype=float)
    selected_links = np.array([sum(len(v) for v in r.selected_links.values()) for r in results], dtype=float)
    selected_rate_mean = np.array([r.selected_rate_mean_mbps for r in results], dtype=float)
    selected_rate_min = np.array([r.selected_rate_min_mbps for r in results], dtype=float)
    selected_rate_p10 = np.array([r.selected_rate_p10_mbps for r in results], dtype=float)
    selected_chi_mean = np.array([r.selected_chi_mean for r in results], dtype=float)
    selected_chi_min = np.array([r.selected_chi_min for r in results], dtype=float)
    selected_chi_p10 = np.array([r.selected_chi_p10 for r in results], dtype=float)
    selected_gamma_db = np.array([r.selected_gamma_comm_mean_db for r in results], dtype=float)
    rate_sat = np.array([r.selected_rate_satisfaction_ratio for r in results], dtype=float)
    chi_sat = np.array([r.selected_chi_ge_min_ratio for r in results], dtype=float)
    feasible_edge_ratio = np.array([r.comm_feasible_edge_ratio for r in results], dtype=float)

    D_satisfied = D_arr >= cfg.D_min
    pd = total_detected / max(total_targets, 1)
    pfa = total_fa / max(total_false, 1)
    pfa_overall = total_fa_overall / max(total_false_overall, 1)
    B_kbit = float(np.mean(overhead_bits)) / 1000.0
    T_ms = float(np.mean(overhead_delay_s)) * 1e3
    mean_D = float(np.mean(D_arr))

    return {
        "P_D": pd,
        "P_D_ci95": pd_ci[:2],
        "P_D_ci95_half_width": pd_ci[2],
        "P_FA": pfa,
        "P_FA_ci95": pfa_ci[:2],
        "P_FA_ci95_half_width": pfa_ci[2],
        "P_FA_overall": pfa_overall,
        "P_FA_overall_ci95": pfa_overall_ci[:2],
        "P_FA_overall_ci95_half_width": pfa_overall_ci[2],
        "B_mean_bits": float(np.mean(overhead_bits)),
        "B_std_bits": float(np.std(overhead_bits)),
        "T_mean_ms": T_ms,
        "T_std_ms": float(np.std(overhead_delay_s) * 1e3),
        "selected_links_mean": float(np.mean(selected_links)),
        "selected_links_std": float(np.std(selected_links)),
        "active_target_ratio_mean": float(np.mean(active / max(cfg.Q, 1))),
        "feasible_target_ratio_mean": float(np.mean(feasible_targets / max(cfg.Q, 1))),
        "feasible_links_mean": float(np.mean(feasible_links)),
        "comm_feasible_edge_ratio_mean": float(np.mean(feasible_edge_ratio)),
        "selected_rate_mean_mbps": float(np.mean(selected_rate_mean)),
        "selected_rate_min_mbps_mean": float(np.mean(selected_rate_min)),
        "selected_rate_p10_mbps_mean": float(np.mean(selected_rate_p10)),
        "selected_chi_mean": float(np.mean(selected_chi_mean)),
        "selected_chi_min_mean": float(np.mean(selected_chi_min)),
        "selected_chi_p10_mean": float(np.mean(selected_chi_p10)),
        "selected_gamma_comm_mean_db": float(np.mean(selected_gamma_db)),
        "selected_rate_satisfaction_ratio_mean": float(np.mean(rate_sat)),
        "selected_chi_ge_min_ratio_mean": float(np.mean(chi_sat)),
        "D_mean": mean_D,
        "D_median": float(np.median(D_arr)),
        "D_p10": float(np.percentile(D_arr, 10)),
        "D_p90": float(np.percentile(D_arr, 90)),
        "D_mean_per_target": np.mean(D_arr, axis=0),
        "D_satisfied_prob_per_target": np.mean(D_satisfied, axis=0),
        "all_targets_satisfied_prob": float(np.mean(np.all(D_satisfied, axis=1))),
        "worst_target_D_mean": float(np.min(np.mean(D_arr, axis=0))),
        "worst_target_satisfied_prob": float(np.min(np.mean(D_satisfied, axis=0))),
        "P_D_per_kbit": pd / max(B_kbit, EPS),
        "P_D_per_ms": pd / max(T_ms, EPS),
        "D_per_kbit": mean_D / max(B_kbit, EPS),
        "D_per_ms": mean_D / max(T_ms, EPS),
    }


def print_summary(summary: Dict[str, Dict[str, Any]]) -> None:
    print("\n========== Lagrangian DOTFS-ISAC Simplified Simulation Summary ==========")
    for method, s in summary.items():
        print(f"\n[{method}]")
        print(f"  P_D                    : {s['P_D']:.4f} "
              f"(95% CI [{s['P_D_ci95'][0]:.4f}, {s['P_D_ci95'][1]:.4f}], +/- {s['P_D_ci95_half_width']:.4f})")
        print(f"  P_FA active            : {s['P_FA']:.4f} "
              f"(95% CI [{s['P_FA_ci95'][0]:.4f}, {s['P_FA_ci95'][1]:.4f}], +/- {s['P_FA_ci95_half_width']:.4f})")
        print(f"  P_FA system-level      : {s['P_FA_overall']:.4f} "
              f"(95% CI [{s['P_FA_overall_ci95'][0]:.4f}, {s['P_FA_overall_ci95'][1]:.4f}], +/- {s['P_FA_overall_ci95_half_width']:.4f})")
        print(f"  Overhead B             : {s['B_mean_bits']:.2f} bit (std {s['B_std_bits']:.2f})")
        print(f"  Overhead T             : {s['T_mean_ms']:.4f} ms (std {s['T_std_ms']:.4f})")
        print(f"  Selected links         : {s['selected_links_mean']:.2f} (std {s['selected_links_std']:.2f})")
        print(f"  Active target ratio    : {s['active_target_ratio_mean']:.4f}")
        print(f"  Feasible target ratio  : {s['feasible_target_ratio_mean']:.4f}")
        print(f"  Feasible links/trial   : {s['feasible_links_mean']:.2f}")
        print(f"  Comm feasible edge ratio: {s['comm_feasible_edge_ratio_mean']:.4f}")
        print(f"  Selected rate mean/min : {s['selected_rate_mean_mbps']:.4f} / {s['selected_rate_min_mbps_mean']:.4f} Mbps")
        print(f"  Selected chi mean/min  : {s['selected_chi_mean']:.4f} / {s['selected_chi_min_mean']:.4f}")
        print(f"  Rate/chi sat. ratio    : {s['selected_rate_satisfaction_ratio_mean']:.4f} / {s['selected_chi_ge_min_ratio_mean']:.4f}")
        print(f"  D mean/median          : {s['D_mean']:.4f} / {s['D_median']:.4f}")
        print(f"  D p10/p90              : {s['D_p10']:.4f} / {s['D_p90']:.4f}")
        print(f"  All targets satisfy    : {s['all_targets_satisfied_prob']:.4f}")
        print(f"  Worst target D mean    : {s['worst_target_D_mean']:.4f}")
        print(f"  Worst target sat. prob : {s['worst_target_satisfied_prob']:.4f}")
        print(f"  Efficiency P_D/kbit    : {s['P_D_per_kbit']:.4f}")
        print(f"  Efficiency P_D/ms      : {s['P_D_per_ms']:.4f}")


def write_csv(summary: Dict[str, Dict[str, Any]], path: Path) -> None:
    scalar_keys = [
        "P_D", "P_FA", "P_FA_overall", "B_mean_bits", "T_mean_ms", "selected_links_mean",
        "active_target_ratio_mean", "feasible_target_ratio_mean", "feasible_links_mean",
        "comm_feasible_edge_ratio_mean",
        "selected_rate_mean_mbps", "selected_rate_min_mbps_mean", "selected_rate_p10_mbps_mean",
        "selected_chi_mean", "selected_chi_min_mean", "selected_chi_p10_mean",
        "selected_gamma_comm_mean_db", "selected_rate_satisfaction_ratio_mean",
        "selected_chi_ge_min_ratio_mean",
        "D_mean", "D_median", "D_p10", "D_p90",
        "all_targets_satisfied_prob", "worst_target_D_mean", "worst_target_satisfied_prob",
        "P_D_per_kbit", "P_D_per_ms", "D_per_kbit", "D_per_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + scalar_keys)
        writer.writeheader()
        for method, s in summary.items():
            row = {"method": method}
            for k in scalar_keys:
                row[k] = s.get(k, "")
            writer.writerow(row)


def run_simulation(cfg: SimConfig) -> Dict[str, Dict[str, Any]]:
    """Run the full Monte Carlo experiment once under the current configuration."""
    all_results: Dict[str, List[MethodResult]] = {m: [] for m in METHODS}
    print_interval = max(1, cfg.num_mc // 10)

    for t in range(cfg.num_mc):
        trial_results = run_one_trial(cfg, t)
        for method, res in trial_results.items():
            all_results[method].append(res)

        if cfg.verbose and ((t + 1) % print_interval == 0 or (t + 1) == cfg.num_mc):
            prop = all_results["proposed_lagrangian"]
            det = sum(r.detected for r in prop)
            tot = sum(r.total_targets for r in prop)
            fa = sum(r.false_alarm for r in prop)
            tf = sum(r.total_false for r in prop)
            print(
                f"MC {t + 1:4d}/{cfg.num_mc}: "
                f"proposed_lagrangian P_D={det / max(tot, 1):.4f}, "
                f"P_FA={fa / max(tf, 1):.4f}"
            )

    return {method: summarize(results, cfg) for method, results in all_results.items()}


def lambda_sweep(cfg: SimConfig, lambda_values: List[float]) -> List[Dict[str, Any]]:
    """Sweep lambda_c and record proposed_lagrangian metrics.

    The sweep focuses on proposed_lagrangian because lambda_c directly controls
    its marginal selection rule:
        score = alpha_q * DeltaD_ijq - lambda_c * cost_ijq.
    """
    rows: List[Dict[str, Any]] = []

    for lam in lambda_values:
        cfg_lam = copy.deepcopy(cfg)
        cfg_lam.lagrangian_lambda = float(lam)

        print("\n" + "=" * 72)
        print(f"Running lambda-cost sweep: lambda_c = {lam:g}")
        print("=" * 72)

        summary = run_simulation(cfg_lam)
        s_prop = summary["proposed_lagrangian"]

        row = {
            "lambda_cost": float(lam),
            "P_D": s_prop["P_D"],
            "P_D_ci95_low": s_prop["P_D_ci95"][0],
            "P_D_ci95_high": s_prop["P_D_ci95"][1],
            "P_D_ci95_half_width": s_prop["P_D_ci95_half_width"],
            "P_FA_active": s_prop["P_FA"],
            "P_FA_overall": s_prop["P_FA_overall"],
            "B_mean_bits": s_prop["B_mean_bits"],
            "T_mean_ms": s_prop["T_mean_ms"],
            "selected_links_mean": s_prop["selected_links_mean"],
            "active_target_ratio_mean": s_prop["active_target_ratio_mean"],
            "comm_feasible_edge_ratio_mean": s_prop["comm_feasible_edge_ratio_mean"],
            "selected_rate_mean_mbps": s_prop["selected_rate_mean_mbps"],
            "selected_rate_min_mbps_mean": s_prop["selected_rate_min_mbps_mean"],
            "selected_chi_mean": s_prop["selected_chi_mean"],
            "selected_chi_min_mean": s_prop["selected_chi_min_mean"],
            "selected_rate_satisfaction_ratio_mean": s_prop["selected_rate_satisfaction_ratio_mean"],
            "selected_chi_ge_min_ratio_mean": s_prop["selected_chi_ge_min_ratio_mean"],
            "D_mean": s_prop["D_mean"],
            "D_median": s_prop["D_median"],
            "D_p10": s_prop["D_p10"],
            "D_p90": s_prop["D_p90"],
            "all_targets_satisfied_prob": s_prop["all_targets_satisfied_prob"],
            "worst_target_D_mean": s_prop["worst_target_D_mean"],
            "worst_target_satisfied_prob": s_prop["worst_target_satisfied_prob"],
            "P_D_per_kbit": s_prop["P_D_per_kbit"],
            "P_D_per_ms": s_prop["P_D_per_ms"],
            "D_per_kbit": s_prop["D_per_kbit"],
            "D_per_ms": s_prop["D_per_ms"],
        }
        rows.append(row)

        print(
            f"lambda={lam:g} | "
            f"P_D={row['P_D']:.4f}, "
            f"T={row['T_mean_ms']:.4f} ms, "
            f"links={row['selected_links_mean']:.2f}, "
            f"P_D/ms={row['P_D_per_ms']:.4f}, "
            f"worst-sat={row['worst_target_satisfied_prob']:.4f}"
        )

    return rows


def write_lambda_sweep_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write lambda-sweep scalar results to a CSV file."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_lambda_sweep(rows: List[Dict[str, Any]], output_prefix: Path) -> None:
    """Generate figures for lambda-cost sweep.

    Saved figures:
        *_pd_vs_delay.png
        *_links_vs_lambda.png
        *_pd_per_ms_vs_lambda.png
        *_worst_sat_vs_lambda.png
    """
    if not rows:
        return

    import matplotlib.pyplot as plt

    lambda_vals = np.array([r["lambda_cost"] for r in rows], dtype=float)
    P_D = np.array([r["P_D"] for r in rows], dtype=float)
    T_ms = np.array([r["T_mean_ms"] for r in rows], dtype=float)
    selected_links = np.array([r["selected_links_mean"] for r in rows], dtype=float)
    P_D_per_ms = np.array([r["P_D_per_ms"] for r in rows], dtype=float)
    worst_sat = np.array([r["worst_target_satisfied_prob"] for r in rows], dtype=float)

    def savefig(suffix: str) -> None:
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(output_prefix.name + suffix), dpi=300)
        plt.close()

    plt.figure()
    plt.plot(T_ms, P_D, marker="o")
    for x, y, lam in zip(T_ms, P_D, lambda_vals):
        plt.annotate(f"{lam:g}", (x, y), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Mean soft-information exchange delay (ms)")
    plt.ylabel("Detection probability $P_D$")
    plt.title("$P_D$ vs exchange delay under different $\\lambda_c$")
    plt.grid(True, alpha=0.3)
    savefig("_pd_vs_delay.png")

    plt.figure()
    plt.plot(lambda_vals, selected_links, marker="o")
    plt.xlabel("Lagrangian communication price $\\lambda_c$")
    plt.ylabel("Mean selected links")
    plt.title("Selected links vs $\\lambda_c$")
    plt.grid(True, alpha=0.3)
    savefig("_links_vs_lambda.png")

    plt.figure()
    plt.plot(lambda_vals, P_D_per_ms, marker="o")
    plt.xlabel("Lagrangian communication price $\\lambda_c$")
    plt.ylabel("$P_D$ per ms")
    plt.title("Detection-delay efficiency vs $\\lambda_c$")
    plt.grid(True, alpha=0.3)
    savefig("_pd_per_ms_vs_lambda.png")

    plt.figure()
    plt.plot(lambda_vals, worst_sat, marker="o")
    plt.xlabel("Lagrangian communication price $\\lambda_c$")
    plt.ylabel("Worst-target satisfied probability")
    plt.title("Weak-target reliability vs $\\lambda_c$")
    plt.grid(True, alpha=0.3)
    savefig("_worst_sat_vs_lambda.png")


# ============================================================
# Ablation and robustness experiments
# ============================================================

def scalar_summary_row(summary: Dict[str, Dict[str, Any]], method: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact scalar row from a method summary."""
    s = summary[method]
    row: Dict[str, Any] = dict(extra)
    row.update({
        "method": method,
        "P_D": s["P_D"],
        "P_D_ci95_low": s["P_D_ci95"][0],
        "P_D_ci95_high": s["P_D_ci95"][1],
        "P_D_ci95_half_width": s["P_D_ci95_half_width"],
        "P_FA_active": s["P_FA"],
        "P_FA_overall": s["P_FA_overall"],
        "B_mean_bits": s["B_mean_bits"],
        "T_mean_ms": s["T_mean_ms"],
        "selected_links_mean": s["selected_links_mean"],
        "active_target_ratio_mean": s["active_target_ratio_mean"],
        "feasible_target_ratio_mean": s["feasible_target_ratio_mean"],
        "feasible_links_mean": s["feasible_links_mean"],
        "comm_feasible_edge_ratio_mean": s["comm_feasible_edge_ratio_mean"],
        "selected_rate_mean_mbps": s["selected_rate_mean_mbps"],
        "selected_rate_min_mbps_mean": s["selected_rate_min_mbps_mean"],
        "selected_rate_p10_mbps_mean": s["selected_rate_p10_mbps_mean"],
        "selected_chi_mean": s["selected_chi_mean"],
        "selected_chi_min_mean": s["selected_chi_min_mean"],
        "selected_chi_p10_mean": s["selected_chi_p10_mean"],
        "selected_gamma_comm_mean_db": s["selected_gamma_comm_mean_db"],
        "selected_rate_satisfaction_ratio_mean": s["selected_rate_satisfaction_ratio_mean"],
        "selected_chi_ge_min_ratio_mean": s["selected_chi_ge_min_ratio_mean"],
        "D_mean": s["D_mean"],
        "D_median": s["D_median"],
        "D_p10": s["D_p10"],
        "D_p90": s["D_p90"],
        "all_targets_satisfied_prob": s["all_targets_satisfied_prob"],
        "worst_target_D_mean": s["worst_target_D_mean"],
        "worst_target_satisfied_prob": s["worst_target_satisfied_prob"],
        "P_D_per_kbit": s["P_D_per_kbit"],
        "P_D_per_ms": s["P_D_per_ms"],
        "D_per_kbit": s["D_per_kbit"],
        "D_per_ms": s["D_per_ms"],
    })
    return row


def ablation_suite(cfg: SimConfig) -> List[Dict[str, Any]]:
    """Run ablation variants of the proposed Lagrangian selector.

    Variants:
        full: full proposed algorithm.
        w/o_alpha: remove target-priority coefficient alpha_q by setting alpha_q=1.
        w/o_delay_price: remove the -lambda_c*c_ijq communication-price term.
        w/o_comm_error_calib: keep the detection environment polluted but remove
            communication-error calibration from the algorithm-side deflection model.
        w/o_softmin_alpha: use dP_D/dD + deficit only, without soft-min target weighting.
    """
    variants: List[Tuple[str, Dict[str, Any]]] = [
        ("full", {}),
        ("w/o_alpha", {"use_target_priority": False}),
        ("w/o_delay_price", {"use_delay_price": False}),
        ("w/o_comm_error_calib", {"use_comm_error_calibration": False}),
        ("w/o_softmin_alpha", {"use_softmin_alpha": False}),
    ]
    rows: List[Dict[str, Any]] = []

    for name, updates in variants:
        cfg_var = copy.deepcopy(cfg)
        for key, value in updates.items():
            setattr(cfg_var, key, value)

        print("\n" + "=" * 72)
        print(f"Running ablation variant: {name}")
        print("=" * 72)

        summary = run_simulation(cfg_var)
        row = scalar_summary_row(
            summary,
            "proposed_lagrangian",
            {
                "experiment": "ablation",
                "variant": name,
                "use_target_priority": cfg_var.use_target_priority,
                "use_delay_price": cfg_var.use_delay_price,
                "use_comm_error_calibration": cfg_var.use_comm_error_calibration,
                "use_softmin_alpha": cfg_var.use_softmin_alpha,
                "lambda_cost": cfg_var.lagrangian_lambda,
                "mu_deficit": cfg_var.lagrangian_mu,
            },
        )
        rows.append(row)
        print(
            f"{name} | P_D={row['P_D']:.4f}, T={row['T_mean_ms']:.4f} ms, "
            f"links={row['selected_links_mean']:.2f}, P_D/ms={row['P_D_per_ms']:.4f}, "
            f"worst-sat={row['worst_target_satisfied_prob']:.4f}"
        )

    return rows


def write_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write a list of scalar metric rows to CSV."""
    if not rows:
        return
    # Preserve first-row order, then append any keys that appear later.
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_ablation(rows: List[Dict[str, Any]], output_prefix: Path) -> None:
    """Generate bar charts for ablation study."""
    if not rows:
        return
    import matplotlib.pyplot as plt

    labels = [str(r["variant"]) for r in rows]
    x = np.arange(len(labels))

    def barplot(metric: str, ylabel: str, suffix: str) -> None:
        values = np.array([float(r[metric]) for r in rows], dtype=float)
        plt.figure(figsize=(max(7, 1.4 * len(labels)), 4.8))
        plt.bar(x, values)
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"Ablation study: {ylabel}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(output_prefix.name + suffix), dpi=300)
        plt.close()

    barplot("P_D", "Detection probability $P_D$", "_pd.png")
    barplot("T_mean_ms", "Mean exchange delay (ms)", "_delay.png")
    barplot("P_D_per_ms", "$P_D$ per ms", "_pd_per_ms.png")
    barplot("worst_target_satisfied_prob", "Worst-target satisfied probability", "_worst_sat.png")


def robustness_sweep(
        cfg: SimConfig,
        robustness_type: str,
        values: List[Any],
        methods_to_record: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run robustness sweeps and record selected methods.

    Supported robustness_type:
        comm_model: values are erasure/biased/flip.
        error_sigma: values are soft_error_sigma_scale.
        residual_direct: values are residual_direct_factor.
    """
    if methods_to_record is None:
        methods_to_record = ["proposed_lagrangian", "sense_sinr", "single_best", "all_neighbor"]

    rows: List[Dict[str, Any]] = []
    for value in values:
        cfg_var = copy.deepcopy(cfg)
        if robustness_type == "comm_model":
            cfg_var.comm_error_model = str(value)  # type: ignore[assignment]
            condition_value: Any = str(value)
        elif robustness_type == "error_sigma":
            cfg_var.soft_error_sigma_scale = float(value)
            condition_value = float(value)
        elif robustness_type == "residual_direct":
            cfg_var.residual_direct_factor = float(value)
            condition_value = float(value)
        else:
            raise ValueError(f"Unknown robustness_type={robustness_type!r}.")

        print("\n" + "=" * 72)
        print(f"Running robustness sweep: {robustness_type} = {condition_value}")
        print("=" * 72)

        summary = run_simulation(cfg_var)
        for method in methods_to_record:
            if method not in summary:
                continue
            row = scalar_summary_row(
                summary,
                method,
                {
                    "experiment": "robustness",
                    "robustness_type": robustness_type,
                    "condition_value": condition_value,
                    "comm_error_model": cfg_var.comm_error_model,
                    "soft_error_sigma_scale": cfg_var.soft_error_sigma_scale,
                    "residual_direct_factor": cfg_var.residual_direct_factor,
                },
            )
            rows.append(row)

        sp = summary["proposed_lagrangian"]
        ss = summary["sense_sinr"]
        print(
            f"{robustness_type}={condition_value} | "
            f"proposed P_D={sp['P_D']:.4f}, T={sp['T_mean_ms']:.2f} ms, worst={sp['worst_target_satisfied_prob']:.4f}; "
            f"sense_sinr P_D={ss['P_D']:.4f}, T={ss['T_mean_ms']:.2f} ms, worst={ss['worst_target_satisfied_prob']:.4f}"
        )

    return rows


def plot_robustness(rows: List[Dict[str, Any]], output_prefix: Path) -> None:
    """Generate line plots for robustness sweep."""
    if not rows:
        return
    import matplotlib.pyplot as plt

    robustness_type = str(rows[0]["robustness_type"])
    methods = []
    for r in rows:
        method = str(r["method"])
        if method not in methods:
            methods.append(method)

    # Preserve input condition order.
    conditions = []
    for r in rows:
        val = r["condition_value"]
        if val not in conditions:
            conditions.append(val)

    def condition_to_float(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return float(conditions.index(v))

    x = np.array([condition_to_float(v) for v in conditions], dtype=float)
    xlabels = [str(v) for v in conditions]

    def lineplot(metric: str, ylabel: str, suffix: str) -> None:
        plt.figure(figsize=(7.2, 4.8))
        for method in methods:
            vals = []
            for cond in conditions:
                match = [r for r in rows if str(r["method"]) == method and r["condition_value"] == cond]
                vals.append(float(match[0][metric]) if match else np.nan)
            plt.plot(x, vals, marker="o", label=method)
        if robustness_type == "residual_direct":
            plt.xscale("log")
        elif robustness_type == "comm_model":
            plt.xticks(x, xlabels)
        plt.xlabel(robustness_type)
        plt.ylabel(ylabel)
        plt.title(f"Robustness: {ylabel} vs {robustness_type}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(output_prefix.name + suffix), dpi=300)
        plt.close()

    lineplot("P_D", "Detection probability $P_D$", "_pd.png")
    lineplot("T_mean_ms", "Mean exchange delay (ms)", "_delay.png")
    lineplot("P_D_per_ms", "$P_D$ per ms", "_pd_per_ms.png")
    lineplot("worst_target_satisfied_prob", "Worst-target satisfied probability", "_worst_sat.png")


def default_config() -> SimConfig:
    """Return the canonical default configuration.

    SimConfig already stores the default values.  Keeping this function thin
    avoids duplicated defaults drifting out of sync.
    """
    return SimConfig()


def communication_constraint_sweep(
        cfg: SimConfig,
        R_min_values: List[float],
        methods_to_record: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Sweep the minimum communication-rate requirement R_min.

    This experiment exposes the communication side of the sensing-centric ISAC
    model. As R_min increases, fewer UAV-to-UAV links satisfy the communication
    constraint, which can affect selected-link reliability, exchange delay, and
    final sensing detection probability.
    """
    if methods_to_record is None:
        methods_to_record = ["proposed_lagrangian", "sense_sinr", "single_best", "all_neighbor"]

    rows: List[Dict[str, Any]] = []
    for R_min in R_min_values:
        cfg_var = copy.deepcopy(cfg)
        cfg_var.R_min = float(R_min)
        print("\n" + "=" * 72)
        print(f"Running communication constraint sweep: R_min = {R_min:g} bit/s")
        print("=" * 72)

        summary = run_simulation(cfg_var)
        for method in methods_to_record:
            if method not in summary:
                continue
            row = scalar_summary_row(
                summary,
                method,
                {
                    "experiment": "comm_sweep",
                    "R_min_bps": float(R_min),
                    "R_min_mbps": float(R_min) / 1e6,
                },
            )
            rows.append(row)

        sp = summary["proposed_lagrangian"]
        print(
            f"R_min={R_min:g} | proposed P_D={sp['P_D']:.4f}, "
            f"T={sp['T_mean_ms']:.2f} ms, links={sp['selected_links_mean']:.2f}, "
            f"rate={sp['selected_rate_mean_mbps']:.3f} Mbps, chi={sp['selected_chi_mean']:.3f}, "
            f"feasible-edge={sp['comm_feasible_edge_ratio_mean']:.3f}"
        )

    return rows


def plot_comm_sweep(rows: List[Dict[str, Any]], output_prefix: Path) -> None:
    """Generate communication-constraint sweep figures.

    In addition to the full delay plot, this function also saves a zoomed delay
    plot without the all-neighbor upper-resource baseline.  The all-neighbor
    delay can be orders of magnitude larger and otherwise visually hides the
    resource-constrained methods.
    """
    if not rows:
        return
    import matplotlib.pyplot as plt

    methods = []
    for r in rows:
        m = str(r["method"])
        if m not in methods:
            methods.append(m)

    R_vals = []
    for r in rows:
        val = float(r["R_min_mbps"])
        if val not in R_vals:
            R_vals.append(val)
    x = np.array(R_vals, dtype=float)

    def get_vals(method: str, metric: str) -> List[float]:
        vals: List[float] = []
        for R in R_vals:
            match = [r for r in rows if str(r["method"]) == method and np.isclose(float(r["R_min_mbps"]), R, rtol=0.0, atol=1e-9)]
            vals.append(float(match[0][metric]) if match else np.nan)
        return vals

    def lineplot(metric: str, ylabel: str, suffix: str, methods_subset: Optional[List[str]] = None, logy: bool = False) -> None:
        plt.figure(figsize=(7.2, 4.8))
        active_methods = methods_subset if methods_subset is not None else methods
        for method in active_methods:
            if method not in methods:
                continue
            plt.plot(x, get_vals(method, metric), marker="o", label=method)
        plt.xlabel("Minimum communication rate $R_{min}$ (Mbit/s)")
        plt.ylabel(ylabel)
        plt.title(f"Communication constraint sweep: {ylabel}")
        if logy:
            plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(output_prefix.name + suffix), dpi=300)
        plt.close()

    lineplot("P_D", "Detection probability $P_D$", "_pd.png")
    lineplot("T_mean_ms", "Mean exchange delay (ms)", "_delay.png")
    lineplot("T_mean_ms", "Mean exchange delay (ms)", "_delay_log.png", logy=True)
    zoom_methods = [m for m in methods if m != "all_neighbor"]
    lineplot("T_mean_ms", "Mean exchange delay (ms)", "_delay_zoom.png", methods_subset=zoom_methods)
    lineplot("selected_rate_mean_mbps", "Selected-link mean rate (Mbit/s)", "_selected_rate.png")
    lineplot("selected_chi_mean", "Selected-link mean reliability $\\chi$", "_selected_chi.png")

    # Feasible communication-edge ratio is a network-level condition and is the
    # same for all methods under the same R_min.  Plot it once instead of drawing
    # duplicate method curves.
    feasible_vals = []
    for R in R_vals:
        match = [r for r in rows if np.isclose(float(r["R_min_mbps"]), R, rtol=0.0, atol=1e-9)]
        feasible_vals.append(float(match[0]["comm_feasible_edge_ratio_mean"]) if match else np.nan)
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(x, feasible_vals, marker="o")
    plt.xlabel("Minimum communication rate $R_{min}$ (Mbit/s)")
    plt.ylabel("Feasible communication-edge ratio")
    plt.title("Communication constraint sweep: feasible edge ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(output_prefix.name + "_feasible_edge_ratio.png"), dpi=300)
    plt.close()


# ============================================================
# Additional fair/OTFS-DD experiments
# ============================================================


def fair_ablation_suite(cfg: SimConfig, fair_max_total_links: int) -> List[Dict[str, Any]]:
    """Run ablation variants under a fixed total-link budget.

    This experiment addresses the concern that some variants, especially
    w/o_delay_price and w/o_alpha, can obtain higher raw P_D simply by selecting
    more cooperative links.  Here all variants share the same max_total_links cap.
    """
    variants: List[Tuple[str, Dict[str, Any]]] = [
        ("full", {}),
        ("w/o_alpha", {"use_target_priority": False}),
        ("w/o_delay_price", {"use_delay_price": False}),
        ("w/o_comm_error_calib", {"use_comm_error_calibration": False}),
        ("w/o_softmin_alpha", {"use_softmin_alpha": False}),
    ]
    rows: List[Dict[str, Any]] = []

    for name, updates in variants:
        cfg_var = copy.deepcopy(cfg)
        cfg_var.max_total_links = int(fair_max_total_links)
        for key, value in updates.items():
            setattr(cfg_var, key, value)

        print("\n" + "=" * 72)
        print(f"Running fair ablation variant: {name} with max_total_links={cfg_var.max_total_links}")
        print("=" * 72)

        summary = run_simulation(cfg_var)
        row = scalar_summary_row(
            summary,
            "proposed_lagrangian",
            {
                "experiment": "fair_ablation",
                "variant": name,
                "fair_max_total_links": cfg_var.max_total_links,
                "use_target_priority": cfg_var.use_target_priority,
                "use_delay_price": cfg_var.use_delay_price,
                "use_comm_error_calibration": cfg_var.use_comm_error_calibration,
                "use_softmin_alpha": cfg_var.use_softmin_alpha,
                "lambda_cost": cfg_var.lagrangian_lambda,
                "mu_deficit": cfg_var.lagrangian_mu,
            },
        )
        rows.append(row)
        print(
            f"{name} | P_D={row['P_D']:.4f}, T={row['T_mean_ms']:.4f} ms, "
            f"links={row['selected_links_mean']:.2f}, P_D/ms={row['P_D_per_ms']:.4f}, "
            f"worst-sat={row['worst_target_satisfied_prob']:.4f}"
        )

    return rows


def dd_ablation_suite(cfg: SimConfig) -> List[Dict[str, Any]]:
    """Evaluate the contribution of OTFS delay-Doppler modeling terms.

    Variants:
        full_dd: default DD-aware model.
        w/o_dd_validity: disable hard DD-bin validity filtering.
        w/o_dd_fractional_loss: ignore fractional DD bin mismatch loss.
        w/o_dd_collision_penalty: ignore DD-bin collision penalty.
        sensing_only_no_dd: disable all three DD mechanisms.

    This experiment is intended to support an OTFS-aware, rather than purely
    geometry-only, interpretation of the simplified simulator.
    """
    variants: List[Tuple[str, Dict[str, Any]]] = [
        ("full_dd", {}),
        ("w/o_dd_validity", {"use_otfs_bin_validity": False}),
        ("w/o_dd_fractional_loss", {"enable_dd_fractional_penalty": False}),
        ("w/o_dd_collision_penalty", {"enable_dd_collision_penalty": False}),
        (
            "w/o_all_dd_effects",
            {
                "use_otfs_bin_validity": False,
                "enable_dd_fractional_penalty": False,
                "enable_dd_collision_penalty": False,
            },
        ),
    ]
    rows: List[Dict[str, Any]] = []

    for name, updates in variants:
        cfg_var = copy.deepcopy(cfg)
        for key, value in updates.items():
            setattr(cfg_var, key, value)

        print("\n" + "=" * 72)
        print(f"Running OTFS-DD ablation variant: {name}")
        print("=" * 72)

        summary = run_simulation(cfg_var)
        for method in ["proposed_lagrangian", "sense_sinr", "single_best", "all_neighbor"]:
            row = scalar_summary_row(
                summary,
                method,
                {
                    "experiment": "dd_ablation",
                    "variant": name,
                    "use_otfs_bin_validity": cfg_var.use_otfs_bin_validity,
                    "enable_dd_fractional_penalty": cfg_var.enable_dd_fractional_penalty,
                    "enable_dd_collision_penalty": cfg_var.enable_dd_collision_penalty,
                },
            )
            rows.append(row)
        sp = summary["proposed_lagrangian"]
        print(
            f"{name} | proposed P_D={sp['P_D']:.4f}, T={sp['T_mean_ms']:.2f} ms, "
            f"links={sp['selected_links_mean']:.2f}, feasible-links={sp['feasible_links_mean']:.2f}"
        )

    return rows


def plot_dd_ablation(rows: List[Dict[str, Any]], output_prefix: Path) -> None:
    """Generate figures for OTFS-DD ablation."""
    if not rows:
        return
    import matplotlib.pyplot as plt

    variants: List[str] = []
    methods: List[str] = []
    for r in rows:
        v = str(r["variant"])
        m = str(r["method"])
        if v not in variants:
            variants.append(v)
        if m not in methods:
            methods.append(m)

    x = np.arange(len(variants))

    def grouped_bar(metric: str, ylabel: str, suffix: str) -> None:
        width = 0.8 / max(len(methods), 1)
        plt.figure(figsize=(max(8, 1.5 * len(variants)), 4.8))
        for idx, method in enumerate(methods):
            vals = []
            for variant in variants:
                match = [r for r in rows if str(r["method"]) == method and str(r["variant"]) == variant]
                vals.append(float(match[0][metric]) if match else np.nan)
            offset = (idx - (len(methods) - 1) / 2.0) * width
            plt.bar(x + offset, vals, width=width, label=method)
        plt.xticks(x, variants, rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"OTFS-DD ablation: {ylabel}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_prefix.with_name(output_prefix.name + suffix), dpi=300)
        plt.close()

    grouped_bar("P_D", "Detection probability $P_D$", "_pd.png")
    grouped_bar("T_mean_ms", "Mean exchange delay (ms)", "_delay.png")
    grouped_bar("selected_links_mean", "Mean selected links", "_links.png")
    grouped_bar("feasible_links_mean", "Feasible links per trial", "_feasible_links.png")
    grouped_bar("P_D_per_ms", "$P_D$ per ms", "_pd_per_ms.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simplified Lagrangian-guided DOTFS-ISAC simulation")
    p.add_argument("--num-mc", type=int, default=None, help="number of Monte Carlo trials")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--csv", type=str, default=None, help="optional CSV output path for a single run")

    p.add_argument("--lambda-cost", type=float, default=None, help="Lagrangian price for 1 ms delay")
    p.add_argument("--mu-deficit", type=float, default=None, help="target deficit multiplier")
    p.add_argument("--softmin-tau", type=float, default=None)
    p.add_argument("--no-softmin-alpha", action="store_true")

    p.add_argument("--max-total-links", type=int, default=None)
    p.add_argument("--max-links-per-target", type=int, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--isac-power-model", type=str, choices=["sensing_only", "joint_waveform", "reliable_comm_assisted"], default=None, help="sensing power model used in the simplified ISAC abstraction")
    p.add_argument("--comm-direct-leakage-factor", type=float, default=None, help="optional i->j sensing leakage factor in communication SINR")
    p.add_argument("--h0-error-bias-scale", type=float, default=None, help="H0-side bias scale for biased communication-error model")
    p.add_argument("--D-min", type=float, default=None)

    # Lambda-cost sweep experiment.
    p.add_argument(
        "--lambda-sweep",
        action="store_true",
        help="run a lambda-cost sweep experiment",
    )
    p.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=[0.001, 0.003, 0.005, 0.010, 0.020],
        help="lambda-cost values for sweep",
    )
    p.add_argument(
        "--sweep-csv",
        type=str,
        default="lambda_sweep_results.csv",
        help="CSV output path for lambda sweep",
    )
    p.add_argument(
        "--sweep-plot-prefix",
        type=str,
        default=None,
        help="optional output prefix for lambda sweep figures",
    )

    # Ablation experiment.
    p.add_argument(
        "--ablation",
        action="store_true",
        help="run ablation study for proposed_lagrangian",
    )
    p.add_argument(
        "--ablation-csv",
        type=str,
        default="ablation_results.csv",
        help="CSV output path for ablation study",
    )
    p.add_argument(
        "--ablation-plot-prefix",
        type=str,
        default=None,
        help="optional output prefix for ablation figures",
    )

    # Robustness experiment.
    p.add_argument(
        "--robustness",
        action="store_true",
        help="run robustness sweep",
    )
    p.add_argument(
        "--robustness-type",
        type=str,
        choices=["comm_model", "error_sigma", "residual_direct"],
        default="comm_model",
        help="robustness sweep type",
    )
    p.add_argument(
        "--comm-error-models",
        type=str,
        nargs="+",
        default=["erasure", "biased", "flip"],
        help="communication error models for robustness sweep",
    )
    p.add_argument(
        "--sigma-scale-values",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 4.0],
        help="soft_error_sigma_scale values for robustness sweep",
    )
    p.add_argument(
        "--residual-direct-values",
        type=float,
        nargs="+",
        default=[1e-5, 1e-4, 1e-3, 1e-2],
        help="residual_direct_factor values for robustness sweep",
    )
    p.add_argument(
        "--robustness-csv",
        type=str,
        default="robustness_results.csv",
        help="CSV output path for robustness sweep",
    )
    p.add_argument(
        "--robustness-plot-prefix",
        type=str,
        default=None,
        help="optional output prefix for robustness figures",
    )

    # Communication-side ISAC sweep.
    p.add_argument(
        "--comm-sweep",
        action="store_true",
        help="sweep the minimum communication-rate constraint R_min",
    )
    p.add_argument(
        "--R-min-values",
        type=float,
        nargs="+",
        default=[1e5, 2e5, 5e5, 1e6, 2e6],
        help="R_min values in bit/s for communication constraint sweep",
    )
    p.add_argument(
        "--comm-sweep-csv",
        type=str,
        default="comm_sweep_results.csv",
        help="CSV output path for communication constraint sweep",
    )
    p.add_argument(
        "--comm-sweep-plot-prefix",
        type=str,
        default=None,
        help="optional output prefix for communication constraint sweep figures",
    )

    # Fair ablation under a fixed selected-link budget.
    p.add_argument(
        "--fair-ablation",
        action="store_true",
        help="run ablation study with a fixed max_total_links budget",
    )
    p.add_argument(
        "--fair-max-total-links",
        type=int,
        default=26,
        help="fixed max_total_links used for fair ablation",
    )
    p.add_argument(
        "--fair-ablation-csv",
        type=str,
        default="fair_ablation_results.csv",
        help="CSV output path for fair ablation study",
    )
    p.add_argument(
        "--fair-ablation-plot-prefix",
        type=str,
        default=None,
        help="optional output prefix for fair ablation figures",
    )

    # OTFS delay-Doppler ablation.
    p.add_argument(
        "--dd-ablation",
        action="store_true",
        help="run OTFS delay-Doppler ablation study",
    )
    p.add_argument(
        "--dd-ablation-csv",
        type=str,
        default="dd_ablation_results.csv",
        help="CSV output path for OTFS-DD ablation study",
    )
    p.add_argument(
        "--dd-ablation-plot-prefix",
        type=str,
        default=None,
        help="optional output prefix for OTFS-DD ablation figures",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()

    if args.num_mc is not None:
        cfg.num_mc = args.num_mc
    if args.seed is not None:
        cfg.seed = args.seed
    if args.quiet:
        cfg.verbose = False
    if args.lambda_cost is not None:
        cfg.lagrangian_lambda = args.lambda_cost
    if args.mu_deficit is not None:
        cfg.lagrangian_mu = args.mu_deficit
    if args.softmin_tau is not None:
        cfg.softmin_tau = args.softmin_tau
    if args.no_softmin_alpha:
        cfg.use_softmin_alpha = False
    if args.max_total_links is not None:
        cfg.max_total_links = args.max_total_links
    if args.max_links_per_target is not None:
        cfg.max_links_per_target = args.max_links_per_target
    if args.rho is not None:
        cfg.rho = args.rho
    if args.isac_power_model is not None:
        cfg.isac_power_model = args.isac_power_model  # type: ignore[assignment]
    if args.comm_direct_leakage_factor is not None:
        cfg.comm_direct_leakage_factor = args.comm_direct_leakage_factor
    if args.h0_error_bias_scale is not None:
        cfg.h0_error_bias_scale = args.h0_error_bias_scale
    if args.D_min is not None:
        cfg.D_min = args.D_min

    if args.comm_sweep:
        rows = communication_constraint_sweep(cfg, list(args.R_min_values))
        out_csv = Path(args.comm_sweep_csv)
        write_rows_csv(rows, out_csv)
        print(f"\nCommunication sweep CSV saved to: {out_csv}")

        if args.comm_sweep_plot_prefix is not None:
            plot_prefix = Path(args.comm_sweep_plot_prefix)
            plot_comm_sweep(rows, plot_prefix)
            print(f"Communication sweep figures saved with prefix: {plot_prefix}")
        return

    if args.fair_ablation:
        rows = fair_ablation_suite(cfg, args.fair_max_total_links)
        out_csv = Path(args.fair_ablation_csv)
        write_rows_csv(rows, out_csv)
        print(f"\nFair ablation CSV saved to: {out_csv}")

        if args.fair_ablation_plot_prefix is not None:
            plot_prefix = Path(args.fair_ablation_plot_prefix)
            plot_ablation(rows, plot_prefix)
            print(f"Fair ablation figures saved with prefix: {plot_prefix}")
        return

    if args.dd_ablation:
        rows = dd_ablation_suite(cfg)
        out_csv = Path(args.dd_ablation_csv)
        write_rows_csv(rows, out_csv)
        print(f"\nOTFS-DD ablation CSV saved to: {out_csv}")

        if args.dd_ablation_plot_prefix is not None:
            plot_prefix = Path(args.dd_ablation_plot_prefix)
            plot_dd_ablation(rows, plot_prefix)
            print(f"OTFS-DD ablation figures saved with prefix: {plot_prefix}")
        return

    if args.ablation:
        rows = ablation_suite(cfg)
        out_csv = Path(args.ablation_csv)
        write_rows_csv(rows, out_csv)
        print(f"\nAblation CSV saved to: {out_csv}")

        if args.ablation_plot_prefix is not None:
            plot_prefix = Path(args.ablation_plot_prefix)
            plot_ablation(rows, plot_prefix)
            print(f"Ablation figures saved with prefix: {plot_prefix}")
        return

    if args.robustness:
        if args.robustness_type == "comm_model":
            robustness_values = list(args.comm_error_models)
        elif args.robustness_type == "error_sigma":
            robustness_values = list(args.sigma_scale_values)
        elif args.robustness_type == "residual_direct":
            robustness_values = list(args.residual_direct_values)
        else:
            raise ValueError(f"Unknown robustness_type={args.robustness_type!r}.")

        rows = robustness_sweep(cfg, args.robustness_type, robustness_values)
        out_csv = Path(args.robustness_csv)
        write_rows_csv(rows, out_csv)
        print(f"\nRobustness CSV saved to: {out_csv}")

        if args.robustness_plot_prefix is not None:
            plot_prefix = Path(args.robustness_plot_prefix)
            plot_robustness(rows, plot_prefix)
            print(f"Robustness figures saved with prefix: {plot_prefix}")
        return

    if args.lambda_sweep:
        rows = lambda_sweep(cfg, list(args.lambda_values))
        out_csv = Path(args.sweep_csv)
        write_lambda_sweep_csv(rows, out_csv)
        print(f"\nLambda sweep CSV saved to: {out_csv}")

        if args.sweep_plot_prefix is not None:
            plot_prefix = Path(args.sweep_plot_prefix)
            plot_lambda_sweep(rows, plot_prefix)
            print(f"Lambda sweep figures saved with prefix: {plot_prefix}")
        return

    summary = run_simulation(cfg)
    print_summary(summary)

    if args.csv is not None:
        out = Path(args.csv)
        write_csv(summary, out)
        print(f"\nCSV saved to: {out}")


if __name__ == "__main__":
    main()
