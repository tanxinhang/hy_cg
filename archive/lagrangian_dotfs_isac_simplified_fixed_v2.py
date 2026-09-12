#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lagrangian-guided simplified DOTFS-ISAC cooperative sensing simulation.

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
"""

from __future__ import annotations

import argparse
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

    # Communication constraints
    R_min: float = 2.0e5
    chi_min: float = 0.30
    enforce_chi_min: bool = False
    comm_leakage_from_sensing: float = 0.05

    # Sensing/detection
    Pfa_target: float = 0.05
    D_min: float = 3.0
    soft_mu_scale: float = 8.0
    soft_sigma0: float = 1.0
    soft_sigma_floor: float = 0.25
    soft_error_sigma_scale: float = 3.0
    enable_comm_error_pollution: bool = True
    comm_error_model: Literal["erasure", "flip", "biased"] = "erasure"
    soft_error_flip_scale: float = 1.0
    soft_error_bias_scale: float = 0.5
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
    lagrangian_mu: float = 1.0       # target-deficit multiplier
    use_softmin_alpha: bool = True
    softmin_tau: float = 0.10
    alpha_floor: float = 0.0
    alpha_cap: float = 10.0

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
    false_alarm: int
    total_false: int
    overhead_bits: float
    overhead_delay_s: float
    selected_links: Dict[int, List[Link]]
    D_fuse_per_target: np.ndarray
    active_targets: int
    feasible_targets: int
    feasible_links: int


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

            gamma = signal / (n0 + interf + EPS)
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
            for k in range(M):
                if k == j or k == i:
                    continue
                residual_direct += cfg.residual_direct_factor * P[k] * base.direct_gain[k, j]

            residual_multi = 0.0
            for k in range(M):
                if k == i or k == j:
                    continue
                residual_multi += cfg.residual_multi_uav_factor * P_sense[k] * base.direct_gain[k, j]

            residual_total = residual_self + residual_direct + residual_multi
            rinr[i, j] = residual_total / (n0 + EPS)
            sigma0[i, j] = cfg.soft_sigma0 * math.sqrt(1.0 + cfg.rinr_sigma_factor * rinr[i, j])

            effective_sensing_power = P_sense[i] + chi_comm[i, j] * P_comm[i]

            for q in range(Q):
                if cfg.use_otfs_bin_validity and (not base.valid_dd[i, j, q]):
                    continue

                raw_signal = P[i] * base.target_gain[i, j, q] * G_proc
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

    if not cfg.enable_comm_error_pollution:
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


def compute_weights(cfg: SimConfig, tables: LinkTables, q: int, links: List[Link], mode: str = "beta") -> Dict[Link, float]:
    if not links:
        return {}

    if mode == "equal":
        return {link: 1.0 / len(links) for link in links}

    vals: List[float] = []
    for link in links:
        i, j = link
        if mode == "deflection":
            mu_eff = effective_h1_mean_for_link(cfg, tables, link, q)
            vals.append(max(mu_eff, 0.0) / (tables.sigma0[i, j] ** 2 + EPS))
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
        var0 += (w ** 2) * h0_variance_for_link(cfg, tables, link)

    return float((max(mean_gap, 0.0) ** 2) / (var0 + EPS))


def target_alpha(cfg: SimConfig, D_fuse: np.ndarray) -> np.ndarray:
    """
    Target marginal value alpha_q.

    alpha_q = dR/dD_q + mu * normalized_deficit

    If use_softmin_alpha=True, dR/dD_q is weighted by soft-min over predicted Pd,
    so weak targets receive larger priority.
    """
    D = np.asarray(D_fuse, dtype=float)
    pd = pd_from_deflection(cfg, D)
    dpd = d_pd_d_D(cfg, D)

    if cfg.use_softmin_alpha:
        tau = max(cfg.softmin_tau, EPS)
        logits = -(pd - np.min(pd)) / tau
        logits = logits - np.max(logits)  # stable softmax; max logit becomes 0
        w = np.exp(logits)
        w = w / max(float(np.sum(w)), EPS)
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
                score = alpha[q] * marginal_D - cfg.lagrangian_lambda * cost_ms

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

        # Stop early if all targets meet the target deflection.
        if np.all(D_fuse >= cfg.D_min):
            break

    return selected, D_fuse


def select_all_neighbor(cfg: SimConfig, base: BaseGains, tables: LinkTables) -> Tuple[Dict[int, List[Link]], np.ndarray]:
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
            scores = np.array([-base.d_uu[i, j] for i, j in links])
            order = np.argsort(scores)[::-1]
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
    return float(rng.normal(0.0, sigma_err))


def evaluate_detection(
    cfg: SimConfig,
    tables: LinkTables,
    selected: Dict[int, List[Link]],
    rng: np.random.Generator,
    method: str,
) -> Tuple[int, int, int, int]:
    detected = 0
    false_alarm = 0
    total_targets = cfg.Q
    total_false = cfg.Q * cfg.num_false_per_target
    base_thr = threshold_from_pfa(cfg)

    for q in range(cfg.Q):
        links = selected.get(q, [])
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
                false_alarm += 1

    return detected, total_targets, false_alarm, total_false


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


def rng_for_method(cfg: SimConfig, trial_index: int, method: str) -> np.random.Generator:
    offset = METHOD_RNG_OFFSETS[method]
    seed = cfg.seed + 12345 + 1000003 * trial_index + offset
    return np.random.default_rng(seed)


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

    detected, total_targets, fa, total_false = evaluate_detection(cfg, tables, selected, rng, method)
    feasible_targets, feasible_links = feasible_stats(cfg, base, tables)

    return MethodResult(
        name=method,
        detected=detected,
        total_targets=total_targets,
        false_alarm=fa,
        total_false=total_false,
        overhead_bits=total_overhead_bits(cfg, selected),
        overhead_delay_s=total_overhead_delay_s(cfg, tables, selected),
        selected_links=selected,
        D_fuse_per_target=D,
        active_targets=active_target_count(selected),
        feasible_targets=feasible_targets,
        feasible_links=feasible_links,
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
    total_fa = sum(r.false_alarm for r in results)
    total_false = sum(r.total_false for r in results)
    pd_ci = binomial_ci95(total_detected, total_targets)
    pfa_ci = binomial_ci95(total_fa, total_false)

    D_arr = np.vstack([r.D_fuse_per_target for r in results])
    overhead_bits = np.array([r.overhead_bits for r in results], dtype=float)
    overhead_delay_s = np.array([r.overhead_delay_s for r in results], dtype=float)
    active = np.array([r.active_targets for r in results], dtype=float)
    feasible_targets = np.array([r.feasible_targets for r in results], dtype=float)
    feasible_links = np.array([r.feasible_links for r in results], dtype=float)
    selected_links = np.array([sum(len(v) for v in r.selected_links.values()) for r in results], dtype=float)

    D_satisfied = D_arr >= cfg.D_min
    pd = total_detected / max(total_targets, 1)
    pfa = total_fa / max(total_false, 1)
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
        "B_mean_bits": float(np.mean(overhead_bits)),
        "B_std_bits": float(np.std(overhead_bits)),
        "T_mean_ms": T_ms,
        "T_std_ms": float(np.std(overhead_delay_s) * 1e3),
        "selected_links_mean": float(np.mean(selected_links)),
        "selected_links_std": float(np.std(selected_links)),
        "active_target_ratio_mean": float(np.mean(active / max(cfg.Q, 1))),
        "feasible_target_ratio_mean": float(np.mean(feasible_targets / max(cfg.Q, 1))),
        "feasible_links_mean": float(np.mean(feasible_links)),
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
        print(f"  P_FA                   : {s['P_FA']:.4f} "
              f"(95% CI [{s['P_FA_ci95'][0]:.4f}, {s['P_FA_ci95'][1]:.4f}], +/- {s['P_FA_ci95_half_width']:.4f})")
        print(f"  Overhead B             : {s['B_mean_bits']:.2f} bit (std {s['B_std_bits']:.2f})")
        print(f"  Overhead T             : {s['T_mean_ms']:.4f} ms (std {s['T_std_ms']:.4f})")
        print(f"  Selected links         : {s['selected_links_mean']:.2f} (std {s['selected_links_std']:.2f})")
        print(f"  Active target ratio    : {s['active_target_ratio_mean']:.4f}")
        print(f"  Feasible target ratio  : {s['feasible_target_ratio_mean']:.4f}")
        print(f"  Feasible links/trial   : {s['feasible_links_mean']:.2f}")
        print(f"  D mean/median          : {s['D_mean']:.4f} / {s['D_median']:.4f}")
        print(f"  D p10/p90              : {s['D_p10']:.4f} / {s['D_p90']:.4f}")
        print(f"  All targets satisfy    : {s['all_targets_satisfied_prob']:.4f}")
        print(f"  Worst target D mean    : {s['worst_target_D_mean']:.4f}")
        print(f"  Worst target sat. prob : {s['worst_target_satisfied_prob']:.4f}")
        print(f"  Efficiency P_D/kbit    : {s['P_D_per_kbit']:.4f}")
        print(f"  Efficiency P_D/ms      : {s['P_D_per_ms']:.4f}")


def write_csv(summary: Dict[str, Dict[str, Any]], path: Path) -> None:
    scalar_keys = [
        "P_D", "P_FA", "B_mean_bits", "T_mean_ms", "selected_links_mean",
        "active_target_ratio_mean", "feasible_target_ratio_mean", "feasible_links_mean",
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


def default_config() -> SimConfig:
    return SimConfig(
        M=15,
        Q=10,
        num_mc=200,
        rho=0.80,
        D_min=3.0,
        max_links_per_target=6,
        max_total_links=60,
        lagrangian_lambda=0.005,
        lagrangian_mu=1.0,
        use_softmin_alpha=True,
        softmin_tau=0.10,
        beta_chi_power=2.0,
        beta_delay_exponent=0.5,
        enforce_chi_min=False,
        enable_comm_error_pollution=True,
        comm_error_model="erasure",
        seed=2026,
        verbose=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simplified Lagrangian-guided DOTFS-ISAC simulation")
    p.add_argument("--num-mc", type=int, default=None, help="number of Monte Carlo trials")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--csv", type=str, default=None, help="optional CSV output path")

    p.add_argument("--lambda-cost", type=float, default=None, help="Lagrangian price for 1 ms delay")
    p.add_argument("--mu-deficit", type=float, default=None, help="target deficit multiplier")
    p.add_argument("--softmin-tau", type=float, default=None)
    p.add_argument("--no-softmin-alpha", action="store_true")

    p.add_argument("--max-total-links", type=int, default=None)
    p.add_argument("--max-links-per-target", type=int, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--D-min", type=float, default=None)
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
    if args.D_min is not None:
        cfg.D_min = args.D_min

    all_results: Dict[str, List[MethodResult]] = {m: [] for m in METHODS}

    for t in range(cfg.num_mc):
        trial_results = run_one_trial(cfg, t)
        for method, res in trial_results.items():
            all_results[method].append(res)

        print_interval = max(1, cfg.num_mc // 10)
        if cfg.verbose and ((t + 1) % print_interval == 0 or (t + 1) == cfg.num_mc):
            prop = all_results["proposed_lagrangian"]
            det = sum(r.detected for r in prop)
            tot = sum(r.total_targets for r in prop)
            fa = sum(r.false_alarm for r in prop)
            tf = sum(r.total_false for r in prop)
            print(f"MC {t+1:4d}/{cfg.num_mc}: "
                  f"proposed_lagrangian P_D={det/max(tot,1):.4f}, "
                  f"P_FA={fa/max(tf,1):.4f}")

    summary = {method: summarize(results, cfg) for method, results in all_results.items()}
    print_summary(summary)

    if args.csv is not None:
        out = Path(args.csv)
        write_csv(summary, out)
        print(f"\nCSV saved to: {out}")


if __name__ == "__main__":
    main()
