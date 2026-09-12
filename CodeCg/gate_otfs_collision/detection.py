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
from .kernels import *
from .kernels import _component_from_power_and_template
from .assignment import *

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
