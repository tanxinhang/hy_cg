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


def _skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    mu = float(np.mean(x))
    sig = float(np.std(x))
    if sig <= EPS:
        return 0.0
    return float(np.mean(((x - mu) / sig) ** 3))


def _excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 0.0
    mu = float(np.mean(x))
    sig = float(np.std(x))
    if sig <= EPS:
        return 0.0
    return float(np.mean(((x - mu) / sig) ** 4) - 3.0)


def _hill_tail_alpha(x: np.ndarray, frac: float) -> float:
    """Hill tail-index estimate for positive collision-strength samples."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 5:
        return float("nan")
    xs = np.sort(x)
    k = max(2, int(math.ceil(float(frac) * xs.size)))
    k = min(k, xs.size - 1)
    tail = xs[-k:]
    threshold = xs[-k - 1]
    if threshold <= 0:
        return float("nan")
    hill_inv_alpha = float(np.mean(np.log((tail + EPS) / (threshold + EPS))))
    if hill_inv_alpha <= EPS:
        return float("inf")
    return float(1.0 / hill_inv_alpha)


def collision_summary(cfg: SimConfig, edges: pd.DataFrame, num_task_edges: int) -> Dict[str, float]:
    if len(edges) == 0:
        return dict(num_task_edges=num_task_edges, num_pairs=0, num_collisions=0, rho_col=0.0,
                    eta_top=0.0, eta_top_raw=0.0, eta_top_norm=0.0, N_DD=0, N_DDA=0,
                    mean_C=0.0, p95_C=0.0, max_C=0.0,
                    mean_C_norm=0.0, p95_C_norm=0.0, max_C_norm=0.0,
                    skew_C_norm=0.0, excess_kurt_C_norm=0.0, hill_alpha_C_norm=np.nan,
                    max_to_mean_C_norm=0.0, dda_over_dd=0.0, angle_removed_frac=0.0,
                    collision_cost_raw=0.0, collision_cost_norm=0.0,
                    doppler_alias_ratio=0.0)
    C = edges["C"].to_numpy(dtype=float)
    Cn = edges["C_norm"].to_numpy(dtype=float) if "C_norm" in edges.columns else C
    N_DD = int(np.sum(edges["omega_dd"].to_numpy() > cfg.dd_min))
    N_DDA = int(np.sum((edges["omega_dd"].to_numpy() > cfg.dd_min) & (edges["omega_ang"].to_numpy() > cfg.ang_min)))
    eta_raw = top_energy_fraction(C, cfg.top_frac)
    eta_norm = top_energy_fraction(Cn, cfg.top_frac)
    mean_cn = float(np.mean(Cn))
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
        mean_C_norm=mean_cn,
        p95_C_norm=float(np.percentile(Cn, 95)),
        max_C_norm=float(np.max(Cn)),
        skew_C_norm=_skewness(Cn),
        excess_kurt_C_norm=_excess_kurtosis(Cn),
        hill_alpha_C_norm=_hill_tail_alpha(Cn, cfg.top_frac),
        max_to_mean_C_norm=float(np.max(Cn) / (mean_cn + EPS)),
        dda_over_dd=float(N_DDA / (N_DD + EPS)),
        angle_removed_frac=float((N_DD - N_DDA) / (N_DD + EPS)) if N_DD > 0 else 0.0,
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


def tracking_gospa_like(cfg: SimConfig, pos_err_xy: np.ndarray, track_lost: np.ndarray) -> Dict[str, float]:
    """Identity-aware GOSPA-like score for the closed-loop tracking gate.

    The simulator keeps target IDs fixed, so this is not a full unlabeled-set
    GOSPA implementation.  It is a tracking-quality metric that combines
    localization error and lost-track penalties without letting RMSE and loss
    hide each other.  For target q, a non-lost track contributes
    min(e_q, c)^p; a lost track contributes c^p / alpha.  By default the score
    is normalized by Q, making it interpretable as a per-target meter-valued
    metric.
    """
    pos_err_xy = np.asarray(pos_err_xy, dtype=float)
    lost = np.asarray(track_lost, dtype=bool)
    Qn = max(int(pos_err_xy.size), 1)
    c = max(float(cfg.gospa_cutoff_m), EPS)
    p = max(float(cfg.gospa_p), 1.0)
    alpha = max(float(cfg.gospa_alpha), EPS)
    denom = float(Qn if cfg.gospa_normalize_by_q else 1.0)
    active = ~lost
    loc_terms = np.minimum(pos_err_xy[active], c) ** p
    miss_count = int(np.sum(lost))
    miss_term = float(miss_count) * (c ** p) / alpha
    loc_sum = float(np.sum(loc_terms))
    total = loc_sum + miss_term
    gospa = (total / denom) ** (1.0 / p)
    loc = (loc_sum / denom) ** (1.0 / p) if loc_sum > 0 else 0.0
    miss = (miss_term / denom) ** (1.0 / p) if miss_term > 0 else 0.0
    clipped_all = (float(np.sum(np.minimum(pos_err_xy, c) ** p)) / denom) ** (1.0 / p)
    return dict(
        gospa_pos_m=float(gospa),
        gospa_loc_m=float(loc),
        gospa_miss_m=float(miss),
        gospa_miss_count=float(miss_count),
        clipped_pos_rmse_m=float(clipped_all),
    )


def assignment_global_utility(cfg: SimConfig, U: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Collision-unaware capacity-constrained global utility assignment.

    This baseline solves the exactly-K_tgt_min task allocation that maximizes
    sum z[j,q] U[j,q] under UAV load and visibility constraints.  It is useful
    for separating the gain of global assignment itself from the gain of
    DD-collision awareness.
    """
    M, Q = U.shape
    n_rows = Q * cfg.k_tgt_min
    n_cols = M * cfg.k_uav_max
    if n_rows == 0 or n_cols == 0 or n_rows > n_cols:
        return assignment_strongest(cfg, U, v)
    # rows are required target slots, columns are repeated UAV capacity slots
    row_targets = [q for q in range(Q) for _ in range(cfg.k_tgt_min)]
    col_uavs = [j for j in range(M) for _ in range(cfg.k_uav_max)]
    BIG = 1e9
    cost = np.full((n_rows, n_cols), BIG, dtype=float)
    for r, q in enumerate(row_targets):
        for cidx, j in enumerate(col_uavs):
            if v[j, q]:
                cost[r, cidx] = -float(U[j, q])
    try:
        from scipy.optimize import linear_sum_assignment
        rr, cc = linear_sum_assignment(cost)
        if len(rr) < n_rows or np.any(cost[rr, cc] >= BIG / 2):
            return assignment_strongest(cfg, U, v)
        z = np.zeros((M, Q), dtype=int)
        for r, cidx in zip(rr, cc):
            q = row_targets[int(r)]
            j = col_uavs[int(cidx)]
            z[j, q] += 1
        z = (z > 0).astype(int)
        # Repeated target-slot assignments to the same UAV could reduce target count.
        # This is rare because UAV slots are distinct but j-q duplicates are possible;
        # fall back to strongest if exact coverage is lost.
        if is_feasible(cfg, z, v):
            return z
    except Exception:
        pass
    return assignment_strongest(cfg, U, v)


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
