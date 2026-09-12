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
from copy import deepcopy
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
from .detection import *
from .detection import (
    _detect_energy_components,
    _detect_coherent_matched_components,
    _detect_noncoherent_matched_components,
    _coherent_signal_vector,
    _expected_noncoherent_signal_power,
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
                            mode: str) -> Tuple[np.ndarray, np.ndarray, float, float, bool, str, float, float]:
    """Standard KF or DD reliability-weighted information-form update.

    V15 robust-gate logic fixes a blind spot of ordinary Mahalanobis gating:
    if a biased DD peak reports a huge ``pos_std``, the covariance used by the
    gate becomes huge as well and km-level residuals can look statistically
    normal.  We therefore keep the usual innovation statistic for diagnostics,
    but allow three additional gate families:

      1) absolute xy residual / velocity residual gates;
      2) self-reported measurement-std rejection and optional std cap for the
         Mahalanobis gate only;
      3) DD cross-norm risk gate.

    Returns:
        x_new, P_new, gamma, innovation_d2, gated, gate_reason,
        abs_pos_resid_m, abs_vel_resid_mps
    """
    _, H, _ = _kf_cv_matrices(cfg)
    z = np.r_[meas_pos, meas_vel].astype(float)
    raw_pos_std = max(float(pos_std), EPS)
    raw_vel_std = max(float(vel_std), EPS)
    pos_var = raw_pos_std ** 2
    vel_var = raw_vel_std ** 2
    R = np.diag([pos_var, pos_var, 1e-12, vel_var, vel_var, 1e-12])
    innov = z - H @ x_pred
    abs_pos_resid_m = float(np.linalg.norm(innov[:2]))
    abs_vel_resid_mps = float(np.linalg.norm(innov[3:5]))

    # Gate diagnostics.  Optionally cap the self-reported measurement std only
    # for the gate statistic; the actual update still uses the physical/raw R
    # unless the gate action modifies it.
    gate_pos_std = raw_pos_std
    if float(cfg.kf_gate_pos_std_cap_m) > 0:
        gate_pos_std = min(gate_pos_std, float(cfg.kf_gate_pos_std_cap_m))
    gate_vel_std = raw_vel_std
    if float(cfg.kf_gate_vel_std_cap_mps) > 0:
        gate_vel_std = min(gate_vel_std, float(cfg.kf_gate_vel_std_cap_mps))
    R_gate = np.diag([
        max(gate_pos_std, EPS) ** 2,
        max(gate_pos_std, EPS) ** 2,
        1e-12,
        max(gate_vel_std, EPS) ** 2,
        max(gate_vel_std, EPS) ** 2,
        1e-12,
    ])
    S_gate = _ensure_spd(H @ P_pred @ H.T + R_gate)
    innovation_d2 = float(innov.T @ np.linalg.pinv(S_gate) @ innov)

    reasons: List[str] = []
    if np.isfinite(innovation_d2) and innovation_d2 > float(cfg.kf_gate_chi2):
        reasons.append("maha")
    if float(cfg.kf_abs_pos_gate_m) > 0 and abs_pos_resid_m > float(cfg.kf_abs_pos_gate_m):
        reasons.append("abs_pos")
    if float(cfg.kf_abs_vel_gate_mps) > 0 and abs_vel_resid_mps > float(cfg.kf_abs_vel_gate_mps):
        reasons.append("abs_vel")
    if float(cfg.kf_reject_pos_std_m) > 0 and raw_pos_std > float(cfg.kf_reject_pos_std_m):
        reasons.append("pos_std")
    c = float(cross_val) if np.isfinite(cross_val) else 0.0
    if float(cfg.kf_reject_cross_norm) > 0 and c > float(cfg.kf_reject_cross_norm):
        reasons.append("cross")

    gated = bool(cfg.kf_innovation_gate and len(reasons) > 0)
    gate_reason = "|".join(reasons) if gated else ""
    if gated:
        action = str(cfg.kf_gate_action).lower()
        if action in ("skip", "drop", "reject"):
            return x_pred.copy(), _ensure_spd(P_pred.copy()), 1.0, innovation_d2, True, gate_reason, abs_pos_resid_m, abs_vel_resid_mps
        R = R * max(float(cfg.kf_outlier_r_inflate), 1.0)

    mode = str(mode).lower()
    gamma = 1.0
    if mode == "dd_rwif":
        pd_c = float(np.clip(pd_val, 0.0, 1.0))
        gamma = (pd_c ** float(cfg.dd_weight_alpha)) * math.exp(-float(cfg.dd_weight_beta) * max(c, 0.0))
        gamma = float(np.clip(gamma, float(cfg.dd_weight_gamma_min), 1.0))
        Pinv = np.linalg.pinv(_ensure_spd(P_pred))
        Rinv = np.linalg.pinv(_ensure_spd(R))
        Lambda = _ensure_spd(Pinv + gamma * (H.T @ Rinv @ H))
        eta = Pinv @ x_pred + gamma * (H.T @ Rinv @ z)
        P_new = _ensure_spd(np.linalg.pinv(Lambda))
        x_new = P_new @ eta
    else:
        S = _ensure_spd(H @ P_pred @ H.T + R)
        K = P_pred @ H.T @ np.linalg.pinv(S)
        x_new = x_pred + K @ innov
        P_new = _ensure_spd((np.eye(6) - K @ H) @ P_pred @ (np.eye(6) - K @ H).T + K @ R @ K.T)
    x_new[2] = cfg.target_alt_m
    x_new[5] = 0.0
    return x_new, P_new, gamma, innovation_d2, gated, gate_reason, abs_pos_resid_m, abs_vel_resid_mps



def _tracking_gpda_update_one(cfg: SimConfig, x_pred: np.ndarray, P_pred: np.ndarray,
                              cand_pos: List[np.ndarray], cand_vel: List[np.ndarray], cand_prior: List[float],
                              pos_std: float, vel_std: float,
                              pd_val: float, cross_val: float,
                              mode: str) -> Tuple[np.ndarray, np.ndarray, float, float, bool, str, float, float, float, float, float, int]:
    """GPDA-like probabilistic candidate association update.

    This is a lightweight PDA/GPDA-style backend for the structured DD peak
    ambiguity experiments.  When biased/clutter DD pollution creates a competing
    peak, the tracker can be given a small candidate set (e.g., clean peak plus
    polluted peak) and updates by moment-matching over candidate-specific KF or
    DD-RWIF updates.  It is not a full JPDA implementation over multiple tracks;
    target IDs remain fixed in this simulator.  Its role is to test whether
    probabilistic peak association can absorb structured DD false peaks better
    than a single-measurement KF/RWIF backend.

    Returns the common update diagnostics plus GPDA diagnostics:
        x_new, P_new, gamma_mean, innovation_d2_mean, gated_all, gate_reason,
        abs_pos_resid_of_best, abs_vel_resid_of_best,
        best_assoc_prob, miss_assoc_prob, assoc_entropy, candidate_count
    """
    cand_pos = [np.asarray(cp, dtype=float).copy() for cp in cand_pos]
    cand_vel = [np.asarray(cv, dtype=float).copy() for cv in cand_vel]
    pri = np.asarray(cand_prior, dtype=float)
    if len(cand_pos) == 0 or len(cand_vel) == 0 or pri.size == 0:
        return (x_pred.copy(), _ensure_spd(P_pred.copy()), 1.0, np.nan, True, "gpda_no_candidate",
                np.nan, np.nan, 0.0, 1.0, 0.0, 0)
    if pri.size != len(cand_pos):
        pri = np.ones(len(cand_pos), dtype=float)
    pri = np.maximum(pri, float(cfg.gpda_likelihood_floor))
    pri = pri / (np.sum(pri) + EPS)

    # Candidate-specific deterministic update mode.
    sub_mode = "dd_rwif" if _is_dd_assoc_mode(mode) else "kf"

    xs: List[np.ndarray] = []
    Ps: List[np.ndarray] = []
    gammas: List[float] = []
    d2s: List[float] = []
    gateds: List[bool] = []
    reasons: List[str] = []
    abs_pos_vals: List[float] = []
    abs_vel_vals: List[float] = []
    scores: List[float] = []
    for i, (mp, mv) in enumerate(zip(cand_pos, cand_vel)):
        mp = mp.copy(); mv = mv.copy()
        mp[2] = cfg.target_alt_m
        mv[2] = 0.0
        x_i, P_i, gamma_i, d2_i, gated_i, reason_i, abs_pos_i, abs_vel_i = _tracking_kf_update_one(
            cfg, x_pred, P_pred, mp, mv, pos_std, vel_std, pd_val, cross_val, sub_mode
        )
        xs.append(x_i); Ps.append(P_i); gammas.append(float(gamma_i)); d2s.append(float(d2_i) if np.isfinite(d2_i) else np.inf)
        gateds.append(bool(gated_i)); reasons.append(str(reason_i)); abs_pos_vals.append(float(abs_pos_i)); abs_vel_vals.append(float(abs_vel_i))
        # Use the gate statistic as an association likelihood.  Hard-gated
        # candidates get zero likelihood under skip/reject semantics; inflated
        # candidates remain possible but naturally less likely through d2.
        if gated_i and str(cfg.kf_gate_action).lower() in ("skip", "drop", "reject"):
            like = 0.0
        else:
            d2_clip = min(float(d2_i) if np.isfinite(d2_i) else 700.0, 700.0)
            like = math.exp(-0.5 * d2_clip)
        scores.append(float(pri[i]) * max(like, 0.0))

    scores_arr = np.asarray(scores, dtype=float)
    # Miss/no-valid-measurement hypothesis.  It prevents GPDA from being forced
    # to follow a dubious candidate set under strong gating or very low Pd.
    pd_c = float(np.clip(pd_val, 0.0, 1.0))
    miss_score = max(float(cfg.gpda_missed_prior), 0.0) * max(1.0 - pd_c, 0.05)
    total = float(np.sum(scores_arr) + miss_score)
    if (not np.isfinite(total)) or total <= 0.0:
        return (x_pred.copy(), _ensure_spd(P_pred.copy()), 1.0, float(np.nanmin(d2s)) if d2s else np.nan,
                True, "gpda_all_rejected", float(np.nanmin(abs_pos_vals)) if abs_pos_vals else np.nan,
                float(np.nanmin(abs_vel_vals)) if abs_vel_vals else np.nan, 0.0, 1.0, 0.0, len(cand_pos))
    betas = scores_arr / total
    beta0 = miss_score / total

    # Moment-matched PDA state/covariance mixture.
    x_mix = beta0 * x_pred.copy()
    for b, x_i in zip(betas, xs):
        x_mix += float(b) * x_i
    x_mix[2] = cfg.target_alt_m
    x_mix[5] = 0.0

    P_mix = beta0 * (P_pred + np.outer(x_pred - x_mix, x_pred - x_mix))
    for b, x_i, P_i in zip(betas, xs, Ps):
        dx = x_i - x_mix
        P_mix += float(b) * (P_i + np.outer(dx, dx))
    P_mix = _ensure_spd(P_mix)

    # Diagnostics.  We report the best non-miss candidate's residual/gating.
    best_idx = int(np.argmax(betas)) if betas.size else 0
    best_prob = float(betas[best_idx]) if betas.size else 0.0
    probs_all = np.r_[beta0, betas]
    entropy = float(-np.sum(probs_all * np.log(np.maximum(probs_all, EPS))))
    gamma_mean = float(np.sum(betas * np.asarray(gammas))) if betas.size else 1.0
    d2_mean = float(np.sum(betas * np.asarray(d2s))) if betas.size and np.all(np.isfinite(d2s)) else (float(d2s[best_idx]) if d2s else np.nan)
    # Treat the measurement as gated only if all candidate measurements were unusable
    # and the miss hypothesis dominates.  Candidate-level gate rates are exposed via trace.
    gated_all = bool(best_prob <= 0.0 and beta0 > 0.5)
    gate_reason = "gpda_all_rejected" if gated_all else ("gpda_candidate_gates:" + ";".join([r for r in reasons if r]))
    return (x_mix, P_mix, gamma_mean, d2_mean, gated_all, gate_reason,
            float(abs_pos_vals[best_idx]) if abs_pos_vals else np.nan,
            float(abs_vel_vals[best_idx]) if abs_vel_vals else np.nan,
            best_prob, float(beta0), entropy, len(cand_pos))



def _is_single_pda_mode(mode: str) -> bool:
    """V17-style per-track PDA/GPDA-like probabilistic peak association."""
    m = str(mode).lower().replace("-", "_")
    return m in ("pda", "gpda", "dd_pda", "dd_gpda")


def _is_shared_jpda_mode(mode: str) -> bool:
    """V18 shared-measurement JPDA-like backend.

    ``jpda`` and ``dd_jpda`` are aliases for the shared-measurement version in
    V18, while ``pda/gpda`` remain single-track probabilistic peak-association
    ablations.
    """
    m = str(mode).lower().replace("-", "_")
    return m in ("jpda", "jpda_shared", "dd_jpda", "dd_jpda_shared")


def _is_dd_assoc_mode(mode: str) -> bool:
    m = str(mode).lower().replace("-", "_")
    return m.startswith("dd_") or m in ("ddgpda", "ddjpda")


def _connected_jpda_components(W: np.ndarray) -> List[Tuple[List[int], List[int]]]:
    """Connected components of the track-measurement validation graph."""
    W = np.asarray(W, dtype=float)
    Q, M = W.shape
    visited_t = np.zeros(Q, dtype=bool)
    visited_m = np.zeros(M, dtype=bool)
    comps: List[Tuple[List[int], List[int]]] = []
    track_to_meas = [set(np.where(W[q] > 0)[0].tolist()) for q in range(Q)]
    meas_to_track = [set(np.where(W[:, m] > 0)[0].tolist()) for m in range(M)]
    for q0 in range(Q):
        if visited_t[q0]:
            continue
        tq = [q0]
        mq: List[int] = []
        tracks: set[int] = set()
        meas: set[int] = set()
        visited_t[q0] = True
        while tq or mq:
            if tq:
                q = tq.pop()
                tracks.add(q)
                for m in track_to_meas[q]:
                    if not visited_m[m]:
                        visited_m[m] = True
                        mq.append(m)
            else:
                m = mq.pop()
                meas.add(m)
                for q in meas_to_track[m]:
                    if not visited_t[q]:
                        visited_t[q] = True
                        tq.append(q)
        comps.append((sorted(tracks), sorted(meas)))
    return comps


def _enumerate_component_marginals(Wc: np.ndarray, missc: np.ndarray, max_events: int) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Exact JPDA marginals for one connected component.

    The component weight is the product of per-track association likelihoods.
    A track can either be missed or associated to one measurement, and each
    measurement can be assigned to at most one track.  If enumeration exceeds
    ``max_events``, fall back to row-wise PDA-like marginals and mark exact=False.
    """
    Wc = np.asarray(Wc, dtype=float)
    missc = np.maximum(np.asarray(missc, dtype=float), EPS)
    nT, nM = Wc.shape
    betas = np.zeros((nT, nM), dtype=float)
    beta0 = np.zeros(nT, dtype=float)
    if nT == 0:
        return betas, beta0, 0, True
    if nM == 0 or not np.any(Wc > 0):
        beta0[:] = 1.0
        return betas, beta0, 1, True

    deg = np.sum(Wc > 0, axis=1)
    order = sorted(range(nT), key=lambda i: int(deg[i]))
    assign = [-1] * nT
    total = 0.0
    n_events = 0
    exact = True

    def rec(level: int, used: set[int], weight: float) -> None:
        nonlocal total, n_events, exact
        if n_events > max_events:
            exact = False
            return
        if level >= nT:
            n_events += 1
            total += weight
            for i, m in enumerate(assign):
                if m < 0:
                    beta0[i] += weight
                else:
                    betas[i, m] += weight
            return
        i = order[level]
        # Miss hypothesis.
        assign[i] = -1
        rec(level + 1, used, weight * missc[i])
        if not exact:
            return
        # Valid measurement hypotheses.
        for m in np.where(Wc[i] > 0)[0].tolist():
            if m in used:
                continue
            assign[i] = int(m)
            used.add(int(m))
            rec(level + 1, used, weight * Wc[i, m])
            used.remove(int(m))
            if not exact:
                return
        assign[i] = -1

    rec(0, set(), 1.0)
    if (not exact) or total <= 0.0 or (not np.isfinite(total)):
        denom = missc + np.sum(Wc, axis=1) + EPS
        beta0 = missc / denom
        betas = Wc / denom[:, None]
        return betas, beta0, int(n_events), False
    betas /= total
    beta0 /= total
    return betas, beta0, int(n_events), True


def _jpda_joint_marginals(cfg: SimConfig, W: np.ndarray, miss_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Compute shared-measurement JPDA marginals by exact component enumeration."""
    W = np.asarray(W, dtype=float)
    miss_scores = np.asarray(miss_scores, dtype=float)
    Q, M = W.shape
    beta = np.zeros((Q, M), dtype=float)
    beta0 = np.ones(Q, dtype=float)
    total_events = 0
    exact_components = 0
    comps = _connected_jpda_components(W)
    for tracks, meas in comps:
        if not tracks:
            continue
        Wc = W[np.ix_(tracks, meas)] if meas else np.zeros((len(tracks), 0), dtype=float)
        missc = np.maximum(miss_scores[tracks], EPS)
        bc, b0c, nev, exact = _enumerate_component_marginals(Wc, missc, int(cfg.jpda_max_joint_events))
        total_events += int(nev)
        exact_components += int(bool(exact))
        for ii, q in enumerate(tracks):
            beta0[q] = b0c[ii]
            for jj, m in enumerate(meas):
                beta[q, m] = bc[ii, jj]
    exact_frac = float(exact_components / max(len(comps), 1))
    return beta, beta0, int(total_events), exact_frac


def _tracking_jpda_shared_update_all(
    cfg: SimConfig,
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    measurements: List[Dict[str, object]],
    pos_std_vec: np.ndarray,
    vel_std_vec: np.ndarray,
    pd_vec: np.ndarray,
    cross_vec: np.ndarray,
    mode: str,
) -> Dict[str, np.ndarray | float]:
    """Shared-measurement JPDA-like update for all tracks.

    Unlike the V17 per-track PDA/GPDA update, this function builds a common
    validation matrix over all tracks and all scene-level measurement candidates,
    enumerates feasible joint association events per connected component, and
    uses the resulting marginal probabilities for moment-matched KF/DD-RWIF
    updates.  This is still a compact simulator implementation, but it contains
    the defining JPDA ingredient missing in V17: cross-track measurement
    exclusivity over a shared measurement pool.
    """
    Q = int(x_pred.shape[0])
    M = len(measurements)
    sub_mode = "dd_rwif" if _is_dd_assoc_mode(mode) else "kf"
    x_cand: List[List[Optional[np.ndarray]]] = [[None for _ in range(M)] for _ in range(Q)]
    P_cand: List[List[Optional[np.ndarray]]] = [[None for _ in range(M)] for _ in range(Q)]
    gamma_cand = np.ones((Q, M), dtype=float)
    d2_cand = np.full((Q, M), np.nan, dtype=float)
    gated_cand = np.zeros((Q, M), dtype=bool)
    reason_cand = np.full((Q, M), "", dtype=object)
    abs_pos_cand = np.full((Q, M), np.nan, dtype=float)
    abs_vel_cand = np.full((Q, M), np.nan, dtype=float)
    W = np.zeros((Q, M), dtype=float)
    for q in range(Q):
        for m, meas in enumerate(measurements):
            mp = np.asarray(meas["pos"], dtype=float)
            mv = np.asarray(meas["vel"], dtype=float)
            prior_m = max(float(meas.get("prior", 1.0)), float(cfg.gpda_likelihood_floor))
            x_i, P_i, gamma_i, d2_i, gated_i, reason_i, abs_pos_i, abs_vel_i = _tracking_kf_update_one(
                cfg, x_pred[q], P_pred[q], mp, mv,
                float(pos_std_vec[q]), float(vel_std_vec[q]), float(pd_vec[q]), float(cross_vec[q]), sub_mode
            )
            x_cand[q][m] = x_i
            P_cand[q][m] = P_i
            gamma_cand[q, m] = float(gamma_i)
            d2_cand[q, m] = float(d2_i) if np.isfinite(d2_i) else np.inf
            gated_cand[q, m] = bool(gated_i)
            reason_cand[q, m] = str(reason_i)
            abs_pos_cand[q, m] = float(abs_pos_i)
            abs_vel_cand[q, m] = float(abs_vel_i)
            if gated_i:
                like = 0.0
            else:
                d2_clip = min(float(d2_i) if np.isfinite(d2_i) else 700.0, 700.0)
                like = math.exp(-0.5 * d2_clip)
                if sub_mode == "dd_rwif":
                    like *= max(float(gamma_i), float(cfg.dd_weight_gamma_min))
            W[q, m] = prior_m * max(float(pd_vec[q]), 0.0) * max(like, 0.0)
    miss_scores = np.maximum(float(cfg.gpda_missed_prior) * np.maximum(1.0 - np.clip(pd_vec, 0.0, 1.0), 0.05), EPS)
    beta, beta0, n_events, exact_frac = _jpda_joint_marginals(cfg, W, miss_scores)

    x_new = np.zeros_like(x_pred)
    P_new = np.zeros_like(P_pred)
    gamma_vals = np.ones(Q, dtype=float)
    innovation_d2_vals = np.full(Q, np.nan, dtype=float)
    gated_flags = np.zeros(Q, dtype=bool)
    gate_reason_vals = np.full(Q, "", dtype=object)
    abs_pos_resid_vals = np.full(Q, np.nan, dtype=float)
    abs_vel_resid_vals = np.full(Q, np.nan, dtype=float)
    best_assoc_prob_vals = np.zeros(Q, dtype=float)
    miss_assoc_prob_vals = beta0.copy()
    assoc_entropy_vals = np.zeros(Q, dtype=float)
    candidate_count_vals = np.sum(W > 0, axis=1).astype(float)

    for q in range(Q):
        bq = beta[q]
        b0 = float(beta0[q])
        x_mix = b0 * x_pred[q].copy()
        for m in range(M):
            if bq[m] > 0 and x_cand[q][m] is not None:
                x_mix += float(bq[m]) * x_cand[q][m]
        x_mix[2] = cfg.target_alt_m
        x_mix[5] = 0.0
        P_mix = b0 * (P_pred[q] + np.outer(x_pred[q] - x_mix, x_pred[q] - x_mix))
        for m in range(M):
            if bq[m] > 0 and x_cand[q][m] is not None and P_cand[q][m] is not None:
                dx = x_cand[q][m] - x_mix
                P_mix += float(bq[m]) * (P_cand[q][m] + np.outer(dx, dx))
        x_new[q] = x_mix
        P_new[q] = _ensure_spd(P_mix)
        if np.sum(bq) > 0:
            best_idx = int(np.argmax(bq))
            best_assoc_prob_vals[q] = float(bq[best_idx])
            gamma_vals[q] = float(np.sum(bq * gamma_cand[q]) / (np.sum(bq) + EPS))
            finite_d2 = np.where(np.isfinite(d2_cand[q]), d2_cand[q], 0.0)
            innovation_d2_vals[q] = float(np.sum(bq * finite_d2) / (np.sum(bq) + EPS))
            abs_pos_resid_vals[q] = float(abs_pos_cand[q, best_idx])
            abs_vel_resid_vals[q] = float(abs_vel_cand[q, best_idx])
            used_reasons = [str(reason_cand[q, m]) for m in range(M) if bq[m] > 1e-9 and str(reason_cand[q, m])]
            gate_reason_vals[q] = "jpda_candidate_gates:" + ";".join(used_reasons) if used_reasons else ""
        else:
            gated_flags[q] = True
            gate_reason_vals[q] = "jpda_shared_no_valid_candidate"
        probs = np.r_[b0, bq[bq > 0]]
        assoc_entropy_vals[q] = float(-np.sum(probs * np.log(np.maximum(probs, EPS))))
    return dict(
        x_new=x_new,
        P_new=P_new,
        gamma_vals=gamma_vals,
        innovation_d2_vals=innovation_d2_vals,
        gated_flags=gated_flags,
        gate_reason_vals=gate_reason_vals,
        abs_pos_resid_vals=abs_pos_resid_vals,
        abs_vel_resid_vals=abs_vel_resid_vals,
        gpda_candidate_count_vals=candidate_count_vals,
        gpda_best_assoc_prob_vals=best_assoc_prob_vals,
        gpda_miss_assoc_prob_vals=miss_assoc_prob_vals,
        gpda_assoc_entropy_vals=assoc_entropy_vals,
        jpda_joint_event_count=float(n_events),
        jpda_exact_component_frac=float(exact_frac),
        jpda_num_measurements=float(M),
    )

def _normalize_xy_dir(v: np.ndarray) -> np.ndarray:
    """Return a finite unit vector in the xy plane."""
    out = np.asarray(v, dtype=float).copy()
    if out.size < 3:
        tmp = np.zeros(3, dtype=float)
        tmp[:out.size] = out
        out = tmp
    out[2] = 0.0
    n = float(np.linalg.norm(out[:2]))
    if not np.isfinite(n) or n < EPS:
        out[:2] = np.array([1.0, 0.0])
    else:
        out[:2] /= n
    return out


def structured_pollution_probability(cfg: SimConfig, cross_val: float) -> float:
    """Potential biased/clutter peak probability induced by DD cross pollution.

    In no-pollution mode this deliberately returns zero.  This avoids confusing
    diagnostics where pollution is disabled but a hypothetical probability is
    still reported as if it were active.
    """
    mode = str(cfg.meas_pollution).lower()
    if mode in ("none", "off", "false"):
        return 0.0
    c = float(cross_val) if np.isfinite(cross_val) else 0.0
    c = max(c, 0.0)
    p_pol = float(cfg.pollution_pmax) * c / (c + float(cfg.pollution_c0) + EPS)
    return float(np.clip(p_pol, 0.0, 1.0))


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
    p_pol = structured_pollution_probability(cfg, cross_val)
    if mode in ("none", "off", "false") or u_pollute >= p_pol:
        return meas_pos, meas_vel, False, p_pol
    dir_pos_unit = _normalize_xy_dir(dir_pos_unit)
    dir_vel_unit = _normalize_xy_dir(dir_vel_unit)
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

def _apply_assignment_hysteresis(cfg: SimConfig, method: str, sc_pred: Scenario,
                                 z_new: np.ndarray, z_prev: Optional[np.ndarray],
                                 U: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, bool, float, float, float]:
    """Optionally keep the previous collision-aware assignment.

    V14 applies hysteresis only to the explicit ``collision_aware_hys`` method,
    so that the original ``collision_aware`` curve remains an unregularized
    baseline.  Feasibility and coverage are handled before objective comparison:
    a worse-coverage new assignment is not accepted when ``hys_force_coverage``
    is enabled.

    Returns:
        z_used, hysteresis_kept_prev, obj_new, obj_prev, threshold
    """
    if (not cfg.assignment_hysteresis) or z_prev is None:
        return z_new, False, np.nan, np.nan, np.nan
    if method not in ("collision_aware_hys", "global_utility_hys"):
        return z_new, False, np.nan, np.nan, np.nan

    prev_feasible = is_feasible(cfg, z_prev, v)
    new_feasible = is_feasible(cfg, z_new, v)
    if prev_feasible and not new_feasible:
        return z_prev.copy(), True, np.nan, np.nan, np.nan
    if (not prev_feasible) and new_feasible:
        return z_new, False, np.nan, np.nan, np.nan
    if (not prev_feasible) and (not new_feasible):
        return z_new, False, np.nan, np.nan, np.nan

    def cov_tuple(z: np.ndarray) -> Tuple[int, int]:
        # Higher is better: first Kmin-covered targets, then one-receiver-covered targets.
        return (int(np.sum(z.sum(axis=0) >= cfg.k_tgt_min)), int(np.sum(z.sum(axis=0) >= 1)))

    cov_new = cov_tuple(z_new)
    cov_prev = cov_tuple(z_prev)
    if cfg.hys_force_coverage:
        if cov_new > cov_prev:
            return z_new, False, np.nan, np.nan, np.nan
        if cov_new < cov_prev:
            return z_prev.copy(), True, np.nan, np.nan, np.nan

    # Compare the underlying non-temporal objective to avoid double-counting
    # switch cost.  Hysteresis itself is the temporal regularizer.
    if method == "global_utility_hys":
        obj_new = float(np.sum(z_new * U))
        obj_prev = float(np.sum(z_prev * U))
    else:
        local = SimConfig(**asdict(cfg))
        local.lambda_switch = 0.0
        local.assignment_hysteresis = False
        obj_new = objective(local, sc_pred, z_new, U, use_full_kernel=False, z_prev=None)
        obj_prev = objective(local, sc_pred, z_prev, U, use_full_kernel=False, z_prev=None)
    threshold = float(cfg.hys_abs_margin) + float(cfg.hys_rel_margin) * max(abs(obj_prev), 1.0)
    if (obj_new - obj_prev) <= threshold:
        return z_prev.copy(), True, float(obj_new), float(obj_prev), float(threshold)
    return z_new, False, float(obj_new), float(obj_prev), float(threshold)

def assign_by_method(cfg: SimConfig, method: str, sc_pred: Scenario, U: np.ndarray, v: np.ndarray, rng: np.random.Generator, z_prev: Optional[np.ndarray] = None) -> np.ndarray:
    """Dispatch assignment by method name.

    Paper baselines (random / global_utility / box_dd / dd_only / proposed) are
    the primary methods.  Legacy names (nearest, strongest, collision_aware*)
    are kept for backward compatibility with supplemental experiments.
    """
    if method == "random":
        return assignment_random(cfg, U, v, rng)
    if method == "nearest":
        return assignment_nearest(cfg, sc_pred, v)
    if method == "strongest":
        return assignment_strongest(cfg, U, v)
    if method in ("global_utility", "global_utility_hys"):
        return assignment_global_utility(cfg, U, v)
    # proposed = full Dirichlet DD-angle collision-aware (the main method)
    if method in ("collision_aware", "collision_aware_hys", "collision_aware_switch", "proposed"):
        return assignment_collision_aware_greedy(cfg, sc_pred, U, v, z_prev=z_prev)
    # dd_only = Dirichlet DD collision-aware, no angle overlap (sigma_theta = inf)
    if method == "dd_only":
        dd_cfg = deepcopy(cfg)
        dd_cfg.sigma_theta_deg = float("inf")
        return assignment_collision_aware_greedy(dd_cfg, sc_pred, U, v, z_prev=z_prev)
    # box_dd = collision-aware with hard box DD overlap (no_spread proxy kernel)
    if method == "box_dd":
        box_cfg = deepcopy(cfg)
        box_cfg.proxy_kernel_mode = "no_spread"
        return assignment_collision_aware_greedy(box_cfg, sc_pred, U, v, z_prev=z_prev)
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
                ang = angle_overlap(cfg, sc, j, qp, q)
                I += bistatic_gain(cfg, sc, i, j, qp) * ov * ang
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


def _tracking_worker(job: Tuple[SimConfig, int]) -> Dict[str, List[Dict[str, float]]]:
    """Worker for one closed-loop tracking MC trial.

    V10 uses common random numbers (CRN) across methods for the tracking layer:
    initial track errors, prediction noises, detection uniforms, and measurement
    unit noises are method-independent.  Evaluation RNGs remain method-specific
    and are isolated from tracking randomness.
    """
    cfg, mc = job
    # Paper baseline methods — the primary comparison set for tracking.
    # proposed vs dd_only is the central question; box_dd, global_utility,
    # and random provide the ablation ladder.
    methods = ["random", "global_utility", "box_dd", "dd_only", "proposed"]
    # Supplemental lower-bound baselines
    methods.extend(["nearest", "strongest"])
    if cfg.assignment_hysteresis:
        methods.append("collision_aware_hys")
    if cfg.lambda_switch > 0:
        methods.append("collision_aware_switch")
    rows: List[Dict[str, float]] = []
    trace_rows: List[Dict[str, float]] = []
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
        if str(cfg.tracker_mode).lower() in ("kf", "kalman", "dd_rwif", "rwif", "information", "pda", "gpda", "dd_pda", "dd_gpda", "jpda", "jpda_shared", "dd_jpda", "dd_jpda_shared"):
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
            z_prev_state = st["z_prev"]
            z_prev_for_objective = z_prev_state if method == "collision_aware_switch" else None
            z_candidate = assign_by_method(cfg, method, sc_pred, U, v, rng_assign, z_prev=z_prev_for_objective)
            z, hys_kept_prev, hys_obj_new, hys_obj_prev, hys_threshold = _apply_assignment_hysteresis(
                cfg, method, sc_pred, z_candidate, z_prev_state, U, v
            )

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
            pollution_probs_potential = np.array([structured_pollution_probability(cfg, cv) for cv in cross_vec], dtype=float)
            covered = tgt_sorted["covered"].to_numpy(dtype=bool) if "covered" in tgt_sorted else (z.sum(axis=0) > 0)
            finite_cross = tgt_sorted["full_cross_norm_mean"].to_numpy(dtype=float)
            polluted_flags = np.zeros(cfg.tracking_Q, dtype=bool)
            pollution_probs = np.zeros(cfg.tracking_Q, dtype=float)
            gamma_vals = np.ones(cfg.tracking_Q, dtype=float)
            innovation_d2_vals = np.full(cfg.tracking_Q, np.nan, dtype=float)
            gated_flags = np.zeros(cfg.tracking_Q, dtype=bool)
            gate_reason_vals = np.full(cfg.tracking_Q, "", dtype=object)
            abs_pos_resid_vals = np.full(cfg.tracking_Q, np.nan, dtype=float)
            abs_vel_resid_vals = np.full(cfg.tracking_Q, np.nan, dtype=float)
            gpda_candidate_count_vals = np.zeros(cfg.tracking_Q, dtype=float)
            gpda_best_assoc_prob_vals = np.full(cfg.tracking_Q, np.nan, dtype=float)
            gpda_miss_assoc_prob_vals = np.full(cfg.tracking_Q, np.nan, dtype=float)
            gpda_assoc_entropy_vals = np.full(cfg.tracking_Q, np.nan, dtype=float)
            tracker_mode = str(cfg.tracker_mode).lower()
            jpda_joint_event_count = np.nan
            jpda_exact_component_frac = np.nan
            jpda_num_measurements = np.nan
            if _is_shared_jpda_mode(tracker_mode):
                if x_pred is None or P_pred is None:
                    raise RuntimeError("Shared JPDA tracker requires predicted state/covariance")
                measurements: List[Dict[str, object]] = []
                for q in range(cfg.tracking_Q):
                    if not detected[q]:
                        continue
                    if cfg.tracking_crn:
                        meas_pos = sc_true.tgt_pos[q] + pos_std_vec[q] * meas_pos_eps[q]
                        meas_vel = sc_true.tgt_vel[q] + vel_std_vec[q] * meas_vel_eps[q]
                    else:
                        meas_pos = sc_true.tgt_pos[q] + rng_assign.normal(0.0, pos_std_vec[q], size=3)
                        meas_vel = sc_true.tgt_vel[q] + rng_assign.normal(0.0, vel_std_vec[q], size=3)
                    meas_pos[2] = cfg.target_alt_m
                    meas_vel[2] = 0.0
                    meas_pos_clean = meas_pos.copy()
                    meas_vel_clean = meas_vel.copy()
                    meas_pos_obs, meas_vel_obs, polluted, p_pol = _apply_structured_dd_measurement_pollution(
                        cfg, meas_pos, meas_vel, sc_true, q, cross_vec[q],
                        pollution_uniform[q] if cfg.tracking_crn else float(rng_assign.random()),
                        pollution_pos_dir[q] if cfg.tracking_crn else rng_assign.normal(0.0, 1.0, size=3),
                        pollution_vel_dir[q] if cfg.tracking_crn else rng_assign.normal(0.0, 1.0, size=3),
                        rng_assign,
                    )
                    polluted_flags[q] = polluted
                    pollution_probs[q] = p_pol
                    if bool(cfg.gpda_include_clean_candidate) and bool(polluted):
                        measurements.append(dict(pos=meas_pos_clean, vel=meas_vel_clean,
                                                 prior=max(1.0 - float(p_pol), float(cfg.gpda_likelihood_floor)),
                                                 source_target=q, kind="clean"))
                        measurements.append(dict(pos=meas_pos_obs, vel=meas_vel_obs,
                                                 prior=max(float(p_pol), float(cfg.gpda_likelihood_floor)),
                                                 source_target=q, kind="polluted"))
                    else:
                        measurements.append(dict(pos=meas_pos_obs, vel=meas_vel_obs, prior=1.0,
                                                 source_target=q, kind=("polluted" if polluted else "clean")))
                jpda_out = _tracking_jpda_shared_update_all(
                    cfg, x_pred, P_pred, measurements, pos_std_vec, vel_std_vec, pd_vec, cross_vec, tracker_mode
                )
                st["x"] = jpda_out["x_new"]
                st["P"] = jpda_out["P_new"]
                st["track_pos"], st["track_vel"] = _pos_vel_from_states(st["x"], cfg)
                gamma_vals = np.asarray(jpda_out["gamma_vals"], dtype=float)
                innovation_d2_vals = np.asarray(jpda_out["innovation_d2_vals"], dtype=float)
                gated_flags = np.asarray(jpda_out["gated_flags"], dtype=bool)
                gate_reason_vals = np.asarray(jpda_out["gate_reason_vals"], dtype=object)
                abs_pos_resid_vals = np.asarray(jpda_out["abs_pos_resid_vals"], dtype=float)
                abs_vel_resid_vals = np.asarray(jpda_out["abs_vel_resid_vals"], dtype=float)
                gpda_candidate_count_vals = np.asarray(jpda_out["gpda_candidate_count_vals"], dtype=float)
                gpda_best_assoc_prob_vals = np.asarray(jpda_out["gpda_best_assoc_prob_vals"], dtype=float)
                gpda_miss_assoc_prob_vals = np.asarray(jpda_out["gpda_miss_assoc_prob_vals"], dtype=float)
                gpda_assoc_entropy_vals = np.asarray(jpda_out["gpda_assoc_entropy_vals"], dtype=float)
                jpda_joint_event_count = float(jpda_out["jpda_joint_event_count"])
                jpda_exact_component_frac = float(jpda_out["jpda_exact_component_frac"])
                jpda_num_measurements = float(jpda_out["jpda_num_measurements"])
                for q in range(cfg.tracking_Q):
                    if gpda_miss_assoc_prob_vals[q] > 0.5:
                        st["miss"][q] += 1
                    else:
                        st["miss"][q] = 0
            else:
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
                        # Keep the clean peak before structured DD pollution.  The
                        # PDA/GPDA-like backend can see both the clean and biased/clutter
                        # peaks as a small candidate set, mimicking a DD peak detector
                        # that exposes multiple local maxima.  Deterministic KF/RWIF
                        # backends still use the selected/observed peak only.
                        meas_pos_clean = meas_pos.copy()
                        meas_vel_clean = meas_vel.copy()
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
                                raise RuntimeError("KF/DD-RWIF/PDA/GPDA tracker requires predicted state/covariance")
                            is_gpda_mode = _is_single_pda_mode(tracker_mode)
                            if is_gpda_mode:
                                # Candidate association baseline.  In this simplified
                                # simulator target identities are fixed; the GPDA-like
                                # update is therefore a probabilistic DD-peak association
                                # baseline rather than a full multi-track JPDA over all
                                # cross-target hypotheses.
                                cand_pos = [meas_pos]
                                cand_vel = [meas_vel]
                                cand_prior = [1.0]
                                if bool(cfg.gpda_include_clean_candidate) and bool(polluted):
                                    cand_pos = [meas_pos_clean, meas_pos]
                                    cand_vel = [meas_vel_clean, meas_vel]
                                    cand_prior = [max(1.0 - float(p_pol), float(cfg.gpda_likelihood_floor)),
                                                  max(float(p_pol), float(cfg.gpda_likelihood_floor))]
                                x_new, P_new, gamma, innovation_d2, gated, gate_reason, abs_pos_resid, abs_vel_resid, best_prob, miss_prob, entropy, cand_count = _tracking_gpda_update_one(
                                    cfg, x_pred[q], P_pred[q], cand_pos, cand_vel, cand_prior,
                                    pos_std_vec[q], vel_std_vec[q], pd_vec[q], cross_vec[q], tracker_mode
                                )
                                innovation_d2_vals[q] = innovation_d2
                                gated_flags[q] = gated
                                gate_reason_vals[q] = gate_reason
                                abs_pos_resid_vals[q] = abs_pos_resid
                                abs_vel_resid_vals[q] = abs_vel_resid
                                gpda_candidate_count_vals[q] = float(cand_count)
                                gpda_best_assoc_prob_vals[q] = float(best_prob)
                                gpda_miss_assoc_prob_vals[q] = float(miss_prob)
                                gpda_assoc_entropy_vals[q] = float(entropy)
                            else:
                                mode_update = "dd_rwif" if tracker_mode in ("dd_rwif", "rwif", "information") else "kf"
                                if cfg.tracking_reinit_lost and was_lost:
                                    # V14+: lost-track reinitialization is no longer allowed to bypass
                                    # innovation gating.  If the reinit measurement is an innovation
                                    # outlier, follow the configured gate action: skip keeps prediction,
                                    # inflate performs a weak robust update instead of hard reset.
                                    gated = False
                                    innovation_d2 = np.nan
                                    gamma = 1.0
                                    gate_reason = ""
                                    abs_pos_resid = np.nan
                                    abs_vel_resid = np.nan
                                    if cfg.gate_reinit_measurement and cfg.kf_innovation_gate:
                                        x_gate, P_gate, gamma_gate, innovation_d2, gated, gate_reason, abs_pos_resid, abs_vel_resid = _tracking_kf_update_one(
                                            cfg, x_pred[q], P_pred[q], meas_pos, meas_vel,
                                            pos_std_vec[q], vel_std_vec[q], pd_vec[q], cross_vec[q], mode_update
                                        )
                                        if gated:
                                            x_new, P_new, gamma = x_gate, P_gate, gamma_gate
                                        else:
                                            x_new = np.r_[meas_pos, meas_vel].astype(float)
                                            P_new = np.diag([
                                                pos_std_vec[q]**2, pos_std_vec[q]**2, 1e-12,
                                                vel_std_vec[q]**2, vel_std_vec[q]**2, 1e-12,
                                            ])
                                    else:
                                        x_new = np.r_[meas_pos, meas_vel].astype(float)
                                        P_new = np.diag([
                                            pos_std_vec[q]**2, pos_std_vec[q]**2, 1e-12,
                                            vel_std_vec[q]**2, vel_std_vec[q]**2, 1e-12,
                                        ])
                                    innovation_d2_vals[q] = innovation_d2
                                    gated_flags[q] = gated
                                    gate_reason_vals[q] = gate_reason
                                    abs_pos_resid_vals[q] = abs_pos_resid
                                    abs_vel_resid_vals[q] = abs_vel_resid
                                else:
                                    x_new, P_new, gamma, innovation_d2, gated, gate_reason, abs_pos_resid, abs_vel_resid = _tracking_kf_update_one(
                                        cfg, x_pred[q], P_pred[q], meas_pos, meas_vel,
                                        pos_std_vec[q], vel_std_vec[q], pd_vec[q], cross_vec[q], mode_update
                                    )
                                    innovation_d2_vals[q] = innovation_d2
                                    gated_flags[q] = gated
                                    gate_reason_vals[q] = gate_reason
                                    abs_pos_resid_vals[q] = abs_pos_resid
                                    abs_vel_resid_vals[q] = abs_vel_resid
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
                                raise RuntimeError("KF/DD-RWIF/GPDA tracker requires predicted state/covariance")
                            st["x"][q] = x_pred[q]
                            st["P"][q] = P_pred[q]
                            st["track_pos"][q] = pred_pos[q]
                            st["track_vel"][q] = pred_vel[q]
                        st["miss"][q] += 1
            # Optional per-target trace for diagnosing catastrophic outliers, pollution,
            # switching, and innovation gating.
            if cfg.save_measurement_trace:
                for q_trace in range(cfg.tracking_Q):
                    prev_col = np.zeros(cfg.tracking_M, dtype=int) if z_prev_state is None else z_prev_state[:, q_trace].astype(int)
                    cur_col = z[:, q_trace].astype(int)
                    trace_rows.append(dict(
                        mc=mc, frame=frame, method=method, target=q_trace,
                        detected=bool(detected[q_trace]),
                        pd_target=float(pd_vec[q_trace]),
                        covered=bool(covered[q_trace]),
                        pos_std=float(pos_std_vec[q_trace]),
                        vel_std=float(vel_std_vec[q_trace]),
                        full_cross_norm=float(cross_vec[q_trace]) if np.isfinite(cross_vec[q_trace]) else np.nan,
                        pollution_prob=float(pollution_probs[q_trace]),
                        pollution_prob_potential=float(pollution_probs_potential[q_trace]),
                        polluted=bool(polluted_flags[q_trace]),
                        dd_info_weight=float(gamma_vals[q_trace]),
                        innovation_d2=float(innovation_d2_vals[q_trace]) if np.isfinite(innovation_d2_vals[q_trace]) else np.nan,
                        abs_pos_resid_m=float(abs_pos_resid_vals[q_trace]) if np.isfinite(abs_pos_resid_vals[q_trace]) else np.nan,
                        abs_vel_resid_mps=float(abs_vel_resid_vals[q_trace]) if np.isfinite(abs_vel_resid_vals[q_trace]) else np.nan,
                        gated=bool(gated_flags[q_trace]),
                        gate_reason=str(gate_reason_vals[q_trace]),
                        gate_reason_abs_pos=bool("abs_pos" in str(gate_reason_vals[q_trace])),
                        gate_reason_abs_vel=bool("abs_vel" in str(gate_reason_vals[q_trace])),
                        gate_reason_pos_std=bool("pos_std" in str(gate_reason_vals[q_trace])),
                        gate_reason_cross=bool("cross" in str(gate_reason_vals[q_trace])),
                        gate_reason_maha=bool("maha" in str(gate_reason_vals[q_trace])),
                        pos_error=float(np.linalg.norm(st["track_pos"][q_trace, :2] - sc_true.tgt_pos[q_trace, :2])),
                        miss_count=int(st["miss"][q_trace]),
                        assignment_changed=bool(not np.array_equal(prev_col, cur_col)),
                        num_receivers=int(z[:, q_trace].sum()),
                        gpda_candidate_count=float(gpda_candidate_count_vals[q_trace]),
                        gpda_best_assoc_prob=float(gpda_best_assoc_prob_vals[q_trace]) if np.isfinite(gpda_best_assoc_prob_vals[q_trace]) else np.nan,
                        gpda_miss_assoc_prob=float(gpda_miss_assoc_prob_vals[q_trace]) if np.isfinite(gpda_miss_assoc_prob_vals[q_trace]) else np.nan,
                        gpda_assoc_entropy=float(gpda_assoc_entropy_vals[q_trace]) if np.isfinite(gpda_assoc_entropy_vals[q_trace]) else np.nan,
                        jpda_joint_event_count=float(jpda_joint_event_count) if np.isfinite(jpda_joint_event_count) else np.nan,
                        jpda_exact_component_frac=float(jpda_exact_component_frac) if np.isfinite(jpda_exact_component_frac) else np.nan,
                        jpda_num_measurements=float(jpda_num_measurements) if np.isfinite(jpda_num_measurements) else np.nan,
                        hysteresis_kept_prev=bool(hys_kept_prev),
                    ))

            pos_err = np.linalg.norm(st["track_pos"][:, :2] - sc_true.tgt_pos[:, :2], axis=1)
            vel_err = np.linalg.norm(st["track_vel"][:, :2] - sc_true.tgt_vel[:, :2], axis=1)
            track_lost = (pos_err > cfg.track_loss_pos_thr_m) | (st["miss"] >= cfg.track_loss_miss_thr)
            gospa_metrics = tracking_gospa_like(cfg, pos_err, track_lost)
            sw = switching_cost(z, st["z_prev"])
            st["z_prev"] = z.copy()
            rows.append(dict(
                mc=mc, frame=frame, method=method,
                mean_track_pos_rmse_m=float(np.sqrt(np.mean(pos_err**2))),
                median_track_pos_error_m=float(np.median(pos_err)),
                p90_track_pos_error_m=float(np.percentile(pos_err, 90)),
                max_track_pos_error_m=float(np.max(pos_err)),
                mean_track_vel_rmse_mps=float(np.sqrt(np.mean(vel_err**2))),
                median_track_vel_error_mps=float(np.median(vel_err)),
                p90_track_vel_error_mps=float(np.percentile(vel_err, 90)),
                track_loss_rate=float(np.mean(track_lost)),
                gospa_pos_m=gospa_metrics["gospa_pos_m"],
                gospa_loc_m=gospa_metrics["gospa_loc_m"],
                gospa_miss_m=gospa_metrics["gospa_miss_m"],
                gospa_miss_count=gospa_metrics["gospa_miss_count"],
                clipped_pos_rmse_m=gospa_metrics["clipped_pos_rmse_m"],
                mean_pd_target=float(np.mean(pd_vec)),
                p05_pd_target=float(np.percentile(pd_vec, 5)),
                detected_target_count=float(np.sum(detected)),
                polluted_measurement_count=float(np.sum(polluted_flags & detected)),
                pollution_expected_count=float(np.sum(pollution_probs[detected])) if np.any(detected) else 0.0,
                pollution_expected_count_detected=float(np.sum(pollution_probs[detected])) if np.any(detected) else 0.0,
                polluted_measurement_rate=float(np.mean(polluted_flags[detected])) if np.any(detected) else 0.0,
                pollution_prob_mean=float(np.mean(pollution_probs[detected])) if np.any(detected) else 0.0,
                pollution_prob_detected_mean=float(np.mean(pollution_probs[detected])) if np.any(detected) else 0.0,
                pollution_prob_detected_over_all_targets=float(np.sum(pollution_probs[detected]) / max(cfg.tracking_Q, 1)),
                pollution_prob_potential_all_mean=float(np.mean(pollution_probs_potential)),
                pollution_prob_all_mean=float(np.mean(pollution_probs)),
                dd_info_weight_mean=float(np.mean(gamma_vals[detected])) if np.any(detected) else np.nan,
                gated_measurement_count=float(np.sum(gated_flags & detected)),
                gated_measurement_rate=float(np.mean(gated_flags[detected])) if np.any(detected) else 0.0,
                abs_pos_gate_count=float(sum(("abs_pos" in str(r)) for r in gate_reason_vals[detected])) if np.any(detected) else 0.0,
                pos_std_gate_count=float(sum(("pos_std" in str(r)) for r in gate_reason_vals[detected])) if np.any(detected) else 0.0,
                cross_gate_count=float(sum(("cross" in str(r)) for r in gate_reason_vals[detected])) if np.any(detected) else 0.0,
                maha_gate_count=float(sum(("maha" in str(r)) for r in gate_reason_vals[detected])) if np.any(detected) else 0.0,
                abs_pos_resid_mean_m=float(np.nanmean(abs_pos_resid_vals[detected])) if np.any(detected) and np.isfinite(abs_pos_resid_vals[detected]).any() else np.nan,
                abs_pos_resid_p95_m=float(np.nanpercentile(abs_pos_resid_vals[detected], 95)) if np.any(detected) and np.isfinite(abs_pos_resid_vals[detected]).any() else np.nan,
                abs_pos_resid_max_m=float(np.nanmax(abs_pos_resid_vals[detected])) if np.any(detected) and np.isfinite(abs_pos_resid_vals[detected]).any() else np.nan,
                innovation_d2_mean=float(np.nanmean(innovation_d2_vals[detected])) if np.any(detected) and np.isfinite(innovation_d2_vals[detected]).any() else np.nan,
                innovation_d2_p95=float(np.nanpercentile(innovation_d2_vals[detected], 95)) if np.any(detected) and np.isfinite(innovation_d2_vals[detected]).any() else np.nan,
                innovation_d2_max=float(np.nanmax(innovation_d2_vals[detected])) if np.any(detected) and np.isfinite(innovation_d2_vals[detected]).any() else np.nan,
                gpda_candidate_count_mean=float(np.nanmean(gpda_candidate_count_vals[detected])) if np.any(detected) and np.isfinite(gpda_candidate_count_vals[detected]).any() else np.nan,
                gpda_best_assoc_prob_mean=float(np.nanmean(gpda_best_assoc_prob_vals[detected])) if np.any(detected) and np.isfinite(gpda_best_assoc_prob_vals[detected]).any() else np.nan,
                gpda_miss_assoc_prob_mean=float(np.nanmean(gpda_miss_assoc_prob_vals[detected])) if np.any(detected) and np.isfinite(gpda_miss_assoc_prob_vals[detected]).any() else np.nan,
                gpda_assoc_entropy_mean=float(np.nanmean(gpda_assoc_entropy_vals[detected])) if np.any(detected) and np.isfinite(gpda_assoc_entropy_vals[detected]).any() else np.nan,
                jpda_joint_event_count=float(jpda_joint_event_count) if np.isfinite(jpda_joint_event_count) else np.nan,
                jpda_exact_component_frac=float(jpda_exact_component_frac) if np.isfinite(jpda_exact_component_frac) else np.nan,
                jpda_num_measurements=float(jpda_num_measurements) if np.isfinite(jpda_num_measurements) else np.nan,
                hysteresis_kept_prev=float(hys_kept_prev),
                hysteresis_obj_new=float(hys_obj_new) if np.isfinite(hys_obj_new) else np.nan,
                hysteresis_obj_prev=float(hys_obj_prev) if np.isfinite(hys_obj_prev) else np.nan,
                hysteresis_threshold=float(hys_threshold) if np.isfinite(hys_threshold) else np.nan,
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
    return {"rows": rows, "trace_rows": trace_rows}


def run_tracking_experiment(cfg: SimConfig, out: Path) -> None:
    # Paper baseline methods — must match _tracking_worker.
    methods = ["random", "global_utility", "box_dd", "dd_only", "proposed"]
    methods.extend(["nearest", "strongest"])
    if cfg.assignment_hysteresis:
        methods.append("collision_aware_hys")
    if cfg.lambda_switch > 0:
        methods.append("collision_aware_switch")
    jobs: List[Tuple[SimConfig, int]] = [(cfg, mc) for mc in range(cfg.tracking_mc)]
    job_results = _run_parallel_jobs(_tracking_worker, jobs, cfg.num_workers, "Tracking")
    records: List[Dict[str, float]] = []
    trace_records: List[Dict[str, float]] = []
    for result in job_results:
        if isinstance(result, dict):
            records.extend(result.get("rows", []))
            trace_records.extend(result.get("trace_rows", []))
        else:  # backward-compatible fallback
            records.extend(result)
    df = pd.DataFrame(records)
    df.to_csv(out / "tracking_trials.csv", index=False)
    if cfg.save_measurement_trace and trace_records:
        pd.DataFrame(trace_records).to_csv(out / "tracking_measurement_trace.csv", index=False)

    # Compact paper-facing summary. The raw tracking_trials.csv remains the full source table.
    agg_cols = {
        "mean_track_pos_rmse_m": "mean",
        "median_track_pos_error_m": "mean",
        "p90_track_pos_error_m": "mean",
        "max_track_pos_error_m": "mean",
        "track_loss_rate": "mean",
        "gospa_pos_m": "mean",
        "gospa_loc_m": "mean",
        "gospa_miss_m": "mean",
        "gospa_miss_count": "mean",
        "clipped_pos_rmse_m": "mean",
        "mean_pd_target": "mean",
        "p05_pd_target": "mean",
        "full_cross_norm_mean": "mean",
        "full_cross_norm_p90": "mean",
        "uncovered_target_count": "mean",
        "detected_target_count": "mean",
        "polluted_measurement_count": "mean",
        "pollution_expected_count": "mean",
        "pollution_expected_count_detected": "mean",
        "polluted_measurement_rate": "mean",
        "pollution_prob_mean": "mean",
        "pollution_prob_detected_mean": "mean",
        "pollution_prob_detected_over_all_targets": "mean",
        "pollution_prob_potential_all_mean": "mean",
        "pollution_prob_all_mean": "mean",
        "dd_info_weight_mean": "mean",
        "gated_measurement_count": "mean",
        "gated_measurement_rate": "mean",
        "abs_pos_gate_count": "mean",
        "pos_std_gate_count": "mean",
        "cross_gate_count": "mean",
        "maha_gate_count": "mean",
        "abs_pos_resid_mean_m": "mean",
        "abs_pos_resid_p95_m": "mean",
        "abs_pos_resid_max_m": "mean",
        "innovation_d2_mean": "mean",
        "innovation_d2_p95": "mean",
        "innovation_d2_max": "mean",
        "gpda_candidate_count_mean": "mean",
        "gpda_best_assoc_prob_mean": "mean",
        "gpda_miss_assoc_prob_mean": "mean",
        "gpda_assoc_entropy_mean": "mean",
        "jpda_joint_event_count": "mean",
        "jpda_exact_component_frac": "mean",
        "jpda_num_measurements": "mean",
        "hysteresis_kept_prev": "mean",
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
        "gospa_pos_m": "mean",
        "gospa_loc_m": "mean",
        "gospa_miss_m": "mean",
        "gospa_miss_count": "mean",
        "clipped_pos_rmse_m": "mean",
        "mean_pd_target": "mean",
        "p05_pd_target": "mean",
        "full_cross_norm_mean": "mean",
        "coverage_ratio": "mean",
        "switch_rate": "mean",
    }).reset_index()
    if cfg.save_diagnostics:
        trial_metrics.to_csv(out / "tracking_trial_level_metrics.csv", index=False)
    # Paired diagnostics: proposed vs dd_only (the central tracking comparison).
    # Also produce proposed vs strongest as a sanity check.
    paired_df = pd.DataFrame()
    for ref_method, ref_label in [("dd_only", "dd_only"), ("strongest", "strongest")]:
        if not {"proposed", ref_method}.issubset(set(trial_metrics["method"])):
            continue
        ca = trial_metrics[trial_metrics.method == "proposed"].set_index("mc")
        st = trial_metrics[trial_metrics.method == ref_method].set_index("mc")
        common = sorted(set(ca.index).intersection(st.index))
        paired_rows = []
        for metric in ["mean_pd_target", "p05_pd_target", "mean_track_pos_rmse_m", "median_track_pos_error_m", "p90_track_pos_error_m", "gospa_pos_m", "gospa_loc_m", "gospa_miss_m", "gospa_miss_count", "track_loss_rate"]:
            delta = ca.loc[common, metric].to_numpy() - st.loc[common, metric].to_numpy()
            lower_is_better = metric not in ("mean_pd_target", "p05_pd_target")
            if lower_is_better:
                win = np.mean(delta < 0)
            else:
                win = np.mean(delta > 0)
            paired_rows.append(dict(
                metric=metric,
                reference=ref_label,
                delta_mean=float(np.nanmean(delta)),
                delta_median=float(np.nanmedian(delta)),
                win_rate_proposed=float(win),
                n_pairs=len(common),
            ))
        paired_df = pd.concat([paired_df, pd.DataFrame(paired_rows)], ignore_index=True)
    if cfg.save_diagnostics and not paired_df.empty:
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
            print("\n[Tracking] Paired diagnostics: proposed vs dd_only / strongest")
            print(paired_df.to_string(index=False))

    for metric, ylabel, fname in [
        ("mean_track_pos_rmse_m", "Mean track position RMSE (m)", "tracking_pos_rmse_time.png"),
        ("p90_track_pos_error_m", "p90 track position error (m)", "tracking_p90_pos_error_time.png"),
        ("max_track_pos_error_m", "Mean of per-trial max position error (m)", "tracking_max_pos_error_time.png"),
        ("gospa_pos_m", "GOSPA-like position score (m)", "tracking_gospa_time.png"),
        ("track_loss_rate", "Track loss rate", "tracking_loss_rate_time.png"),
        ("switch_rate", "Assignment switch rate", "tracking_switch_rate_time.png"),
        ("p05_pd_target", "p05 target $P_D$", "tracking_p05_pd_time.png"),
        ("uncovered_target_count", "Uncovered targets", "tracking_uncovered_targets_time.png"),
        ("polluted_measurement_rate", "Polluted measurement rate", "tracking_polluted_rate_time.png"),
        ("gated_measurement_rate", "Innovation-gated measurement rate", "tracking_gated_rate_time.png"),
        ("hysteresis_kept_prev", "Hysteresis keep-previous rate", "tracking_hysteresis_rate_time.png"),
    ]:
        if metric not in summary.columns:
            continue
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
    """Two-regime Gate-2 density label.

    Earlier versions used a three-regime scheme with a dense-collision threshold
    that was unreachable for the simulated deployments.  Gate 2 now separates
    low-collision deployments from sparse-collision deployments and reports
    right-skew / concentration of collision *strength* separately.
    """
    if rho < cfg.regime_i_max:
        return "I_low_collision"
    return "II_sparse_collision"


def _pooled_hill_alpha_from_values(values: np.ndarray, frac: float) -> Tuple[float, int, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 20:
        return float("nan"), int(x.size), float("nan")
    xs = np.sort(x)
    k = max(2, int(math.ceil(float(frac) * xs.size)))
    k = min(k, xs.size - 1)
    tail = xs[-k:]
    threshold = xs[-k - 1]
    if threshold <= 0:
        return float("nan"), int(k), float(threshold)
    inv_alpha = float(np.mean(np.log((tail + EPS) / (threshold + EPS))))
    alpha = float(1.0 / inv_alpha) if inv_alpha > EPS else float("inf")
    return alpha, int(k), float(threshold)


def _write_gate2_pooled_hill_summary(edge_records: List[float], out: Path) -> None:
    vals = np.asarray(edge_records, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    rows = []
    for frac in (0.05, 0.10, 0.20):
        alpha, k, threshold = _pooled_hill_alpha_from_values(vals, frac)
        rows.append(dict(
            tail_frac=frac,
            n_edges=int(vals.size),
            k_tail=k,
            threshold_C_norm=threshold,
            pooled_hill_alpha=alpha,
        ))
    pd.DataFrame(rows).to_csv(out / "gate2_pooled_hill_summary.csv", index=False)


def run_gate2(cfg: SimConfig, out: Path) -> None:
    base_rng = make_rng(cfg.seed + 1000)
    records = []
    pooled_cnorm: List[float] = []
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
                            if len(edges) and "C_norm" in edges.columns:
                                pooled_cnorm.extend(edges["C_norm"].to_numpy(dtype=float).tolist())
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
        max_C_norm=("max_C_norm", "mean"),
        skew_C_norm_mean=("skew_C_norm", "mean"),
        excess_kurt_C_norm_mean=("excess_kurt_C_norm", "mean"),
        max_to_mean_C_norm_mean=("max_to_mean_C_norm", "mean"),
        strength_top_frac_mean=("eta_top_norm", "mean"),
        N_DD_mean=("N_DD", "mean"),
        N_DDA_mean=("N_DDA", "mean"),
        dda_over_dd_mean=("dda_over_dd", "mean"),
        angle_removed_frac_mean=("angle_removed_frac", "mean"),
        alias_ratio_mean=("doppler_alias_ratio", "mean"),
        sparse_collision_frac=("regime", lambda s: float(np.mean(s == "II_sparse_collision"))),
        low_collision_frac=("regime", lambda s: float(np.mean(s == "I_low_collision"))),
    ).reset_index()
    summary.to_csv(out / "gate2_regime_summary.csv", index=False)

    strength_summary = pd.DataFrame([dict(
        rho_col_mean=float(df["rho_col"].mean()) if len(df) else 0.0,
        rho_col_p95=float(np.percentile(df["rho_col"], 95)) if len(df) else 0.0,
        rho_col_max=float(df["rho_col"].max()) if len(df) else 0.0,
        C_norm_skew_mean=float(df["skew_C_norm"].mean()) if len(df) else 0.0,
        C_norm_excess_kurt_mean=float(df["excess_kurt_C_norm"].mean()) if len(df) else 0.0,
        C_norm_max_to_mean=float(df["max_to_mean_C_norm"].mean()) if len(df) else 0.0,
        top_frac_energy_mean=float(df["eta_top_norm"].mean()) if len(df) else 0.0,
        dda_over_dd_mean=float(df["dda_over_dd"].mean()) if len(df) else 0.0,
        angle_removed_frac_mean=float(df["angle_removed_frac"].mean()) if len(df) else 0.0,
    )])
    strength_summary.to_csv(out / "gate2_strength_tail_summary.csv", index=False)
    _write_gate2_pooled_hill_summary(pooled_cnorm, out)

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

    # Two-regime fractions vs sigma for largest M,Q.
    sub2 = summary[(summary.M == max(Ms)) & (summary.Q == max(Qs)) & (summary.K_tgt_min == max(ktmins)) & (summary.K_uav_max == max(kumaxs))]
    if len(sub2):
        sub2 = sub2.sort_values("sigma_theta_deg")
        labels = ["inf" if x >= 900 else f"{x:g}" for x in sub2.sigma_theta_deg]
        x = np.arange(len(sub2))
        plt.figure(figsize=(7, 4.5))
        b1 = sub2["low_collision_frac"].values
        b2 = sub2["sparse_collision_frac"].values
        plt.bar(x, b1, label="Low collision")
        plt.bar(x, b2, bottom=b1, label="Sparse collision")
        plt.xticks(x, labels)
        plt.xlabel(r"Angle resolution $\sigma_\theta$ (deg)")
        plt.ylabel("Trial fraction")
        plt.title(f"Gate 2: Collision-density regimes, M={max(Ms)}, Q={max(Qs)}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "gate2_regime_fraction_vs_angle.png", dpi=cfg.dpi)
        plt.close()

    # Strength-tail / concentration diagnostics.
    if len(df):
        plt.figure(figsize=(6.3, 4.5))
        vals = np.asarray(df["mean_C_norm"], dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            plt.hist(np.log10(vals + EPS), bins=30, alpha=0.75)
        plt.xlabel(r"$\log_{10}$ mean normalized collision strength")
        plt.ylabel("Trial count")
        plt.title("Gate 2: collision strength is right-skewed, not density-heavy-tailed")
        plt.tight_layout()
        plt.savefig(out / "gate2_cnorm_strength_hist.png", dpi=cfg.dpi)
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
