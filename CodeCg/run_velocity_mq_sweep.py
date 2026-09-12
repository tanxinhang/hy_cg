#!/usr/bin/env python3
"""Velocity sweep + high-density M/Q scan, using existing experiment infrastructure.

1. Velocity sweep (single-frame Gate1): 0,20,50,100,150 m/s with paper baselines.
2. High-density Gate2: Q=10,12,15,18 with M=6,9, covering sparse through dense regimes.
"""
from __future__ import annotations

import sys, os, math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from gate_otfs_collision.config import SimConfig, parse_float_list, parse_int_list
from gate_otfs_collision.scenario import make_rng, generate_clustered_scenario
from gate_otfs_collision.assignment import (
    assignment_collision_aware_greedy,
    assignment_global_utility,
    assignment_random,
    assignment_nearest,
    assignment_strongest,
    build_visibility,
    build_collision_edges,
    collision_summary,
    task_edge_list,
    is_feasible,
    objective,
)
from gate_otfs_collision.kernels import (
    bistatic_delay_doppler,
    stable_method_offset,
)
from gate_otfs_collision.detection import evaluate_assignment_full_link
from gate_otfs_collision.experiments import _make_gate1_jittered_scenario, _run_parallel_jobs


def _set_target_speed(sc, speed: float, rng):
    """Rescale target velocities to `speed` while preserving direction."""
    sc = deepcopy(sc)
    for q in range(sc.Q):
        cur = float(np.linalg.norm(sc.tgt_vel[q, :2]))
        if cur > 1e-9:
            sc.tgt_vel[q, :2] = sc.tgt_vel[q, :2] / cur * speed
        else:
            ang = rng.uniform(0, 2 * np.pi)
            sc.tgt_vel[q, :2] = speed * np.array([np.cos(ang), np.sin(ang)])
    return sc


def _paper_assignments_for_scene(cfg, sc, rng):
    """Return dict of paper-baseline assignments for a single scene."""
    _, u, v = build_visibility(cfg, sc)
    out = {}
    out["random"] = assignment_random(cfg, u, v, rng)
    out["global_utility"] = assignment_global_utility(cfg, u, v)
    # box_dd
    box_cfg = deepcopy(cfg)
    box_cfg.proxy_kernel_mode = "no_spread"
    out["box_dd"] = assignment_collision_aware_greedy(box_cfg, sc, u, v)
    # dd_only
    dd_cfg = deepcopy(cfg)
    dd_cfg.sigma_theta_deg = float("inf")
    out["dd_only"] = assignment_collision_aware_greedy(dd_cfg, sc, u, v)
    # proposed
    out["proposed"] = assignment_collision_aware_greedy(cfg, sc, u, v)
    return out, u, v


def _single_velocity_worker(job):
    """Worker for one (speed, mc) combination."""
    cfg, speed, mc = job
    rng = make_rng(cfg.seed + 6000 + 100 * int(speed) + mc)
    local = SimConfig(**asdict(cfg))
    local.target_speed_mps = max(float(speed), 0.0)
    local.eval_kernel_mode = cfg.eval_kernel_mode
    base = _make_gate1_jittered_scenario(local, rng)
    sc = _set_target_speed(base, speed, rng)
    z_by_method, u, v = _paper_assignments_for_scene(local, sc, rng)

    rows = []
    for method, z in z_by_method.items():
        eval_rng = make_rng(cfg.seed + 6500 + 1000 * mc + 100 * int(speed) + stable_method_offset(method))
        full = evaluate_assignment_full_link(local, sc, z, eval_rng)
        edges_proxy = build_collision_edges(local, sc, z, use_full_kernel=False)
        summ = collision_summary(local, edges_proxy, len(task_edge_list(z)))
        vals, aliases = [], []
        for i in range(sc.M):
            for j in range(sc.M):
                for q in range(sc.Q):
                    _, nu = bistatic_delay_doppler(local, sc, i, j, q)
                    vals.append(abs(nu))
                    aliases.append(float(abs(nu) > local.doppler_max_hz))
        row = dict(mc=mc, speed=speed, method=method,
                   feasible=is_feasible(local, z, v),
                   coverage_ratio=float(np.mean(z.sum(axis=0) >= local.k_tgt_min)),
                   num_task_edges=int(len(task_edge_list(z))),
                   mean_abs_doppler_hz=float(np.mean(vals)) if vals else 0.0,
                   max_abs_doppler_hz=float(np.max(vals)) if vals else 0.0,
                   doppler_alias_ratio=float(np.mean(aliases)) if aliases else 0.0)
        row.update(summ)
        row.update(full)
        rows.append(row)
    return {"rows": rows}


def run_velocity_sweep(cfg: SimConfig, out: Path):
    speeds = parse_float_list(cfg.velocity_list_mps)
    jobs = []
    for speed in speeds:
        for mc in range(cfg.velocity_mc):
            jobs.append((cfg, speed, mc))
    print(f"Velocity sweep: {len(speeds)} speeds x {cfg.velocity_mc} MCs = {len(jobs)} jobs, workers={cfg.num_workers}")
    results = _run_parallel_jobs(_single_velocity_worker, jobs, cfg.num_workers, "Velocity")
    all_rows = []
    for r in results:
        if isinstance(r, dict):
            all_rows.extend(r.get("rows", []))
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "velocity_paper_trials.csv", index=False)
    summary = df.groupby(["speed", "method"]).agg(
        pd_mean=("pd_mean", "mean"), pd_p05=("pd_p05", "mean"),
        full_cross_norm_mean=("full_cross_norm_mean", "mean"),
        collision_cost_norm=("collision_cost_norm", "mean"),
        mean_abs_doppler_hz=("mean_abs_doppler_hz", "mean"),
        max_abs_doppler_hz=("max_abs_doppler_hz", "mean"),
        doppler_alias_ratio=("doppler_alias_ratio", "mean"),
        coverage_ratio=("coverage_ratio", "mean"),
    ).reset_index()
    summary.to_csv(out / "velocity_paper_summary.csv", index=False)
    print("\n=== Velocity Sweep Summary ===")
    for speed in speeds:
        sub = summary[summary.speed == speed]
        print(f"\nSpeed={speed} m/s:")
        for _, r in sub.iterrows():
            print(f"  {r['method']:<20s} Pd={r['pd_mean']:.4f}  cross={r['full_cross_norm_mean']:.4f}  alias={r['doppler_alias_ratio']:.3f}")


def run_high_density_gate2(cfg: SimConfig, out: Path):
    """Gate2 scan with extended Q range to push collision density."""
    base_rng = make_rng(cfg.seed + 1000)
    records = []
    pooled_cnorm = []
    Ms = parse_int_list(cfg.m_list)
    Qs = parse_int_list(cfg.q_list)
    ktmins = parse_int_list(cfg.ktmin_list)
    kumaxs = parse_int_list(cfg.kumax_list)
    sigmas = parse_float_list(cfg.sigma_theta_list)

    total = len(Ms) * len(Qs) * len(ktmins) * len(kumaxs) * len(sigmas) * cfg.gate2_mc
    print(f"Gate2 HD scan: {total} total iterations, {cfg.gate2_mc} MCs per config")

    n = 0
    for M in Ms:
        for Q in Qs:
            for kt in ktmins:
                for ku in kumaxs:
                    for sig in sigmas:
                        for mc in range(cfg.gate2_mc):
                            n += 1
                            seed = int(base_rng.integers(0, 2**31 - 1))
                            rng = make_rng(seed)
                            local = SimConfig(**asdict(cfg))
                            local.k_tgt_min = kt
                            local.k_uav_max = ku
                            local.sigma_theta_deg = sig
                            sc = generate_clustered_scenario(local, M, Q, rng)
                            _, U, v = build_visibility(local, sc)
                            z = assignment_strongest(local, U, v)
                            edges = build_collision_edges(local, sc, z, use_full_kernel=False)
                            if len(edges) and "C_norm" in edges.columns:
                                pooled_cnorm.extend(edges["C_norm"].to_numpy(dtype=float).tolist())
                            summ = collision_summary(local, edges, len(task_edge_list(z)))
                            coverage = float(np.mean(z.sum(axis=0) >= local.k_tgt_min))
                            row = dict(M=M, Q=Q, K_tgt_min=kt, K_uav_max=ku,
                                       sigma_theta_deg=(999.0 if math.isinf(sig) else sig), mc=mc,
                                       coverage_ratio=coverage)
                            row.update(summ)
                            records.append(row)
                            if n % 200 == 0:
                                print(f"  [{n}/{total}] M={M} Q={Q} kt={kt} ku={ku} sig={sig}")

    df = pd.DataFrame(records)
    df.to_csv(out / "gate2_hdensity_trials.csv", index=False)
    group_cols = ["M", "Q", "K_tgt_min", "K_uav_max", "sigma_theta_deg"]
    summary = df.groupby(group_cols).agg(
        rho_col_mean=("rho_col", "mean"), rho_col_std=("rho_col", "std"),
        eta_top_mean=("eta_top", "mean"), eta_top_norm_mean=("eta_top_norm", "mean"),
        coverage_mean=("coverage_ratio", "mean"),
        mean_C_norm=("mean_C_norm", "mean"), p95_C_norm=("p95_C_norm", "mean"),
        N_DD_mean=("N_DD", "mean"), N_DDA_mean=("N_DDA", "mean"),
        angle_removed_frac_mean=("angle_removed_frac", "mean"),
    ).reset_index()
    summary.to_csv(out / "gate2_hdensity_summary.csv", index=False)

    # Show key rows: high Q, low M
    high_q = summary[summary.Q >= 15].sort_values(["M", "Q", "sigma_theta_deg"])
    print("\n=== High-Density Gate2 (Q>=15) ===")
    print(high_q[["M", "Q", "sigma_theta_deg", "rho_col_mean", "mean_C_norm", "angle_removed_frac_mean"]].to_string(index=False))


def main():
    cfg = SimConfig()
    out = Path("./outputs_velocity_and_mq")

    import shutil
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pd.DataFrame([asdict(cfg)]).to_csv(out / "config.csv", index=False)

    if "--velocity" in sys.argv or "--all" in sys.argv or len(sys.argv) == 1:
        run_velocity_sweep(cfg, out)
    if "--gate2" in sys.argv or "--all" in sys.argv or len(sys.argv) == 1:
        run_high_density_gate2(cfg, out)
    print(f"\nAll outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
