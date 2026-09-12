#!/usr/bin/env python3
"""Velocity sweep with tracking: proposed vs dd_only at 0,20,50,100,150 m/s.

For each target speed, regenerates the scenario, rescales target velocities, runs
both proposed and dd_only through the full tracking loop, and reports per-speed
Pd / GOSPA / track-loss comparisons.

Uses Dirichlet kernels + multi-worker parallelism for speed.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from gate_otfs_collision.config import SimConfig, EPS, parse_float_list
from gate_otfs_collision.scenario import make_rng, generate_clustered_scenario
from gate_otfs_collision.assignment import (
    assignment_collision_aware_greedy,
    build_visibility,
    task_edge_list,
    switching_cost,
)
from gate_otfs_collision.kernels import stable_method_offset
from gate_otfs_collision.experiments import (
    evaluate_assignment_targets,
    _tracking_predict_filter,
    tracking_gospa_like,
    _run_parallel_jobs,
    propagate_scene,
    copy_scene_with_targets,
)


def _velocity_tracking_worker(job):
    """One MC trial at a given target speed for both proposed and dd_only."""
    cfg, mc, speed, methods = job
    rng = make_rng(cfg.seed + 90000 + 1000 * int(speed) + mc)
    sc_true = generate_clustered_scenario(cfg, cfg.tracking_M, cfg.tracking_Q, rng)
    # Rescale target speeds
    for q in range(sc_true.Q):
        cur_spd = np.linalg.norm(sc_true.tgt_vel[q, :2])
        if cur_spd > EPS:
            sc_true.tgt_vel[q, :2] = sc_true.tgt_vel[q, :2] / cur_spd * speed
        else:
            angle = rng.uniform(0, 2 * np.pi)
            sc_true.tgt_vel[q, :2] = speed * np.array([np.cos(angle), np.sin(angle)])
    true0_pos = sc_true.tgt_pos.copy()
    true0_vel = sc_true.tgt_vel.copy()

    rows = []
    states = {}
    for method in methods:
        rng_init = make_rng(cfg.seed + 91000 + 1000 * int(speed) + 100 * mc + stable_method_offset(method))
        init_pos = true0_pos + rng_init.normal(0.0, cfg.track_init_pos_std_m, size=true0_pos.shape)
        init_vel = true0_vel + rng_init.normal(0.0, cfg.track_init_vel_std_mps, size=true0_vel.shape)
        state_dict = dict(
            track_pos=init_pos.copy(), track_vel=init_vel.copy(),
            miss=np.zeros(cfg.tracking_Q, dtype=int), z_prev=None,
        )
        state_dict["track_pos"][:, 2] = cfg.target_alt_m
        state_dict["track_vel"][:, 2] = 0.0
        states[method] = state_dict

    for frame in range(cfg.num_frames):
        rng_frame = make_rng(cfg.seed + 92000 + 10000 * mc + frame)
        if frame > 0:
            sc_true = propagate_scene(cfg, sc_true, rng_frame)

        rng_crn = make_rng(cfg.seed + 92500 + 10000 * mc + frame)
        pred_pos_eps = rng_crn.normal(0.0, 1.0, size=sc_true.tgt_pos.shape)
        pred_vel_eps = rng_crn.normal(0.0, 1.0, size=sc_true.tgt_vel.shape)
        det_uniform = rng_crn.random(cfg.tracking_Q)
        meas_pos_eps = rng_crn.normal(0.0, 1.0, size=sc_true.tgt_pos.shape)
        meas_vel_eps = rng_crn.normal(0.0, 1.0, size=sc_true.tgt_vel.shape)

        for method in methods:
            st = states[method]
            rng_assign = make_rng(cfg.seed + 93000 + 10000 * mc + 100 * frame + stable_method_offset(method))
            pred_pos, pred_vel, _, _ = _tracking_predict_filter(cfg, st, pred_pos_eps, pred_vel_eps, rng_assign)
            sc_pred = copy_scene_with_targets(sc_true, pred_pos, pred_vel)
            _, U, v = build_visibility(cfg, sc_pred)
            # Assign: proposed = default, dd_only = sigma_theta=inf
            if method == "dd_only":
                assign_cfg = deepcopy(cfg)
                assign_cfg.sigma_theta_deg = float("inf")
            else:
                assign_cfg = cfg
            z = assignment_collision_aware_greedy(assign_cfg, sc_pred, U, v)

            rng_eval = make_rng(cfg.seed + 93500 + 10000 * mc + 100 * frame + stable_method_offset(method))
            tgt = evaluate_assignment_targets(cfg, sc_true, z, rng_eval)
            tgt_sorted = tgt.sort_values("target")
            pd_vec = np.nan_to_num(tgt_sorted["pd_target"].to_numpy(dtype=float), nan=0.0)
            pos_std_vec = tgt_sorted["rmse_pos_proxy_m"].to_numpy(dtype=float)
            vel_std_vec = tgt_sorted["rmse_vel_proxy_mps"].to_numpy(dtype=float)
            pos_std_vec = np.where(np.isfinite(pos_std_vec), pos_std_vec, cfg.track_loss_pos_thr_m)
            vel_std_vec = np.where(np.isfinite(vel_std_vec), vel_std_vec, 10.0 * cfg.track_meas_vel_floor_mps)

            detected = det_uniform < pd_vec
            cross_vec = tgt_sorted["full_cross_norm_mean"].to_numpy(dtype=float)
            covered = z.sum(axis=0) > 0

            for q in range(cfg.tracking_Q):
                was_lost = (np.linalg.norm(st["track_pos"][q, :2] - sc_true.tgt_pos[q, :2]) > cfg.track_loss_pos_thr_m) or (st["miss"][q] >= cfg.track_loss_miss_thr)
                if detected[q]:
                    meas_pos = sc_true.tgt_pos[q] + pos_std_vec[q] * meas_pos_eps[q]
                    meas_vel = sc_true.tgt_vel[q] + vel_std_vec[q] * meas_vel_eps[q]
                    meas_pos[2] = cfg.target_alt_m
                    meas_vel[2] = 0.0
                    if was_lost and cfg.tracking_reinit_lost:
                        st["track_pos"][q] = meas_pos.copy()
                        st["track_vel"][q] = meas_vel.copy()
                        st["miss"][q] = 0
                    else:
                        alpha_p = cfg.tracking_alpha_pos
                        alpha_v = cfg.tracking_alpha_vel
                        st["track_pos"][q] = alpha_p * st["track_pos"][q] + (1 - alpha_p) * meas_pos
                        st["track_vel"][q] = alpha_v * st["track_vel"][q] + (1 - alpha_v) * meas_vel
                        st["miss"][q] = 0
                else:
                    st["miss"][q] += 1

            pos_err = np.linalg.norm(st["track_pos"][:, :2] - sc_true.tgt_pos[:, :2], axis=1)
            track_lost = (pos_err > cfg.track_loss_pos_thr_m) | (st["miss"] >= cfg.track_loss_miss_thr)
            gospa_metrics = tracking_gospa_like(cfg, pos_err, track_lost)

            rows.append(dict(
                mc=mc, frame=frame, method=method, speed=speed,
                mean_track_pos_rmse_m=float(np.sqrt(np.mean(pos_err**2))),
                median_track_pos_error_m=float(np.median(pos_err)),
                p90_track_pos_error_m=float(np.percentile(pos_err, 90)),
                max_track_pos_error_m=float(np.max(pos_err)),
                track_loss_rate=float(np.mean(track_lost)),
                gospa_pos_m=gospa_metrics["gospa_pos_m"],
                gospa_loc_m=gospa_metrics["gospa_loc_m"],
                gospa_miss_m=gospa_metrics["gospa_miss_m"],
                gospa_miss_count=gospa_metrics["gospa_miss_count"],
                mean_pd_target=float(np.mean(pd_vec)),
                p05_pd_target=float(np.percentile(pd_vec, 5)),
                full_cross_norm_mean=float(np.nanmean(np.where(np.isfinite(cross_vec), cross_vec, np.nan))),
                coverage_ratio=float(np.mean(covered)),
            ))
    return {"rows": rows}


def main():
    cfg = SimConfig()
    cfg.tracking_mc = 20
    cfg.num_frames = 30
    cfg.num_workers = 10
    cfg.inner_mc = 100  # reduced for speed; Pd mean is unbiased
    speeds = [0, 20, 50, 100, 150]
    methods = ["proposed", "dd_only"]
    out_dir = Path("./outputs_velocity_tracking")

    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    jobs = []
    for speed in speeds:
        for mc in range(cfg.tracking_mc):
            jobs.append((cfg, mc, speed, methods))

    print(f"Running velocity sweep with tracking: {len(speeds)} speeds x {cfg.tracking_mc} MCs x {len(methods)} methods = {len(jobs)} jobs")
    print(f"Workers: {cfg.num_workers}")

    results = _run_parallel_jobs(_velocity_tracking_worker, jobs, cfg.num_workers, "Velocity+Tracking")

    all_rows = []
    for r in results:
        if isinstance(r, dict):
            all_rows.extend(r.get("rows", []))
        else:
            all_rows.extend(r)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "velocity_tracking_trials.csv", index=False)

    # Summary: final frame per speed per method
    final = df[df.frame == df.frame.max()]
    summary = final.groupby(["speed", "method"]).agg(
        gospa_pos_m=("gospa_pos_m", "mean"),
        gospa_loc_m=("gospa_loc_m", "mean"),
        gospa_miss_m=("gospa_miss_m", "mean"),
        gospa_miss_count=("gospa_miss_count", "mean"),
        track_loss_rate=("track_loss_rate", "mean"),
        mean_track_pos_rmse_m=("mean_track_pos_rmse_m", "mean"),
        mean_pd_target=("mean_pd_target", "mean"),
        p05_pd_target=("p05_pd_target", "mean"),
        full_cross_norm_mean=("full_cross_norm_mean", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "velocity_tracking_summary.csv", index=False)

    print("\n=== Velocity Sweep with Tracking: Final Frame Summary ===")
    print(summary.to_string(index=False))

    # Paired comparison: proposed - dd_only at each speed
    print("\n=== proposed - dd_only at each speed ===")
    for speed in speeds:
        sub = final[final.speed == speed]
        prop = sub[sub.method == "proposed"]
        dd = sub[sub.method == "dd_only"]
        if len(prop) and len(dd):
            print(f"\nSpeed={speed} m/s:")
            for col in ["gospa_pos_m", "gospa_loc_m", "gospa_miss_m", "track_loss_rate", "mean_pd_target"]:
                pv = prop[col].mean()
                dv = dd[col].mean()
                print(f"  {col}: proposed={pv:.4f}  dd_only={dv:.4f}  delta={pv-dv:+.4f}")

    print(f"\nSaved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
