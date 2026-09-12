from __future__ import annotations

import argparse
import sys
import json
import shutil
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

from .config import *


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate experiments for OTFS DD task-edge collision modeling (V18 shared-measurement JPDA tracking backend)")
    parser.add_argument("--gate",
                        choices=["gate1", "gate2", "both", "difficulty", "phase", "kernel", "velocity", "angle", "pred",
                                 "tracking", "otfs", "supplement", "paper", "all"], default="both")
    parser.add_argument("--out-dir", default="./gate_otfs_collision_outputs")
    parser.add_argument("--fresh-out-dir", action="store_true",
                        help="Delete out-dir before running, preventing mixed old/new CSV and figure outputs")
    parser.add_argument("--save-diagnostics", action="store_true",
                        help="Save verbose diagnostic CSV files in addition to trials/summary outputs")
    parser.add_argument("--save-all-detectors", action="store_true",
                        help="Compute and save all detector Pd variants; otherwise only the selected primary detector is evaluated")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gate1-mc", type=int, default=10)
    parser.add_argument("--gate2-mc", type=int, default=5)
    parser.add_argument("--difficulty-mc", type=int, default=10,
                        help="MC trials per noise-scale point in the difficulty sweep")
    parser.add_argument("--phase-mc", type=int, default=10, help="MC trials per phase mode in the phase-model sweep")
    parser.add_argument("--kernel-mc", type=int, default=10,
                        help="MC trials per kernel mode in the OTFS-kernel ablation")
    parser.add_argument("--velocity-mc", type=int, default=10,
                        help="MC trials per speed point in the velocity/Doppler sweep")
    parser.add_argument("--angle-mc", type=int, default=10,
                        help="MC trials per angle-resolution point in the angle-usefulness gate")
    parser.add_argument("--pred-mc", type=int, default=10,
                        help="MC trials per prediction-error point in the prediction robustness gate")
    parser.add_argument("--tracking-mc", type=int, default=5, help="MC trials in the multi-frame tracking gate")
    parser.add_argument("--inner-mc", type=int, default=200)
    parser.add_argument("--n-tau", type=int, default=64)
    parser.add_argument("--n-nu", type=int, default=32)
    parser.add_argument("--sigma-theta-deg", type=float, default=20.0)
    parser.add_argument("--sigma-theta-list", default="10,20,40,inf")
    parser.add_argument("--difficulty-noise-scales", default="0.3,1,3,10,30",
                        help="Comma-separated multiplicative scales applied to snr_floor for difficulty sweep")
    parser.add_argument("--phase-mode-list", default="coherent,random_phase,noncoherent_power",
                        help="Comma-separated cooperative phase modes for phase sweep")
    parser.add_argument("--kernel-eval-modes", default="no_spread,gaussian,dirichlet,actual_otfs",
                        help="Comma-separated DD-kernel modes for OTFS role ablation")
    parser.add_argument("--velocity-list-mps", default="0,20,50,100,150",
                        help="Comma-separated target speeds for velocity/Doppler sweep")
    parser.add_argument("--pred-pos-sigma-list", default="0,5,10,20,50",
                        help="Comma-separated target position prediction-error std values in meters")
    parser.add_argument("--pred-vel-sigma-list", default="0,1,3,5,10",
                        help="Comma-separated target velocity prediction-error std values in m/s")
    parser.add_argument("--tracking-M", type=int, default=15, help="Number of UAVs for the closed-loop tracking gate")
    parser.add_argument("--tracking-Q", type=int, default=10,
                        help="Number of targets for the closed-loop tracking gate")
    parser.add_argument("--num-frames", type=int, default=30, help="Number of frames in the closed-loop tracking gate")
    parser.add_argument("--dt-frame", type=float, default=0.2, help="Frame interval in seconds for tracking gate")
    parser.add_argument("--lambda-switch", type=float, default=0.0,
                        help="Assignment switching penalty used by collision_aware_switch in tracking gate")
    parser.add_argument("--no-tracking-crn", action="store_true",
                        help="Disable common-random-number tracking primitives; mainly for ablation/debugging")
    parser.add_argument("--target-pd-fusion", choices=["mean", "or"], default="mean",
                        help="Target-level Pd fusion in tracking gate: conservative mean or noncoherent OR fusion")
    parser.add_argument("--tracking-reinit-lost", action="store_true",
                        help="Reinitialize a lost track when it is detected again")
    parser.add_argument("--tracking-alpha-pos", type=float, default=0.65,
                        help="Alpha-filter position update gain in tracking gate")
    parser.add_argument("--tracking-alpha-vel", type=float, default=0.55,
                        help="Alpha-filter velocity update gain in tracking gate")
    parser.add_argument("--tracker-mode",
                        choices=["alpha", "kf", "dd_rwif", "pda", "gpda", "dd_pda", "dd_gpda", "jpda", "jpda_shared",
                                 "dd_jpda", "dd_jpda_shared"], default="alpha",
                        help="Tracking update model: alpha filter, standard CV-KF, DD-RWIF, single-track PDA/GPDA-like, or shared-measurement JPDA-like update")
    parser.add_argument("--meas-pollution", choices=["none", "biased_peak", "clutter_peak"], default="none",
                        help="Structured DD-collision measurement pollution model for V12 falsification tests")
    parser.add_argument("--pollution-pmax", type=float, default=0.35,
                        help="Max probability of structured DD measurement pollution")
    parser.add_argument("--pollution-c0", type=float, default=3.0, help="Ccross scale in p_pollute=pmax*C/(C+C0)")
    parser.add_argument("--pollution-pos-bias-m", type=float, default=80.0,
                        help="Position bias magnitude for biased_peak pollution")
    parser.add_argument("--pollution-vel-bias-mps", type=float, default=8.0,
                        help="Velocity bias magnitude for biased_peak pollution")
    parser.add_argument("--dd-weight-alpha", type=float, default=1.0,
                        help="Pd exponent used by DD-RWIF reliability weight")
    parser.add_argument("--dd-weight-beta", type=float, default=0.15,
                        help="Ccross exponent factor exp(-beta*C) used by DD-RWIF")
    parser.add_argument("--dd-weight-gamma-min", type=float, default=0.05,
                        help="Minimum information weight for DD-RWIF detected measurements")
    parser.add_argument("--kf-process-pos-std-m", type=float, default=1.0, help="CV-KF process position std per frame")
    parser.add_argument("--kf-process-vel-std-mps", type=float, default=0.4,
                        help="CV-KF process velocity std per frame")
    parser.add_argument("--assignment-hysteresis", action="store_true",
                        help="Apply temporal hysteresis to collision-aware assignments")
    parser.add_argument("--hys-rel-margin", type=float, default=0.0,
                        help="Relative objective margin for assignment hysteresis")
    parser.add_argument("--hys-abs-margin", type=float, default=0.0,
                        help="Absolute objective margin for assignment hysteresis")
    parser.add_argument("--no-hys-force-coverage", action="store_true",
                        help="Do not override hysteresis when new assignment improves coverage")
    parser.add_argument("--kf-innovation-gate", action="store_true",
                        help="Enable Mahalanobis innovation gating for KF/DD-RWIF updates")
    parser.add_argument("--kf-gate-chi2", type=float, default=13.3,
                        help="Chi-square threshold for robust innovation gating; 13.3 is roughly 99%% for effective 4-D xy/vx-vy updates")
    parser.add_argument("--kf-outlier-r-inflate", type=float, default=50.0,
                        help="Measurement covariance inflation factor for gated outliers")
    parser.add_argument("--kf-gate-action", choices=["inflate", "skip"], default="inflate",
                        help="Action when the innovation gate is exceeded")
    parser.add_argument("--kf-abs-pos-gate-m", type=float, default=0.0,
                        help="Direct xy-position residual gate in meters; <=0 disables. Useful for high-variance biased peaks")
    parser.add_argument("--kf-abs-vel-gate-mps", type=float, default=0.0,
                        help="Direct xy-velocity residual gate in m/s; <=0 disables")
    parser.add_argument("--kf-gate-pos-std-cap-m", type=float, default=0.0,
                        help="Cap pos_std only inside Mahalanobis gate; <=0 disables")
    parser.add_argument("--kf-gate-vel-std-cap-mps", type=float, default=0.0,
                        help="Cap vel_std only inside Mahalanobis gate; <=0 disables")
    parser.add_argument("--kf-reject-pos-std-m", type=float, default=0.0,
                        help="Gate detected measurements whose reported pos_std exceeds this value; <=0 disables")
    parser.add_argument("--kf-reject-cross-norm", type=float, default=0.0,
                        help="Gate detected measurements whose full_cross_norm exceeds this value; <=0 disables")
    parser.add_argument("--no-gate-reinit-measurement", action="store_true",
                        help="Allow lost-track reinitialization to bypass innovation gating; mainly for ablation/debugging")
    parser.add_argument("--save-measurement-trace", action="store_true",
                        help="Save per-target tracking_measurement_trace.csv for outlier diagnostics")
    parser.add_argument("--gospa-cutoff-m", type=float, default=300.0,
                        help="Cutoff distance for the identity-aware GOSPA-like tracking metric")
    parser.add_argument("--gospa-p", type=float, default=2.0, help="p order for the GOSPA-like metric")
    parser.add_argument("--gospa-alpha", type=float, default=2.0,
                        help="alpha denominator for missed-track penalty in the GOSPA-like metric")
    parser.add_argument("--no-gospa-normalize-by-q", action="store_true",
                        help="Report unnormalized total GOSPA-like score instead of per-target normalized score")
    parser.add_argument("--include-global-assignment-baselines", action="store_true",
                        help="Add collision-unaware global utility baselines to the tracking gate")
    parser.add_argument("--gpda-missed-prior", type=float, default=0.05,
                        help="Prior mass for the missed/no-valid-measurement hypothesis in GPDA-like updates")
    parser.add_argument("--gpda-likelihood-floor", type=float, default=1e-12,
                        help="Numerical floor for GPDA candidate likelihoods")
    parser.add_argument("--no-gpda-clean-candidate", action="store_true",
                        help="Do not expose the clean candidate to PDA/JPDA when a structured polluted peak is generated; useful for ablation")
    parser.add_argument("--jpda-max-joint-events", type=int, default=200000,
                        help="Maximum exact joint-association events per connected JPDA component before row-wise fallback")
    parser.add_argument("--proxy-kernel-mode", choices=["no_spread", "gaussian", "dirichlet", "actual_otfs"],
                        default="dirichlet", help="Optimization-layer DD kernel used outside kernel ablations")
    parser.add_argument("--gaussian-sigma-k", type=float, default=0.65,
                        help="Gaussian kernel Doppler-bin width for kernel ablation")
    parser.add_argument("--gaussian-sigma-l", type=float, default=0.65,
                        help="Gaussian kernel delay-bin width for kernel ablation")
    parser.add_argument("--m-list", default="9,12,15,18")
    parser.add_argument("--q-list", default="6,8,10,12")
    parser.add_argument("--ktmin-list", default="2,3")
    parser.add_argument("--kumax-list", default="2,3")
    parser.add_argument("--lambda-col-list", default="0,0.1,0.3,1,3,10",
                        help="Comma-separated lambda_col values for the paper sensitivity sweep")
    parser.add_argument("--lambda-mc", type=int, default=20,
                        help="MC trials per lambda_col value in the paper sensitivity sweep")
    parser.add_argument("--correlation-mc", type=int, default=20,
                        help="MC trials for the Cbar-vs-Delta-Pd proxy validation")
    parser.add_argument("--bootstrap-iters", type=int, default=2000,
                        help="Bootstrap resamples for paired confidence intervals")
    parser.add_argument("--paper-include-tracking", action="store_true",
                        help="Include the closed-loop tracking experiment in --gate paper; off by default because it can be slow")
    parser.add_argument("--lambda-col", type=float, default=1.0)
    parser.add_argument("--c-norm-min", type=float, default=0.1,
                        help="Effective collision threshold C/protected_self; 0.1 = -10 dB")
    parser.add_argument("--mu-div", type=float, default=0.15)
    parser.add_argument("--include-oracle-gate1", action="store_true",
                        help="Enable exact small-scale oracle in Gate 1; can be slow")
    parser.add_argument("--raw-collision-cost", action="store_true",
                        help="Use raw collision energy instead of normalized collision in the assignment objective")
    parser.add_argument("--eval-kernel-mode",
                        choices=["no_spread", "gaussian", "dirichlet", "actual_otfs", "broadened_dirichlet"],
                        default="actual_otfs", help="Gate-1 full-link evaluation kernel")
    parser.add_argument("--detector-mode",
                        choices=["noncoherent_matched", "matched", "coherent_matched", "energy", "both"],
                        default="noncoherent_matched",
                        help="Primary Pd: practical noncoherent matched-filter bank by default; matched/coherent_matched is coherent upper bound")
    parser.add_argument("--coop-phase-mode", choices=["random_phase", "coherent", "noncoherent_power"],
                        default="random_phase",
                        help="Phase model for same-target cooperative bistatic returns; coherent is an ideal upper bound")
    parser.add_argument("--overlap-round-bins", type=float, default=0.01,
                        help="DD-bin quantization for actual-OTFS kernel cache")
    parser.add_argument("--candidate-pool-size", type=int, default=10,
                        help="Collision-aware candidate pool used when coalition enumeration is truncated")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of multiprocessing workers for independent MC jobs; use 1 for serial execution")
    parser.add_argument("--no-clear-worker-caches", action="store_true",
                        help="Do not clear actual-OTFS caches after each worker job (not recommended for long tracking runs)")
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
        assignment_hysteresis=args.assignment_hysteresis,
        hys_rel_margin=args.hys_rel_margin,
        hys_abs_margin=args.hys_abs_margin,
        hys_force_coverage=(not args.no_hys_force_coverage),
        kf_innovation_gate=args.kf_innovation_gate,
        kf_gate_chi2=args.kf_gate_chi2,
        gate_reinit_measurement=(not args.no_gate_reinit_measurement),
        kf_outlier_r_inflate=args.kf_outlier_r_inflate,
        kf_gate_action=args.kf_gate_action,
        kf_abs_pos_gate_m=args.kf_abs_pos_gate_m,
        kf_abs_vel_gate_mps=args.kf_abs_vel_gate_mps,
        kf_gate_pos_std_cap_m=args.kf_gate_pos_std_cap_m,
        kf_gate_vel_std_cap_mps=args.kf_gate_vel_std_cap_mps,
        kf_reject_pos_std_m=args.kf_reject_pos_std_m,
        kf_reject_cross_norm=args.kf_reject_cross_norm,
        save_measurement_trace=args.save_measurement_trace,
        gospa_cutoff_m=args.gospa_cutoff_m,
        gospa_p=args.gospa_p,
        gospa_alpha=args.gospa_alpha,
        gospa_normalize_by_q=(not args.no_gospa_normalize_by_q),
        include_global_assignment_baselines=args.include_global_assignment_baselines,
        gpda_missed_prior=args.gpda_missed_prior,
        gpda_likelihood_floor=args.gpda_likelihood_floor,
        gpda_include_clean_candidate=(not args.no_gpda_clean_candidate),
        jpda_max_joint_events=args.jpda_max_joint_events,
        inner_mc=args.inner_mc,
        n_tau=args.n_tau,
        n_nu=args.n_nu,
        sigma_theta_deg=args.sigma_theta_deg,
        sigma_theta_list=args.sigma_theta_list,
        m_list=args.m_list,
        q_list=args.q_list,
        ktmin_list=args.ktmin_list,
        kumax_list=args.kumax_list,
        lambda_col_list=args.lambda_col_list,
        lambda_mc=args.lambda_mc,
        correlation_mc=args.correlation_mc,
        bootstrap_iters=args.bootstrap_iters,
        paper_include_tracking=args.paper_include_tracking,
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
    import pandas as pd

    pd.DataFrame([cfg_dict]).to_csv(out / "config.csv", index=False)
    manifest = dict(
        script_version="v18",
        run_id=cfg.run_id,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        argv=sys.argv,
        config=cfg_dict,
    )
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    from .experiments import (
        run_gate1,
        run_gate2,
        run_difficulty_sweep,
        run_prediction_robustness,
        run_tracking_experiment,
        run_phase_sweep,
        run_angle_usefulness_sweep,
        run_kernel_ablation,
        run_velocity_sweep,
    )
    from .paper_simulation import run_paper_experiments

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
    if args.gate in ("paper", "all"):
        run_paper_experiments(cfg, out)

    print(f"\nAll outputs saved to: {out.resolve()}")
