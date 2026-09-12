from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
    eval_kernel_mode: str = "dirichlet"  # no_spread | gaussian | dirichlet | actual_otfs | broadened_dirichlet
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
    tracking_mc: int = 20
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
    tracker_mode: str = "alpha"           # alpha | kf | dd_rwif | gpda | dd_gpda
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

    # V14 temporal assignment and robust-update controls
    assignment_hysteresis: bool = False      # keep previous CA assignment unless objective improvement is meaningful
    hys_rel_margin: float = 0.0              # relative objective margin for hysteresis
    hys_abs_margin: float = 0.0              # absolute objective margin for hysteresis
    hys_force_coverage: bool = True          # always accept new assignment if it improves coverage/uncovered count
    kf_innovation_gate: bool = False         # robust innovation gate for KF/DD-RWIF updates
    kf_gate_chi2: float = 13.3               # approx chi-square threshold for effective 4-D xy/vx-vy measurement at 99%
    gate_reinit_measurement: bool = True       # apply innovation gate before hard lost-track reinitialization
    kf_outlier_r_inflate: float = 50.0       # R inflation factor when innovation is gated
    kf_gate_action: str = "inflate"          # inflate | skip
    # V15: extra robust gates for high self-reported-variance biased peaks.
    # Values <= 0 disable the corresponding gate/cap.
    kf_abs_pos_gate_m: float = 0.0            # direct xy residual gate, e.g., 500 m
    kf_abs_vel_gate_mps: float = 0.0          # direct xy velocity residual gate
    kf_gate_pos_std_cap_m: float = 0.0        # cap pos_std only inside Mahalanobis gate
    kf_gate_vel_std_cap_mps: float = 0.0      # cap vel_std only inside Mahalanobis gate
    kf_reject_pos_std_m: float = 0.0          # gate measurements whose reported pos_std is too large
    kf_reject_cross_norm: float = 0.0         # gate measurements whose DD cross norm is too large
    save_measurement_trace: bool = False     # save per-target tracking_measurement_trace.csv

    # V16 tracking metrics / baselines
    gospa_cutoff_m: float = 300.0           # cutoff distance for identity-aware GOSPA-like metric
    gospa_p: float = 2.0                    # GOSPA p-norm order
    gospa_alpha: float = 2.0                # missed-track penalty denominator
    gospa_normalize_by_q: bool = True       # report per-target normalized GOSPA-like metric
    include_global_assignment_baselines: bool = False  # add collision-unaware global utility baselines

    # V17 GPDA-like candidate-association tracking backend
    gpda_missed_prior: float = 0.05          # prior mass for missed/no-valid-measurement hypothesis
    gpda_likelihood_floor: float = 1e-12     # numerical floor for candidate likelihoods
    gpda_include_clean_candidate: bool = True # when a structured polluted peak occurs, expose the clean+polluted candidate pair to GPDA
    jpda_max_joint_events: int = 200000      # maximum exact joint-association events per connected component before row-wise fallback

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

    # Paper-oriented simulation bundle
    lambda_col_list: str = "0,0.1,0.3,1,3,10"
    lambda_mc: int = 20
    correlation_mc: int = 20
    bootstrap_iters: int = 2000
    paper_include_tracking: bool = True

    # Runtime / parallel execution
    num_workers: int = 10
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
