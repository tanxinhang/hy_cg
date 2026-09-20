"""Configuration for the DOTFS-ISAC cooperative-sensing simulator.

Design note
-----------
The original single-file prototype accumulated a single flat ``SimConfig`` with
~60 fields plus a ~30-flag CLI, most of which existed only to describe
*experiment variants* (which penalty is switched off, which robustness axis is
swept, ...).  Here configuration is split by meaning:

* :class:`Scale`, :class:`Geometry`, :class:`Waveform`, :class:`Radio`,
  :class:`CommCfg`, :class:`Detect` and :class:`DD` describe one *physical
  setting*.  They are what a reviewer would look up in a system-model table.
* :class:`Selector` holds the parameters of the proposed link-selection rule.
* :class:`Run` holds Monte-Carlo / reproducibility / reporting knobs.

Experiment variants are deliberately *not* extra top-level flags.  They are
named dotted-path override dictionaries in :mod:`isac_sim.experiments`, applied
through :func:`apply_overrides`.  Adding a new ablation therefore never touches
this file.

Every field keeps the semantics of the v10 prototype so that numbers remain
reproducible; only the *organisation* changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from numbers import Integral
from typing import Any, Dict, Iterator, Literal, Tuple
import math

# --------------------------------------------------------------------------
# Type aliases shared across the package
# --------------------------------------------------------------------------
Link = Tuple[int, int]

MethodName = Literal[
    "rcs_robust_bundle_cg",
    "joint_bundle_cg",
    "joint_bundle_cg_exact_llr",
    "fixed_fusion_bundle",
    "local_only_bundle",
    "proposed_lagrangian",
    "all_neighbor",
    "random",
    "nearest",
    "shortest_bistatic",
    "raw_sense_sinr",
    "sense_sinr",
    "sense_sinr_budgeted",
    "single_best",
    "topk_deflection",
    "global_topk_deflection",
    "cost_aware_greedy",
    "exact_marginal_greedy",
    "proposed_c2f",
    "proposed_c2f_adaptive",
    "proposed_c2f_adaptive_pd",
    "proposed_c2f_adaptive_pd_distributed",
    "proposed_c2f_adaptive_pd_robust",
    "proposed_c2f_adaptive_pd_calibrated",
    "proposed_c2f_adaptive_pd_fusion_polish",
    "proposed_c2f_pd",
    "proposed_c2f_full",
    "proposed_c2f_full_pd",
]

CommErrorModel = Literal["erasure", "gaussian_replacement", "flip", "biased"]
IsacPowerModel = Literal["sensing_only", "joint_waveform", "reliable_comm_assisted"]


# --------------------------------------------------------------------------
# Physical setting
# --------------------------------------------------------------------------
@dataclass
class Scale:
    """Numbers of cooperating UAVs and targets."""

    M: int = 15
    Q: int = 10


@dataclass
class Geometry:
    """Deployment box and kinematic ranges."""

    area_xy: float = 4000.0
    h_uav_min: float = 800.0
    h_uav_max: float = 1200.0
    h_target_min: float = 700.0
    h_target_max: float = 1500.0
    comm_range: float = 2500.0
    uav_speed_min: float = 20.0
    uav_speed_max: float = 60.0
    target_speed_min: float = 30.0
    target_speed_max: float = 150.0


@dataclass
class Waveform:
    """OTFS delay-Doppler grid and carrier."""

    N: int = 64
    L: int = 64
    delta_f: float = 30e3
    T: float = 1.0 / 30e3
    fc: float = 5.9e9
    c: float = 3e8


@dataclass
class WaveformImpairments:
    """Post-matched-filter impairment ratios for robustness experiments.

    All fields default to the ideal receiver, so paper/V1 presets are unchanged.
    INR values are relative to the existing noise-plus-residual denominator.
    Synchronisation errors are expressed in delay/Doppler bin units.
    """

    enable: bool = False
    # If false in belief mode, truth/detection still sees the impairments but
    # the scheduler builds its sensing table under the ideal-waveform model.
    # This is the explicit unaware-scheduler counterfactual.
    scheduler_aware: bool = True
    clutter_inr: float = 0.0
    multipath_inr: float = 0.0
    unresolved_target_inr: float = 0.0
    sync_delay_bins: float = 0.0
    sync_doppler_bins: float = 0.0


@dataclass
class Radio:
    """Transmit power, noise floor, ISAC power split and residual interference."""

    P_default: float = 1.0
    rho: float = 0.80  # sensing power fraction of the joint waveform
    # Optional per-UAV sensing fractions; each UAV still has P_default watts.
    # None preserves the historical uniform split exactly.
    rho_by_uav: tuple[float, ...] | None = None
    noise_psd_dbm_hz: float = -174.0
    noise_figure_db: float = 7.0
    # sensing_only            : only rho*P contributes to the sensing echo
    # joint_waveform          : the whole P contributes
    # reliable_comm_assisted  : rho*P plus a reliability-weighted comm. part
    isac_power_model: IsacPowerModel = "sensing_only"
    # Residual self / direct-link-cancellation / multi-UAV interference floors.
    # Only used by ``interference.coupling="legacy"``; the coupled model builds
    # the sensing interference from the shared interference field instead.
    residual_self_factor: float = 1e-14
    residual_direct_factor: float = 1e-4
    residual_multi_uav_factor: float = 1e-10
    rinr_sigma_factor: float = 0.15

    # --- Radar hardware link budget -------------------------------------
    # Directional gain on the desired bistatic echo path.  These quantities
    # were historically absent, which forced target_rcs to absorb the entire
    # antenna/system budget.  Values are in dB/dBi and are converted once as
    # G_hw = 10**((G_tx + G_rx - L_sys)/10).  Zero defaults preserve every
    # released result exactly.
    radar_tx_gain_dbi: float = 0.0
    radar_rx_gain_dbi: float = 0.0
    radar_system_loss_db: float = 0.0
    # Optional net-gain abstraction used only for preregistered sensitivity
    # sweeps. When set, it overrides the component sum above. This avoids
    # inventing a Tx/Rx split before an antenna platform has been specified.
    radar_net_gain_db: float | None = None

    # --- SINR denominator guard ------------------------------------------
    # Every SINR is ``signal / (n0 + interference + guard)``.  The guard exists
    # only to avoid dividing by zero, so it MUST be negligible compared with the
    # noise power.  The historical constant ``model.EPS = 1e-12`` was not: for
    # the default radio settings ``n0 = 3.83e-14 W``, i.e. the guard was 26x the
    # noise floor and silently suppressed every sensing SINR by 14.33 dB --
    # which is also what hid the sensing interference term entirely.
    #   "noise_relative": guard = n0 * 10**(eps_rel_db/10), i.e. a fixed number
    #                     of dB below the noise floor, correct at any power
    #                     scale.  **Default since the model correction.**
    #   "legacy":         guard = 1e-12.  Scale-blind; kept only so that the
    #                     frozen pre-correction results stay reproducible via
    #                     ``PRESETS["legacy"]``.
    eps_mode: str = "noise_relative"
    eps_rel_db: float = -30.0


@dataclass
class CommCfg:
    """UAV-to-UAV communication constraint and packetisation."""

    R_min: float = 2.0e5
    chi_min: float = 0.30
    enforce_chi_min: bool = False
    comm_leakage_from_sensing: float = 0.05
    comm_direct_leakage_factor: float = 0.0
    K_candidates: int = 4
    b_d: float = 160.0
    # How the communication interference term is built.
    # "full_concurrent": every UAV is assumed to transmit at full comm power.
    #     Conservative worst case; independent of which links are selected.
    # "active_set": post-selection sensitivity model.  Only transmitters in the
    #     selected reporting set contribute payload interference during final
    #     evaluation.  Candidate selection still starts from the conservative
    #     table, so this is deliberately labelled an ablation rather than a
    #     self-consistent endogenous-interference optimizer.
    # "orthogonal": report payloads are strictly time/frequency orthogonal, so
    #     no other report payload interferes with a reporting leg.  Continuous
    #     sensing-waveform leakage remains present.  This is the self-consistent
    #     companion of ``mac_model="serial"`` used by the conference release.
    interference_model: str = "full_concurrent"

    # --- Reporting reliability model -------------------------------------
    # "heuristic":  chi = gamma / (gamma + gamma_req).  A monotone smooth
    #     mapping with no probabilistic meaning (legacy behaviour, kept for
    #     bit-exact parity with the frozen results).
    # "fbl":        finite-blocklength packet error probability
    #                   eps ~ Q( (C(gamma) - k/n) / sqrt(V(gamma)/n) ),
    #               chi = 1 - eps.  Gives chi a genuine *packet success
    #               probability* meaning and makes the payload size k and the
    #               blocklength n first-class parameters.
    reliability_model: str = "heuristic"
    # Blocklength in channel uses of one soft-information report packet.
    # Only used by ``reliability_model="fbl"``.
    n_block: int = 2048
    # Latency bookkeeping.
    # "payload":      T = k / R    (legacy: delay follows the link rate).
    # "blocklength":  T = n / B    (the block occupies n channel uses at
    #                 bandwidth B, so the latency is rate-independent).
    latency_model: str = "payload"
    # --- Reporting MAC ---------------------------------------------------
    # "serial":    T = sum_l B_l / R_l      (strict time-division).
    # "parallel":  T = max_l B_l / R_l      (all reports concurrent).
    # "slot":      conflict-graph colouring (shared transmitter or receiver
    #              conflicts); reports in the same slot are concurrent, slots
    #              run sequentially.  Under "slot" the communication SINR of a
    #              leg is recomputed counting *only* its co-slot transmitters
    #              as interferers, so the interference model and the latency
    #              model describe one and the same MAC.
    #
    # "serial" / "parallel" keep the legacy behaviour: their latency follows
    # the MAC but their SINR uses ``interference_model`` (default
    # "full_concurrent").  The self-consistent choice is "slot".
    mac_model: str = "serial"


@dataclass
class Interference:
    """How communication and sensing share one spectrum.

    In an ISAC network a UAV radiates a *single* joint waveform into the same
    band, so a receiver at node ``j`` sees the very same set of concurrent
    transmitters twice: once against its communication signal and once against
    its sensing echo.  A model that gives communication a full interference term
    while letting sensing observe (almost) none is therefore not an ISAC model
    -- it removes exactly the coupling that makes ISAC hard, and with it any
    reason for the two functions to be co-designed.

    ``coupling`` selects the bookkeeping.

    * ``"legacy"`` -- communication interference follows
      ``comm.interference_model`` while sensing interference is a set of
      *decoupled* residual floors (``radio.residual_*``).  Those floors neither
      follow the active transmitter set nor reuse the direct-path gains, and the
      strongest interferer of all -- the illuminator's own direct path -- is
      explicitly excluded from the sum.  Kept only for bit-exact parity with the
      frozen results.

    * ``"shared_spectrum"`` -- both receivers are built from *one* interference
      field over the same active transmitter set and the same direct gains.  At
      node ``j``,

          I_comm(i,j) = sum_{k != i, j} ( P_comm[k] + c_leak * P_sense[k] ) g_kj
          I_sense(j)  = kappa_self * P[j] + kappa_dc * sum_{k != j} P_tx[k] g_kj

      with ``P_tx[k] = P_sense[k] + P_comm[k]`` the radiated ISAC power of UAV
      ``k``.  Two properties of this coupling are worth stating explicitly.

      1. The illuminator ``i`` *is* an interferer on the sensing side: its
         direct path competes with its own target echo, which is the classical
         near-far problem of bistatic sensing.  On the communication side the
         same UAV is the wanted signal source.  This is a difference in role,
         not in physics.
      2. ``kappa_dc`` is the direct-path cancellation the sensing receiver can
         achieve, not a hand-tuned floor.  Cooperative ISAC knows every
         illuminator's waveform, so the deterministic direct path can be
         reconstructed and subtracted; the residual is then set by
         reference-aided channel-estimation accuracy.  With ``N_p`` reference
         elements at per-element SNR ``gamma_p`` the achievable depth is
         ``min(10 log10(N_p gamma_p), kappa_hw)``, where ``kappa_hw`` is the
         analogue/RF ceiling that more reference cannot remove.  The derivation,
         its numerical check against the production link tables, and the
         literature anchors for ``kappa_hw`` are in ``KAPPA_DERIVATION.md``
         (tool: ``tools/derive_kappa_pilot_budget.py``).  Note the raw near-far
         ratio is a *geometric* quantity -- 41.3 dB on the legacy paper geometry
         versus 50.9 dB on the current headline scenario -- so ``kappa_dc``
         decides whether the sensing task survives and may not be quoted
         without the geometry it was calibrated on.

    The sensing waveform is assumed to be radiated continuously (that is the
    ISAC premise), so sensing interference is independent of the reporting
    schedule unless ``sense_gate_by_active_tx`` is enabled; only the
    communication payload is gated by which UAVs actually report.
    """

    coupling: str = "shared_spectrum"
    # Direct-path cancellation at the sensing receiver, in dB.  This is
    # *interference suppression*, not a signal gain: because every illuminator's
    # waveform is known in cooperative ISAC, the deterministic direct path can
    # be reconstructed and subtracted, and the residual is set by
    # reference-aided channel-estimation accuracy.  It does not touch the echo
    # term.
    #
    # Derived, not tuned: with N_p reference elements at per-element SNR
    # gamma_p, least-squares subtraction leaves a residual of 1/(N_p gamma_p) of
    # the direct power, so the achievable depth is
    # min(10 log10(N_p gamma_p), kappa_hw), with kappa_hw the analogue/RF
    # ceiling that more reference cannot remove (published digital-domain
    # full-duplex ISAC reaches ~60 dB).  The shipped 40 dB is exactly
    # N_p*gamma_p = 1e4, e.g. 100 reference elements at 20 dB -- 2.4% of this
    # waveform's 4096 elements.  Derivation, numerical check and literature
    # anchors: KAPPA_DERIVATION.md, tools/derive_kappa_pilot_budget.py.
    #
    # Calibration history (both numbers are the median of
    # ``residual direct field / processed echo``, i.e. the echo *after* the N*L
    # processing gain, taken from the production link tables):
    #   * legacy paper geometry (4 km, RCS 50 m^2): 48.0 dB.  The original 40 dB
    #     default was calibrated there, where it leaves the residual only
    #     2.6 dB above the processed echo -- i.e. roughly level, as intended.
    #   * current headline scenario (600 m, RCS 0.1 m^2): 52.2 dB.  The ratio is
    #     a *geometric* quantity, so it does not carry over: at 40 dB the
    #     residual is still about 10 dB ABOVE the processed echo, and the
    #     sensing SINR floor (self-residual + noise) is reached near 60 dB.
    #   Both requirements sit below the ~60 dB hardware ceiling, i.e. both are
    #   reachable from an ordinary cooperative reference budget.
    #
    # Consequence: kappa is a feasibility prerequisite, not a tuning knob that
    # buys performance.  Measured at 600 m / RCS 0.1 with MC=200:
    #   kappa=0  -> P_D 0.0505 against P_FA 0.0495 (no detection capability),
    #   kappa=20 -> P_D 0.0890,  kappa=40 -> P_D 0.6830.
    # The sweep covers 0-80 dB; quoting any kappa without also quoting the
    # geometry it was calibrated on is meaningless.
    direct_cancellation_db: float = 40.0
    # Follow the active transmitter set on the sensing side as well.  Physically
    # unnecessary (illumination is continuous) but useful to isolate the effect.
    sense_gate_by_active_tx: bool = False


@dataclass
class Fusion:
    """Where the soft statistics of a target meet.

    The sensing chain is ``i -> q -> j``: UAV ``i`` illuminates, UAV ``j``
    receives the echo and *produces* the soft statistic.  The statistic then
    has to be reported somewhere to be fused.

    * ``mode="tx"`` (legacy): the statistic is reported back to the sensing
      initiator ``i``.  This makes the reporting direction ``j -> i`` and
      quietly reuses the sensing pair as the reporting pair -- the modelling
      shortcut the reviewers objected to.
    * ``mode="explicit"``: each target ``q`` is assigned an explicit fusion
      UAV ``f_q`` and every statistic is reported ``j -> f_q``.  The sensing
      pair and the reporting relation are fully decoupled, and the reporting
      reliability / rate / latency are those of ``(j, f_q)``.

    ``rule`` picks the fusion UAV when ``mode="explicit"``:
    ``"max_in_rate"`` (largest total incoming rate from the candidate
    receivers), ``"max_min_rate"`` (max-min fairness) or
    ``"nearest_target"`` (closest to the predicted target position) or
    ``"nearest_target_capacitated"`` (minimum-distance assignment under the
    same per-UAV target capacity as the proposed outer problem).
    ``"nearest_centroid"`` is retained as a backward-compatible alias for
    ``"nearest_target"``. ``"capacitated_value"`` maximizes a detector-
    quality proxy over target--fusion pairs subject to a hard per-UAV target
    capacity.  The capacity is a system budget, not a scalarization weight.
    """

    mode: str = "tx"
    rule: str = "max_in_rate"
    # Maximum number of targets assigned to one fusion UAV. ``-1`` is
    # unbounded and preserves all released V1 configurations.
    max_targets_per_uav: int = -1
    # Homogeneous per-UAV CPU model. A non-negative rate activates the budget
    # C_f = F_f * T_proc; negative keeps released configurations unbounded.
    cpu_rate_cycles_per_s: float = -1.0
    processing_window_s: float = 0.01
    cpu_fixed_cycles: float = 0.0
    cpu_per_observation_cycles: float = 1.0
    cpu_cubic_cycles: float = 0.0


@dataclass
class Corr:
    """Observation-correlation model for the fused soft statistic.

    The legacy composite deflection assumes ``Cov(s_a, s_b) = 0`` for
    ``a != b``, which a multi-UAV bistatic network manifestly violates:
    pairs share transmitters, receivers, the target's RCS fluctuation and
    possibly neighbouring delay-Doppler support.  With the correlation model
    on, the fusion uses

        D_q = delta^T Sigma^{-1} delta,      w* propto Sigma^{-1} delta

    with ``Sigma = S R S`` and ``R`` built from a *factor* model, i.e.
    ``R = rho_tx * 1{i = i'} + rho_rx * 1{j = j'} + rho_target
    + rho_dd * 1{adjacent DD support}`` on the off-diagonal and unit diagonal.
    Because it is a factor model the result is positive semi-definite by
    construction as long as the ``rho``s are non-negative and sum to at most
    one -- the remainder is the idiosyncratic (independent) share.
    """

    enable: bool = False
    rho_tx: float = 0.15
    rho_rx: float = 0.15
    rho_target: float = 0.10
    rho_dd: float = 0.10


@dataclass
class ActiveSensing:
    """Discrete active-observation design for active evidence acquisition.

    Each mode jointly declares a sensing-power multiplier, an absolute number
    of independent looks, and whether the refined DD receiver is used.  Energy
    is measured in normalized look-power units.  Aspect angles define a finite,
    predeclared uncertainty set; they are not fitted to trial outcomes.
    """

    enable: bool = False
    mode_names: tuple[str, ...] = ("eco", "nominal", "intensive")
    power_scales: tuple[float, ...] = (0.5, 1.0, 1.25)
    looks: tuple[int, ...] = (8, 16, 32)
    refined: tuple[bool, ...] = (False, True, True)
    energy_budget_per_target: float = 64.0
    energy_budget_per_uav: float = 256.0
    max_tx_observations_per_uav: int = -1
    matched_filter_cycles_per_look: float = 1.0
    llr_cycles_per_look: float = 0.25
    dd_refine_cycles: float = 16.0
    max_candidates_per_pair: int = 8
    candidate_strategy: str = "scenario_union"
    scenario_topk_per_scenario: int = 1
    complementary_pair_topk: int = 2
    complete_pool_max_links: int = 12
    branch_node_limit: int = 200_000
    information_metric: str = "forward_kl"
    energy_price: float = 0.0
    report_price: float = 0.0
    aspect_enable: bool = True
    aspect_angles_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    aspect_floor: float = 0.20


@dataclass
class Detect:
    """Sensing-side detection model and soft-information statistics."""

    Pfa_target: float = 0.05
    D_min: float = 3.0
    # Per-target design point used by the selector and lexicographic oracle.
    # This is distinct from the empirical weak-target operating requirement
    # used to freeze a resource-surface budget.
    pd_required: float = 0.95
    weak_pd_required: float = 0.80
    # --- Soft-statistic model --------------------------------------------
    # "gaussian": legacy ``mu = kappa_mu * log(1 + gamma)`` with a hand-set
    #     ``soft_mu_scale``.  There is no derivation for it -- it only encodes
    #     "higher sensing SINR should give a larger soft mean".
    # "llr":      the soft statistic is the *centred local log-likelihood
    #     ratio* of the delay-Doppler matched-filter / local-energy output.
    #     For independent fast-fluctuation looks in complex Gaussian noise,
    #     the integrated bin energy is Gamma distributed, giving
    #     exponential, giving
    #         ell  = -ln(1+gamma) + x * gamma/(1+gamma),  x = |z|^2 / sigma_n^2
    #         delta = E1[ell] - E0[ell] = gamma^2/(1+gamma)
    #         sigma0^2 = Var0[ell]      = gamma^2/(1+gamma)^2
    #         D_10   = D_KL(p1||p0)     = gamma - ln(1+gamma)
    #     i.e. *no free parameter at all*, and the per-link information gain
    #     is literally a Kullback-Leibler divergence.
    soft_stat_model: str = "gaussian"
    # Number of OTFS frames incoherently integrated inside one CPI.  Only used
    # by ``soft_stat_model="llr"``, where the single-link deflection is
    # ``L * gamma^2``.  A physical parameter (CPI length), not a tuning knob.
    n_looks: int = 16
    soft_mu_scale: float = 8.0
    soft_sigma0: float = 1.0
    soft_sigma_floor: float = 0.25
    soft_error_sigma_scale: float = 3.0
    # Deterministic Monte-Carlo quadrature size used by the common calibrated
    # multi-report detector under the true-erasure model.
    fused_calibration_samples: int = 8192
    # Experimental exact-mixture calibration for the released
    # ``gaussian_replacement`` reporting channel.  False keeps the frozen
    # Cornish--Fisher path bit-exact; True uses deterministic quadrature of the
    # implemented centred-Gamma / Gaussian replacement mixture.
    exact_gaussian_replacement_threshold: bool = False
    # Environment-level pollution used by the Monte-Carlo detector.
    enable_comm_error_pollution: bool = True
    # ``gaussian_replacement`` is the released V1 surrogate: failed packets
    # are replaced by zero-mean uncertainty.  ``erasure`` is a true drop and
    # contributes an exact zero statistic.
    comm_error_model: CommErrorModel = "gaussian_replacement"
    soft_error_flip_scale: float = 1.0
    soft_error_bias_scale: float = 0.5
    h0_error_bias_scale: float = 0.0
    # Independent H1 noise/report realizations per target.  One preserves the
    # released Monte-Carlo protocol; larger values are for conditional-P_D
    # audits at a fixed geometry and selected set.
    num_h1_per_target: int = 1
    num_false_per_target: int = 30
    # Channel abstraction
    path_loss_exp: float = 2.0
    shadow_std_db: float = 1.0
    rician_K_db: float = 15.0
    target_rcs: float = 50.0
    sensing_processing_gain: float | None = None
    # --- Target RCS model -------------------------------------------------
    # "iid":        every (i, j, q) draws its own exponential fluctuation, so
    #               the same target looks like a different object to every
    #               bistatic pair (legacy behaviour, kept for parity).
    # "swerling1":  one exponential realisation per (target, CPI), shared by
    #               every pair -- the physically correct latent-target reading.
    #               An optional bistatic aspect factor g(theta_i, theta_j)
    #               is applied on top.
    # "mean":       use the mean RCS in the link budget.  This is the correct
    #               companion of the local independent-look LLR, whose H1 energy
    #               distribution already marginalizes the RCS fluctuation;
    #               drawing RCS here as well would count it twice.
    rcs_model: str = "iid"
    rcs_aspect_enable: bool = False


@dataclass
class Prior:
    """Target-state prediction-uncertainty model.

    When ``sigma_pos_m > 0`` or ``sigma_vel_mps > 0`` the
    ``prior-sweep`` experiment perturbs each MC trial's target state by
    independent Gaussian draws of the given standard deviation.  Default
    values leave both sigmas at zero so the simulator uses the *true*
    target state, matching the original behaviour.

    ``dt_s`` is the look-ahead used by :func:`isac_sim.prior.predicted_geometry`
    when a constant-velocity prediction is requested.
    """

    sigma_pos_m: float = 0.0
    sigma_vel_mps: float = 0.0
    dt_s: float = 0.1
    # --- Truth vs belief --------------------------------------------------
    # ``belief_mode=False`` (legacy): the *perturbed* target state is written
    # back into the geometry, so the scheduler and the physical world share
    # one state.  That answers "how does P_D degrade if the tracker is
    # wrong", but it is NOT the reviewers' question -- they asked how a
    # scheduler that may only *see* a belief performs against the true
    # target.
    #
    # ``belief_mode=True``: two states are carried side by side,
    #     truth  x_{q,t}        -> echo generation, DD, sensing gain, detector
    #     belief b_q = N(xhat, P) -> link selection, DD search window, budget
    # which is the honest ``predict -> schedule -> sense -> update`` loop.
    belief_mode: bool = False
    # Standard deviation of the *belief* error used to build (xhat, P).  The
    # belief is the truth corrupted by these sigmas, so sigma = 0 degenerates
    # to a perfect tracker.
    belief_sigma_pos_m: float = 150.0
    belief_sigma_vel_mps: float = 15.0
    # Confidence mass of the horizontal Gaussian position-error ball used by
    # the optional robust detector-PD selector.  For a 2-D isotropic belief,
    # r = sigma * sqrt(-2 log(1-confidence)).
    robust_position_confidence: float = 0.50
    # Ellipsoidal DD search gate derived from the predicted covariance.  A
    # selected link captures the truth when its delay and Doppler residuals are
    # within this many standard deviations (plus half a quantisation bin).
    search_gate_sigma: float = 3.0
    # What the scheduler knows about target RCS.  ``"realized"`` is the legacy
    # oracle-like path; ``"mean"`` uses E[sigma_q] and prevents a current-CPI
    # RCS realization from leaking into pre-sensing scheduling decisions.
    scheduler_rcs: str = "realized"
    # Lower endpoint of the target-class RCS uncertainty interval, expressed
    # as a fraction of ``detect.target_rcs``.  The RCS-robust bundle method
    # prices every column at this endpoint.  It is an epistemic design bound,
    # not a current-CPI RCS observation or a tunable objective weight.
    rcs_lower_factor: float = 0.5


@dataclass
class DD:
    """OTFS delay-Doppler validity / fractional-loss / collision mechanisms."""

    use_otfs_bin_validity: bool = True
    enable_dd_fractional_penalty: bool = True
    enable_dd_collision_penalty: bool = True
    dd_collision_alpha: float = 1.0


# --------------------------------------------------------------------------
# Algorithm
# --------------------------------------------------------------------------
@dataclass
class Refine:
    """Coarse-to-fine DD-grid refinement (paper Table I: W, kappa_dd, eta_min,
    L_short).  Disabled by default so existing numerical behaviour is unchanged.

    ``window_kernel`` selects how the local-window energy is evaluated:

    * ``"dirichlet"`` -- periodic Dirichlet leakage (default).  This is the
      physics-faithful choice and the one used by the sibling
      ``gate_otfs_collision`` package.
    * ``"sinc"``      -- analytic sinc window; useful as an ablation.

    ``apply_to_all`` turns off the shortlist screening so every feasible link
    is refined.  This is the *full local refinement* comparison used to
    quantify the ``41.1%`` fine-grid-evaluation saving reported in the paper.
    """

    enable: bool = False
    half_width: int = 1
    kappa_dd: float = 0.45
    eta_min: float = 0.25
    shortlist_size: int = 25
    window_kernel: str = "dirichlet"
    apply_to_all: bool = False
    # --- Refinement model -------------------------------------------------
    # "interp":  eta^f = min{1, max[eta_min, eta^c + kappa_dd (eta^loc -
    #            eta^c)]}.  Two hand-set parameters (kappa_dd, eta_min) with
    #            no waveform interpretation -- exactly the heuristic the
    #            reviewers flagged.  Kept as the default for parity.
    # "window":  the fine estimator resolves the fractional delay-Doppler
    #            offset and therefore *recovers* the local-window energy, so
    #            eta^f = eta^loc.  No free parameter; C2F becomes a genuine
    #            computational-acceleration mechanism (cheap main-bin coarse
    #            screening -> waveform-based local refinement) instead of a
    #            way to inflate the sensing gain of the selected links.
    mode: str = "interp"


@dataclass
class Selector:
    """Detector-marginal observation selection and its behaviour switches.

    Released V1 uses a scalar report price.  The V1.1 successor disables that
    price and lets the hard resource budgets below define feasibility.

    The four ``use_*`` flags are *not* tuning knobs: each one disables exactly
    one mechanism and is only ever flipped by an ablation variant.  They live
    here (rather than in the physical groups) because they change the algorithm,
    not the scenario.
    """

    # Marginal-value price: score = alpha_q * dD - lambda_c * delay_ms.
    lambda_c: float = 0.005
    mu_deficit: float = 1.0
    use_softmin_alpha: bool = True
    softmin_tau: float = 0.10
    alpha_floor: float = 0.0
    alpha_cap: float = 10.0
    # ``first_order`` reproduces the historical alpha_q * DeltaD rule.
    # ``exact_utility`` evaluates the actual fair utility increment and is the
    # paper-canonical rule; its greedy and exhaustive oracle share one objective.
    # ``detector_pd`` evaluates the post-report H0/H1 moments and the
    # implemented CF-corrected threshold under the same communication price.
    score_mode: str = "first_order"
    # Historical early exit at D_min is retained for legacy reproduction.  It
    # is disabled in the canonical release because it can stop while the stated
    # utility still has a positive feasible marginal gain.
    stop_at_D_min: bool = True
    # Ablation switches.
    use_target_priority: bool = True
    use_delay_price: bool = True
    use_comm_error_calibration: bool = True
    # Experimental structural constraint: admit one local observation for
    # each serviceable target before allocating remote reports. This encodes
    # "remote complements local" without a fitted scalar reward.
    require_local_anchor: bool = False
    # Resource limits.
    max_links_per_target: int = 6
    max_total_links: int = 60
    # --- Coordinated-radiation price --------------------------------------
    # Every distinct UAV that radiates during the sensing observation re-injects
    # its own direct-path leakage into every receiver.  The released objective
    # maximises each target's marginal detection and never asks how many nodes it
    # is waking up, so it happily spreads 25 links over 10-11 illuminators when 3
    # would do (measured, 500 m / RCS 0.2: capping the count lifts the worst-target
    # P_D from 0.205 to ~0.79, and even a random 3-node cap beats the reference).
    #
    # ``tx_penalty`` charges this price in utility units when a candidate would
    # introduce a *new* radiating node; ``max_tx_nodes`` is the hard-cap dual.
    # Inter-UAV signalling is assumed ideal and instantaneous (see the paper's
    # assumption list).  Both default to "off", so the frozen release path stays
    # bit-exact.
    tx_penalty: float = 0.0
    max_tx_nodes: int | None = None
    # Counterfactual control for auditing how strongly results depend on
    # zero-report-cost evidence produced at the fusion UAV.  ``-1`` keeps the
    # nominal unlimited policy, ``0`` forbids local evidence, and a positive
    # value caps its count independently for each target.
    max_local_observations_per_target: int = -1
    # Hard system-wide cap on inter-UAV reports.  This is independent of the
    # observation cap because local evidence consumes sensing/computation but
    # no reporting slot.  ``-1`` leaves the nominal selector unconstrained.
    max_remote_reports: int = -1
    # Hard processing budgets.  Each selected (i,j,q) observation consumes one
    # receiver-processing unit at j and one fusion-processing unit at f_q.
    # ``-1`` leaves the corresponding resource unbounded.
    max_observations_per_receiver: int = -1
    max_observations_per_fusion_uav: int = -1
    candidate_topk_per_target: int = 40
    min_marginal_D: float = 0.0
    # V1.2 target--fusion--bundle column generation controls.
    bundle_shortlist_per_type: int = 4
    bundle_cg_max_iterations: int = 8
    bundle_pricing_tolerance: float = 1e-8
    bundle_exact_pricing_max_candidates: int = 10


@dataclass
class Coordination:
    """Inter-UAV radiation coordination on the *release* selection path.

    The released selector scores every candidate against a sensing denominator
    in which **every** UAV radiates, so a node that illuminates nothing still
    injects its direct-path leakage into every receiver.  Coordination closes
    that loop as a fixed point::

        round 0: mask = None (everyone radiates)  -> select
        round k: mask = illuminators of round k-1 -> rebuild tables -> reselect

    It is the same map that ``coordination.select_with_coordination`` runs, but
    wired into :func:`isac_sim.simulate.run_method_on_trial` so the released
    entry point -- not only ``tools/*`` -- can report it.  The measured lever is
    large and it is pure protocol: no extra hardware gain.  Quote it only from
    the *release* entry point (600 m / RCS 0.1, G_hw = 0 dB, kappa = 40 dB,
    MC = 1000, ``proposed_c2f_adaptive_pd`` + ``selector.max_tx_nodes = 3``)::

        P_D        0.6938 -> 0.8501
        worst P_D  0.6650 -> 0.8380     (paired, same seed)

    The larger ``tools/*`` figure (0.85 -> 0.95) is a different 口径 and must
    not be quoted next to these -- see ``COORDINATION_WIRING.md``.

    ``enable`` defaults to ``False`` so the frozen release path stays
    bit-exact.  Turning it on requires ``interference.sense_gate_by_active_tx``
    (validated, not assumed): without the gate ``compute_link_tables`` ignores
    the mask and coordination would silently be a no-op -- the failure mode
    that made an earlier "coordination" number unreportable.

    ``rounds`` is a *budget*, not a promise.  The mask map is deterministic but
    has no convergence proof; measured on 6 seeds it settled after 2-4 rounds,
    and the membership (not the count) was what moved.  A repeated mask is
    reported as a cycle rather than silently burning the budget.
    """

    enable: bool = False
    rounds: int = 6


@dataclass
class Cancellation:
    """Receiver-side direct-path interference cancellation (TP-UIC V1).

    ``interference.direct_cancellation_db`` is a *constant*: it asserts that a
    sensing receiver removes a fixed 40 dB of the aggregated direct field,
    whatever the geometry, the reference budget or the target state.  Measured
    against the production link tables that constant is a *requirement*
    (``KAPPA_DERIVATION.md``), not a description of a receiver.

    This block replaces the constant by the output of an executable estimator:
    target-preserving, uncertainty-aware interference cancellation (TP-UIC).
    The receiver reconstructs every active illuminator's direct contribution
    from its *known* cooperative waveform, but fits it only inside the subspace
    orthogonal to the local target manifold, so that the interference estimate
    cannot absorb a weak target.  The leftover interference is then split into

      * interference deliberately retained because it projects onto the target
        subspace (the price of protection), and
      * estimation-uncertainty residual, described by the posterior covariance.

    The realised cancellation depth is therefore an *output*: the same
    algorithm reads 40 dB on one geometry and 20 dB on another, and states its
    own target-retention ratio alongside.

    ``enable`` defaults to ``False``: with the gate closed every released
    number is bit-exact, exactly as for ``coordination.enable``.  Nothing in
    this block is read outside :mod:`isac_sim.cancellation`.
    """

    enable: bool = False
    # How ``model.compute_link_tables`` forms the residual direct field (the term
    # the frozen constant ``interference.direct_cancellation_db`` describes):
    #
    # ``"off"``     the frozen constant, bit for bit.  The default.
    # ``"predict"`` the analytic bridge :func:`cancellation.predict_cancellation`,
    #               per receiver: the dimension-ratio retention plus the
    #               reference-budget estimation term.  It is *optimistic*
    #               (measured 5x on the 600 m scenario) and exists so the chain
    #               can be closed end to end, not to report a result.
    # ``"measure"`` refused by ``compute_link_tables``.  Measuring the residual
    #               needs the trial geometry, which that function is not given;
    #               build ``cancellation.measure_residual_fraction(cfg, geom,
    #               base)`` and pass the array to ``compute_link_tables`` as
    #               ``residual_fraction_by_receiver``.  That measured array -- not
    #               this field -- is what the module's documentation allows a
    #               scheduler to consume.
    mode: str = "off"
    # --- reference budget -------------------------------------------------
    # Coherent CPIs integrated for the direct-path channel estimate.  The
    # estimator runs on the reference observation and is applied to the current
    # observation, so this is a genuine receiver-side budget and not a fudge:
    # ``n_cpi=1`` means "one CPI, no free gain".  Predicted depth grows as
    # 10 log10(n_cpi) with the per-illuminator INR held fixed.
    n_cpi: int = 1
    # --- interference dictionary ------------------------------------------
    # Fractional-DD tangent columns per active illuminator.  Default 0, and
    # that default is a *physical* statement rather than a convenience: in a
    # cooperative network the serving UAVs' positions are shared, so the
    # direct-path delay and Doppler are computed, not estimated.  Adding
    # tangent columns models the residual mismatch left by synchronisation and
    # oscillator error, at a measurable cost in depth -- use it as a robustness
    # axis, not as the baseline.
    interference_tangent_order: int = 0
    # --- target protection ------------------------------------------------
    protect_targets: bool = True
    # How many targets one receiver protects.
    #
    # This is the load-bearing design choice, and the first TP-UIC experiment
    # found it the hard way: protecting *every* believed echo at a receiver
    # means the union of 140 tangent spaces (15 UAVs x 10 targets), whose rank
    # on the 600 m scenario is 195 of 4096 bins.  That subspace turned out to
    # contain the entire 14-dimensional direct-path subspace -- every
    # illuminator's kernel was >=99.9% inside it -- so stage-1 cancellation
    # collapsed to zero depth.  Protecting a target costs interference-learning
    # space, and the cost is convex in the number of protected targets.
    #
    # The default protects the ``max_protected_targets`` echoes with the
    # *lowest* echo-to-total-field ratio at this receiver: exactly the weak
    # targets the method is for.  Set it to ``0`` to protect everything (the
    # unconstrained variant, kept for the ablation that documents the collapse).
    max_protected_targets: int = 3
    # 0 = protect the predicted DD centre only; 1 = also protect the first
    # derivatives d/df_delay and d/df_doppler, i.e. the tangent space of the
    # target manifold.  1 is the meaningful setting: the target will not sit
    # exactly on its predicted bin, and a centre-only window makes no statement
    # about how fast the response leaves it.
    tangent_order: int = 1
    # Finite-difference step, in DD-bin units, for the tangent columns.
    tangent_step_bins: float = 0.05
    # Experimental belief-covariance protection.  When enabled, the hard
    # protection basis includes deterministic 95% marginal DD/bearing sigma
    # points in addition to the centre tangent space.  Off by default so the
    # released receiver remains bit-exact.
    covariance_protection: bool = False
    # --- estimator --------------------------------------------------------
    # Channel prior variance of the complex direct coefficient (unit-power
    # Rician LOS component).  ``None`` disables the prior, leaving plain
    # (protected) least squares.
    prior_variance: float | None = 1.0
    # Soft target-protection weight.  Zero is exactly unprotected ridge LS;
    # larger values increasingly penalise reconstructed direct interference in
    # the believed target subspace.  Used by the experimental ``soft_tpuic``
    # arm at fixed n_cpi=1.
    soft_protection_mu: float = 1.0
    # Belief-only adaptive soft protection.  Each multiplier is screened
    # against the hard TP-UIC arm for predicted residual and risk survival.
    # Disabled by default because each grid point requires another solve.
    adaptive_soft_enable: bool = False
    adaptive_soft_mu_grid: Tuple[float, ...] = (
        0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0,
    )
    adaptive_soft_risk_slack: float = 0.0
    adaptive_soft_residual_quantile: float = 0.80
    # --- detector calibration ---------------------------------------------
    # Charge the *belief error* to the residual covariance ``C_res``.
    #
    # Every echo template is built at the **believed** target state while the
    # echo arrives from the true one, so each template is misplaced by
    # ``delta_l * da/dl + delta_k * da/dk`` to first order, with ``delta_l`` and
    # ``delta_k`` the position and velocity error expressed in DD bins.  The
    # echoes therefore leak through the templates into the residual, which is
    # what lifts the measured ``P_FA`` above its nominal level.
    #
    # This is **not** the coefficient prior above: the prior describes the
    # receiver's uncertainty about the interference amplitudes it is *fitting*;
    # this term describes the geometry of a template that *cannot be placed
    # exactly*.  Charging one for the other is what leaves the detector
    # mis-calibrated by exactly the amount being measured.
    #
    # Off by default so every released number stays bit-exact; the flag exists so
    # that both the benefit (calibrated ``P_FA``) and the cost (a slightly
    # inflated ``C_res``, hence a little ``P_D``) can be measured rather than
    # asserted.
    belief_error_in_cres: bool = False
    # --- stage 2 ----------------------------------------------------------
    joint_refine: bool = True
    # Local DD search half-width used to form the detection statistic.
    search_half_width: int = 1
    # --- accounting -------------------------------------------------------
    hw_ceiling_db: float = 60.0


@dataclass
class Aperture:
    """Receive array: adds the *angular* dimension to the DD sensing model.

    The released model is DD-only, and the angular probe
    (``ANGULAR_IDENTIFIABILITY_PROBE.md``) measured what that costs: with one
    element the *typical* per-link escape fraction is ``4.1e-04``, i.e. targets
    that share a delay-Doppler cell are not separable at all, and the whole
    "identifiability" story is about the receiver having a spatial dimension.

    This block adds that dimension where it enters the **production** number: the
    delay-Doppler collision penalty (``model.build_base_gains``).  Two targets in
    one DD cell currently mask each other completely, so the link's echo power is
    divided by the number of co-bin targets.  With an array the pair is masked by
    ``|A(delta_u)|^2`` instead, which is ``1`` for a single element and shrinks as
    the pair separates in angle -- so the penalty becomes a *soft* count that
    reduces exactly to today's integer count at ``m_rx = 1``.

    **What this deliberately does not do.**  It does not add the array's
    collection gain (``10 log10 m`` on the desired echo).  That is a link-budget
    quantity, indistinguishable in its effect from ``radar_net_gain_db``, and
    folding it in here would make any measured improvement ambiguous between
    "more selective" and "simply louder".  See :mod:`isac_sim.aperture`.

    ``enable`` defaults to ``False`` and ``m_rx`` to ``1``: with the gate closed
    every released number is bit-exact.
    """

    enable: bool = False
    # Receive elements of a half-wavelength ULA.  1 == the released DD-only model.
    m_rx: int = 1
    # Body axis the 1-D array resolves along (0 = x, 1 = y).  Fixed, not fitted:
    # a 1-D array resolves one axis and has a front/back ambiguity, so "the
    # better axis" is an oracle choice this model is not allowed to make.
    axis: int = 0


@dataclass
class Run:
    """Monte-Carlo, reproducibility and reporting knobs."""

    num_mc: int = 200
    seed: int = 2026
    verbose: bool = True
    # Independent trials can be evaluated in separate processes. Results are
    # consumed in trial-index order, so changing this value does not change the
    # random streams or numerical output.
    workers: int = 1
    # Wall-clock metrics are intentionally opt-in because they are not
    # deterministic across worker counts and must not contaminate V1 summaries.
    record_runtime: bool = False


# --------------------------------------------------------------------------
# Root configuration
# --------------------------------------------------------------------------
@dataclass
class Config:
    scale: Scale = field(default_factory=Scale)
    geometry: Geometry = field(default_factory=Geometry)
    waveform: Waveform = field(default_factory=Waveform)
    waveform_impairments: WaveformImpairments = field(default_factory=WaveformImpairments)
    radio: Radio = field(default_factory=Radio)
    comm: CommCfg = field(default_factory=CommCfg)
    detect: Detect = field(default_factory=Detect)
    dd: DD = field(default_factory=DD)
    refine: Refine = field(default_factory=Refine)
    prior: Prior = field(default_factory=Prior)
    selector: Selector = field(default_factory=Selector)
    run: Run = field(default_factory=Run)
    fusion: Fusion = field(default_factory=Fusion)
    corr: Corr = field(default_factory=Corr)
    active_sensing: ActiveSensing = field(default_factory=ActiveSensing)
    interference: Interference = field(default_factory=Interference)
    coordination: Coordination = field(default_factory=Coordination)
    cancellation: Cancellation = field(default_factory=Cancellation)
    aperture: Aperture = field(default_factory=Aperture)

    # Backward-compatible aliases used by the model code, so the math reads the
    # same way as in the prototype.  These are properties, not stored fields.
    @property
    def M(self) -> int:
        return self.scale.M

    @property
    def Q(self) -> int:
        return self.scale.Q


def default_config() -> Config:
    """Return the canonical default configuration."""
    return Config()


# --------------------------------------------------------------------------
# Named configuration bundles
# --------------------------------------------------------------------------
# A preset is a *coherent* set of overrides that switches the simulator from the
# frozen legacy abstraction to the physically consistent model.  Presets are
# deliberately not CLI flags: they are plain override dictionaries, so they can
# also be composed into an experiment variant.
PRESETS: Dict[str, Dict[str, Any]] = {
    # The frozen pre-correction model: scale-blind SINR guard plus decoupled
    # sensing-residual floors.  It is no longer the default.
    #
    # Contract, stated honestly: this preset pins only the two keys below, so it
    # restores the *interference / SINR-guard* layer and nothing else.  It does
    # NOT pin the detector-side model, which was revised after this preset was
    # written (the between-group term of the law of total variance moved from
    # the deflection denominator into the H1 variance, and ``comm_error_model``
    # changed value from ``erasure`` to ``gaussian_replacement``).  Historical
    # CSVs produced before that revision -- including the archived
    # ``.workbuddy/baseline_historical_603b61d/main/main.csv`` -- are therefore
    # NOT reproducible from this preset alone.  The frozen regression baseline
    # ``.workbuddy/baseline/main/main.csv`` was re-frozen against the current
    # model on 2026-09-16 and is what this preset is checked against.
    "legacy": {
        "interference.coupling": "legacy",
        "radio.eps_mode": "legacy",
    },
    # The corrected coupled model with the historical post-selection active-set
    # sensitivity path.  This is an ABLATION of the paper-canonical MAC, not a
    # replacement for it.  Use it when an experiment must be immune to future
    # default changes while varying the active-set assumption.
    "isac-consistent": {
        "interference.coupling": "shared_spectrum",
        "radio.eps_mode": "noise_relative",
        "comm.interference_model": "active_set",
    },
    # The same, with the strict (waveform-derived) soft statistic and the
    # finite-blocklength reliability model.
    "isac-consistent-strict": {
        "interference.coupling": "shared_spectrum",
        "radio.eps_mode": "noise_relative",
        "comm.interference_model": "active_set",
        "detect.soft_stat_model": "llr",
        "comm.reliability_model": "fbl",
        "fusion.mode": "explicit",
    },
    # Single source of truth for paper figures/tables.  The reporting packets
    # are orthogonal, while the sensing waveform remains continuously radiated;
    # this preserves the defining ISAC coupling without an endogenous active-set
    # fixed point.  Legacy/active-set variants remain available as ablations.
    "paper-canonical": {
        "geometry.uav_speed_min": 30.0,
        "geometry.uav_speed_max": 60.0,
        "geometry.target_speed_min": 50.0,
        "geometry.target_speed_max": 90.0,
        "interference.coupling": "shared_spectrum",
        "radio.eps_mode": "noise_relative",
        "comm.interference_model": "orthogonal",
        "comm.mac_model": "serial",
        "comm.reliability_model": "fbl",
        "comm.latency_model": "blocklength",
        "comm.enforce_chi_min": True,
        "fusion.mode": "explicit",
        "detect.soft_stat_model": "llr",
        "detect.rcs_model": "mean",
        "prior.belief_mode": True,
        "prior.scheduler_rcs": "mean",
        "refine.enable": True,
        "refine.mode": "window",
        "selector.score_mode": "exact_utility",
        "selector.stop_at_D_min": False,
    },
}

# Frozen V1 candidate.  It deliberately inherits every physical, detector,
# refinement and selector setting from the paper release and changes only the
# target-specific fusion destination rule.  Keeping this relationship
# executable prevents future paper-preset edits from creating an accidental,
# undocumented second experimental protocol.
PRESETS["target-local-v1"] = {
    **PRESETS["paper-canonical"],
    "fusion.rule": "nearest_target",
}

# Budget-driven successor.  Numerical resource budgets are deliberately not
# baked into the preset: experiments must state them as physical scenario
# inputs instead of selecting a favourable point after a sweep.
PRESETS["capacitated-target-fusion-v1.1"] = {
    **PRESETS["target-local-v1"],
    "fusion.rule": "capacitated_value",
    "detect.comm_error_model": "erasure",
    "selector.use_delay_price": False,
    "selector.lambda_c": 0.0,
}

# Optimization-only successor: inherit the complete V1.1 physical/detector
# model and remove the experimental mandatory-anchor restriction.
PRESETS["joint-bundle-v1.2"] = {
    **PRESETS["capacitated-target-fusion-v1.1"],
    "selector.require_local_anchor": False,
}

# Algebraic decomposition of the historical 50 m^2 default into a small-target
# RCS and an explicit radar hardware budget.  This is a calibration bridge, not
# an independently validated hardware design: 0.1 m^2 * 10^(27/10) = 50.12 m^2.
PRESETS["small-uav-link-budget-bridge"] = {
    **PRESETS["paper-canonical"],
    "detect.target_rcs": 0.1,
    "radio.radar_tx_gain_dbi": 16.0,
    "radio.radar_rx_gain_dbi": 16.0,
    "radio.radar_system_loss_db": 5.0,
}

# Three physically named small-UAV scenario scales.  They inherit
# paper-canonical directly and therefore carry NO radar hardware budget
# (radar_net_gain_db = 0 dB: 0 dBi tx / 0 dBi rx / 0 dB loss).  The 27 dB
# budget belongs to small-uav-link-budget-bridge only and is deliberately not
# shared here.  Only geometry, connectivity, and target-class mean RCS change.
# Aspect fluctuation remains disabled until its angular law is calibrated
# against measurements.
PRESETS["small-uav-dense-s1"] = {
    **PRESETS["paper-canonical"],
    "geometry.area_xy": 1000.0,
    "geometry.h_uav_min": 200.0,
    "geometry.h_uav_max": 500.0,
    "geometry.h_target_min": 200.0,
    "geometry.h_target_max": 500.0,
    "geometry.comm_range": 1200.0,
    "detect.target_rcs": 0.1,
}

# Compact nominal-RCS scenario requested for an 800 m horizontal deployment.
# This is an area side length, not a hard or fixed bistatic leg distance.
PRESETS["small-uav-compact-800m"] = {
    **PRESETS["paper-canonical"],
    "geometry.area_xy": 800.0,
    "geometry.h_uav_min": 200.0,
    "geometry.h_uav_max": 500.0,
    "geometry.h_target_min": 200.0,
    "geometry.h_target_max": 500.0,
    "geometry.comm_range": 1000.0,
    "detect.target_rcs": 0.05,
}

PRESETS["small-uav-nominal-s2"] = {
    **PRESETS["paper-canonical"],
    "geometry.area_xy": 2000.0,
    "geometry.h_uav_min": 300.0,
    "geometry.h_uav_max": 600.0,
    "geometry.h_target_min": 200.0,
    "geometry.h_target_max": 800.0,
    "geometry.comm_range": 1800.0,
    "detect.target_rcs": 0.05,
}

PRESETS["small-uav-sparse-s3"] = {
    **PRESETS["paper-canonical"],
    "geometry.area_xy": 4000.0,
    "geometry.h_uav_min": 500.0,
    "geometry.h_uav_max": 1200.0,
    "geometry.h_target_min": 500.0,
    "geometry.h_target_max": 1200.0,
    "geometry.comm_range": 2500.0,
    "detect.target_rcs": 0.02,
}

# The only manuscript/result identity currently allowed to carry headline
# claims.  V1.1 remains a gated local successor until its preregistered tests
# pass; scripts and audits can import this constant instead of guessing.
HEADLINE_RELEASE_PRESET = "target-local-v1"

# Isolated successor protocol for the first waveform-calibration phase.  It
# deliberately inherits the frozen V1 operating point and changes only the
# waveform-adapter switch and deterministic threshold-calibration resolution.
# Impairment magnitudes remain zero until an experiment supplies an explicit,
# auditable scenario; this preset must never be used to overwrite V1 results.
PRESETS["target-local-waveform-v2-phase1"] = {
    **PRESETS["target-local-v1"],
    "waveform_impairments.enable": True,
    "detect.fused_calibration_samples": 16_384,
}


def apply_preset(cfg: Config, name: str) -> Config:
    """Return ``cfg`` with the named preset applied.

    >>> cfg = apply_preset(Config(), "isac-consistent")
    """
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return apply_overrides(cfg, PRESETS[name])


def validate_config(cfg: Config) -> None:
    """Reject internally inconsistent physical-model combinations.

    Historical presets remain runnable, but new configurations fail loudly
    when their MAC and payload-interference assumptions describe different
    systems.
    """
    if cfg.scale.M < 2:
        raise ValueError("scale.M must be at least two")
    if cfg.scale.Q < 1:
        raise ValueError("scale.Q must be at least one")
    if cfg.run.num_mc < 1:
        raise ValueError("run.num_mc must be at least one")
    if not math.isfinite(cfg.detect.Pfa_target) or not 0.0 < cfg.detect.Pfa_target < 1.0:
        raise ValueError("detect.Pfa_target must lie in (0, 1)")
    if cfg.detect.n_looks < 1:
        raise ValueError("detect.n_looks must be at least one")
    if cfg.detect.num_h1_per_target < 1:
        raise ValueError("detect.num_h1_per_target must be at least one")
    if cfg.detect.num_false_per_target < 1:
        raise ValueError("detect.num_false_per_target must be at least one")
    corr_values = (
        cfg.corr.rho_tx,
        cfg.corr.rho_rx,
        cfg.corr.rho_target,
        cfg.corr.rho_dd,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in corr_values):
        raise ValueError("correlation coefficients must be finite and non-negative")
    if sum(corr_values) > 1.0 + 1e-12:
        raise ValueError("correlation coefficients must sum to at most one")
    active = cfg.active_sensing
    mode_lengths = {
        len(active.mode_names), len(active.power_scales),
        len(active.looks), len(active.refined),
    }
    if len(mode_lengths) != 1 or not active.mode_names:
        raise ValueError("active sensing mode fields must have equal non-zero lengths")
    if not all(isinstance(name, str) and name.strip() for name in active.mode_names):
        raise ValueError("active sensing mode names must be non-empty strings")
    if len(set(active.mode_names)) != len(active.mode_names):
        raise ValueError("active sensing mode names must be unique")
    if not all(math.isfinite(value) and value > 0.0 for value in active.power_scales):
        raise ValueError("active sensing power scales must be finite and positive")
    if not all(
        isinstance(value, Integral) and not isinstance(value, bool) and value >= 1
        for value in active.looks
    ):
        raise ValueError("active sensing look counts must be positive integers")
    if not all(isinstance(value, bool) for value in active.refined):
        raise ValueError("active sensing refinement flags must be Boolean")
    if (
        not math.isfinite(active.energy_budget_per_target)
        or active.energy_budget_per_target <= 0.0
    ):
        raise ValueError("active sensing energy budget must be finite and positive")
    if (
        not math.isfinite(active.energy_budget_per_uav)
        or active.energy_budget_per_uav <= 0.0
    ):
        raise ValueError("active sensing per-UAV energy budget must be finite and positive")
    if active.max_tx_observations_per_uav < -1:
        raise ValueError("active sensing transmitter cap must be -1 or non-negative")
    if not all(
        math.isfinite(value) and value >= 0.0 for value in (
            active.matched_filter_cycles_per_look,
            active.llr_cycles_per_look,
            active.dd_refine_cycles,
        )
    ):
        raise ValueError("active sensing computation costs must be finite and non-negative")
    if active.max_candidates_per_pair < 1 or active.branch_node_limit < 1:
        raise ValueError("active sensing search limits must be positive")
    if active.candidate_strategy not in {"scenario_union", "robust_singleton", "full"}:
        raise ValueError(
            "active sensing candidate strategy must be scenario_union, robust_singleton, or full"
        )
    if (
        active.scenario_topk_per_scenario < 0
        or active.complementary_pair_topk < 0
        or active.complete_pool_max_links < 1
    ):
        raise ValueError("active sensing candidate-family limits are invalid")
    if active.information_metric not in {"forward_kl", "jeffreys"}:
        raise ValueError("active sensing information metric must be 'forward_kl' or 'jeffreys'")
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (active.energy_price, active.report_price)
    ):
        raise ValueError("active sensing prices must be finite and non-negative")
    if not active.aspect_angles_deg or not all(
        math.isfinite(value) for value in active.aspect_angles_deg
    ):
        raise ValueError("active sensing aspect angles must be finite and non-empty")
    if not math.isfinite(active.aspect_floor) or not 0.0 < active.aspect_floor <= 1.0:
        raise ValueError("active sensing aspect floor must lie in (0, 1]")

    interference_model = cfg.comm.interference_model.lower()
    mac_model = cfg.comm.mac_model.lower()
    if interference_model not in {"full_concurrent", "active_set", "orthogonal"}:
        raise ValueError(
            f"Unknown comm.interference_model={cfg.comm.interference_model!r}; "
            "expected 'full_concurrent', 'active_set', or 'orthogonal'"
        )
    if mac_model not in {"serial", "parallel", "slot"}:
        raise ValueError(
            f"Unknown comm.mac_model={cfg.comm.mac_model!r}; "
            "expected 'serial', 'parallel', or 'slot'"
        )
    if interference_model == "orthogonal" and mac_model != "serial":
        raise ValueError(
            "comm.interference_model='orthogonal' requires comm.mac_model='serial'"
        )
    if cfg.interference.sense_gate_by_active_tx:
        # ``sense_gate_by_active_tx`` declares "a node outside active_tx_mask
        # radiates nothing".  That statement is only complete under the
        # orthogonal 口径, where no report payload is radiated during the
        # sensing observation: ``compute_link_tables`` gates the interference
        # fields (and, since the echo-gate fix, the observation itself) but
        # never the wanted communication signal.  Under a concurrent-payload
        # 口径 a muted reporter would lose its interference while keeping its
        # signal, i.e. the schedule would read better than it can be.  Binding
        # the gate to the 口径 where it is self-consistent is what makes
        # ``active_tx_mask`` mean exactly one thing -- the set of radiating
        # illuminators -- instead of two depending on the caller.
        if interference_model != "orthogonal":
            raise ValueError(
                "interference.sense_gate_by_active_tx=True requires "
                "comm.interference_model='orthogonal' (got "
                f"{cfg.comm.interference_model!r}): under a concurrent-payload "
                "口径 the gate silences interference but not the communication "
                "signal, so it cannot describe a radiation set."
            )
        if cfg.interference.coupling != "shared_spectrum":
            raise ValueError(
                "interference.sense_gate_by_active_tx=True requires "
                "interference.coupling='shared_spectrum' (got "
                f"{cfg.interference.coupling!r}); under the legacy coupling the "
                "mask is not consulted at all, so the gate would be a silent "
                "no-op."
            )
    if cfg.coordination.enable:
        # The gate is what makes the mask physically meaningful; without it
        # ``compute_link_tables`` never consults ``active_tx_mask`` and the
        # coordination loop would re-select against an unchanged denominator --
        # a silent no-op that has already produced one unreportable number.
        if not cfg.interference.sense_gate_by_active_tx:
            raise ValueError(
                "coordination.enable=True requires "
                "interference.sense_gate_by_active_tx=True; otherwise "
                "active_tx_mask is ignored by compute_link_tables and the "
                "coordination fixed point cannot change anything."
            )
        if cfg.coordination.rounds < 1:
            raise ValueError("coordination.rounds must be at least one")
    if cfg.prior.scheduler_rcs.lower() not in {"realized", "mean"}:
        raise ValueError(
            f"Unknown prior.scheduler_rcs={cfg.prior.scheduler_rcs!r}; "
            "expected 'realized' or 'mean'"
        )
    if not 0.0 < cfg.prior.rcs_lower_factor <= 1.0:
        raise ValueError("prior.rcs_lower_factor must lie in (0, 1]")
    if int(cfg.cancellation.n_cpi) < 1:
        raise ValueError("cancellation.n_cpi must be positive")
    if cfg.cancellation.adaptive_soft_enable:
        grid = tuple(float(v) for v in cfg.cancellation.adaptive_soft_mu_grid)
        if not grid or any(not math.isfinite(v) or v < 0.0 for v in grid):
            raise ValueError(
                "adaptive soft mu grid must be finite, non-empty, and non-negative"
            )
        slack = float(cfg.cancellation.adaptive_soft_risk_slack)
        if not math.isfinite(slack) or not 0.0 <= slack <= 1.0:
            raise ValueError("adaptive soft risk slack must lie in [0, 1]")
        quantile = float(cfg.cancellation.adaptive_soft_residual_quantile)
        if not math.isfinite(quantile) or not 0.5 <= quantile < 1.0:
            raise ValueError(
                "adaptive soft residual quantile must lie in [0.5, 1)"
            )
    if int(cfg.cancellation.tangent_order) not in (0, 1):
        raise ValueError(
            "cancellation.tangent_order currently supports only 0 or 1"
        )
    if int(cfg.cancellation.interference_tangent_order) not in (0, 1):
        raise ValueError(
            "cancellation.interference_tangent_order currently supports only 0 or 1"
        )
    if not math.isfinite(cfg.cancellation.tangent_step_bins) or (
        cfg.cancellation.tangent_step_bins <= 0.0
    ):
        raise ValueError("cancellation.tangent_step_bins must be positive")
    if cfg.selector.score_mode.lower() not in {"first_order", "exact_utility", "detector_pd"}:
        raise ValueError(
            f"Unknown selector.score_mode={cfg.selector.score_mode!r}; "
            "expected 'first_order', 'exact_utility', or 'detector_pd'"
        )
    if not 0.0 < cfg.detect.pd_required <= 1.0:
        raise ValueError("detect.pd_required must lie in (0, 1]")
    if not 0.0 < cfg.detect.weak_pd_required <= 1.0:
        raise ValueError("detect.weak_pd_required must lie in (0, 1]")
    if cfg.detect.target_rcs <= 0.0:
        raise ValueError("detect.target_rcs must be positive and is measured in m^2")
    if cfg.radio.rho_by_uav is not None:
        if (len(cfg.radio.rho_by_uav) != cfg.scale.M or
                not all(math.isfinite(x) and 0.0 < x < 1.0 for x in cfg.radio.rho_by_uav)):
            raise ValueError("radio.rho_by_uav must have M finite fractions in (0, 1)")
    radar_db_terms = (
        cfg.radio.radar_tx_gain_dbi,
        cfg.radio.radar_rx_gain_dbi,
        cfg.radio.radar_system_loss_db,
    )
    if not all(math.isfinite(value) for value in radar_db_terms):
        raise ValueError("radar antenna gains and system loss must be finite")
    if cfg.radio.radar_system_loss_db < 0.0:
        raise ValueError("radio.radar_system_loss_db must be non-negative")
    if (cfg.radio.radar_net_gain_db is not None
            and not math.isfinite(cfg.radio.radar_net_gain_db)):
        raise ValueError("radio.radar_net_gain_db must be finite when specified")
    if cfg.detect.fused_calibration_samples < 512:
        raise ValueError("detect.fused_calibration_samples must be at least 512")
    impairment_values = (
        cfg.waveform_impairments.clutter_inr,
        cfg.waveform_impairments.multipath_inr,
        cfg.waveform_impairments.unresolved_target_inr,
    )
    if any(value < 0.0 for value in impairment_values):
        raise ValueError("waveform impairment INR values must be non-negative")
    if cfg.fusion.rule.lower() not in {
        "max_in_rate", "max_min_rate", "nearest_target", "nearest_centroid",
        "nearest_target_capacitated",
        "capacitated_value", "capacitated_pd_lookahead",
    }:
        raise ValueError(
            f"Unknown fusion.rule={cfg.fusion.rule!r}; expected 'max_in_rate', "
            "'max_min_rate', 'nearest_target', 'nearest_target_capacitated', "
            "'capacitated_value', or 'capacitated_pd_lookahead'"
        )
    if cfg.fusion.max_targets_per_uav < -1 or cfg.fusion.max_targets_per_uav == 0:
        raise ValueError("fusion.max_targets_per_uav must be -1 or positive")
    if not math.isfinite(cfg.fusion.processing_window_s) or cfg.fusion.processing_window_s <= 0.0:
        raise ValueError("fusion.processing_window_s must be positive")
    if (
        not math.isfinite(cfg.fusion.cpu_rate_cycles_per_s)
        or cfg.fusion.cpu_rate_cycles_per_s == 0.0
    ):
        raise ValueError("fusion.cpu_rate_cycles_per_s must be negative (off) or positive")
    cpu_costs = (
        cfg.fusion.cpu_fixed_cycles,
        cfg.fusion.cpu_per_observation_cycles,
        cfg.fusion.cpu_cubic_cycles,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in cpu_costs):
        raise ValueError("fusion CPU cost coefficients must be finite and non-negative")
    if cfg.prior.search_gate_sigma < 0:
        raise ValueError("prior.search_gate_sigma must be non-negative")
    if not 0.0 < cfg.prior.robust_position_confidence < 1.0:
        raise ValueError("prior.robust_position_confidence must lie in (0, 1)")
    if cfg.run.workers < 1:
        raise ValueError("run.workers must be at least one")
    if cfg.selector.max_local_observations_per_target < -1:
        raise ValueError(
            "selector.max_local_observations_per_target must be -1 or non-negative"
        )
    if cfg.selector.max_remote_reports < -1:
        raise ValueError("selector.max_remote_reports must be -1 or non-negative")
    if cfg.selector.max_observations_per_receiver < -1:
        raise ValueError(
            "selector.max_observations_per_receiver must be -1 or non-negative"
        )
    if cfg.selector.max_observations_per_fusion_uav < -1:
        raise ValueError(
            "selector.max_observations_per_fusion_uav must be -1 or non-negative"
        )
    if cfg.selector.bundle_shortlist_per_type < 1:
        raise ValueError("selector.bundle_shortlist_per_type must be positive")
    if cfg.selector.bundle_cg_max_iterations < 1:
        raise ValueError("selector.bundle_cg_max_iterations must be positive")
    if cfg.selector.bundle_pricing_tolerance < 0.0:
        raise ValueError("selector.bundle_pricing_tolerance must be non-negative")
    if cfg.selector.bundle_exact_pricing_max_candidates < 0:
        raise ValueError(
            "selector.bundle_exact_pricing_max_candidates must be non-negative"
        )


# --------------------------------------------------------------------------
# Dotted-path override machinery
# --------------------------------------------------------------------------
def _resolve(cfg: Any, path: str) -> Tuple[Any, str]:
    parts = path.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        if not is_dataclass(obj) or not hasattr(obj, part):
            raise KeyError(f"no configuration group {part!r} in path {path!r}")
        obj = getattr(obj, part)
    return obj, parts[-1]


def _coerce(current: Any, value: Any) -> Any:
    """Cast ``value`` to the type of the existing field."""
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if current is None:  # e.g. sensing_processing_gain
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Return a deep copy of ``cfg`` with dotted-path ``overrides`` applied.

    Example
    -------
    >>> apply_overrides(cfg, {"selector.lambda_c": 0.02, "dd.use_otfs_bin_validity": False})
    """
    import copy as _copy

    out = _copy.deepcopy(cfg)
    for path, value in overrides.items():
        owner, name = _resolve(out, path)
        if not hasattr(owner, name):
            raise KeyError(f"unknown configuration field {path!r}")
        setattr(owner, name, _coerce(getattr(owner, name), value))
    return out


def iter_leaf_paths(cfg: Config) -> Iterator[Tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every scalar leaf, for logging/diffing."""

    def walk(obj: Any, prefix: str) -> Iterator[Tuple[str, Any]]:
        for f in fields(obj):
            value = getattr(obj, f.name)
            path = f"{prefix}{f.name}"
            if is_dataclass(value):
                yield from walk(value, f"{path}.")
            else:
                yield path, value

    yield from walk(cfg, "")
