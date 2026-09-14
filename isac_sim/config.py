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
from typing import Any, Dict, Iterator, Literal, Tuple

# --------------------------------------------------------------------------
# Type aliases shared across the package
# --------------------------------------------------------------------------
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
    "topk_deflection",
    "global_topk_deflection",
    "cost_aware_greedy",
    "exact_marginal_greedy",
    "proposed_c2f",
    "proposed_c2f_adaptive",
    "proposed_c2f_adaptive_pd",
    "proposed_c2f_pd",
    "proposed_c2f_full",
]

CommErrorModel = Literal["erasure", "flip", "biased"]
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
class Radio:
    """Transmit power, noise floor, ISAC power split and residual interference."""

    P_default: float = 1.0
    rho: float = 0.80  # sensing power fraction of the joint waveform
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
         reconstructed and subtracted; the residual is set by channel-estimation
         accuracy.  The raw near-far ratio is 30-40 dB, so ``kappa_dc`` is
         exactly the quantity that decides whether the sensing task survives --
         hence it is swept by the ``interference-consistency`` experiment
         instead of being asserted.

    The sensing waveform is assumed to be radiated continuously (that is the
    ISAC premise), so sensing interference is independent of the reporting
    schedule unless ``sense_gate_by_active_tx`` is enabled; only the
    communication payload is gated by which UAVs actually report.
    """

    coupling: str = "shared_spectrum"
    # Direct-path cancellation at the sensing receiver, in dB.  40 dB is the
    # near-far ratio measured for the paper geometry (41.3 dB), i.e. it cancels
    # the direct path down to the echo power level; the sweep covers 0-80 dB.
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
    ``"nearest_target"`` (closest to the predicted target position).
    ``"nearest_centroid"`` is retained as a backward-compatible alias for
    ``"nearest_target"``.
    """

    mode: str = "tx"
    rule: str = "max_in_rate"


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
class Detect:
    """Sensing-side detection model and soft-information statistics."""

    Pfa_target: float = 0.05
    D_min: float = 3.0
    # --- Soft-statistic model --------------------------------------------
    # "gaussian": legacy ``mu = kappa_mu * log(1 + gamma)`` with a hand-set
    #     ``soft_mu_scale``.  There is no derivation for it -- it only encodes
    #     "higher sensing SINR should give a larger soft mean".
    # "llr":      the soft statistic is the *centred local log-likelihood
    #     ratio* of the delay-Doppler matched-filter / local-energy output.
    #     For a Swerling-I target in complex Gaussian noise the bin energy is
    #     exponential, giving
    #         ell  = -ln(1+gamma) + x * gamma/(1+gamma),  x = |z|^2 / sigma_n^2
    #         delta = E1[ell] - E0[ell] = gamma^2/(1+gamma)
    #         sigma0^2 = Var0[ell]      = gamma^2/(1+gamma)^2
    #         J      = D_KL(p1||p0)     = gamma - ln(1+gamma)
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
    # Environment-level pollution used by the Monte-Carlo detector.
    enable_comm_error_pollution: bool = True
    comm_error_model: CommErrorModel = "erasure"
    soft_error_flip_scale: float = 1.0
    soft_error_bias_scale: float = 0.5
    h0_error_bias_scale: float = 0.0
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
    #               companion of the local Swerling-I LLR, whose H1 energy
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
    # Ellipsoidal DD search gate derived from the predicted covariance.  A
    # selected link captures the truth when its delay and Doppler residuals are
    # within this many standard deviations (plus half a quantisation bin).
    search_gate_sigma: float = 3.0
    # What the scheduler knows about target RCS.  ``"realized"`` is the legacy
    # oracle-like path; ``"mean"`` uses E[sigma_q] and prevents a current-CPI
    # RCS realization from leaking into pre-sensing scheduling decisions.
    scheduler_rcs: str = "realized"


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
    """Proposed Lagrangian link-selection rule and its behaviour switches.

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
    # ``detector_pd`` is an experimental matched-budget rule that evaluates the
    # post-report H0/H1 moments and the implemented CF-corrected threshold.
    score_mode: str = "first_order"
    # Historical early exit at D_min is retained for legacy reproduction.  It
    # is disabled in the canonical release because it can stop while the stated
    # utility still has a positive feasible marginal gain.
    stop_at_D_min: bool = True
    # Ablation switches.
    use_target_priority: bool = True
    use_delay_price: bool = True
    use_comm_error_calibration: bool = True
    # Resource limits.
    max_links_per_target: int = 6
    max_total_links: int = 60
    candidate_topk_per_target: int = 40
    min_marginal_D: float = 0.0


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


# --------------------------------------------------------------------------
# Root configuration
# --------------------------------------------------------------------------
@dataclass
class Config:
    scale: Scale = field(default_factory=Scale)
    geometry: Geometry = field(default_factory=Geometry)
    waveform: Waveform = field(default_factory=Waveform)
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
    interference: Interference = field(default_factory=Interference)

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
    # sensing-residual floors.  Kept so that every historical result stays
    # reproducible, but it is no longer the default.
    "legacy": {
        "interference.coupling": "legacy",
        "radio.eps_mode": "legacy",
    },
    # The corrected coupled model with the historical post-selection active-set
    # sensitivity path.  It is not the paper-canonical MAC.
    # is what an experiment that must be immune to future default changes should
    # use.  ``comm.interference_model`` is set to the paper's active-set form.
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
    if cfg.prior.scheduler_rcs.lower() not in {"realized", "mean"}:
        raise ValueError(
            f"Unknown prior.scheduler_rcs={cfg.prior.scheduler_rcs!r}; "
            "expected 'realized' or 'mean'"
        )
    if cfg.selector.score_mode.lower() not in {"first_order", "exact_utility", "detector_pd"}:
        raise ValueError(
            f"Unknown selector.score_mode={cfg.selector.score_mode!r}; "
            "expected 'first_order', 'exact_utility', or 'detector_pd'"
        )
    if cfg.fusion.rule.lower() not in {
        "max_in_rate", "max_min_rate", "nearest_target", "nearest_centroid"
    }:
        raise ValueError(
            f"Unknown fusion.rule={cfg.fusion.rule!r}; expected 'max_in_rate', "
            "'max_min_rate', or 'nearest_target'"
        )
    if cfg.prior.search_gate_sigma < 0:
        raise ValueError("prior.search_gate_sigma must be non-negative")
    if cfg.run.workers < 1:
        raise ValueError("run.workers must be at least one")


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
