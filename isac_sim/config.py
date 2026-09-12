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
    residual_self_factor: float = 1e-14
    residual_direct_factor: float = 1e-4
    residual_multi_uav_factor: float = 1e-10
    rinr_sigma_factor: float = 0.15


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


@dataclass
class Detect:
    """Sensing-side detection model and soft-information statistics."""

    Pfa_target: float = 0.05
    D_min: float = 3.0
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
    # Ablation switches.
    use_target_priority: bool = True
    use_delay_price: bool = True
    use_comm_error_calibration: bool = True
    # Resource limits.
    max_links_per_target: int = 6
    max_total_links: int = 60
    candidate_topk_per_target: int = 40
    min_marginal_D: float = 0.0
    # Candidate-ranking heuristic (beta).
    beta_scale: float = 5.0
    beta_chi_power: float = 2.0
    beta_delay_exponent: float = 0.5


@dataclass
class Run:
    """Monte-Carlo, reproducibility and reporting knobs."""

    num_mc: int = 200
    seed: int = 2026
    verbose: bool = True


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
