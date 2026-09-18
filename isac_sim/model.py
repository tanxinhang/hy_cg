"""Link model: geometry, channel gains and the derived per-link quantity tables.

This module is a faithful port of the v10 prototype's channel layer.  The
arithmetic and the order of random draws are unchanged, so runs remain
bit-for-bit compatible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import Config
from .fbl import chi_from_gamma as _chi_from_gamma
from .llr import soft_mean as _soft_mean
from .llr import soft_var0 as _soft_var0

EPS = 1e-12


# ==========================================================================
# Elementary math helpers
# ==========================================================================
def wavelength(cfg: Config) -> float:
    return cfg.waveform.c / cfg.waveform.fc


def bandwidth(cfg: Config) -> float:
    return cfg.waveform.N * cfg.waveform.delta_f


def noise_power(cfg: Config) -> float:
    psd_w_hz = 10.0 ** ((cfg.radio.noise_psd_dbm_hz - 30.0) / 10.0)
    nf_linear = 10.0 ** (cfg.radio.noise_figure_db / 10.0)
    return psd_w_hz * bandwidth(cfg) * nf_linear


def radar_hardware_gain(cfg: Config) -> float:
    """Linear desired-echo gain from radar Tx/Rx gain and system loss.

    ``target_gain`` deliberately remains the propagation/RCS term so that RCS
    keeps its physical unit of square metres.  Hardware enters the received
    echo separately and defaults to one for exact historical reproducibility.
    """
    r = cfg.radio
    net_db = (
        r.radar_net_gain_db
        if r.radar_net_gain_db is not None
        else r.radar_tx_gain_dbi + r.radar_rx_gain_dbi - r.radar_system_loss_db
    )
    return 10.0 ** (net_db / 10.0)


def denominator_guard(cfg: Config, n0: float) -> float:
    """Additive guard for SINR denominators.

    The guard exists only to avoid a division by zero; it must therefore be
    negligible against the noise floor.  ``model.EPS = 1e-12`` is not -- for the
    default radio it is 26x ``n0``, which suppresses every sensing SINR by
    14.33 dB and hides the sensing interference term entirely.  Under
    ``radio.eps_mode="noise_relative"`` the guard is placed ``eps_rel_db`` below
    the noise power, which is scale-free and harmless at any transmit power.

    An unknown ``eps_mode`` is an error rather than a silent fallback: a typo
    would otherwise quietly reproduce the legacy behaviour and invalidate a
    whole sweep without any visible symptom.
    """
    mode = cfg.radio.eps_mode
    if mode == "noise_relative":
        return n0 * 10.0 ** (cfg.radio.eps_rel_db / 10.0)
    if mode == "legacy":
        return EPS
    raise ValueError(
        f"Unknown radio.eps_mode={mode!r}; expected 'legacy' or 'noise_relative'"
    )


def path_gain(d: np.ndarray, cfg: Config) -> np.ndarray:
    lam = wavelength(cfg)
    d = np.maximum(d, 1.0)
    return (lam / (4.0 * np.pi)) ** 2 / (d ** cfg.detect.path_loss_exp)


def rician_power_gain(shape: Tuple[int, ...], K_db: float, rng: np.random.Generator) -> np.ndarray:
    K = 10.0 ** (K_db / 10.0)
    los = np.sqrt(K / (K + 1.0))
    nlos = np.sqrt(1.0 / (K + 1.0)) * (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ) / np.sqrt(2.0)
    return np.abs(los + nlos) ** 2


def qfunc(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return 0.5 * math.erfc(float(arr.item()) / math.sqrt(2.0))
    vals = [0.5 * math.erfc(float(v) / math.sqrt(2.0)) for v in arr.ravel()]
    return np.array(vals, dtype=float).reshape(arr.shape)


def normal_pdf(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    out = np.exp(-0.5 * arr * arr) / math.sqrt(2.0 * math.pi)
    if out.ndim == 0:
        return float(out.item())
    return out


def threshold_from_pfa(cfg: Config) -> float:
    in_built = {0.10: 1.2816, 0.05: 1.6449, 0.01: 2.3263, 0.001: 3.0902}
    for pfa, thr in in_built.items():
        if abs(cfg.detect.Pfa_target - pfa) < 1e-12:
            return thr
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if qfunc(mid) > cfg.detect.Pfa_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pd_from_deflection(cfg: Config, D: np.ndarray | float) -> np.ndarray:
    eta = threshold_from_pfa(cfg)
    D_arr = np.asarray(D, dtype=float)
    return np.asarray(qfunc(eta - np.sqrt(np.maximum(D_arr, 0.0))), dtype=float)


def d_pd_d_D(cfg: Config, D: np.ndarray | float) -> np.ndarray:
    """Derivative of ``Q(eta - sqrt(D))`` with respect to ``D``."""
    eta = threshold_from_pfa(cfg)
    D_arr = np.asarray(D, dtype=float)
    sqrtD = np.sqrt(np.maximum(D_arr, 1e-4))
    z = eta - sqrtD
    return np.asarray(normal_pdf(z), dtype=float) / (2.0 * sqrtD)


def binomial_ci95(success: int, total: int) -> Tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    z = 1.96
    n = float(total)
    p = success / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi, 0.5 * (hi - lo)


# ==========================================================================
# Containers
# ==========================================================================
@dataclass
class Geometry:
    p_uav: np.ndarray
    v_uav: np.ndarray
    p_tgt: np.ndarray
    v_tgt: np.ndarray


@dataclass
class BaseGains:
    """Random geometry-dependent quantities that do not depend on the algorithm."""

    edge_mask: np.ndarray
    direct_gain: np.ndarray
    d_uu: np.ndarray
    d_uav_tgt: np.ndarray
    tau: np.ndarray
    doppler: np.ndarray
    delay_bin: np.ndarray
    doppler_bin: np.ndarray
    valid_dd: np.ndarray
    dd_frac_loss: np.ndarray
    dd_collision_count: np.ndarray
    target_gain: np.ndarray
    geom_factor: np.ndarray
    # Azimuth of the bistatic viewing bisector in the horizontal target frame.
    # Used only by the opt-in active-evidence aspect-scenario model.
    aspect_azimuth: np.ndarray
    # Per-(i,j,q) RCS realisation.  Stored so a second geometry (e.g. the
    # scheduler's *belief*) can reuse the exact same physical channel -- the
    # UAV-UAV fading and the target RCS are physical quantities and must not be
    # re-drawn just because the tracker's belief differs from the truth.
    rcs_fluct: np.ndarray
    # C2F DD-refinement arrays (paper eq. coarse/fine_dd_gain).  When
    # ``cfg.refine`` is disabled both fields equal ``dd_frac_loss`` so the
    # existing sensing-SINR computation stays bit-exact.
    eta_loc: np.ndarray
    eta_fine: np.ndarray
    # Fractional DD centres in axis units (signed) for the eta^loc window sum.
    delay_frac: np.ndarray
    doppler_frac: np.ndarray


@dataclass
class LinkTables:
    """Per-link quantities derived from :class:`BaseGains` and the power split."""

    gamma_comm: np.ndarray
    rate: np.ndarray
    chi_comm: np.ndarray
    feasible_comm: np.ndarray
    raw_gamma_sense: np.ndarray
    gamma_sense: np.ndarray
    rinr: np.ndarray
    mu_soft: np.ndarray
    sigma0: np.ndarray
    # Per-(i,j,q) H0 variance of the soft statistic.  In the legacy Gaussian
    # model this equals ``sigma0[i,j]^2`` broadcast over ``q``; in the LLR model
    # it is *derived* from the sensing SINR (``L*gamma^2/(1+gamma)^2``).
    var0_q: np.ndarray


def rescale_sensing_tables_for_rcs(
    cfg: Config,
    tables: LinkTables,
    factor: float | np.ndarray,
) -> LinkTables:
    """Return a non-mutating RCS-counterfactual sensing table.

    For fixed geometry, power, interference, and waveform impairments, RCS is
    linear in the desired echo power and therefore in sensing SINR. ``factor``
    may be scalar or one positive multiplier per target. Communication fields
    are unchanged. Soft-statistic moments are recomputed from the scaled SINR
    rather than scaled directly, preserving the nonlinear LLR model.
    """
    factors = np.asarray(factor, dtype=float)
    if factors.ndim == 0:
        factors = np.full(cfg.scale.Q, float(factors), dtype=float)
    if factors.shape != (cfg.scale.Q,):
        raise ValueError(
            f"RCS factor must be scalar or shape ({cfg.scale.Q},), got {factors.shape}"
        )
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("RCS factors must be finite and strictly positive")

    scale = factors.reshape(1, 1, cfg.scale.Q)
    raw_gamma = np.asarray(tables.raw_gamma_sense, dtype=float) * scale
    gamma = np.asarray(tables.gamma_sense, dtype=float) * scale
    mu_soft = np.asarray(_soft_mean(cfg, gamma), dtype=float)
    pair_var0 = np.broadcast_to(
        np.asarray(tables.sigma0, dtype=float)[:, :, None] ** 2,
        gamma.shape,
    )
    var0_q = np.asarray(_soft_var0(cfg, gamma, pair_var0), dtype=float)
    return LinkTables(
        gamma_comm=tables.gamma_comm,
        rate=tables.rate,
        chi_comm=tables.chi_comm,
        feasible_comm=tables.feasible_comm,
        raw_gamma_sense=raw_gamma,
        gamma_sense=gamma,
        rinr=tables.rinr,
        mu_soft=mu_soft,
        sigma0=tables.sigma0,
        var0_q=var0_q,
    )


# ==========================================================================
# Geometry sampling
# ==========================================================================
def generate_geometry(cfg: Config, rng: np.random.Generator) -> Geometry:
    g = cfg.geometry
    M, Q = cfg.scale.M, cfg.scale.Q

    p_uav = np.zeros((M, 3))
    p_uav[:, :2] = rng.uniform(0.0, g.area_xy, size=(M, 2))
    p_uav[:, 2] = rng.uniform(g.h_uav_min, g.h_uav_max, size=M)

    p_tgt = np.zeros((Q, 3))
    p_tgt[:, :2] = rng.uniform(0.0, g.area_xy, size=(Q, 2))
    p_tgt[:, 2] = rng.uniform(g.h_target_min, g.h_target_max, size=Q)

    v_uav = np.zeros((M, 3))
    sp_u = rng.uniform(g.uav_speed_min, g.uav_speed_max, size=M)
    ang_u = rng.uniform(0.0, 2.0 * np.pi, size=M)
    v_uav[:, 0] = sp_u * np.cos(ang_u)
    v_uav[:, 1] = sp_u * np.sin(ang_u)

    v_tgt = np.zeros((Q, 3))
    sp_t = rng.uniform(g.target_speed_min, g.target_speed_max, size=Q)
    ang_t = rng.uniform(0.0, 2.0 * np.pi, size=Q)
    v_tgt[:, 0] = sp_t * np.cos(ang_t)
    v_tgt[:, 1] = sp_t * np.sin(ang_t)

    geom = Geometry(p_uav=p_uav, v_uav=v_uav, p_tgt=p_tgt, v_tgt=v_tgt)

    # Target-state prior perturbation.  When both sigmas are zero the simulator
    # consumes the geometry untouched and the RNG stream is unchanged -- so
    # default-config runs remain bit-exact with the parity golden values.
    if cfg.prior.sigma_pos_m > 0 or cfg.prior.sigma_vel_mps > 0:
        from .prior import perturbed_geometry

        geom = perturbed_geometry(
            cfg,
            geom,
            cfg.prior.sigma_pos_m,
            cfg.prior.sigma_vel_mps,
            rng,
        )

    return geom


# ==========================================================================
# Geometry-dependent gains and OTFS delay-Doppler bins
# ==========================================================================
def build_base_gains(
    cfg: Config,
    geom: Geometry,
    rng: np.random.Generator,
    channel: BaseGains | None = None,
    rcs_view: str = "realized",
) -> BaseGains:
    """Build the base gains for a geometry.

    ``channel``, when given, supplies the UAV-UAV physical realisation to reuse.
    With ``rcs_view="realized"`` it also supplies target RCS (legacy behavior).
    With ``rcs_view="mean"`` the target gain uses the configured mean RCS, so a
    scheduler cannot observe the current-CPI RCS realization before sensing.
    """
    M, Q = cfg.scale.M, cfg.scale.Q
    d = cfg.detect
    w = cfg.waveform
    lam = wavelength(cfg)

    if channel is not None:
        d_uu = channel.d_uu
        edge_mask = channel.edge_mask
        direct_gain = channel.direct_gain
    else:
        diff_uu = geom.p_uav[:, None, :] - geom.p_uav[None, :, :]
        d_uu = np.linalg.norm(diff_uu, axis=-1)
        edge_mask = (d_uu <= cfg.geometry.comm_range) & (~np.eye(M, dtype=bool))

        direct_gain = path_gain(d_uu, cfg)
        direct_gain *= 10.0 ** (rng.normal(0.0, d.shadow_std_db, size=(M, M)) / 10.0)
        direct_gain *= rician_power_gain((M, M), d.rician_K_db, rng)
        np.fill_diagonal(direct_gain, 0.0)

    diff_uav_tgt = geom.p_uav[:, None, :] - geom.p_tgt[None, :, :]
    d_uav_tgt = np.linalg.norm(diff_uav_tgt, axis=-1)

    tau = np.zeros((M, M, Q))
    doppler = np.zeros((M, M, Q))
    delay_bin = np.zeros((M, M, Q), dtype=int)
    doppler_bin = np.zeros((M, M, Q), dtype=int)
    valid_dd = np.zeros((M, M, Q), dtype=bool)
    dd_frac_loss = np.ones((M, M, Q), dtype=float)
    delay_frac_full = np.zeros((M, M, Q), dtype=float)
    doppler_frac_full = np.zeros((M, M, Q), dtype=float)
    l_float_grid = np.zeros((M, M, Q), dtype=float)
    k_float_grid = np.zeros((M, M, Q), dtype=float)
    target_gain = np.zeros((M, M, Q))
    geom_factor = np.zeros((M, M, Q))
    aspect_azimuth = np.zeros((M, M, Q))
    rcs_fluct = np.zeros((M, M, Q))

    # Swerling-I target: one exponential RCS realisation per (target, CPI),
    # shared by every bistatic pair observing that target.  Only drawn when
    # ``detect.rcs_model == "swerling1"`` so the default (iid) path keeps its
    # exact RNG stream.
    rcs_shared: np.ndarray | None = None
    if channel is None and d.rcs_model == "swerling1":
        rcs_shared = rng.exponential(scale=d.target_rcs, size=Q)
    elif channel is None and d.rcs_model == "mean":
        rcs_shared = np.full(Q, d.target_rcs, dtype=float)

    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            for q in range(Q):
                p_i, p_j, p_q = geom.p_uav[i], geom.p_uav[j], geom.p_tgt[q]
                v_i, v_j, v_q = geom.v_uav[i], geom.v_uav[j], geom.v_tgt[q]

                vec_iq = p_q - p_i
                vec_jq = p_q - p_j
                d_iq = max(float(np.linalg.norm(vec_iq)), 1.0)
                d_jq = max(float(np.linalg.norm(vec_jq)), 1.0)
                u_iq = vec_iq / d_iq
                u_jq = vec_jq / d_jq

                tau_ijq = (d_iq + d_jq) / w.c
                rr_tx = float(np.dot(v_q - v_i, u_iq))
                rr_rx = float(np.dot(v_q - v_j, u_jq))
                nu_ijq = (rr_tx + rr_rx) / lam

                tau[i, j, q] = tau_ijq
                doppler[i, j, q] = nu_ijq

                l_float = tau_ijq * w.L * w.delta_f
                k_float = nu_ijq * w.N * w.T
                l_bin = int(np.round(l_float))
                k_bin = int(np.round(k_float))
                delay_bin[i, j, q] = l_bin
                doppler_bin[i, j, q] = k_bin
                valid_dd[i, j, q] = (0 <= l_bin < w.L) and (-(w.N // 2) <= k_bin < w.N // 2)

                delay_frac = abs(l_float - l_bin)
                doppler_frac = abs(k_float - k_bin)
                dd_frac_loss[i, j, q] = float((np.sinc(delay_frac) ** 2) * (np.sinc(doppler_frac) ** 2))
                # Signed fractional offsets (needed by eta_local_sinc / Dirichlet).
                delay_frac_full[i, j, q] = float(l_float - l_bin)
                doppler_frac_full[i, j, q] = float(k_float - k_bin)
                l_float_grid[i, j, q] = float(l_float)
                k_float_grid[i, j, q] = float(k_float)

                a = (p_i - p_q) / d_iq
                b = (p_j - p_q) / d_jq
                sin_angle = float(np.linalg.norm(np.cross(a, b)))
                geom_factor[i, j, q] = 0.1 + 0.9 * min(max(sin_angle, 0.0), 1.0)
                bisector_xy = a[:2] + b[:2]
                if float(np.linalg.norm(bisector_xy)) <= 1e-12:
                    bisector_xy = a[:2]
                aspect_azimuth[i, j, q] = float(
                    np.arctan2(bisector_xy[1], bisector_xy[0])
                )

                if rcs_view == "mean":
                    rcs_fluct[i, j, q] = float(d.target_rcs)
                    if d.rcs_aspect_enable:
                        rcs_fluct[i, j, q] *= 0.2 + 0.8 * min(max(sin_angle, 0.0), 1.0)
                elif channel is not None:
                    rcs_fluct[i, j, q] = float(channel.rcs_fluct[i, j, q])
                elif rcs_shared is not None:
                    rcs_fluct[i, j, q] = float(rcs_shared[q])
                    if d.rcs_aspect_enable:
                        # Bistatic aspect factor: forward/back-scatter sees the
                        # full RCS, near-specular sees less.
                        rcs_fluct[i, j, q] *= 0.2 + 0.8 * min(max(sin_angle, 0.0), 1.0)
                else:
                    rcs_fluct[i, j, q] = rng.exponential(scale=d.target_rcs)
                target_gain[i, j, q] = lam ** 2 * rcs_fluct[i, j, q] / ((4.0 * np.pi) ** 3 * d_iq ** 2 * d_jq ** 2)

    dd_collision_count = np.ones((M, M, Q), dtype=float)
    if cfg.dd.enable_dd_collision_penalty:
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                bins: Dict[Tuple[int, int], List[int]] = {}
                for q in range(Q):
                    if not valid_dd[i, j, q]:
                        continue
                    key = (int(delay_bin[i, j, q]), int(doppler_bin[i, j, q]))
                    bins.setdefault(key, []).append(q)
                for qs in bins.values():
                    for q in qs:
                        dd_collision_count[i, j, q] = float(len(qs))

    # C2F DD-refinement (paper eq. coarse/fine_dd_gain).  When refinement is
    # disabled both arrays equal ``dd_frac_loss`` so the existing path is
    # bit-exact.  Computation is cheap: 2 * (2W+1)^2 Dirichlet evaluations per
    # link, with the leakage distribution cached on a 1e-3 grid.
    from .dd import eta_local_array, eta_fine_array

    if cfg.refine.enable or cfg.refine.apply_to_all:
        eta_loc = eta_local_array(
            cfg,
            np.where(valid_dd, l_float_grid, np.nan),
            np.where(valid_dd, k_float_grid, np.nan),
        )
        eta_fine = eta_fine_array(cfg, dd_frac_loss, eta_loc)
    else:
        eta_loc = dd_frac_loss.copy()
        eta_fine = dd_frac_loss.copy()

    return BaseGains(
        edge_mask=edge_mask,
        direct_gain=direct_gain,
        d_uu=d_uu,
        d_uav_tgt=d_uav_tgt,
        tau=tau,
        doppler=doppler,
        delay_bin=delay_bin,
        doppler_bin=doppler_bin,
        valid_dd=valid_dd,
        dd_frac_loss=dd_frac_loss,
        dd_collision_count=dd_collision_count,
        target_gain=target_gain,
        geom_factor=geom_factor,
        aspect_azimuth=aspect_azimuth,
        rcs_fluct=rcs_fluct,
        eta_loc=eta_loc,
        eta_fine=eta_fine,
        delay_frac=delay_frac_full,
        doppler_frac=doppler_frac_full,
    )


# ==========================================================================
# Communication and sensing SINR tables
# ==========================================================================
def packet_bits_for_target(cfg: Config, q: int) -> float:
    """Payload bits carried per candidate packet (currently target-independent)."""
    return float(max(cfg.comm.K_candidates, 0) * cfg.comm.b_d)


def compute_link_tables(
    cfg: Config,
    base: BaseGains,
    dd_gain: np.ndarray | None = None,
    active_tx_mask: np.ndarray | None = None,
    reuse_from: LinkTables | None = None,
    sensing_power_scale_by_uav: np.ndarray | None = None,
) -> LinkTables:
    """Per-link communication / sensing-SINR quantities.

    ``dd_gain`` is an optional ``(M, M, Q)`` override for the DD gain.  When
    omitted (the default) the original coarse ``dd_frac_loss`` is used so the
    trial outputs stay bit-exact.  The C2F selector passes ``base.eta_fine``
    to obtain the fine-grain sensing-SINR table used at the fine stage.

    ``active_tx_mask`` is an optional ``(M,)`` boolean array marking the UAVs
    that are concurrently transmitting.  When ``cfg.comm.interference_model``
    is ``"active_set"`` only those UAVs contribute to the communication
    interference term, so a smaller selected link set sees less interference.
    ``None`` (the default, and the ``"full_concurrent"`` model) keeps the
    original worst-case behaviour where every UAV contributes.  Under
    ``interference_model="orthogonal"`` report payload interference is zero,
    while continuous sensing-waveform leakage remains in the denominator.

    ``reuse_from`` is a performance fast path for the ``active_set`` model.
    Under the default ``sensing_only`` (or ``joint_waveform``) power model the
    sensing-side quantities -- ``gamma_sense``, ``mu_soft``, ``sigma0``,
    ``rinr``, ``raw_gamma_sense`` -- do **not** depend on the communication
    interference, so they can be copied verbatim from the table built at
    selection time and only the O(M^2) communication block needs recomputing.
    This removes the O(M^2 * Q) sensing loop (by far the dominant cost) from
    every re-evaluation.  It is ignored for ``reliable_comm_assisted``, where
    the sensing power itself depends on ``chi_comm``.

    ``sensing_power_scale_by_uav`` applies active-mode power at the transmitter
    before either desired-signal or leakage fields are formed.  Thus raising a
    UAV's sensing power strengthens its own echoes and increases the
    interference seen by other receivers.  The default all-one vector exactly
    preserves the released passive model.
    """
    M, Q = cfg.scale.M, cfg.scale.Q
    r, c, d = cfg.radio, cfg.comm, cfg.detect
    dd_used = base.dd_frac_loss if dd_gain is None else dd_gain

    # Sensing quantities are independent of the communication interference
    # except for the "reliable_comm_assisted" power model, where the effective
    # sensing power is inflated by chi_comm.
    # Sensing quantities depend on ``dd_gain``, so the fast path is only valid
    # when the caller did not override it (the active_set re-evaluation passes
    # ``dd_gain=None`` and reuses the selection-stage sensing block).
    can_reuse_sensing = (
        reuse_from is not None
        and r.rho_by_uav is None
        and sensing_power_scale_by_uav is None
        and dd_gain is None
        and r.isac_power_model != "reliable_comm_assisted"
        # Under active-set coupling the sensing denominator contains the active
        # report payloads and must be rebuilt.  Orthogonal reporting contains no
        # payload term during the sensing observation, so its sensing block is
        # schedule-independent and remains reusable.
        and not (
            cfg.interference.coupling == "shared_spectrum"
            and active_tx_mask is not None
        )
    )

    P = np.full(M, r.P_default, dtype=float)
    rho = r.rho
    if r.rho_by_uav is not None:
        rho = np.asarray(r.rho_by_uav, dtype=float)
        if rho.shape != (M,) or not np.all(np.isfinite(rho)) or np.any((rho <= 0) | (rho >= 1)):
            raise ValueError(f"rho_by_uav must contain {M} finite fractions in (0, 1)")
    P_sense = rho * P
    if sensing_power_scale_by_uav is not None:
        power_scale = np.asarray(sensing_power_scale_by_uav, dtype=float)
        if power_scale.shape != (M,) or not np.all(np.isfinite(power_scale)):
            raise ValueError(f"sensing power scale must be finite with shape ({M},)")
        if np.any(power_scale <= 0.0):
            raise ValueError("sensing power scales must be strictly positive")
        P_sense = P_sense * power_scale
    P_comm = (1.0 - rho) * P

    n0 = noise_power(cfg)
    eps_den = denominator_guard(cfg, n0)
    B = bandwidth(cfg)
    gamma_req = max(2.0 ** (c.R_min / B) - 1.0, EPS)

    # ---- One interference field, shared by both receivers -----------------
    # Under ``interference.coupling="shared_spectrum"`` the communication and
    # the sensing receiver at node j are built from the same concurrent
    # transmitters, the same direct-path gains and the same radiated powers.
    # The three field vectors below do not depend on the illuminator i, only on
    # the receiving node j, so they are computed once per table build.
    # ``direct_gain`` has a zero diagonal, hence ``P @ G`` sums over k != j.
    shared = cfg.interference.coupling == "shared_spectrum"
    if not shared and cfg.interference.coupling != "legacy":
        raise ValueError(
            f"Unknown interference.coupling={cfg.interference.coupling!r}; "
            f"expected 'legacy' or 'shared_spectrum'"
        )
    if shared:
        ic = cfg.interference
        # Every UAV continuously radiates its sensing component.  Payload power
        # follows the actual reporting MAC: all nodes in full-concurrent mode,
        # selected reporters in active-set mode, and no payload during the
        # sensing observation in the conference release's orthogonal phase.
        if c.interference_model == "orthogonal":
            P_rad_sense = P_sense
            P_rad_pay = np.zeros_like(P_comm)
        elif active_tx_mask is not None:
            P_rad_sense = P_sense + P_comm * active_tx_mask
            P_rad_pay = P_comm * active_tx_mask
        else:
            P_rad_sense = P_sense + P_comm
            P_rad_pay = P_comm
        P_leak = P_sense                                  # always radiated
        # --- Coordination gate --------------------------------------------
        # A muted node radiates NOTHING: not its sensing waveform, and not its
        # leakage into the other receivers. The gated power must reproduce the
        # UNGATED expression of the current 口径 -- the previous implementation
        # always used ``(P_sense + P_comm) * mask``, which under the orthogonal
        # release口径 silently switched the radiated power from ``P_sense``
        # (= rho*P) to ``P_sense + P_comm`` (= P): 25% more interference, i.e.
        # enabling coordination made the baseline worse before the coordination
        # could help. With the fix, gating with an all-True mask is a no-op.
        # Only active when the gate is enabled, so the frozen default path is
        # bit-exact.
        # ``gate_echo`` mirrors the block below into the *observation* path: a
        # muted node radiates nothing, so its echo cannot be received either.
        # Without it the gate deletes a muted illuminator's interference but
        # keeps its observation, so a mask that omits an illuminator the
        # schedule still uses is credited with detection it could not achieve
        # (measured before the fix: with every node muted, 2100 of 2250
        # sensing-SINR entries were still positive).
        gate_echo = bool(ic.sense_gate_by_active_tx and active_tx_mask is not None)
        if gate_echo:
            if c.interference_model == "orthogonal":
                P_rad_sense = P_sense * active_tx_mask
                P_rad_pay = np.zeros_like(P_comm)
            else:
                P_rad_sense = (P_sense + P_comm) * active_tx_mask
                P_rad_pay = P_comm * active_tx_mask
            P_leak = P_sense * active_tx_mask
        I_sense_field = P_rad_sense @ base.direct_gain      # (M,) direct-path field at j
        I_pay_field = P_rad_pay @ base.direct_gain          # (M,) report-payload field at j
        I_leak_field = P_leak @ base.direct_gain            # (M,) sensing-waveform leakage at j
        kappa_dc = 10.0 ** (-ic.direct_cancellation_db / 10.0)
    else:
        I_sense_field = I_pay_field = I_leak_field = None
        P_leak = None
        kappa_dc = 0.0
        gate_echo = False

    gamma_comm = np.zeros((M, M))
    rate = np.zeros((M, M))
    chi_comm = np.zeros((M, M))
    feasible_comm = np.zeros((M, M), dtype=bool)

    for i in range(M):
        for j in range(M):
            if i == j or not base.edge_mask[i, j]:
                continue

            signal = P_comm[i] * base.direct_gain[i, j]
            if shared:
                # Same field as the sensing receiver at j, minus i's own
                # contribution -- i is the wanted source on this link, not an
                # interferer.  The subtracted terms must match the powers that
                # were summed into the fields, otherwise a silent transmitter
                # would subtract a contribution it never made and the
                # interference could turn negative.
                interf = (
                    I_pay_field[j]
                    - P_rad_pay[i] * base.direct_gain[i, j]
                    + c.comm_leakage_from_sensing
                    * (I_leak_field[j] - P_leak[i] * base.direct_gain[i, j])
                )
            else:
                interf = 0.0
                for k in range(M):
                    if k == i or k == j:
                        continue
                    # "active_set" model: a UAV only interferes if it is actually
                    # transmitting (i.e. it is the sending end of a selected
                    # reporting link).  Silent UAVs contribute nothing.
                    if active_tx_mask is not None and not active_tx_mask[k]:
                        continue
                    interf += (P_comm[k] + c.comm_leakage_from_sensing * P_sense[k]) * base.direct_gain[k, j]

            direct_leakage = c.comm_direct_leakage_factor * P_sense[i] * base.direct_gain[i, j]
            gamma = signal / (n0 + interf + direct_leakage + eps_den)
            gamma_comm[i, j] = gamma
            rate[i, j] = B * np.log2(1.0 + gamma)
            # Reliability under the configured model: the legacy heuristic
            # ``gamma/(gamma+gamma_req)`` or the finite-blocklength success
            # probability ``1 - Q(...)``.  Dispatched through fbl.py so the two
            # stay bit-compatible when ``reliability_model="heuristic"``.
            chi_comm[i, j] = _chi_from_gamma(cfg, gamma, gamma_req)

    # chi_comm = gamma / (gamma + gamma_req) lies in [0, 1) by construction.
    # Clip once here (vectorised, O(M^2)) so the fusion hot path -- which is
    # called millions of times -- never pays for a scalar np.clip.
    np.clip(chi_comm, 0.0, 1.0, out=chi_comm)

    # Feasibility of the *reporting* link.  The soft statistic s_{ijq} is
    # produced at the receiving UAV j and reported back to the transmitting
    # UAV i, so the reporting direction is j -> i and feasibility is judged by
    # rate[j, i] / chi_comm[j, i] (NOT rate[i, j]).
    for i in range(M):
        for j in range(M):
            if i == j or not base.edge_mask[i, j]:
                continue
            if c.enforce_chi_min:
                feasible_comm[i, j] = (rate[j, i] >= c.R_min) and (chi_comm[j, i] >= c.chi_min)
            else:
                feasible_comm[i, j] = (rate[j, i] >= c.R_min)

    # ---- Fast path -------------------------------------------------------
    # Only the communication block depends on the active transmitting set, so
    # the (much larger) sensing block can be reused.  This is what makes the
    # "active_set" model affordable inside the Monte-Carlo loop.
    if can_reuse_sensing:
        return LinkTables(
            gamma_comm=gamma_comm,
            rate=rate,
            chi_comm=chi_comm,
            feasible_comm=feasible_comm,
            raw_gamma_sense=reuse_from.raw_gamma_sense,
            gamma_sense=reuse_from.gamma_sense,
            rinr=reuse_from.rinr,
            mu_soft=reuse_from.mu_soft,
            sigma0=reuse_from.sigma0,
            var0_q=reuse_from.var0_q,
        )

    raw_gamma_sense = np.zeros((M, M, Q))
    gamma_sense = np.zeros((M, M, Q))
    rinr = np.zeros((M, M))
    mu_soft = np.zeros((M, M, Q))
    var0_q = np.zeros((M, M, Q))
    sigma0 = np.full((M, M), d.soft_sigma0)

    G_proc = cfg.waveform.N * cfg.waveform.L if d.sensing_processing_gain is None else d.sensing_processing_gain
    G_hw = radar_hardware_gain(cfg)

    for i in range(M):
        for j in range(M):
            if i == j:
                continue

            residual_self = r.residual_self_factor * P[j]
            if shared:
                # Same interferer set, same gains and same radiated powers as the
                # communication receiver at j -- only the receiver's suppression
                # differs.  Note that the illuminator i is NOT excluded: its
                # direct path is precisely the near-far term that a bistatic
                # sensing receiver has to cancel.
                residual_direct = kappa_dc * float(I_sense_field[j])
                residual_multi = 0.0
            else:
                residual_direct = 0.0
                residual_multi = 0.0
                # Direct-link cancellation error and multi-UAV sensing leakage share
                # the same interferer set, so both are accumulated in one loop.
                for k in range(M):
                    if k == i or k == j:
                        continue
                    residual_direct += r.residual_direct_factor * P[k] * base.direct_gain[k, j]
                    residual_multi += r.residual_multi_uav_factor * P_sense[k] * base.direct_gain[k, j]

            residual_total = residual_self + residual_direct + residual_multi
            rinr[i, j] = residual_total / (n0 + eps_den)
            sigma0[i, j] = d.soft_sigma0 * math.sqrt(1.0 + r.rinr_sigma_factor * rinr[i, j])

            if r.isac_power_model == "sensing_only":
                effective_sensing_power = P_sense[i]
            elif r.isac_power_model == "joint_waveform":
                effective_sensing_power = P_sense[i] + P_comm[i]
            elif r.isac_power_model == "reliable_comm_assisted":
                effective_sensing_power = P_sense[i] + chi_comm[i, j] * P_comm[i]
            else:
                raise ValueError(f"Unknown isac_power_model={r.isac_power_model!r}")

            if gate_echo and not active_tx_mask[i]:
                # Muted illuminator: no waveform was radiated on this CPI, so
                # there is nothing to receive.  Both ``signal`` and
                # ``raw_signal`` are built from this factor, so one assignment
                # covers the gated observation and its raw SINR baseline.
                effective_sensing_power = 0.0

            for q in range(Q):
                if cfg.dd.use_otfs_bin_validity and (not base.valid_dd[i, j, q]):
                    continue

                # Keep the raw sensing-SINR baseline consistent with the selected ISAC power model.
                raw_signal = effective_sensing_power * base.target_gain[i, j, q] * G_proc * G_hw
                raw_gamma_sense[i, j, q] = raw_signal / (n0 + eps_den)

                collision_penalty = 1.0 / (max(base.dd_collision_count[i, j, q], 1.0) ** cfg.dd.dd_collision_alpha)
                dd_loss = dd_used[i, j, q] if cfg.dd.enable_dd_fractional_penalty else 1.0

                signal = effective_sensing_power * base.target_gain[i, j, q] * G_proc * G_hw * collision_penalty * dd_loss
                waveform_capture = 1.0
                waveform_inr = 0.0
                if cfg.waveform_impairments.enable:
                    wi = cfg.waveform_impairments
                    waveform_capture = float(
                        np.sinc(wi.sync_delay_bins) ** 2
                        * np.sinc(wi.sync_doppler_bins) ** 2
                    )
                    unresolved = max(base.dd_collision_count[i, j, q] - 1.0, 0.0)
                    waveform_inr = float(
                        wi.clutter_inr
                        + wi.multipath_inr
                        + wi.unresolved_target_inr * unresolved
                    )
                gamma = (
                    signal * waveform_capture
                    / ((n0 + residual_total + eps_den) * (1.0 + waveform_inr))
                )
                gamma_sense[i, j, q] = gamma
                # Soft statistic under the configured model: the legacy
                # ``kappa_mu * log(1+gamma)`` Gaussian mean, or the centred
                # local LLR mean ``L*gamma^2/(1+gamma)``.  Dispatched through
                # llr.py so the two stay bit-compatible in "gaussian" mode.
                mu_soft[i, j, q] = _soft_mean(cfg, gamma)
                # H0 variance: legacy broadcasts the pair-level sigma0^2, the
                # LLR model derives it from the SINR.
                var0_q[i, j, q] = _soft_var0(cfg, gamma, float(sigma0[i, j]) ** 2)

    return LinkTables(
        gamma_comm=gamma_comm,
        rate=rate,
        chi_comm=chi_comm,
        feasible_comm=feasible_comm,
        raw_gamma_sense=raw_gamma_sense,
        gamma_sense=gamma_sense,
        rinr=rinr,
        mu_soft=mu_soft,
        sigma0=sigma0,
        var0_q=var0_q,
    )
