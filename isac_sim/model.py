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
    beta: np.ndarray
    mu_soft: np.ndarray
    sigma0: np.ndarray


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
def build_base_gains(cfg: Config, geom: Geometry, rng: np.random.Generator) -> BaseGains:
    M, Q = cfg.scale.M, cfg.scale.Q
    d = cfg.detect
    w = cfg.waveform
    lam = wavelength(cfg)

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

                rcs_fluct = rng.exponential(scale=d.target_rcs)
                target_gain[i, j, q] = lam ** 2 * rcs_fluct / ((4.0 * np.pi) ** 3 * d_iq ** 2 * d_jq ** 2)

                a = (p_i - p_q) / d_iq
                b = (p_j - p_q) / d_jq
                sin_angle = float(np.linalg.norm(np.cross(a, b)))
                geom_factor[i, j, q] = 0.1 + 0.9 * np.clip(sin_angle, 0.0, 1.0)

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
) -> LinkTables:
    """Per-link communication / sensing-SINR quantities.

    ``dd_gain`` is an optional ``(M, M, Q)`` override for the DD gain.  When
    omitted (the default) the original coarse ``dd_frac_loss`` is used so the
    trial outputs stay bit-exact.  The C2F selector passes ``base.eta_fine``
    to obtain the fine-grain sensing-SINR table used at the fine stage.
    """
    M, Q = cfg.scale.M, cfg.scale.Q
    r, c, d = cfg.radio, cfg.comm, cfg.detect
    dd_used = base.dd_frac_loss if dd_gain is None else dd_gain

    P = np.full(M, r.P_default, dtype=float)
    P_sense = r.rho * P
    P_comm = (1.0 - r.rho) * P

    n0 = noise_power(cfg)
    B = bandwidth(cfg)
    gamma_req = max(2.0 ** (c.R_min / B) - 1.0, EPS)

    gamma_comm = np.zeros((M, M))
    rate = np.zeros((M, M))
    chi_comm = np.zeros((M, M))
    feasible_comm = np.zeros((M, M), dtype=bool)

    for i in range(M):
        for j in range(M):
            if i == j or not base.edge_mask[i, j]:
                continue

            signal = P_comm[i] * base.direct_gain[i, j]
            interf = 0.0
            for k in range(M):
                if k == i or k == j:
                    continue
                interf += (P_comm[k] + c.comm_leakage_from_sensing * P_sense[k]) * base.direct_gain[k, j]

            direct_leakage = c.comm_direct_leakage_factor * P_sense[i] * base.direct_gain[i, j]
            gamma = signal / (n0 + interf + direct_leakage + EPS)
            gamma_comm[i, j] = gamma
            rate[i, j] = B * np.log2(1.0 + gamma)
            chi_comm[i, j] = gamma / (gamma + gamma_req + EPS)

            if c.enforce_chi_min:
                feasible_comm[i, j] = (rate[i, j] >= c.R_min) and (chi_comm[i, j] >= c.chi_min)
            else:
                feasible_comm[i, j] = (rate[i, j] >= c.R_min)

    raw_gamma_sense = np.zeros((M, M, Q))
    gamma_sense = np.zeros((M, M, Q))
    rinr = np.zeros((M, M))
    beta = np.zeros((M, M, Q))
    mu_soft = np.zeros((M, M, Q))
    sigma0 = np.full((M, M), d.soft_sigma0)

    G_proc = cfg.waveform.N * cfg.waveform.L if d.sensing_processing_gain is None else d.sensing_processing_gain

    for i in range(M):
        for j in range(M):
            if i == j or not base.edge_mask[i, j]:
                continue

            residual_self = r.residual_self_factor * P[j]
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
            rinr[i, j] = residual_total / (n0 + EPS)
            sigma0[i, j] = d.soft_sigma0 * math.sqrt(1.0 + r.rinr_sigma_factor * rinr[i, j])

            if r.isac_power_model == "sensing_only":
                effective_sensing_power = P_sense[i]
            elif r.isac_power_model == "joint_waveform":
                effective_sensing_power = P[i]
            elif r.isac_power_model == "reliable_comm_assisted":
                effective_sensing_power = P_sense[i] + chi_comm[i, j] * P_comm[i]
            else:
                raise ValueError(f"Unknown isac_power_model={r.isac_power_model!r}")

            for q in range(Q):
                if cfg.dd.use_otfs_bin_validity and (not base.valid_dd[i, j, q]):
                    continue

                # Keep the raw sensing-SINR baseline consistent with the selected ISAC power model.
                raw_signal = effective_sensing_power * base.target_gain[i, j, q] * G_proc
                raw_gamma_sense[i, j, q] = raw_signal / (n0 + EPS)

                collision_penalty = 1.0 / (max(base.dd_collision_count[i, j, q], 1.0) ** cfg.dd.dd_collision_alpha)
                dd_loss = dd_used[i, j, q] if cfg.dd.enable_dd_fractional_penalty else 1.0

                signal = effective_sensing_power * base.target_gain[i, j, q] * G_proc * collision_penalty * dd_loss
                gamma = signal / (n0 + residual_total + EPS)
                gamma_sense[i, j, q] = gamma
                mu_soft[i, j, q] = d.soft_mu_scale * np.log1p(gamma)

                if feasible_comm[i, j]:
                    chi_eff = float(np.clip(chi_comm[i, j], 0.0, 1.0))
                    delay_ms = 1e3 * packet_bits_for_target(cfg, q) / max(rate[i, j], EPS)
                    beta[i, j, q] = (
                        cfg.selector.beta_scale
                        * np.log1p(gamma)
                        * base.geom_factor[i, j, q]
                        * (chi_eff ** cfg.selector.beta_chi_power)
                        / ((1.0 + rinr[i, j]) * ((delay_ms + EPS) ** cfg.selector.beta_delay_exponent))
                    )

    return LinkTables(
        gamma_comm=gamma_comm,
        rate=rate,
        chi_comm=chi_comm,
        feasible_comm=feasible_comm,
        raw_gamma_sense=raw_gamma_sense,
        gamma_sense=gamma_sense,
        rinr=rinr,
        beta=beta,
        mu_soft=mu_soft,
        sigma0=sigma0,
    )
