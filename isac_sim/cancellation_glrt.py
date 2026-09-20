"""TP-UIC V1.1 -- target-conditioned whitened GLRT on the TP-UIC residual.

Why the V1 detector could not answer the question
-------------------------------------------------
V1 scored every arm with the energy of the *whole* protected block,

    T = ||U_w^H r||^2 / rank(U_w),

which on the 600 m scenario asks "is there energy in this DD patch".  Ten
scatterers share a handful of delay bins there, so dropping target ``q`` leaves
nine other echoes inside the very same subspace and ``T(H1) ~ T(H0)`` by
construction -- measured ``+0.06 .. +0.52 dB`` where the physics promises
``+17 dB``.  The detector, not the canceller, was the broken link.

V1.1 changes the *question* and leaves the canceller alone:

    H0,q:  r = A_-q a_-q + v
    H1,q:  r = A_q a_q + A_-q a_-q + v,        v ~ CN(0, C_res)

with ``(r, C_res)`` handed over by the canceller.  Whiten, project out the other
targets' manifolds, and test only what is left of target ``q``::

    r~    = C_res^{-1/2} r
    A~_x  = C_res^{-1/2} T_e A_x
    P_-q  = I - A~_-q A~_-q^dagger
    B_q   = P_-q C_res^{-1/2} T_e A_q
    T_q   = r~^H P_{B_q} r~                                              (1)

so the formal claim becomes "given the other nine targets and the residual
interference, does target ``q`` still supply evidence" instead of "is this patch
occupied".  The chain the paper gets to state is

    TP-UIC -> (r, C_res, eta_surv) -> whiten -> nuisance projection -> T_q

i.e. the canceller does not merely subtract power; it exports the statistics the
detector needs, and the two are co-designed rather than bolted together.

Every arm is an affine map, exactly
-----------------------------------
All arms of :mod:`isac_sim.cancellation` act as

    removed = F y + c,     F = B s B^H,   rank(F) <= d or d + |candidates|

because ``protected_map``/``joint_refine`` solve a small normal-equation system
whose right-hand side is linear in ``y``.  Two arms are not estimators
(``fixed_kappa`` and ``perfect_channel`` remove a *known* vector), so ``F = 0``
and the direct field is merely scaled.  ``F`` is recovered by **probing**: the
map is applied to an orthonormal basis spanning its range and its row space, and
the resulting small matrix reproduces it exactly -- the probe error is measured
and asserted, not assumed.  ``tests/test_cancellation_glrt.py`` pins the probe
against ``cancellation_arms`` itself, which is what keeps this file honest if V1
is ever edited.

Residual covariance, exactly and cheaply
----------------------------------------
With ``T_d`` the transfer applied to the direct field and ``T_e`` the one applied
to the echo,

    r = T_d x + T_e s + (I - F) n - c + (M X) e

where ``e`` is the coefficient error of a *reference-based* estimate.  For an
estimator arm ``T_e = I - F`` and ``T_d = I - F``; for an oracle arm ``T_e = I``
(the echo passes through untouched) and ``T_d = direct_gain I``.  Taking the
direct coefficients as Gaussian with diagonal variance ``R_diag`` -- the belief
eq. (3) already integrates over -- the interference-plus-noise covariance is

    C_res = sigma^2 I + T_d X R_diag X^H T_d^H + (M X) C_h (M X)^H / n_cpi
          = sigma^2 I + U S U^H,   U = [B, T_d X diag(sqrt R), M X L]      (2)

**identity plus low rank**, so ``C_res^{-1/2}`` is applied exactly by
eigendecomposing the ~60 x 60 restriction to ``span(U)``: no ``K x K`` matrix is
ever formed, which is what makes whitening affordable on 4096 bins.

The ``sigma^2 I`` floor is a modelling decision worth stating.  A canceller that
fits its coefficients on the very observation it cleans also removes the noise in
the fitted subspace, so its residual covariance is *singular* on ``span(M X)``
(14 of 4096 dimensions here) -- a null space no detector may whiten into.
``cancellation.n_cpi`` already declares the coefficients come from a reference
budget, and that reading gives the coefficient error independent noise, a genuine
full-rank floor, and lets the reference budget appear in the detection metric at
all.  V1 measured that the reference budget buys no cancellation *depth* (the
retention term dominates); here it buys *calibration*, which is the term a
detector actually consumes.

``direct_variance="prior"`` uses ``cancellation.prior_variance``: the released
belief, ``sigma_h^2 = 1``, consistent with the unit-amplitude direct coefficients
the observation builder draws.  ``"zero"`` is the ablation in which the receiver
trusts its own estimate completely; what it costs in detection is the
residual-covariance calibration error, and it is reported rather than hidden.

Identifiability, measured rather than asserted
---------------------------------------------
``rho_c = ||P_-q C^{-1/2} a_c||^2 / ||C^{-1/2} a_c||^2`` is the fraction of a
template's own whitened matched-filter energy that survives the nuisance
projection, i.e. ``1 - rho`` is the masked fraction.  It is scale free and
bounded in ``[0, 1]``, with two limits pinned by tests: ``rho = 0`` exactly when
the template lies in the other targets' span, ``rho = 1`` when it is orthogonal
to it.  ``xi_rel_q`` is the matrix version over the whole template block, and
moving the template *off* the DD grid produces the continuous-DD audit: if
``rho`` collapses for every fractional offset then the co-location is intrinsic
and the answer is more physical resolution, not a better canceller.

What this module does *not* do
------------------------------
* It never touches ``model.py`` and changes no released number.  Nothing here is
  imported on the default path (``cancellation.enable = False``).
* It does not re-tune the canceller.  V1 is frozen; this module only consumes it.
* It claims no novelty for OTFS SIC.  The contribution on offer is that the
  cancellation depth *and* the residual statistics come out of one algorithm,
  and that a target-conditioned detector consumes both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .cancellation import (
    EPS,
    CancellationResult,
    Observation,
    Subspace,
    TargetSource,
    _protection_basis,
    joint_interference_covariance,
    joint_refine,
    lift_dictionary,
    orthonormalise,
    protected_map,
    soft_protected_map,
    steering_derivative,
    tangent_columns,
    target_dictionary,
)

__all__ = [
    "ARM_ORDER",
    "ArmPlan",
    "LowRankCovariance",
    "ResidualModel",
    "TargetGLRT",
    "IdentifiabilityAudit",
    "arm_plans",
    "residual_model",
    "target_conditioned_glrt",
    "identifiability_audit",
    "masking_curve",
    "MaskingCurve",
    "restrict_to_target",
    "glrt_threshold",
    "glrt_p_value",
]

ARM_ORDER: Tuple[str, ...] = (
    "no_ic",
    "fixed_kappa",
    "plain_ls",
    "ridge_ls",
    "protected_ls",
    "tp_uic_stage1",
    "tp_uic_full",
    "perfect_channel",
)


# ==========================================================================
# 1. chi^2 with an even number of degrees of freedom (numpy only)
# ==========================================================================
def _chi2_cdf_even(x: float, half: int) -> float:
    """``P(chi^2_{2*half} <= x)`` -- the Erlang CDF, a finite sum.

    The GLRT dof is always ``2 * rank(B_q)``, i.e. even, because a complex
    direction contributes two real ones.  That keeps the CDF a finite sum and the
    quantile a safe bisection, so this module stays numpy-only: a detector that
    needed ``scipy`` merely to set a threshold would add a dependency the release
    path does not have.
    """
    if x <= 0.0 or half <= 0:
        return 0.0
    h = 0.5 * float(x)
    term = 1.0
    total = 1.0
    for j in range(1, int(half)):
        term *= h / j
        total += term
    return float(1.0 - math.exp(-h) * total)


def _chi2_quantile_even(p: float, half: int) -> float:
    """Inverse of :func:`_chi2_cdf_even` by bisection (monotone, no scipy)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly inside (0, 1), got %r" % (p,))
    lo, hi = 0.0, 2.0 * float(max(half, 1))
    for _ in range(64):
        if _chi2_cdf_even(hi, half) >= p:
            break
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _chi2_cdf_even(mid, half) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def glrt_threshold(p_fa: float, dof_real: int) -> float:
    """CFAR level for ``T = (1/2) chi^2_{dof_real}`` under the stated ``C_res``.

    ``T_q`` is the energy of a whitened unit-variance complex residual seen
    through a ``dof_real / 2`` dimensional subspace, so ``2 T_q`` is chi-square
    with ``dof_real`` degrees of freedom and ``E[T_q] = dof_real / 2``.

    The level is ``0.5 * chi^2_{dof}(1 - p_fa)``, and every factor in that
    expression has been a bug at least once: the threshold is on the *halved*
    statistic ``T`` while the helper returns a quantile of ``2 T``, and it must
    be the ``1 - p_fa`` quantile rather than the ``p_fa`` one.  Placing it at the
    5th percentile of the null fires 95 % of the time; forgetting the 0.5 puts it
    8 dB too high.  Both directions are pinned by
    ``test_threshold_achieves_the_requested_false_alarm_rate``.
    """
    if dof_real <= 0:
        return float("inf")
    if dof_real % 2:
        raise ValueError(
            "dof must be even (a complex direction gives two real ones), got %d"
            % dof_real
        )
    return 0.5 * _chi2_quantile_even(1.0 - p_fa, dof_real // 2)


def glrt_p_value(statistic: float, dof_real: int) -> float:
    """``P(T >= statistic | H0)`` under the stated ``C_res``."""
    if dof_real <= 0:
        return 1.0
    return float(
        max(0.0, 1.0 - _chi2_cdf_even(2.0 * float(statistic), dof_real // 2))
    )


# ==========================================================================
# 2. The arms as affine maps
# ==========================================================================
@dataclass
class ArmPlan:
    """Which estimator an arm runs, so its linear form can be rebuilt.

    Mirrors the arm table of :func:`isac_sim.cancellation.cancellation_arms`
    deliberately.  The candidate support is *read back* from the recorded result
    rather than recomputed, so the only duplicated logic is the small
    subspace/prior table -- and ``tests/test_cancellation_glrt.py`` compares the
    rebuilt form against the real arm on the observation itself.
    """

    name: str
    kind: str  # "none" | "oracle" | "estimator"
    subspace: Subspace
    prior: float | None
    candidates: Tuple[int, ...] = ()
    soft_mu: float | None = None


def _empty_subspace(n_bins: int) -> Subspace:
    return Subspace(U=np.zeros((n_bins, 0), dtype=complex), rank=0)


def _belief_subspace(obs: Observation) -> Subspace:
    U = obs.basis_belief
    return Subspace(U=U, rank=0 if U is None else int(U.shape[1]))


def arm_plans(
    cfg, obs: Observation, results: Dict[str, CancellationResult] | None = None
) -> Dict[str, ArmPlan]:
    """The arm table, in the form the linear probe needs."""
    n_bins = int(obs.y.size)
    empty = _empty_subspace(n_bins)
    belief = _belief_subspace(obs)
    prior = cfg.cancellation.prior_variance if cfg.cancellation.prior_variance else None
    candidates: Tuple[int, ...] = ()
    if results is not None and "tp_uic_full" in results:
        candidates = tuple(int(c) for c in results["tp_uic_full"].candidates)
    plans = {
        "no_ic": ArmPlan("no_ic", "none", empty, None),
        "fixed_kappa": ArmPlan("fixed_kappa", "oracle", empty, None),
        "plain_ls": ArmPlan("plain_ls", "estimator", empty, None),
        "ridge_ls": ArmPlan("ridge_ls", "estimator", empty, prior),
        "protected_ls": ArmPlan("protected_ls", "estimator", belief, None),
        "tp_uic_stage1": ArmPlan("tp_uic_stage1", "estimator", belief, prior),
        "tp_uic_full": ArmPlan("tp_uic_full", "estimator", belief, prior, candidates),
        "perfect_channel": ArmPlan("perfect_channel", "oracle", empty, None),
    }
    if results is not None and "adaptive_soft_tpuic" in results:
        adaptive = results["adaptive_soft_tpuic"]
        if adaptive.soft_mu is None:
            plans["adaptive_soft_tpuic"] = ArmPlan(
                "adaptive_soft_tpuic", "estimator", belief, prior, candidates
            )
        else:
            ids = np.asarray(obs.A_target_ids)
            tested = int(obs.weak_index)
            target_subspace = orthonormalise(obs.A[:, ids == tested])
            plans["adaptive_soft_tpuic"] = ArmPlan(
                "adaptive_soft_tpuic", "estimator", target_subspace, prior,
                soft_mu=float(adaptive.soft_mu),
            )
    return plans


def _removed_vector(cfg, obs: Observation, plan: ArmPlan, v: np.ndarray) -> np.ndarray:
    """The vector a plan's estimator removes from ``v`` -- V1's ``operator``."""
    if plan.kind != "estimator":
        return np.zeros(int(v.size), dtype=complex)
    X, A = obs.X, obs.A
    sigma2 = float(obs.sigma2)
    if plan.candidates:
        A_cand = A[:, list(plan.candidates)]
        h, _a = joint_refine(v, X, A_cand, sigma2, plan.prior, plan.prior or 1.0)
        return X @ h
    if plan.soft_mu is not None:
        h, _cov = soft_protected_map(
            v, X, plan.subspace, sigma2, plan.prior, plan.soft_mu
        )
        return X @ h
    h, _c, mx = protected_map(v, X, plan.subspace, sigma2, plan.prior)
    return mx @ h


def _probe_basis(obs: Observation, plan: ArmPlan) -> np.ndarray:
    """A basis spanning both the range and the row space of the removal map.

    For a protected estimator the removal is ``M X h(M y)``, so it lives entirely
    inside ``span(M X)``; for the joint arm it is ``X h`` with ``h`` depending on
    ``[X, A_cand]^H y``, so ``span([X, A_cand])`` covers image and row space
    alike.  On a basis covering both, the probe reconstructs the map exactly.
    """
    if plan.kind != "estimator":
        return np.zeros((int(obs.y.size), 0), dtype=complex)
    if plan.candidates:
        return np.concatenate([obs.X, obs.A[:, list(plan.candidates)]], axis=1)
    if plan.soft_mu is not None:
        return obs.X
    return plan.subspace.complement_matrix(obs.X)


def _low_rank_form(
    cfg, obs: Observation, plan: ArmPlan, tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Recover ``(basis, small, error)`` with ``F = basis @ small @ basis^H``.

    Probing on an orthonormal basis ``Q`` of the covering subspace gives ``F Q``
    directly; with ``P = Q R`` the small matrix is ``(Q^H F Q) (Q^H P)^{+}``.
    ``error`` is the measured reconstruction residual on the probe columns --
    returned, not assumed, so that a silently rank-deficient probe cannot pass
    for an exact model.

    ``tol`` is deliberately loose (1e-6) rather than machine tight.  The joint
    arm's covering basis holds ``[X, A_cand]`` whose columns can be nearly
    collinear, so ``pinv(R)`` amplifies round-off and the measured error sits
    around 1e-8 on some trials.  That is eight orders below anything that would
    move a reported cancellation depth, while a genuinely non-affine arm shows
    up at O(1e-2).
    """
    n_bins = int(obs.y.size)
    if plan.kind != "estimator":
        return np.zeros((n_bins, 0), dtype=complex), np.zeros((0, 0), dtype=complex), 0.0
    P = _probe_basis(obs, plan)
    if P.shape[1] == 0:
        return np.zeros((n_bins, 0), dtype=complex), np.zeros((0, 0), dtype=complex), 0.0
    Q, R = np.linalg.qr(P)
    removed = np.stack(
        [_removed_vector(cfg, obs, plan, P[:, j]) for j in range(P.shape[1])], axis=1
    )
    small = (Q.conj().T @ removed) @ np.linalg.pinv(R, rcond=1e-12)
    back = Q @ (small @ (Q.conj().T @ P))
    denom = max(float(np.linalg.norm(removed)), EPS)
    error = float(np.linalg.norm(back - removed) / denom)
    if error > tol:
        raise RuntimeError(
            "arm %r: the linear probe does not reproduce the arm (relative error "
            "%.3e > %.1e) -- the arm is not affine, or the covering subspace is "
            "too small" % (plan.name, error, tol)
        )
    return Q, np.asarray(small), error


# ==========================================================================
# 3. sigma^2 I + low rank, whitened exactly
# ==========================================================================
@dataclass
class LowRankCovariance:
    """``C = sigma2 I + W diag(lam) W^H`` with ``W`` orthonormal.

    Kept in eigen form because that is exactly what whitening needs::

        C^{-1/2} v = sigma^{-1} (v + W ((1 + lam/sigma2)^{-1/2} - 1) * (W^H v))

    Eigenvalues equal to ``sigma2`` are simply absent from ``W``.
    """

    sigma2: float
    W: np.ndarray
    lam: np.ndarray
    min_ratio: float

    @property
    def rank(self) -> int:
        return int(self.W.shape[1])

    @property
    def trace(self) -> float:
        n_bins = int(self.W.shape[0])
        return float(self.sigma2 * (n_bins - self.rank) + np.sum(self.sigma2 + self.lam))

    def apply_matrix(self, V: np.ndarray) -> np.ndarray:
        WtV = self.W.conj().T @ V
        return self.sigma2 * V + self.W @ (self.lam[:, None] * WtV)

    def apply(self, v: np.ndarray) -> np.ndarray:
        return self.apply_matrix(np.asarray(v)[:, None])[:, 0]

    def whiten_matrix(self, V: np.ndarray) -> np.ndarray:
        factor = (1.0 + self.lam / self.sigma2) ** -0.5 - 1.0
        return (V + self.W @ (factor[:, None] * (self.W.conj().T @ V))) / math.sqrt(
            self.sigma2
        )

    def whiten(self, v: np.ndarray) -> np.ndarray:
        return self.whiten_matrix(np.asarray(v)[:, None])[:, 0]

    def inverse_matrix(self, V: np.ndarray) -> np.ndarray:
        factor = 1.0 / (1.0 + self.lam / self.sigma2) - 1.0
        return (V + self.W @ (factor[:, None] * (self.W.conj().T @ V))) / self.sigma2

    def inverse(self, v: np.ndarray) -> np.ndarray:
        return self.inverse_matrix(np.asarray(v)[:, None])[:, 0]


@dataclass
class ResidualModel:
    """One arm's affine action plus the covariance it leaves behind.

    ``removed(y) = basis small basis^H y + const``; ``direct_gain`` scales the
    direct field that survives (``1`` for an estimator arm, ``keep`` for
    ``fixed_kappa``, ``0`` for the oracle); ``cov`` is ``C_res`` of eq. (2).
    """

    name: str
    basis: np.ndarray
    small: np.ndarray
    const: np.ndarray
    direct_gain: float
    sigma2: float
    cov: LowRankCovariance
    probe_error: float = 0.0
    direct_variance: str = "prior"
    r_diag: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def remove(self, v: np.ndarray) -> np.ndarray:
        """``F v`` -- the y-dependent part of the removal only."""
        if self.rank == 0:
            return np.zeros_like(v)
        return self.basis @ (self.small @ (self.basis.conj().T @ v))

    def remove_matrix(self, V: np.ndarray) -> np.ndarray:
        if self.rank == 0:
            return np.zeros_like(V)
        return self.basis @ (self.small @ (self.basis.conj().T @ V))

    def residual(self, v: np.ndarray) -> np.ndarray:
        """``v - F v - c`` -- what the receiver is left with."""
        return v - self.remove(v) - self.const

    def signal_transfer(self, A: np.ndarray) -> np.ndarray:
        """``T_e A``: how the arm attenuates a dictionary of echo templates."""
        return A - self.remove_matrix(A)

    def transfer_frobenius(self) -> float:
        """``tr((I-F)(I-F)^H)`` in closed form, for the covariance check."""
        n_bins = int(self.basis.shape[0])
        if self.rank == 0:
            return float(n_bins)
        s = self.small
        return float(
            n_bins
            - 2.0 * np.real(np.trace(s))
            + np.real(np.trace(s.conj().T @ s))
        )


def _belief_bin_sigmas(cfg) -> Tuple[float, float]:
    """``(sigma_delay_bins, sigma_doppler_bins)`` from the declared belief error.

    The receiver's tracker states a position error and a velocity error; the
    detector needs them in the units its templates are indexed by.  The delay
    conversion is the one-way path error ``sigma_p / c`` scaled by the delay-bin
    spacing ``1 / (L * delta_f)``; the Doppler conversion is the bistatic radial
    rate error ``2 sigma_v / lambda`` scaled by the Doppler-bin spacing
    ``1 / (N * T)``.  Both are the *largest* such error over viewing geometry, so
    the term is conservative rather than tuned.
    """
    w = cfg.waveform
    lam = float(w.c) / float(w.fc)
    sigma_delay_bins = (float(cfg.prior.belief_sigma_pos_m) / float(w.c)
                        * float(w.L) * float(w.delta_f))
    sigma_doppler_bins = (2.0 * float(cfg.prior.belief_sigma_vel_mps) / lam
                          * float(w.N) * float(w.T))
    return float(sigma_delay_bins), float(sigma_doppler_bins)


def _belief_error_factor(cfg, obs: Observation) -> np.ndarray:
    """``B`` with ``B B^H = C_belief``: covariance of the template misalignment.

    Up to three columns per believed echo describe delay, Doppler, and bearing
    mismatch.  DD derivatives are lifted by the source steering vector and the
    bearing derivative is ``a_DD (x) da_array/du``.  Their outer products form
    the first-order mismatch covariance.  ``P`` is the believed echo power, so
    a target the receiver can barely see contributes barely any mismatch.

    **The tested target's own sources are excluded.**  ``C_res`` is the
    covariance of the residual *under H0*, and under H0 that target's echo is
    absent, so charging its leakage would inflate the covariance exactly where
    the false-alarm level is set.  The cost is honest and stated: under H1 the
    tested echo's own misalignment is not charged, so this term buys calibrated
    ``P_FA`` at the price of a slightly optimistic ``P_D``.

    Uses the *believed* sources and their believed bins -- the receiver cannot
    know the true ones, and a covariance charged on the truth would be an oracle
    dressed as a calibration.
    """
    sources = obs.targets_belief if obs.targets_belief is not None else obs.targets
    n_bins = int(obs.y.size)
    if not sources:
        return np.zeros((n_bins, 0), dtype=complex)
    sigma_l, sigma_k = _belief_bin_sigmas(cfg)
    if sigma_l <= 0.0 and sigma_k <= 0.0:
        return np.zeros((n_bins, 0), dtype=complex)
    step = float(cfg.cancellation.tangent_step_bins)
    tested = int(getattr(obs, "weak_index", 0))
    cols: List[np.ndarray] = []
    for src in sources:
        if int(src.target) == tested:
            continue
        block = tangent_columns(cfg, float(src.doppler_bin), float(src.delay_bin),
                                1, step)
        if block.shape[1] < 3:
            continue
        centre, d_delay, d_doppler = block[:, 0], block[:, 1], block[:, 2]
        norm = float(np.sqrt(max(np.vdot(centre, centre).real, EPS)))
        amp = math.sqrt(max(float(src.power), 0.0)) / norm
        sigma_l_src = (
            sigma_l if src.sigma_delay_bin is None
            else max(float(src.sigma_delay_bin), 0.0)
        )
        sigma_k_src = (
            sigma_k if src.sigma_doppler_bin is None
            else max(float(src.sigma_doppler_bin), 0.0)
        )
        sigma_u_src = max(float(src.sigma_bearing_u or 0.0), 0.0)
        u_src = float(getattr(src, "u", 0.0))
        if sigma_l_src > 0.0:
            cols.append(lift_dictionary(
                cfg, ((amp * sigma_l_src) * d_delay)[:, None], [u_src]
            )[:, 0])
        if sigma_k_src > 0.0:
            cols.append(lift_dictionary(
                cfg, ((amp * sigma_k_src) * d_doppler)[:, None], [u_src]
            )[:, 0])
        m_rx = int(cfg.aperture.m_rx) if cfg.aperture.enable else 1
        if sigma_u_src > 0.0 and m_rx > 1:
            cols.append(
                (amp * sigma_u_src)
                * np.kron(centre, steering_derivative(m_rx, u_src))
            )
    if not cols:
        return np.zeros((n_bins, 0), dtype=complex)
    return np.stack(cols, axis=1)


def _direct_prior(cfg, obs: Observation, plan: ArmPlan) -> Tuple[str, np.ndarray]:
    """``(label, R_diag)`` -- the belief the receiver states about ``h``.

    The prior is the same for every arm, oracle included, and that is not an
    oversight: what an oracle arm leaves behind is ``direct_gain * x``, whose
    covariance is ``direct_gain^2 X R_h X^H`` with the *same* ``R_h``.  Returning
    zeros here would have told the detector that ``no_ic`` residual is pure noise,
    i.e. that the direct field had been cancelled -- the one thing that arm
    explicitly does not do.
    """
    d = int(obs.X.shape[1])
    value = cfg.cancellation.prior_variance
    return "prior", np.full(d, float(value) if value else 1.0)


def _oracle_gain(cfg, name: str) -> float:
    """How much of the direct field an oracle arm leaves in the residual."""
    if name == "no_ic":
        return 1.0
    if name == "perfect_channel":
        return 0.0
    if name == "fixed_kappa":
        return math.sqrt(
            10.0 ** (-float(cfg.interference.direct_cancellation_db) / 10.0)
        )
    raise KeyError(name)


def _estimator_covariance(cfg, obs: Observation, plan: ArmPlan) -> np.ndarray:
    """Posterior covariance of the interference coefficients, ``C_h``.

    Mirrors the normal equations of ``protected_map``/``joint_refine`` rather
    than reading ``c_diag`` back, because the diagonal throws away the
    correlations between illuminators and the joint arm does not report a
    covariance at all.  The reference budget enters as ``1 / n_cpi``: ``n_cpi``
    coherent CPIs is exactly the statement that the coefficient error is that
    much smaller than a single-CPI self-fit.
    """
    d = int(obs.X.shape[1])
    if plan.kind != "estimator" or d == 0:
        return np.zeros((d, d), dtype=complex)
    sigma2 = float(obs.sigma2)
    pv = plan.prior
    if plan.candidates:
        A_cand = obs.A[:, list(plan.candidates)]
        return joint_interference_covariance(
            obs.X, A_cand, sigma2, pv, pv or 1.0
        )
    if plan.soft_mu is not None:
        px = plan.subspace.project(obs.X)
        gram = obs.X.conj().T @ obs.X + float(plan.soft_mu) * (
            px.conj().T @ px
        )
        precision = gram / sigma2
        if pv:
            precision = precision + (1.0 / float(pv)) * np.eye(d)
        return np.asarray(np.linalg.pinv(precision, rcond=1e-12), dtype=complex)
    mx = plan.subspace.complement_matrix(obs.X)
    gram = mx.conj().T @ mx
    if pv:
        cov = np.linalg.inv((gram / sigma2) + (1.0 / float(pv)) * np.eye(d))
    else:
        cov = np.linalg.pinv(gram, rcond=1e-10) * sigma2
    return np.asarray(cov, dtype=complex)


def _psd_sqrt(C: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """``L`` with ``L L^H = C`` for a Hermitian PSD ``C`` (eigen route)."""
    if C.size == 0:
        return np.zeros_like(C)
    w, V = np.linalg.eigh(0.5 * (C + C.conj().T))
    if w.size and w.min() < -tol * max(abs(w.max()), EPS):
        raise RuntimeError(
            "estimator covariance is not PSD (min eigenvalue %.3e)" % (w.min(),)
        )
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)[None, :]).astype(complex)


def residual_model(
    cfg,
    obs: Observation,
    name: str,
    results: Dict[str, CancellationResult] | None = None,
    *,
    plans: Dict[str, ArmPlan] | None = None,
    direct_variance: str = "prior",
    check: bool = True,
    dictionary: str = "belief",
) -> ResidualModel:
    """Build the affine form and the residual covariance of one arm.

    The covariance is the *reference-based* one::

        C_res = sigma^2 I  +  T_d X R_h X^H T_d^H  +  (M X) C_h (M X)^H / n_cpi
                ^floor      ^direct field kept       ^coefficient error
                            +  B B^H                 (optional, see below)
                            ^belief-error template misalignment

    ``dictionary="truth"`` builds the templates from the true target state, where
    there is no misalignment to charge; ``dictionary="belief"`` (the default, and
    what every released number uses) charges it when
    ``cancellation.belief_error_in_cres`` is set.  Passing one and charging the
    other is the easiest way to make a calibration measurement meaningless.

    The floor is ``sigma^2 I`` and **not** ``sigma^2 (I-F)(I-F)^H``, which is a
    modelling decision worth stating.  A canceller that fits its coefficients on
    the very observation it cleans also removes the noise in the fitted
    subspace, so its residual covariance is *singular* on ``span(M X)`` (14 of
    4096 dimensions here) -- a null space no detector may whiten into.  The
    ``cancellation.n_cpi`` field already says the coefficients come from a
    *reference* budget, and that reading makes the coefficient error independent
    noise, gives ``C_res`` a genuine full-rank floor, and lets the reference
    budget show up in the detection metric at all (V1 measured that it buys no
    cancellation depth, because retention dominates; here it buys calibration).
    """
    if results is None or name not in results:
        raise KeyError("residual_model needs the recorded result of arm %r" % (name,))
    plan = (plans if plans is not None else arm_plans(cfg, obs, results))[name]
    sigma2 = float(obs.sigma2)
    n_bins = int(obs.y.size)
    label, r_diag = _direct_prior(cfg, obs, plan)
    if direct_variance == "zero":
        r_diag = np.zeros_like(r_diag)
        label = "zero"
    elif direct_variance != "prior":
        raise ValueError(
            "direct_variance must be 'prior' or 'zero', got %r" % (direct_variance,)
        )

    if plan.kind == "estimator":
        basis, small, error = _low_rank_form(cfg, obs, plan)
        const = np.zeros(n_bins, dtype=complex)
        gain = 1.0
        subtraction_dictionary = (
            obs.X if (plan.candidates or plan.soft_mu is not None)
            else plan.subspace.complement_matrix(obs.X)
        )
        est_factor = subtraction_dictionary @ _psd_sqrt(
            _estimator_covariance(cfg, obs, plan)
        )
        est_factor = est_factor / math.sqrt(float(cfg.cancellation.n_cpi))
    else:
        basis = np.zeros((n_bins, 0), dtype=complex)
        small = np.zeros((0, 0), dtype=complex)
        const = obs.y - results[name].residual
        gain = _oracle_gain(cfg, name)
        error = 0.0
        est_factor = np.zeros((n_bins, 0), dtype=complex)

    X = obs.X
    ix = X - (basis @ (small @ (basis.conj().T @ X))) if basis.shape[1] else X
    direct_factor = (gain * ix) * np.sqrt(r_diag)[None, :]

    if cfg.cancellation.belief_error_in_cres and dictionary == "belief":
        belief_factor = _belief_error_factor(cfg, obs)
    else:
        belief_factor = np.zeros((n_bins, 0), dtype=complex)

    # Only the *perturbations* enter the low-rank factor.  ``basis`` spans
    # the range of F and must NOT be added: the noise floor is sigma^2 I, not
    # sigma^2 (I-F)(I-F)^H, so F's range carries no covariance of its own --
    # only ``(I-F) X`` (what the canceller leaves of the direct field), the
    # coefficient error and the belief-error misalignment do.  Adding it
    # inflated the condition number to 1e13 and made ``eigh`` report a minimum
    # eigenvalue below the floor.
    blocks = [b for b in (direct_factor, est_factor, belief_factor) if b.shape[1]]

    def c_apply(V: np.ndarray) -> np.ndarray:
        """``C_res @ V``, without ever forming a K x K matrix."""
        out = sigma2 * V
        for b in blocks:
            out = out + b @ (b.conj().T @ V)
        return out

    if blocks:
        Q0, _ = np.linalg.qr(np.concatenate(blocks, axis=1))
        C_red = Q0.conj().T @ c_apply(Q0)
        C_red = 0.5 * (C_red + C_red.conj().T)
        mu, V = np.linalg.eigh(C_red)
    else:
        Q0 = np.zeros((n_bins, 0), dtype=complex)
        mu = np.zeros(0)
        V = np.zeros((0, 0), dtype=complex)
    W = Q0 @ V if Q0.shape[1] else np.zeros((n_bins, 0), dtype=complex)
    min_ratio = float(mu.min() / sigma2) if mu.size else 1.0
    if check and min_ratio < 1.0 - 1e-9:
        raise RuntimeError(
            "arm %r: the modelled residual covariance has an eigenvalue below "
            "the noise floor (min / sigma^2 = %.6f).  C_res is a sum of PSD "
            "terms plus sigma^2 I, so this means the stated belief is "
            "inconsistent with the arm rather than a typo in the experiment."
            % (name, min_ratio)
        )
    keep = np.abs(mu - sigma2) > 1e-12 * sigma2
    cov = LowRankCovariance(
        sigma2=sigma2, W=W[:, keep], lam=mu[keep] - sigma2, min_ratio=min_ratio
    )
    model = ResidualModel(
        name=name,
        basis=basis,
        small=small,
        const=const,
        direct_gain=gain,
        sigma2=sigma2,
        cov=cov,
        probe_error=error,
        direct_variance=label,
        r_diag=r_diag,
    )
    if check:
        # The eigen form must reproduce the analytic trace.  A mismatch means the
        # restriction to span(U) missed part of the perturbation, which would
        # silently mis-whiten every statistic.
        analytic = sigma2 * float(n_bins) + sum(
            float(np.sum(np.abs(b) ** 2)) for b in blocks
        )
        if abs(cov.trace - analytic) > 1e-6 * max(abs(analytic), EPS):
            raise RuntimeError(
                "arm %r: residual-covariance trace mismatch (eigen %.6e vs "
                "analytic %.6e)" % (name, cov.trace, analytic)
            )
    return model


# ==========================================================================
# 4. Target-conditioned whitened GLRT
# ==========================================================================
def _project_out(B_neg: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``(I - B_neg B_neg^dagger) V`` by least squares (rank safe)."""
    if B_neg.shape[1] == 0:
        return V
    coef = np.linalg.lstsq(B_neg, V, rcond=None)[0]
    return V - B_neg @ coef


def _projector_energy(B: np.ndarray, v: np.ndarray) -> float:
    """``v^H P_B v`` for a possibly rank-deficient ``B``."""
    if B.shape[1] == 0:
        return 0.0
    coef = np.linalg.lstsq(B, v, rcond=None)[0]
    fitted = B @ coef
    return float(np.vdot(fitted, fitted).real)


def _numerical_rank(B: np.ndarray, rtol: float = 1e-8) -> int:
    if B.size == 0 or B.shape[1] == 0:
        return 0
    sv = np.linalg.svd(B, compute_uv=False)
    if sv.size == 0 or sv[0] <= 0.0:
        return 0
    return int(np.sum(sv > rtol * sv[0]))


def _generalised_min_eigen(G_keep: np.ndarray, G_full: np.ndarray) -> float:
    """``lambda_min(G_full, G_keep)`` on the range of ``G_keep``.

    The matrix analogue of ``rho``: the worst direction of the template block
    keeps this fraction of its whitened matched-filter energy.  Computed as the
    smallest eigenvalue of ``G_keep^{-1/2} G_full G_keep^{-1/2}``, which is
    symmetric and clamped into ``[0, 1]`` because ``G_full`` is a monotone
    shrinkage of ``G_keep``.
    """
    if G_keep.size == 0 or G_keep.shape[0] == 0:
        return 0.0
    w, V = np.linalg.eigh(0.5 * (G_keep + G_keep.conj().T))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    if w.size == 0 or w.max() <= EPS:
        return 0.0
    keep = w > max(w.max() * 1e-12, EPS)
    if not np.any(keep):
        return 0.0
    Vk = V[:, keep]
    inv_sqrt = Vk @ np.diag(w[keep] ** -0.5) @ Vk.conj().T
    M = inv_sqrt @ G_full @ inv_sqrt
    M = 0.5 * (M + M.conj().T)
    return float(max(0.0, min(1.0, np.linalg.eigvalsh(M).min())))


@dataclass
class TargetGLRT:
    """The V1.1 detector's output for one (arm, observation, target)."""

    arm: str
    target: int
    statistic: float
    dof_real: int
    threshold: float
    p_fa: float
    detected: bool
    rho: np.ndarray  # per template, in [0, 1]
    g_self: np.ndarray  # per template, whitened matched-filter energy
    ncp_unit: float  # expected GLRT defection for unit-amplitude echo
    ncp_best: float  # best-case defection over a unit-norm echo vector
    rho_weighted: float  # energy-weighted escape fraction, in [0, 1]
    xi_q: float  # lambda_min(B_q^H B_q)
    xi_rel_q: float  # scale-free version, in [0, 1]
    n_templates: int
    n_nuisance_columns: int
    nuisance_rank: int
    residual_power: float
    whitened_residual_power: float = 0.0

    @property
    def rho_min(self) -> float:
        return float(self.rho.min()) if self.rho.size else float("nan")

    @property
    def p_value(self) -> float:
        return glrt_p_value(self.statistic, self.dof_real)


def _dictionary(cfg, obs: Observation, which: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(dictionary, column -> target id)`` for the belief or the truth."""
    if which == "belief":
        return obs.A, np.asarray(obs.A_target_ids)
    if which == "truth":
        return target_dictionary(
            cfg, obs.targets, covariance_expanded=False
        ), np.asarray(obs.A_true_target_ids)
    raise ValueError("dictionary must be 'belief' or 'truth', got %r" % (which,))


def _centre_mask(cfg, n_cols: int) -> np.ndarray:
    """Select the centre column of every source block.

    ``tangent_columns`` puts the centre first and the ``2*order`` fractional-DD
    derivatives after it, and the dictionary repeats that layout per source.  A
    tangent column is a *derivative*, i.e. uncertainty about where the echo is,
    not an extra echo signature: mixing the two into the tested block would
    inflate the dof with directions that carry no independent signal and would
    deflate ``rho`` for reasons that have nothing to do with identifiability.
    """
    n_basis = 1 + 2 * int(cfg.cancellation.tangent_order)
    mask = np.zeros(int(n_cols), dtype=bool)
    if n_basis > 0:
        mask[::n_basis] = True
    return mask


def _dictionary_centre_mask(cfg, obs: Observation, which: str, n_cols: int) -> np.ndarray:
    explicit = (
        obs.A_centre_mask if which == "belief" else obs.A_true_centre_mask
    )
    if explicit is None:
        return _centre_mask(cfg, n_cols)
    mask = np.asarray(explicit, dtype=bool)
    if mask.shape != (int(n_cols),):
        raise ValueError(
            "%s dictionary centre mask has shape %r, expected (%d,)"
            % (which, mask.shape, int(n_cols))
        )
    return mask


def _manifold_columns(
    cfg,
    obs: Observation,
    target: int,
    order: int,
    step: float | None = None,
    dictionary: str = "belief",
    subset: Sequence[int] | None = None,
) -> np.ndarray:
    """Fractional-DD tangent columns of every *other* target."""
    sources: Sequence[TargetSource] = (
        (obs.targets_belief if dictionary == "belief" else obs.targets)
        or obs.targets_belief
        or obs.targets
    )
    h = float(cfg.cancellation.tangent_step_bins if step is None else step)
    allowed = None if subset is None else {int(v) for v in subset}
    blocks: List[np.ndarray] = []
    for src in sources:
        if int(src.target) == int(target):
            continue
        if allowed is not None and int(src.target) not in allowed:
            continue
        amp = math.sqrt(max(float(src.power), 0.0))
        dd_block = amp * tangent_columns(
            cfg, src.doppler_bin, src.delay_bin, order, h
        )
        blocks.append(
            lift_dictionary(cfg, dd_block, [float(src.u)] * dd_block.shape[1])
        )
        m_rx = int(cfg.aperture.m_rx) if cfg.aperture.enable else 1
        if m_rx > 1:
            centre = amp * tangent_columns(
                cfg, src.doppler_bin, src.delay_bin, 0, h
            )[:, 0]
            blocks.append(
                np.kron(centre, steering_derivative(m_rx, float(src.u)))[:, None]
            )
    if not blocks:
        return np.zeros((int(obs.y.size), 0), dtype=complex)
    return np.concatenate(blocks, axis=1)


def target_conditioned_glrt(
    cfg,
    obs: Observation,
    result: CancellationResult,
    model: ResidualModel,
    *,
    target: int | None = None,
    p_fa: float = 0.05,
    nuisance_manifold: int = 0,
    manifold_step: float | None = None,
    dictionary: str = "belief",
    template_override: np.ndarray | None = None,
    centre_only: bool = True,
    nuisance_targets: Sequence[int] | None = None,
    null_cov_model: ResidualModel | None = None,
    threshold_override: float | None = None,
) -> TargetGLRT:
    """``T_q`` of eq. (1) plus the identifiability numbers that go with it.

    ``nuisance_manifold`` expands the nuisance block of the *other* targets with
    their fractional-DD tangent columns, i.e. it lets the detector say "the other
    nine are known only up to a continuous delay/Doppler offset".  That is the
    honest version of the question, and comparing ``rho`` at order 0 and order 1
    is what separates "the grid made them collide" from "they really are one
    subspace".
    """
    A, ids = _dictionary(cfg, obs, dictionary)
    tgt = int(obs.weak_index if target is None else target)
    sel_tgt = np.asarray(ids) == tgt
    if not np.any(sel_tgt):
        raise ValueError(
            "target %d has no column in the %s dictionary" % (tgt, dictionary)
        )
    base = (
        _dictionary_centre_mask(cfg, obs, dictionary, A.shape[1])
        if centre_only else np.ones(A.shape[1], dtype=bool)
    )
    sel_q = sel_tgt & base
    if not np.any(sel_q):
        raise ValueError("target %d has no centre column" % (tgt,))

    if template_override is not None:
        A_q = np.asarray(template_override, dtype=complex)
        if A_q.ndim == 1:
            A_q = A_q[:, None]
    else:
        A_q = A[:, sel_q]
    A_neg = np.where(
        np.isin(np.asarray(ids), list(nuisance_targets)) if nuisance_targets is not None
        else ~sel_tgt,
        base,
        False,
    )
    A_neg = A[:, A_neg]
    if nuisance_manifold > 0:
        extra = _manifold_columns(
            cfg, obs, tgt, int(nuisance_manifold), manifold_step, dictionary,
            subset=nuisance_targets,
        )
        if extra.shape[1]:
            A_neg = np.concatenate([A_neg, extra], axis=1)

    cov = model.cov
    w_q = cov.whiten_matrix(model.signal_transfer(A_q))
    w_neg = cov.whiten_matrix(model.signal_transfer(A_neg))
    r_whitened = cov.whiten(model.residual(obs.y))

    B_q = _project_out(w_neg, w_q)
    statistic = _projector_energy(B_q, r_whitened)
    dof_real = 2 * _numerical_rank(B_q)
    threshold = (
        float(threshold_override)
        if threshold_override is not None else glrt_threshold(p_fa, dof_real)
    )
    if (
        threshold_override is None
        and null_cov_model is not None
        and dof_real > 0
    ):
        rank_q = dof_real // 2
        q_basis = np.linalg.qr(B_q)[0][:, :rank_q]
        raw_directions = cov.whiten_matrix(q_basis)
        null_small = raw_directions.conj().T @ (
            null_cov_model.cov.apply_matrix(raw_directions)
        )
        null_eigenvalues = np.maximum(
            np.real(np.linalg.eigvalsh(
                0.5 * (null_small + null_small.conj().T)
            )),
            0.0,
        )
        if np.allclose(
            null_eigenvalues, null_eigenvalues[0], rtol=1e-10, atol=0.0
        ):
            threshold = float(null_eigenvalues[0]) * glrt_threshold(
                p_fa, dof_real
            )
        else:
            # A complex Gaussian quadratic form is a weighted sum of unit
            # exponentials.  Fixed numerical quadrature makes the threshold
            # deterministic and independent of the evaluated H0 draws.
            rng = np.random.default_rng(0x43464152)
            draws = rng.exponential(
                scale=1.0, size=(262144, int(null_eigenvalues.size))
            ) @ null_eigenvalues
            threshold = float(np.quantile(
                draws, 1.0 - float(p_fa), method="higher"
            ))

    g_self = np.real(np.sum(np.abs(w_q) ** 2, axis=0))
    g_keep = np.real(np.sum(np.abs(B_q) ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(g_self > 0.0, g_keep / np.maximum(g_self, EPS), 0.0)
    rho = np.clip(rho, 0.0, 1.0)
    gram_q = B_q.conj().T @ B_q if B_q.shape[1] else np.zeros((0, 0), dtype=complex)
    # Two defection parameters, because they answer different questions.  The
    # echo coefficients have unit amplitude and random phase, so the *expected*
    # defection is the incoherent sum sum_c rho_c g_c.  ``ncp_best`` is the
    # coherent bound (the largest singular value squared), which is what a
    # receiver could reach if it knew the phases -- the gap between the two is
    # the price of not knowing them.
    sv_b = np.linalg.svd(B_q, compute_uv=False) if B_q.shape[1] else np.zeros(0)
    total_self = float(np.sum(g_self))

    return TargetGLRT(
        arm=model.name,
        target=tgt,
        statistic=float(statistic),
        dof_real=dof_real,
        threshold=float(threshold),
        p_fa=float(p_fa),
        detected=bool(statistic > threshold) if np.isfinite(threshold) else False,
        rho=np.asarray(rho, dtype=float),
        g_self=np.asarray(g_self, dtype=float),
        ncp_unit=float(np.sum(rho * g_self)),
        ncp_best=float(sv_b[0] ** 2) if sv_b.size else 0.0,
        rho_weighted=float(np.sum(rho * g_self) / total_self) if total_self > 0 else 0.0,
        xi_q=float(np.linalg.eigvalsh(gram_q).min()) if gram_q.shape[0] else 0.0,
        xi_rel_q=_generalised_min_eigen(w_q.conj().T @ w_q, gram_q),
        n_templates=int(A_q.shape[1]),
        n_nuisance_columns=int(A_neg.shape[1]),
        nuisance_rank=_numerical_rank(w_neg),
        residual_power=float(np.vdot(model.residual(obs.y), model.residual(obs.y)).real),
        whitened_residual_power=float(np.vdot(r_whitened, r_whitened).real),
    )


# ==========================================================================
# 5. Continuous-DD identifiability audit
# ==========================================================================
@dataclass
class IdentifiabilityAudit:
    """``rho`` on the grid, on the manifold, and off the grid."""

    arm: str
    target: int
    dictionary: str
    rho_on_grid: np.ndarray
    rho_manifold: np.ndarray
    rho_manifold2: np.ndarray
    xi_rel_on_grid: float
    xi_rel_manifold: float
    n_templates: int
    n_nuisance_on_grid: int
    n_nuisance_manifold: int
    ncp_unit_on_grid: float
    ncp_unit_manifold: float
    dof_on_grid: int
    dof_manifold: int
    off_grid: Dict[Tuple[float, float], np.ndarray] = field(default_factory=dict)

    @property
    def rho_min_on_grid(self) -> float:
        return float(np.min(self.rho_on_grid)) if self.rho_on_grid.size else float("nan")

    @property
    def rho_min_manifold(self) -> float:
        return (
            float(np.min(self.rho_manifold)) if self.rho_manifold.size else float("nan")
        )

    @property
    def rho_min_manifold2(self) -> float:
        return (
            float(np.min(self.rho_manifold2)) if self.rho_manifold2.size else float("nan")
        )

    @property
    def off_grid_min(self) -> float:
        vals = [float(np.min(v)) for v in self.off_grid.values() if v.size]
        return float(np.min(vals)) if vals else float("nan")

    @property
    def off_grid_median(self) -> float:
        vals = [float(np.median(v)) for v in self.off_grid.values() if v.size]
        return float(np.median(vals)) if vals else float("nan")

    @property
    def off_grid_max(self) -> float:
        vals = [float(np.max(v)) for v in self.off_grid.values() if v.size]
        return float(np.max(vals)) if vals else float("nan")


def identifiability_audit(
    cfg,
    obs: Observation,
    model: ResidualModel,
    result: CancellationResult,
    *,
    target: int | None = None,
    dictionary: str = "belief",
    p_fa: float = 0.05,
    centre_only: bool = True,
    deltas: Iterable[Tuple[float, float]] = (
        (0.4, 0.0),
        (-0.4, 0.0),
        (0.0, 0.4),
        (0.0, -0.4),
        (0.4, 0.4),
        (-0.4, -0.4),
        (0.5, 0.5),
    ),
) -> IdentifiabilityAudit:
    """Is target ``q`` separable, on the grid and in the continuous model?

    ``rho_on_grid`` uses the other targets' on-grid columns as nuisance,
    ``rho_manifold`` adds their first-order tangent families, and ``off_grid``
    moves target ``q``'s own templates by fractional delay/Doppler offsets while
    the others keep their manifolds.  A target that stays masked under every
    offset is *intrinsically* co-located, and the remedy is more physical
    resolution rather than a better canceller.
    """
    A, ids = _dictionary(cfg, obs, dictionary)
    tgt = int(obs.weak_index if target is None else target)
    on_grid = target_conditioned_glrt(
        cfg, obs, result, model, target=tgt, p_fa=p_fa,
        nuisance_manifold=0, dictionary=dictionary, centre_only=centre_only,
    )
    manifold = target_conditioned_glrt(
        cfg, obs, result, model, target=tgt, p_fa=p_fa,
        nuisance_manifold=1, dictionary=dictionary, centre_only=centre_only,
    )
    manifold2 = target_conditioned_glrt(
        cfg, obs, result, model, target=tgt, p_fa=p_fa,
        nuisance_manifold=2, dictionary=dictionary, centre_only=centre_only,
    )

    sources: Sequence[TargetSource] = (
        obs.targets_belief if dictionary == "belief" else obs.targets
    ) or obs.targets
    per_target = [s for s in sources if int(s.target) == tgt]
    step = float(cfg.cancellation.tangent_step_bins)
    base_A_q = A[:, np.asarray(ids) == tgt]

    off_grid: Dict[Tuple[float, float], np.ndarray] = {}
    for dk, dl in deltas:
        if dk == 0.0 and dl == 0.0:
            continue
        cols = [
            math.sqrt(max(float(src.power), 0.0))
            * tangent_columns(
                cfg, float(src.doppler_bin) + float(dk),
                float(src.delay_bin) + float(dl), 0, step,
            )
            for src in per_target
        ]
        template = np.concatenate(cols, axis=1) if cols else base_A_q
        shifted = target_conditioned_glrt(
            cfg, obs, result, model, target=tgt, p_fa=p_fa,
            nuisance_manifold=1, dictionary=dictionary, template_override=template,
            centre_only=centre_only,
        )
        off_grid[(float(dk), float(dl))] = shifted.rho

    return IdentifiabilityAudit(
        arm=model.name,
        target=tgt,
        dictionary=dictionary,
        rho_on_grid=on_grid.rho,
        rho_manifold=manifold.rho,
        rho_manifold2=manifold2.rho,
        xi_rel_on_grid=on_grid.xi_rel_q,
        xi_rel_manifold=manifold.xi_rel_q,
        n_templates=on_grid.n_templates,
        n_nuisance_on_grid=on_grid.n_nuisance_columns,
        n_nuisance_manifold=manifold.n_nuisance_columns,
        ncp_unit_on_grid=on_grid.ncp_unit,
        ncp_unit_manifold=manifold.ncp_unit,
        dof_on_grid=on_grid.dof_real,
        dof_manifold=manifold.dof_real,
        off_grid=off_grid,
    )


# ==========================================================================
# 6. Masking curve: how many co-located scatterers it takes
# ==========================================================================
@dataclass
class MaskingCurve:
    """``rho`` as a function of how many other targets are treated as nuisance.

    The shape of this curve *is* the mechanism.  If ``rho`` collapses gradually
    as neighbours are added, the masking is a union-of-subspaces (patch
    covering) effect and what binds is the *number* of co-located scatterers; if
    it collapses on the first neighbour, the tested target is masked by that one
    target alone and the fix is a different viewing geometry rather than more
    resolution.
    """

    arm: str
    target: int
    dictionary: str
    counts: List[int]
    rho_weighted: List[float]
    rho_min: List[float]
    ncp_unit: List[float]
    distance_bins: List[float]
    neighbours: List[int]


def masking_curve(
    cfg,
    obs: Observation,
    model: ResidualModel,
    result: CancellationResult,
    *,
    target: int | None = None,
    dictionary: str = "belief",
    p_fa: float = 0.05,
    counts: Iterable[int] = (0, 1, 2, 3, 5, 9),
    centre_only: bool = True,
) -> MaskingCurve:
    """``rho`` versus the number of nearest co-located targets taken as nuisance."""
    A, ids = _dictionary(cfg, obs, dictionary)
    tgt = int(obs.weak_index if target is None else target)
    sources: Sequence[TargetSource] = (
        (obs.targets_belief if dictionary == "belief" else obs.targets)
        or obs.targets_belief
        or obs.targets
    )
    positions: Dict[int, List[Tuple[float, float]]] = {}
    for src in sources:
        positions.setdefault(int(src.target), []).append(
            (float(src.doppler_bin), float(src.delay_bin))
        )
    if tgt not in positions:
        raise ValueError("target %d has no source" % (tgt,))
    mine = np.mean(np.asarray(positions[tgt], dtype=float), axis=0)
    others = sorted(q for q in positions if q != tgt)
    dist = {q: float(np.linalg.norm(np.mean(np.asarray(positions[q], dtype=float), axis=0) - mine))
            for q in others}
    order = sorted(others, key=lambda q: dist[q])

    rho_w: List[float] = []
    rho_m: List[float] = []
    ncp: List[float] = []
    reach: List[float] = []
    used: List[int] = []
    for n in counts:
        n = int(min(max(n, 0), len(order)))
        subset = order[:n]
        got = target_conditioned_glrt(
            cfg, obs, result, model, target=tgt, p_fa=p_fa,
            nuisance_manifold=0, dictionary=dictionary, centre_only=centre_only,
            nuisance_targets=subset,
        )
        rho_w.append(got.rho_weighted)
        rho_m.append(got.rho_min)
        ncp.append(got.ncp_unit)
        reach.append(max((dist[q] for q in subset), default=0.0))
        used.append(n)
    return MaskingCurve(
        arm=model.name,
        target=tgt,
        dictionary=dictionary,
        counts=used,
        rho_weighted=rho_w,
        rho_min=rho_m,
        ncp_unit=ncp,
        distance_bins=reach,
        neighbours=[int(q) for q in order],
    )


# ==========================================================================
# 7. Single-target mechanism validation
# ==========================================================================
def restrict_to_target(cfg, obs: Observation, target: int) -> Observation:
    """The same geometry with one echo and a one-target protection basis.

    This is the mechanism-validation observation: it answers "does interference
    cancellation transfer to detection at all" with the multi-target
    identifiability question removed, and nothing else.  It keeps ``X`` and the
    direct field untouched, zeroes every other echo, restricts ``A``/``A_true``
    and the ids to the tested target, and rebuilds both protection bases from
    that target alone so the canceller is never asked to protect phantoms.
    """
    ids = np.asarray(obs.A_true_target_ids)
    sel = ids == int(target)
    if not np.any(sel):
        raise ValueError("target %d has no columns in the true dictionary" % (target,))

    true_sources = [s for s in obs.targets if int(s.target) == int(target)]
    belief_sources = [
        s for s in (obs.targets_belief or []) if int(s.target) == int(target)
    ]
    if not true_sources:
        raise ValueError("target %d has no true source" % (target,))

    A_true = target_dictionary(
        cfg, true_sources, covariance_expanded=False
    )
    alpha_true = np.asarray(obs.alpha_true)[sel].copy()
    s_target = A_true @ alpha_true
    if belief_sources:
        A = target_dictionary(cfg, belief_sources)
        A_ids = np.full(A.shape[1], int(target), dtype=int)
        centre_blocks = []
        for src in belief_sources:
            width = target_dictionary(cfg, [src]).shape[1]
            block = np.zeros(width, dtype=bool)
            if width:
                block[0] = True
            centre_blocks.append(block)
        A_centres = np.concatenate(centre_blocks)
    else:  # a caller that built the observation by hand
        A, A_ids = obs.A, np.full(int(obs.A.shape[1]), int(target), dtype=int)
        A_centres = _centre_mask(cfg, A.shape[1])
    true_centres = _centre_mask(cfg, A_true.shape[1])
    noise = obs.y - obs.x_direct - obs.s_target
    n_bins = int(obs.y.size)
    return replace(
        obs,
        y=obs.x_direct + s_target + noise,
        A=A,
        A_target_ids=A_ids,
        A_true_target_ids=np.full(A_true.shape[1], int(target), dtype=int),
        A_centre_mask=A_centres,
        A_true_centre_mask=true_centres,
        s_target=s_target,
        alpha_true=alpha_true,
        targets=true_sources,
        targets_belief=belief_sources or true_sources,
        basis_belief=_protection_basis(cfg, belief_sources or true_sources, n_bins),
        basis_truth=_protection_basis(cfg, true_sources, n_bins),
        weak_index=int(target),
    )
