"""TP-UIC V1 -- Target-Preserving Uncertainty-aware Interference Cancellation.

Why this module exists
----------------------
``interference.direct_cancellation_db`` (``kappa_dc`` internally) is a single
constant that multiplies the *aggregated* direct field at a sensing receiver
(``model.py``: ``residual_direct = kappa_dc * I_sense_field[j]``).  ``KAPPA_
DERIVATION.md`` shows that constant is a *requirement* the scenario implies, not
a description of a receiver: at 600 m / RCS 0.1 the geometry asks for 52.2 dB
and the shipped 40 dB leaves the residual 10.2 dB above the processed echo.
Nothing in the library actually cancels anything -- ``grep`` for
``cancel|subspace|projection|reconstruct`` finds configuration and comments.

This module makes the residual an *output* of an executable estimator instead.

Signal model
------------
At sensing receiver ``j`` one CPI gives a complex observation in the OTFS
delay--Doppler (DD) domain, vectorised over the ``K = N * L`` grid bins::

    y_j = X_j h_j  +  A_j alpha_j  +  n_j                     (1a)
    X_j = [ sqrt(P_i g_ij) a(theta^dir_ij) ]_{i in A}          (1b)
    A_j = [ sqrt(P_i g^tgt_ijq G_proc G_hw) a(theta^tgt_ijq) ] (1c)

where ``a(theta)`` is the *unit-norm* full-grid OTFS response of a unit-
amplitude scatterer at the fractional DD offset ``theta = (f_doppler, f_delay)``
in bin units -- i.e. exactly the kernel already used by
:mod:`isac_sim.waveform`.  Two normalisation facts make this bridge to the
link-table model exact rather than analogical:

* ``||a(theta)||^2 = 1``, so ``||X_j h||^2 = sum_i P_i g_ij = I_sense_field[j]``
  -- the sample-level input interference power *is* the model's direct field.
* the echo column norm is ``P_i g^tgt G_proc G_hw``, i.e. the same expression
  ``model.py`` uses for ``signal``.  ``G_proc`` appears because the DD-domain
  matched filter integrates the full ``N*L``-element frame coherently, while
  the direct path is not matched-filtered against its own waveform.

The four modules
----------------
**M1 -- interference dictionary.**  Every active illuminator contributes its
known waveform's direct response, optionally with a local fractional-DD
tangent basis (the direct path carries a delay/Doppler of its own; a
single-bin model would mis-fit it).

**M2 -- target tangent protection.**  For each believed target the receiver
protects not only the centre response ``a(theta_hat)`` but its first
derivatives, i.e. the columns of the Jacobian ``J = [a, da/df_delay,
da/df_doppler]``.  Staking ``U = orth([J_1 ... J_Q])`` and ``P = U U^H``,
``M = I - P`` splits the observation into a subspace that may contain target
evidence and a subspace that may be used for interference learning.  Because
``a(theta_true) = a(theta_hat) + J dtheta + O(||dtheta||^2)``, the *un*protected
part of the true target satisfies ``||M a_true|| = O(||dtheta||^2)``: the
protection is a first-order statement with a stated error term, not a magic
window.

**M3 -- protected MAP estimate.**  ``h_hat = argmin ||M(y - X h)||^2 +
sigma_n^2 ||h||^2_{R_h^{-1}}``, with the closed form

    h_hat = (X^H M X + sigma_n^2 R_h^{-1})^{-1} X^H M y          (2a)
    C_h   = (R_h^{-1} + sigma_n^{-2} X^H M X)^{-1}               (2b)

and the *conservative* subtraction ``r1 = y - M X h_hat``.  This has a clean
invariant that the tests pin: since ``U^H M = 0``,

    P r1 = P y                                                   (2c)

-- the part of the observation defined as target-carrying is left untouched by
stage 1.  An interference estimator therefore cannot "fit the target away".

**M4 -- uncertainty accounting.**  The residual is split into what was
*deliberately retained* and what remains because of estimation error::

    I_res = tr(P X R_h X^H P)  +  tr(M X C_h X^H M)               (3)

so "cancellation was not complete" and "the algorithm failed" stop being the
same statement.  The first term is the price paid for protecting the target and
shrinks only if the belief is good; the second is honest estimation noise and
shrinks with the reference budget.  ``C_h`` and the resulting ``I_res`` are the
interface a scheduler would consume.

Equation (3) is written in **total** power (no division by ``K``), because its
measured counterpart ``||P x||^2`` and the model's ``I_sense_field`` are both
totals.  The per-element average in the method note is the same relation divided
on both sides; mixing the two conventions shifts the prediction by
``10 log10(K)`` = 36 dB at ``K = 4096``.  See :func:`residual_accounting`.

**Stage 2 (optional).**  After a first cancellation the receiver has candidate
targets; modelling them *explicitly* lets it subtract the interference that
stage 1 deliberately kept::

    (h_hat, alpha_hat) = argmin ||y - X h - A_Omega alpha||^2
                              + lambda_h ||h||^2 + lambda_t ||alpha||^2   (4a)
    r2 = y - X h_hat_joint                                              (4b)

The target estimate is never subtracted from ``r2``.

Measurement contract
--------------------
The canceller's *detection* numbers are only as good as the hypothesis pair they
are scored on, so the module ships the pair builder::

    obs_h1, obs_h0 = build_observation_pair(...)

Both observations share ``x_direct``, ``X``, ``A`` and every target's echo except
the one under test; only the noise is redrawn.  Do **not** approximate this with
two :func:`build_observation` calls -- each call consumes the rng to draw the
direct phases, so the two hypotheses receive different illumination and every
statistic ratio collapses towards 1 (measured: 1.00--1.15 while the physics
promised +17 dB).  ``test_independent_builds_are_not_a_matched_pair`` pins that
the shortcut really is broken, so the guard cannot be quietly dropped.

What this module is *not*
-------------------------
It is not a new SIC idea: DD-domain SIC for OTFS-ISAC exists (Tang & Yu 2024),
two-stage digital cancellation exists, and target-preserving subspace methods
exist in passive radar.  What is built here is the specific receiver that the
frozen link-table model has been *assuming* -- with the constant replaced by
(2)--(3) and the assumption made measurable.

Numerical conventions
---------------------
* Everything is a flat ``(K,)`` complex vector with ``K = N * L`` and
  ``index = doppler_bin * L + delay_bin``, matching
  :func:`isac_sim.waveform.full_otfs_kernel`'s ``(N, L)`` grid flattened in C
  order (the kernel's own convention).
* ``sigma^2`` is the *per-bin* noise power, so ``model.noise_power(cfg)``.
* ``R_h`` is per-coefficient, diagonal, in units of the coefficient itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import Config
from .waveform import full_otfs_kernel

EPS = 1e-12


# ==========================================================================
# 1. Sources, observation and result containers
# ==========================================================================
@dataclass(frozen=True)
class DirectSource:
    """One active illuminator's direct-path term at the receiver."""

    uav: int
    power: float  # radiated sensing power reaching the direct path, W
    gain: float  # direct-path power gain g_ij (dimensionless)
    doppler_bin: float  # fractional Doppler bin of the UAV-UAV link
    delay_bin: float  # fractional delay bin of the UAV-UAV link

    @property
    def power_at_receiver(self) -> float:
        return self.power * self.gain


@dataclass(frozen=True)
class TargetSource:
    """One believed/true target echo observed by the receiver."""

    uav: int
    target: int
    power: float  # P_sense * target_gain * G_proc * G_hw, i.e. echo power
    doppler_bin: float
    delay_bin: float


@dataclass
class Observation:
    """The DD-domain observation plus everything a canceller may know."""

    y: np.ndarray  # (K,) complex
    X: np.ndarray  # (K, d) direct dictionary
    A: np.ndarray  # (K, Q) believed target dictionary
    x_direct: np.ndarray  # (K,) true direct component  (diagnostics only)
    s_target: np.ndarray  # (K,) true echo component    (diagnostics only)
    h_true: np.ndarray  # (d,) true direct coefficients
    alpha_true: np.ndarray  # (Q,) true echo coefficients
    sigma2: float
    direct: List[DirectSource] = field(default_factory=list)
    targets: List[TargetSource] = field(default_factory=list)
    # The *believed* target sources.  ``targets`` above holds the true ones, so
    # without this field a caller cannot rebuild a belief-side dictionary from a
    # subset of the targets -- which is what the V1.1 continuous-DD
    # identifiability audit needs (it asks whether one target's *local manifold*
    # is still separable once the other targets' manifolds are treated as
    # nuisance).  Defaults to ``None`` so every existing construction site and
    # every recorded result stays valid.
    targets_belief: List[TargetSource] | None = None
    # Protection bases: the receiver only ever uses the belief one.  The truth
    # basis exists so an experiment can price the belief error.
    basis_belief: np.ndarray | None = None
    basis_truth: np.ndarray | None = None
    # Which target is under test (the "weak target").  The statistic is formed
    # on the whole block of columns belonging to this target, because a target
    # is illuminated by several UAVs and a single-column template would
    # understate its detectability.
    weak_index: int = 0
    # Target index of every column of ``A`` and of ``A_true``, so an experiment
    # can select a target's block and drop its echo to form an H0 observation
    # without rebuilding the geometry.
    A_target_ids: np.ndarray | None = None
    A_true_target_ids: np.ndarray | None = None


@dataclass
class CancellationResult:
    """Everything a downstream scheduler would need from the receiver.

    The accounting is a *component transfer*, not a difference of totals.
    Every arm here is a linear map ``f`` on the observation, so the observation
    residual decomposes exactly::

        y - f(y) = [x - f(x)] + [s - f(s)] - f(n)

    and the three bracketed terms are the interference that survived, the echo
    that survived, and the noise the estimator pulled into the observation.  A
    single ``||x - f(y)||^2`` would mix them, and it mixes them in the direction
    that flatters protection -- which is why the fields below are separate.
    """

    name: str
    residual: np.ndarray  # (K,) complex, the cleaned observation y - f(y)
    h_hat: np.ndarray  # (d,) interference coefficient estimate applied to y
    c_h_diag: np.ndarray  # (d,) posterior variance of h_hat
    i_in: float  # ||x_direct||^2, the input direct power
    i_res: float  # i_res_structural + i_res_estimate, the interference left behind
    i_res_structural: float  # ||x_direct - f(x_direct)||^2, what f refuses to remove
    i_res_estimate: float  # ||f(n)||^2, the estimation error pulled in from noise
    i_res_pred: float  # predicted by eq. (3)
    i_res_retained: float  # first term of eq. (3): the price of protection
    i_res_pred_estimate: float  # second term of eq. (3): estimation uncertainty
    eta_protect: float  # ||P s||^2 / ||s||^2 -- protection coverage (design)
    eta_survive: float  # ||s - f(s)||^2 / ||s||^2 -- echo survival (realised)
    noise_enhance_db: float  # 10 log10(||f(n)||^2 / ||n||^2)
    protect_dim: int  # rank of the protection subspace
    n_coefficients: int  # number of complex interference coefficients fit
    matched_stat: float = 0.0  # weak-target statistic at the belief offset
    candidates: Tuple[int, ...] = ()

    @property
    def kappa_db(self) -> float:
        """Measured cancellation depth, i.e. the effective ``kappa_dc``.

        Defined on the *interference* residual only, and that is the whole point
        of splitting it: ``model.compute_link_tables`` puts the noise floor and
        the residual direct field in the same denominator but as separate terms,
        so the constant ``kappa_dc`` never described the noise.  ``I_res`` is
        therefore ``||x - f(x)||^2 + ||f(n)||^2`` -- the two are orthogonal
        (``f(n)`` lies in ``span(M X)``, the structural residual does not), so
        the sum is exact and equals eq. (3) term by term.  Leaving ``f(n)`` out
        would credit every estimator with a depth it does not have, because the
        estimation error is *born* from the noise the estimator fits.
        """
        if self.i_res <= 0.0 or self.i_in <= 0.0:
            return float("inf")
        return 10.0 * math.log10(self.i_in / self.i_res)

    @property
    def kappa_pred_db(self) -> float:
        if self.i_res_pred <= 0.0 or self.i_in <= 0.0:
            return float("inf")
        return 10.0 * math.log10(self.i_in / self.i_res_pred)

    @property
    def calibration_error_db(self) -> float:
        """|measured - predicted| depth, the residual-covariance check."""
        if self.i_res <= 0.0 or self.i_res_pred <= 0.0:
            return float("nan")
        return abs(10.0 * math.log10(self.i_res_pred / self.i_res))


# ==========================================================================
# 2. DD-domain primitives
# ==========================================================================
def wavelength(cfg: Config) -> float:
    return cfg.waveform.c / cfg.waveform.fc


def dd_offset_from_physical(cfg: Config, tau_s: float, nu_hz: float) -> Tuple[float, float]:
    """Physical ``(tau, nu)`` -> fractional ``(doppler_bin, delay_bin)``.

    Identical to the conversion used when ``model.build_base_gains`` fills
    ``doppler_bin`` / ``delay_bin`` for the target echoes, so the sample-level
    observation and the link tables agree bin for bin.
    """
    w = cfg.waveform
    return nu_hz * w.N * w.T, tau_s * w.L * w.delta_f


def direct_link_offset(
    cfg: Config, geom, i: int, j: int
) -> Tuple[float, float]:
    """Fractional DD offset of the UAV ``i`` -> UAV ``j`` direct path.

    Delay is the geometric range over ``c``; Doppler is the *bistatic* radial
    rate ``(v_i - v_j) . u_ij`` over the wavelength.  The target-echo convention
    in ``build_base_gains`` is the same one with the target substituted, so a
    target that happens to sit on the baseline is indistinguishable from a
    direct path -- which is exactly the near-far problem this module is about.
    """
    w = cfg.waveform
    vec = np.asarray(geom.p_uav[j], dtype=float) - np.asarray(geom.p_uav[i], dtype=float)
    dist = max(float(np.linalg.norm(vec)), 1.0)
    unit = vec / dist
    tau = dist / w.c
    nu = float(np.dot(np.asarray(geom.v_uav[i]) - np.asarray(geom.v_uav[j]), unit)) / wavelength(cfg)
    return dd_offset_from_physical(cfg, tau, nu)


def target_link_offset(cfg: Config, geom, i: int, j: int, q: int) -> Tuple[float, float]:
    """Fractional DD offset of the bistatic ``i -> q -> j`` echo path."""
    w = cfg.waveform
    p_i, p_j, p_q = geom.p_uav[i], geom.p_uav[j], geom.p_tgt[q]
    v_i, v_j, v_q = geom.v_uav[i], geom.v_uav[j], geom.v_tgt[q]
    vec_iq, vec_jq = p_q - p_i, p_q - p_j
    d_iq = max(float(np.linalg.norm(vec_iq)), 1.0)
    d_jq = max(float(np.linalg.norm(vec_jq)), 1.0)
    tau = (d_iq + d_jq) / w.c
    rr_tx = float(np.dot(v_q - v_i, vec_iq / d_iq))
    rr_rx = float(np.dot(v_q - v_j, vec_jq / d_jq))
    return dd_offset_from_physical(cfg, tau, (rr_tx + rr_rx) / wavelength(cfg))


def kernel_vector(cfg: Config, doppler_bin: float, delay_bin: float) -> np.ndarray:
    """Unit-norm full-grid DD response of a unit-amplitude scatterer.

    Wraps :func:`isac_sim.waveform.full_otfs_kernel`, which is already
    L2-normalised and cached.  The returned vector is flattened in C order over
    ``(doppler, delay)``, matching :func:`otfs_demodulate`'s output layout.
    """
    w = cfg.waveform
    grid = full_otfs_kernel(
        int(w.N),
        int(w.L),
        float(doppler_bin),
        float(delay_bin),
        float(w.delta_f),
        int(round(w.fc / 1e6)),
    )
    return np.asarray(grid, dtype=complex).reshape(-1)


def tangent_columns(
    cfg: Config,
    doppler_bin: float,
    delay_bin: float,
    order: int,
    step: float,
) -> np.ndarray:
    """``[a, da/df_delay, da/df_doppler, ...]`` as a ``(K, 1 + 2*order)`` block.

    Central finite differences of the *exact* OTFS response.  A one-sided or
    analytic derivative would be smaller but the kernel is already cached, so
    the difference costs nothing and stays honest about the numerical model.
    """
    centre = kernel_vector(cfg, doppler_bin, delay_bin)
    if order <= 0:
        return centre[:, None]
    h = float(max(step, EPS))
    d_delay = (kernel_vector(cfg, doppler_bin, delay_bin + h)
               - kernel_vector(cfg, doppler_bin, delay_bin - h)) / (2.0 * h)
    d_doppler = (kernel_vector(cfg, doppler_bin + h, delay_bin)
                 - kernel_vector(cfg, doppler_bin - h, delay_bin)) / (2.0 * h)
    columns = [centre, d_delay, d_doppler]
    return np.stack(columns, axis=1)


# ==========================================================================
# 3. Subspace projectors (never materialise a K x K matrix)
# ==========================================================================
@dataclass
class Subspace:
    """Orthonormal basis plus its two projectors, applied implicitly."""

    U: np.ndarray  # (K, r) orthonormal columns
    rank: int

    def project(self, x: np.ndarray) -> np.ndarray:
        """``P x`` -- the component inside the target-carrying subspace."""
        if self.rank == 0:
            return np.zeros_like(x)
        return self.U @ (self.U.conj().T @ x)

    def complement(self, x: np.ndarray) -> np.ndarray:
        """``M x = (I - P) x`` -- the component available for interference learning."""
        if self.rank == 0:
            return x
        return x - self.U @ (self.U.conj().T @ x)

    def complement_matrix(self, B: np.ndarray) -> np.ndarray:
        """``M B`` for a matrix ``B`` (K, m), column-wise but batched."""
        if self.rank == 0:
            return B
        return B - self.U @ (self.U.conj().T @ B)


def orthonormalise(
    columns: np.ndarray, tol: float = 1e-6, max_rank: int | None = None
) -> Subspace:
    """Rank-trimmed orthonormal basis via the Gram matrix.

    The Gram route (``m x m`` eigendecomposition, ``m`` = number of candidate
    columns) is used rather than a QR/SVD of the ``K x m`` matrix because ``K``
    is 4096 while ``m`` is a few hundred.

    Columns are **normalised before** the eigenvalue cut, and that is not a
    numerical detail: the DD tangent columns are finite differences and their
    norms scale like ``1/step``, i.e. hundreds of times the centre columns'.
    A cut placed relative to the largest eigenvalue of the *unnormalised* Gram
    therefore admits directions whose only distinction is finite-difference
    noise, and the reported protection rank inflates by 3x (measured: 195
    instead of 62 on the 600 m scenario).  What matters for "may this subspace
    hold target evidence" is correlation, not magnitude, so the cut is applied
    to the correlation matrix with an absolute threshold.
    """
    B = np.asarray(columns, dtype=complex)
    if B.ndim == 1:
        B = B[:, None]
    if B.shape[1] == 0:
        return Subspace(U=np.zeros((B.shape[0], 0), dtype=complex), rank=0)
    norms = np.linalg.norm(B, axis=0)
    keep_cols = norms > EPS
    if not np.any(keep_cols):
        return Subspace(U=np.zeros((B.shape[0], 0), dtype=complex), rank=0)
    B = B[:, keep_cols] / norms[keep_cols][None, :]
    gram = B.conj().T @ B
    gram = 0.5 * (gram + gram.conj().T)
    w, V = np.linalg.eigh(gram)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    keep = w > float(tol)
    if max_rank is not None:
        keep &= np.arange(w.size) < int(max_rank)
    w, V = w[keep], V[:, keep]
    if w.size == 0:
        return Subspace(U=np.zeros((B.shape[0], 0), dtype=complex), rank=0)
    U = (B @ V) / np.sqrt(w)[None, :]
    return Subspace(U=U, rank=int(U.shape[1]))


# ==========================================================================
# 4. Dictionary construction
# ==========================================================================
def direct_dictionary(
    cfg: Config,
    sources: Sequence[DirectSource],
    tangent_order: int | None = None,
    tangent_step: float | None = None,
) -> np.ndarray:
    """``X_j`` of eq. (1b), with an optional fractional-DD tangent extension.

    ``tangent_order = 0`` (the default) gives one column per active illuminator:
    the direct path's delay and Doppler are *computed* from the shared UAV
    positions, so only a complex gain is unknown.  ``tangent_order = n`` adds
    ``2n`` columns per illuminator to absorb synchronisation/oscillator
    mismatch; each added column costs estimation variance (eq. 3) and is
    therefore a robustness axis rather than an improvement.
    """
    c = cfg.cancellation
    order = int(c.interference_tangent_order if tangent_order is None else tangent_order)
    step = float(c.tangent_step_bins if tangent_step is None else tangent_step)
    blocks: List[np.ndarray] = []
    for src in sources:
        amp = math.sqrt(max(src.power_at_receiver, 0.0))
        blocks.append(amp * tangent_columns(cfg, src.doppler_bin, src.delay_bin, order, step))
    if not blocks:
        return np.zeros((cfg.waveform.N * cfg.waveform.L, 0), dtype=complex)
    return np.concatenate(blocks, axis=1)


def target_dictionary(
    cfg: Config, sources: Sequence[TargetSource], tangent_order: int | None = None
) -> np.ndarray:
    """``A_j`` of eq. (1c): one protected tangent set per believed target."""
    c = cfg.cancellation
    order = int(c.tangent_order if tangent_order is None else tangent_order)
    step = float(c.tangent_step_bins)
    blocks: List[np.ndarray] = []
    for src in sources:
        amp = math.sqrt(max(src.power, 0.0))
        blocks.append(amp * tangent_columns(cfg, src.doppler_bin, src.delay_bin, order, step))
    if not blocks:
        return np.zeros((cfg.waveform.N * cfg.waveform.L, 0), dtype=complex)
    return np.concatenate(blocks, axis=1)


# ==========================================================================
# 5. Estimators and cancellation
# ==========================================================================
def protected_map(
    y: np.ndarray,
    X: np.ndarray,
    subspace: Subspace,
    sigma2: float,
    prior_variance: float | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eq. (2): protected (ridge-regularised) least squares.

    Returns ``(h_hat, c_diag, MX)`` where ``MX`` is the projected dictionary
    ``M X``, reused for the subtraction step.  ``prior_variance=None`` drops the
    prior and leaves plain protected LS -- which is the ``R_h^{-1} -> 0`` limit
    the module docstring quotes.

    The estimate is linear in ``y``, and that is what the accounting below
    exploits: the same routine is applied to the direct component, the echo
    component and the noise component separately, so an arm's effect on each is
    known exactly rather than inferred from the total.
    """
    MX = subspace.complement_matrix(X)
    gram = MX.conj().T @ MX
    rhs = MX.conj().T @ y
    sg = _positive_sigma(sigma2)
    if prior_variance is not None and prior_variance > 0.0:
        gram_r = gram + (sg / float(prior_variance)) * np.eye(gram.shape[0])
        cov = np.linalg.inv((gram / sg)
                            + (1.0 / float(prior_variance)) * np.eye(gram.shape[0]))
        h_hat = np.linalg.solve(gram_r, rhs)
    else:
        cov = np.linalg.pinv(gram, rcond=1e-10) * sg
        h_hat = np.linalg.lstsq(gram, rhs, rcond=1e-10)[0]
    return h_hat, np.real(np.diag(cov)), MX


def joint_refine(
    y: np.ndarray,
    X: np.ndarray,
    A_cand: np.ndarray,
    sigma2: float,
    prior_variance: float | None,
    target_prior: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Eq. (4a): simultaneous interference / target fit on a candidate support.

    Both blocks carry a Gaussian prior, so the normal equations stay ``(d + m)``
    square and no ``K x K`` inverse appears.  The target block is only ever used
    to *explain* the observation on the candidate support; it is not subtracted.
    """
    blocks = [X]
    if A_cand.shape[1] > 0:
        blocks.append(A_cand)
    D = np.concatenate(blocks, axis=1)
    gram = D.conj().T @ D
    rhs = D.conj().T @ y
    lam = np.zeros(gram.shape[0])
    if prior_variance is not None and prior_variance > 0.0:
        lam[: X.shape[1]] = float(sigma2) / float(prior_variance)
    if A_cand.shape[1] > 0:
        lam[X.shape[1]:] = float(sigma2) / max(float(target_prior), EPS)
    theta = np.linalg.solve(gram + np.diag(lam), rhs)
    return theta[: X.shape[1]], theta[X.shape[1]:]


def _positive_sigma(sigma2: float) -> float:
    """Guard a variance against the module-level ``EPS``.

    ``EPS = 1e-12`` is not a safe divisor guard here: the default radio's noise
    power is ``3.83e-14`` W, i.e. 26x *smaller* than EPS, so ``max(sigma2, EPS)``
    silently replaces the true variance by a value 26x too large.  The predicted
    residual then comes out 26x too large (measured: 365 instead of 14 noise
    units), which looks like a modelling error and is not one.  ``model.py``
    documents exactly this failure mode for ``radio.eps_mode``; the same trap is
    avoided here by refusing a non-positive variance instead of clamping it.
    """
    value = float(sigma2)
    if not value > 0.0:
        raise ValueError(f"noise variance must be positive, got {sigma2!r}")
    return value


def residual_accounting(
    X: np.ndarray,
    subspace: Subspace,
    sigma2: float,
    prior_variance: float | None,
) -> Tuple[float, float]:
    """Eq. (3) in **total power**, without forming any ``K x K`` matrix.

    ``tr(P X R_h X^H P) = sum_i R_h[i] * ||(X^H U)[i, :]||^2`` because the
    projector is idempotent, and ``tr(M X C_h X^H M) = tr(C_h X^H M X)`` because
    the trace is cyclic.  Both reduce to small matrix products.

    No division by ``K``: the measured counterpart is ``||P x||^2``, a total
    power, and ``I_sense_field`` is a total power too.  The per-element average
    in the method note's eq. (3) is the same statement divided on both sides; the
    two conventions must not be mixed or the prediction is off by ``log10(K)``
    = 36 dB, which is exactly the size of the disagreement the first run showed.
    """
    xu = subspace.U.conj().T @ X if subspace.rank else np.zeros((0, X.shape[1]), dtype=complex)
    r_h = (np.full(X.shape[1], float(prior_variance))
           if prior_variance is not None and prior_variance > 0.0
           else np.zeros(X.shape[1]))
    retained = float(np.sum(r_h * np.sum(np.abs(xu) ** 2, axis=0)))
    mx = subspace.complement_matrix(X)
    gram = mx.conj().T @ mx
    eye = np.eye(gram.shape[0])
    inv_r_h = (eye / float(prior_variance)
               if prior_variance is not None and prior_variance > 0.0
               else np.zeros_like(eye))
    c_h = np.linalg.inv(gram / _positive_sigma(sigma2) + inv_r_h)
    # tr(M X C_h X^H M) = tr(C_h X^H M X): one small trace, not a sum of
    # diagonal products.  The two differ whenever the Gram is not diagonal, and
    # on this scenario it is not (cond ~ 2e3), which is why the elementwise form
    # over-predicted the estimation residual.  The no-prior LS case must reduce
    # to sigma^2 * rank(M X) exactly, and tr(C_h G) does while the elementwise
    # form does not.
    estimate = float(np.real(np.trace(c_h @ gram)))
    return retained, estimate


# ==========================================================================
# 6. The receiver
# ==========================================================================
@dataclass
class _Arm:
    """One comparison arm, stored as its action on the three components."""

    name: str
    sub_direct: np.ndarray
    sub_target: np.ndarray
    sub_noise: np.ndarray
    h_hat: np.ndarray
    c_diag: np.ndarray
    subspace: Subspace
    candidates: Tuple[int, ...] = ()
    # How the residual is predicted for this arm.  ``estimator`` runs eq. (3);
    # the other two arms do not estimate a channel at all, so running the
    # estimator formula on them would invent a residual for an arm that has
    # none -- exactly the 27 dB that the first run reported for ``no_ic``.
    predict: str = "estimator"


def cancellation_arms(
    cfg: Config,
    obs: Observation,
    *,
    weak_target: int | None = None,
    threshold: float = 0.0,
) -> Dict[str, CancellationResult]:
    """Run every comparison arm of the first TP-UIC experiment on one trial.

    The arms are the ones the method note lists, and each isolates a single
    mechanism -- the difference between two arms is attributable:

    ``no_ic``            raw observation (lower bound)
    ``fixed_kappa``      the *frozen model's assumption*: the true direct path
                         suppressed by exactly ``direct_cancellation_db``,
                         with no estimator at all
    ``plain_ls``         unprotected least squares over the full grid
    ``ridge_ls``         unprotected regularised (MAP) least squares
    ``protected_ls``     target protection, plain LS (no prior)
    ``tp_uic_stage1``    target protection + MAP + the uncertainty report
    ``tp_uic_full``      stage 1 followed by the joint refinement of eq. (4)
    ``perfect_channel``  oracle: subtract the true direct path exactly

    ``threshold`` is the CFAR level used to accept a target candidate for the
    joint stage; pass an empirically calibrated H0 level so the candidate set is
    never an oracle.
    """
    X, A, y = obs.X, obs.A, obs.y
    x, s = obs.x_direct, obs.s_target
    n = y - x - s
    sigma2 = float(obs.sigma2)
    n_bins = int(y.size)
    wt = int(obs.weak_index if weak_target is None else weak_target)
    # The weak-target statistic is the energy in the block subspace of that
    # target's columns, normalised by its rank, so its H0 mean is the per-bin
    # noise variance and a CFAR threshold can be set without knowing the
    # geometry.  A single-column template would be wrong here: a target is
    # illuminated by several UAVs and the receiver does not know which path
    # will dominate.
    weak_block = None
    if obs.A_target_ids is not None and obs.A.shape[1]:
        cols = obs.A[:, np.asarray(obs.A_target_ids) == wt]
        if cols.shape[1]:
            weak_block = orthonormalise(cols)
    belief = Subspace(
        U=obs.basis_belief,
        rank=0 if obs.basis_belief is None else obs.basis_belief.shape[1],
    )
    empty = Subspace(U=np.zeros((n_bins, 0), dtype=complex), rank=0)
    prior = cfg.cancellation.prior_variance if cfg.cancellation.prior_variance else None
    parts: Dict[str, _Arm] = {}

    def operator(name: str, subspace: Subspace, pv, candidates: Tuple[int, ...] = ()) -> _Arm:
        """Apply the same linear estimator to each component separately."""
        def solve(v: np.ndarray):
            if candidates:
                A_cand = A[:, list(candidates)] if candidates else np.zeros((n_bins, 0), dtype=complex)
                h, _a = joint_refine(v, X, A_cand, sigma2, pv, pv or 1.0)
                return h, np.zeros(X.shape[1])
            h, c, _mx = protected_map(v, X, subspace, sigma2, pv)
            return h, c

        h_y, c_y = solve(y)
        h_x, _ = solve(x)
        h_s, _ = solve(s)
        h_n, _ = solve(n)
        mx = subspace.complement_matrix(X)
        if candidates:
            return _Arm(name, X @ h_x, X @ h_s, X @ h_n, h_y, c_y, subspace, candidates)
        return _Arm(name, mx @ h_x, mx @ h_s, mx @ h_n, h_y, c_y, subspace, candidates)

    # ---- no_ic ----------------------------------------------------------
    zero = np.zeros(n_bins, dtype=complex)
    parts["no_ic"] = _Arm("no_ic", zero.copy(), zero.copy(), zero.copy(),
                          np.zeros(X.shape[1], dtype=complex), np.zeros(X.shape[1]), empty,
                          predict="none")

    # ---- fixed_kappa: the frozen model's own assumption ------------------
    # The released model states that the residual direct *power* is
    # 10^(-kappa/10) of the input, so the residual amplitude is
    # sqrt(10^(-kappa/10)) and the subtracted amplitude is 1 - sqrt(...).
    # Nothing is estimated; this arm exists only so the constant can be read in
    # the same units as the algorithm.
    keep = math.sqrt(10.0 ** (-float(cfg.interference.direct_cancellation_db) / 10.0))
    parts["fixed_kappa"] = _Arm("fixed_kappa", (1.0 - keep) * x, zero.copy(), zero.copy(),
                                (1.0 - keep) * obs.h_true, np.zeros(X.shape[1]), empty,
                                predict="exact")

    # ---- plain_ls / ridge_ls: no protection ------------------------------
    for name, pv in (("plain_ls", None), ("ridge_ls", prior)):
        parts[name] = operator(name, empty, pv)

    # ---- protected_ls / tp_uic_stage1 ------------------------------------
    for name, pv in (("protected_ls", None), ("tp_uic_stage1", prior)):
        parts[name] = operator(name, belief, pv)

    # ---- tp_uic_full: candidate support fixed by the stage-1 statistic ---
    stage1_residual = y - (parts["tp_uic_stage1"].sub_direct
                           + parts["tp_uic_stage1"].sub_target
                           + parts["tp_uic_stage1"].sub_noise)
    candidate_cols: List[int] = []
    if obs.A_target_ids is not None and A.shape[1]:
        ids = np.asarray(obs.A_target_ids)
        for q in sorted(set(int(v) for v in ids)):
            cols = np.flatnonzero(ids == q)
            block = orthonormalise(A[:, cols])
            if not block.rank:
                continue
            projected = block.U.conj().T @ stage1_residual
            level = float(np.vdot(projected, projected).real) / block.rank
            if level > threshold:
                candidate_cols.extend(int(c) for c in cols)
    candidates = tuple(candidate_cols)
    parts["tp_uic_full"] = operator("tp_uic_full", belief, prior, candidates)

    # ---- perfect_channel -------------------------------------------------
    parts["perfect_channel"] = _Arm("perfect_channel", x.copy(), zero.copy(), zero.copy(),
                                    obs.h_true, np.zeros(X.shape[1]), empty, predict="exact")

    # ---- score -----------------------------------------------------------
    out: Dict[str, CancellationResult] = {}
    s_energy = float(np.vdot(s, s).real)
    n_energy = float(np.vdot(n, n).real)
    for name, arm in parts.items():
        sub_total = arm.sub_direct + arm.sub_target + arm.sub_noise
        residual = y - sub_total
        if arm.predict == "estimator":
            retained, pred_estimate = residual_accounting(X, arm.subspace, sigma2, prior)
            i_res_pred = retained + pred_estimate
        else:
            retained, pred_estimate = 0.0, 0.0
            # ``none``: nothing is removed, so the residual is the input field.
            # ``exact``: the arm knows the direct component, so its depth is a
            # definition rather than a prediction and the calibration error is
            # zero by construction, not by luck.
            i_res_pred = float(np.vdot(x, x).real) if arm.predict == "none" else None
        structural = float(np.vdot(x - arm.sub_direct, x - arm.sub_direct).real)
        estimation = float(np.vdot(arm.sub_noise, arm.sub_noise).real)
        if i_res_pred is None:
            i_res_pred = structural + estimation
        if weak_block is not None and weak_block.rank:
            statistic = float(np.vdot(weak_block.U.conj().T @ residual,
                                      weak_block.U.conj().T @ residual).real) / weak_block.rank
        else:
            statistic = 0.0
        if estimation <= 0.0 or n_energy <= 0.0:
            # An arm that removes nothing removes no noise: report it as -inf
            # rather than letting the EPS guard invent a -22 dB reading.
            noise_db = float("-inf")
        else:
            noise_db = float(10.0 * math.log10(estimation / n_energy))
        out[name] = CancellationResult(
            name=name,
            residual=residual,
            h_hat=arm.h_hat,
            c_h_diag=np.asarray(arm.c_diag, dtype=float),
            i_in=float(np.vdot(x, x).real),
            i_res=structural + estimation,
            i_res_structural=structural,
            i_res_estimate=estimation,
            i_res_pred=i_res_pred,
            i_res_retained=retained,
            i_res_pred_estimate=pred_estimate,
            eta_protect=float(np.vdot(belief.project(s), belief.project(s)).real / max(s_energy, EPS)),
            eta_survive=float(np.vdot(s - arm.sub_target, s - arm.sub_target).real / max(s_energy, EPS)),
            noise_enhance_db=noise_db,
            protect_dim=int(arm.subspace.rank),
            n_coefficients=int(X.shape[1]),
            matched_stat=statistic,
            candidates=arm.candidates,
        )
    return out


# ==========================================================================
# 7. Observation builder (bridge to the link tables)
# ==========================================================================
def build_observation(
    cfg: Config,
    geom_true,
    geom_belief,
    base,
    receiver: int,
    *,
    rng: np.random.Generator,
    sense_power: np.ndarray,
    radiated_power: np.ndarray,
    processing_gain: float,
    hw_gain: float,
    active_mask: np.ndarray | None = None,
    include_echo: bool = True,
    exclude_target: int | None = None,
    weak_index: int = 0,
) -> Observation:
    """Assemble ``(1)`` for one receiver from the *same* geometry as the tables.

    ``geom_belief`` may equal ``geom_true`` (perfect tracker).  The direct path
    never depends on the belief -- only the target dictionary and the protection
    basis do, which is precisely the asymmetry the experiment wants to price.
    """
    M, Q = cfg.scale.M, cfg.scale.Q
    n_bins = int(cfg.waveform.N * cfg.waveform.L)
    if active_mask is None:
        active_mask = np.ones(M, dtype=bool)

    direct: List[DirectSource] = []
    for i in range(M):
        if i == receiver or not active_mask[i]:
            continue
        k_bin, l_bin = direct_link_offset(cfg, geom_true, i, receiver)
        direct.append(
            DirectSource(
                uav=i,
                power=float(radiated_power[i]),
                gain=float(base.direct_gain[i, receiver]),
                doppler_bin=float(k_bin),
                delay_bin=float(l_bin),
            )
        )

    targets_true: List[TargetSource] = []
    targets_belief: List[TargetSource] = []
    true_ids: List[int] = []
    for i in range(M):
        if i == receiver or not active_mask[i]:
            continue
        for q in range(Q):
            gain = float(base.target_gain[i, receiver, q])
            if gain <= 0.0:
                continue
            k_t, l_t = target_link_offset(cfg, geom_true, i, receiver, q)
            k_b, l_b = target_link_offset(cfg, geom_belief, i, receiver, q)
            power = float(sense_power[i]) * gain * float(processing_gain) * float(hw_gain)
            targets_true.append(TargetSource(i, q, power, k_t, l_t))
            targets_belief.append(TargetSource(i, q, power, k_b, l_b))
            true_ids.append(int(q))

    X = direct_dictionary(cfg, direct)
    A = target_dictionary(cfg, targets_belief) if cfg.cancellation.protect_targets else \
        np.zeros((n_bins, 0), dtype=complex)
    A_true = target_dictionary(cfg, targets_true)

    # Truth lives on the *centre* column of each block.  The tangent columns are
    # extra regressors with zero true coefficient, so fitting them can only add
    # estimation variance -- which is exactly the cost the experiment measures
    # in eq. (3).  Writing the truth this way keeps the comparison honest: the
    # fractional-DD basis is a modelling choice, not free information.
    n_basis = 1 + 2 * int(cfg.cancellation.interference_tangent_order)
    h_true = np.zeros(X.shape[1], dtype=complex)
    if X.shape[1]:
        centre = np.arange(0, X.shape[1], n_basis)
        h_true[centre] = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=centre.size))
    alpha_true = np.zeros(A_true.shape[1], dtype=complex)
    if alpha_true.size:
        alpha_true = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=alpha_true.size))
    # One target id per *column*, repeated over each source's basis block, so a
    # caller can select a target's block or drop its echo without knowing the
    # layout.
    n_tgt_basis = 1 + 2 * int(cfg.cancellation.tangent_order)
    ids = np.repeat(np.asarray(true_ids, dtype=int), n_tgt_basis)

    x_direct = X @ h_true
    if include_echo:
        s_target = A_true @ alpha_true
    else:
        s_target = np.zeros(n_bins, dtype=complex)
    if exclude_target is not None:
        drop = ids == int(exclude_target)
        s_target = s_target - A_true[:, drop] @ alpha_true[drop]
        alpha_true = alpha_true.copy()
        alpha_true[drop] = 0.0
    sigma2 = _noise_power(cfg)
    noise = (rng.normal(size=n_bins) + 1j * rng.normal(size=n_bins)) * math.sqrt(sigma2 / 2.0)

    basis_belief = _protection_basis(cfg, targets_belief, n_bins)
    basis_truth = _protection_basis(cfg, targets_true, n_bins)

    return Observation(
        y=x_direct + s_target + noise,
        X=X,
        A=A,
        x_direct=x_direct,
        s_target=s_target,
        h_true=h_true,
        alpha_true=alpha_true,
        sigma2=sigma2,
        direct=direct,
        targets=targets_true,
        targets_belief=targets_belief,
        basis_belief=basis_belief,
        basis_truth=basis_truth,
        weak_index=int(weak_index),
        A_target_ids=ids.copy(),
        A_true_target_ids=ids,
    )


def build_observation_pair(
    cfg: Config,
    geom_true,
    geom_belief,
    base,
    receiver: int,
    *,
    rng: np.random.Generator,
    sense_power: np.ndarray,
    radiated_power: np.ndarray,
    processing_gain: float,
    hw_gain: float,
    exclude_target: int,
    active_mask: np.ndarray | None = None,
    weak_index: int | None = None,
    share_noise: bool = False,
) -> Tuple[Observation, Observation]:
    """``(H1, H0)`` -- one geometry, one direct field, two hypotheses.

    Calling :func:`build_observation` twice does **not** produce a matched pair.
    Each call consumes ``rng`` to draw the direct phases ``h_true``, the echo
    phases ``alpha_true`` and the noise, so the second call gets a *different*
    direct component: measured on trial 0 of the 600 m scenario,
    ``||x_direct||^2`` moved from ``5.59e-10`` to ``8.75e-10``.  The two
    statistics then differ because the illumination changed, not because the
    tested echo was removed, and every detection metric built on them collapses
    to ``P_D ~ P_FA`` -- which is exactly what was observed (statistic ratio
    H1/H0 sat at 1.00--1.15 while the physics promised ``+17 dB``).

    H0 is therefore *derived* from H1 by zeroing the tested target's columns, so
    both observations share ``x_direct``, ``X``, ``A`` and every other target's
    echo by construction.  Only the noise is redrawn (independently by default),
    which is what a CFAR threshold calibrated on the H0 pool needs;
    ``share_noise=True`` reuses H1's noise draw when a paired difference is
    wanted instead of a threshold.

    Other targets' echoes deliberately remain in H0: in this scenario the
    tested target is one of ``Q`` scatterers and the rest are interference, so
    removing them would flatter the detector.
    """
    obs1 = build_observation(
        cfg, geom_true, geom_belief, base, receiver,
        rng=rng, sense_power=sense_power, radiated_power=radiated_power,
        processing_gain=processing_gain, hw_gain=hw_gain,
        active_mask=active_mask, include_echo=True,
        weak_index=int(exclude_target) if weak_index is None else int(weak_index),
    )
    # ``Observation`` keeps the true dictionary implicit in ``targets`` so the
    # type stays light; rebuilding it is deterministic and cheap.  It is the
    # only way to form the tested echo, because the belief dictionary would use
    # the wrong offsets.
    A_true = target_dictionary(cfg, obs1.targets)
    ids = np.asarray(obs1.A_true_target_ids)
    drop = ids == int(exclude_target)
    if not drop.any():
        raise ValueError(
            "exclude_target %d has no column in the true dictionary -- H0 "
            "would silently equal H1 and every detection metric would be "
            "meaningless" % int(exclude_target)
        )
    s_h0 = obs1.s_target - A_true[:, drop] @ obs1.alpha_true[drop]
    alpha_h0 = obs1.alpha_true.copy()
    alpha_h0[drop] = 0.0

    if share_noise:
        noise = obs1.y - obs1.x_direct - obs1.s_target
    else:
        n_bins = int(obs1.y.size)
        noise = (rng.normal(size=n_bins) + 1j * rng.normal(size=n_bins)) * math.sqrt(
            obs1.sigma2 / 2.0
        )
    obs0 = replace(
        obs1,
        y=obs1.x_direct + s_h0 + noise,
        s_target=s_h0,
        alpha_true=alpha_h0,
    )
    return obs1, obs0


def _protection_basis(cfg: Config, sources: Sequence[TargetSource], n_bins: int) -> np.ndarray:
    """``U_j = orth([J_1, ..., J_Q])`` -- module M2.

    Protection is budgeted per *target*, not per echo path: the sources are
    grouped by target index, each target's total echo power at this receiver is
    formed, and the ``max_protected_targets`` weakest are protected through all
    of their illuminating paths.  ``max_protected_targets = 0`` protects
    everything and is kept only because the first experiment needs that arm to
    show what it costs.
    """
    c = cfg.cancellation
    if not c.protect_targets or not sources:
        return np.zeros((n_bins, 0), dtype=complex)

    budget = int(c.max_protected_targets)
    if budget > 0:
        by_target: Dict[int, List[TargetSource]] = {}
        for src in sources:
            by_target.setdefault(int(src.target), []).append(src)
        ranked = sorted(by_target, key=lambda q: sum(s.power for s in by_target[q]))
        wanted = set(ranked[:budget])
        sources = [s for s in sources if int(s.target) in wanted]

    blocks = [
        tangent_columns(
            cfg, s.doppler_bin, s.delay_bin,
            int(c.tangent_order), float(c.tangent_step_bins),
        )
        for s in sources
    ]
    return orthonormalise(np.concatenate(blocks, axis=1)).U


def _noise_power(cfg: Config) -> float:
    from .model import noise_power

    return float(noise_power(cfg))


# ==========================================================================
# 8. The analytic bridge back into the link-table model
# ==========================================================================
def predict_cancellation(
    cfg: Config,
    i_sense_field: np.ndarray,
    n_illuminators: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predicted residual direct field and protection retention, per receiver.

    This is the ``C_TP-UIC`` interface a scheduler would call.  Two terms:

    *retention* -- the direct energy that lies inside the target subspace is
    *deliberately* kept, because subtracting it would subtract target evidence
    too.  It is bounded by the protection rank over the observation dimension,
    ``rho_prot = rank / K``; the experiment measures the realised value and this
    function returns the bound, which is the part a receiver can promise.

    *estimation* -- with ``d`` complex coefficients fitted per CPI the residual
    is ``d * sigma_eff^2`` with ``sigma_eff^2 = n0 / n_cpi``, so the residual
    fraction of the input field is ``d * n0 / (n_cpi * I_sense)``.

    Both are returned as *fractions of the input direct field*, so the caller
    can write ``I_res = fraction * I_sense_field`` and keep the existing
    denominator structure of ``model.compute_link_tables`` untouched.
    """
    c = cfg.cancellation
    d_basis = 1 + 2 * int(c.interference_tangent_order)
    d = np.maximum(np.asarray(n_illuminators, dtype=float), 1.0) * d_basis
    n0 = _noise_power(cfg)
    n_cpi = max(int(c.n_cpi), 1)
    i_sense = np.maximum(np.asarray(i_sense_field, dtype=float), EPS)
    estimate = (d * n0 / n_cpi) / i_sense
    retained = np.full_like(estimate, _protection_leakage_fraction(cfg))
    return estimate + retained, retained


def _protection_leakage_fraction(cfg: Config) -> float:
    """Dimension-ratio *estimate* of the protected fraction of the observation.

    ``n_targets * (M - 1) * (1 + 2*order) / K``.  This is the only figure a
    receiver can write down before seeing the scene, and it is **not** a safe
    bound: it assumes the protection subspace and the interference subspace are
    in general position.  They are not.  Measured on the 600 m scenario the
    ratio reads 3.1% while the realised retention is 17% of the input direct
    field -- an optimistic error of about 5x -- because the direct kernels and
    the target kernels occupy the *same* compact DD region (the whole 600 m
    scene spans roughly 4 delay bins by 10 Doppler bins, so 4096 bins are mostly
    empty and the occupied ones are shared).

    Use it for planning only.  A scheduler must consume the *measured* ``I_res``
    (or a prediction fitted to it), never this number.
    """
    c = cfg.cancellation
    if not c.protect_targets:
        return 0.0
    budget = int(c.max_protected_targets)
    n_targets = cfg.scale.Q if budget <= 0 else min(budget, cfg.scale.Q)
    n_bins = float(cfg.waveform.N * cfg.waveform.L)
    worst = float(n_targets * (cfg.scale.M - 1) * (1 + 2 * int(c.tangent_order)))
    return min(1.0, worst / n_bins)


def kappa_from_budget(
    cfg: Config, i_sense_field: np.ndarray, n_illuminators: np.ndarray
) -> np.ndarray:
    """Effective cancellation depth implied by the receiver design, in dB."""
    fraction, _ = predict_cancellation(cfg, i_sense_field, n_illuminators)
    depth = -10.0 * np.log10(np.clip(fraction, EPS, None))
    return np.minimum(depth, float(cfg.cancellation.hw_ceiling_db))
