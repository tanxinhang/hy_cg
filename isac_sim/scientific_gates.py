"""Theorem-backed validation primitives for the V1.6 mechanism release.

This module intentionally separates identities that can be proved from
performance statements that require a fixed-PFA detector experiment.  The
functions are small enough to serve both as paper equations and executable
scientific release gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .llr import llr_kld


@dataclass(frozen=True)
class LocalLlrAggregation:
    """Sample-wise centralized and partitioned exact-LLR sums."""

    centralized: np.ndarray
    local_sums: tuple[np.ndarray, ...]
    aggregated: np.ndarray

    @property
    def max_abs_error(self) -> float:
        return float(np.max(np.abs(self.centralized - self.aggregated), initial=0.0))


@dataclass(frozen=True)
class CommonLatentLlr:
    """Correct joint and invalid naive-marginal LLRs for a shared latent."""

    joint: np.ndarray
    naive_marginal_sum: np.ndarray

    @property
    def max_abs_gap(self) -> float:
        return float(np.max(np.abs(self.joint - self.naive_marginal_sum), initial=0.0))


@dataclass(frozen=True)
class SymmetricComplementarityResult:
    """Exact two-view robust choices under the Swerling-II KL model."""

    single_robust_information: float
    equal_split_robust_information: float
    optimal_robust_information: float
    choose_cooperation: bool
    low_snr_ratio: float
    low_snr_cooperation_condition: bool


@dataclass(frozen=True)
class CoherentPowerOracle:
    """Fixed-total-power coherent eigenvalue oracle."""

    optimal_power: float
    best_single_power: float
    gain: float
    principal_weights: tuple[complex, ...]


@dataclass(frozen=True)
class ReliabilityProtectionPlan:
    """Exact continuous-knapsack protection plan for aggregate LLR packets."""

    final_success: tuple[float, ...]
    budget_used: float
    expected_received_kl: float


@dataclass(frozen=True)
class PhysicalHeadroomDashboard:
    """Metric-consistent decomposition of physical and algorithmic headroom.

    Values must all use the same metric.  A KL dashboard is an information
    decomposition; a PD dashboard is valid only when every oracle was directly
    optimized/evaluated at the same declared false-alarm operating point.
    """

    metric: str
    best_single: float
    noncoherent_full_oracle: float
    coherent_oracle: float
    restricted_oracle: float
    proposed: float

    def __post_init__(self) -> None:
        if self.metric not in {"kl", "pd"}:
            raise ValueError("metric must be 'kl' or 'pd'")
        values = (
            self.best_single, self.noncoherent_full_oracle,
            self.coherent_oracle, self.restricted_oracle, self.proposed,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("headroom values must be finite")

    @property
    def noncoherent_physical_headroom(self) -> float:
        return self.noncoherent_full_oracle - self.best_single

    @property
    def synchronization_headroom(self) -> float:
        return self.coherent_oracle - self.noncoherent_full_oracle

    @property
    def candidate_loss(self) -> float:
        return self.noncoherent_full_oracle - self.restricted_oracle

    @property
    def algorithm_loss(self) -> float:
        return self.restricted_oracle - self.proposed

    @property
    def total_noncoherent_gap(self) -> float:
        return self.noncoherent_full_oracle - self.proposed

    @property
    def decomposition_error(self) -> float:
        return self.total_noncoherent_gap - (
            self.candidate_loss + self.algorithm_loss
        )

    @property
    def recovery_fraction(self) -> float:
        denominator = self.noncoherent_physical_headroom
        if denominator <= 0.0:
            return float("nan")
        return (self.proposed - self.best_single) / denominator


def aggregate_partitioned_llrs(
    per_observation_llr: np.ndarray,
    partitions: Sequence[Sequence[int]],
) -> LocalLlrAggregation:
    """Aggregate exact LLRs after validating a disjoint, exhaustive partition."""
    values = np.asarray(per_observation_llr, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1 or not np.all(np.isfinite(values)):
        raise ValueError("per_observation_llr must be finite with shape (sample, A)")
    flattened = [int(index) for group in partitions for index in group]
    if sorted(flattened) != list(range(values.shape[1])):
        raise ValueError("partitions must cover every observation exactly once")
    local = tuple(np.sum(values[:, tuple(group)], axis=1) for group in partitions)
    centralized = np.sum(values, axis=1)
    aggregated = np.sum(np.stack(local, axis=0), axis=0)
    return LocalLlrAggregation(centralized, local, aggregated)


def common_latent_gaussian_llrs(
    observations: np.ndarray, latent_variance: float
) -> CommonLatentLlr:
    """Return the shared-latent joint LLR and the incorrect marginal sum.

    H0 has independent unit-variance Gaussian observations.  Under H1 all
    observations share theta~N(0,tau^2) plus independent unit noise.
    """
    values = np.asarray(observations, dtype=float)
    tau2 = float(latent_variance)
    if values.ndim != 2 or values.shape[1] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("observations must be finite with shape (sample, m>=2)")
    if not np.isfinite(tau2) or tau2 <= 0.0:
        raise ValueError("latent_variance must be finite and positive")
    count = values.shape[1]
    joint = (
        -0.5 * np.log1p(count * tau2)
        + 0.5 * tau2 / (1.0 + count * tau2)
        * np.sum(values, axis=1) ** 2
    )
    naive = (
        -0.5 * count * np.log1p(tau2)
        + 0.5 * tau2 / (1.0 + tau2)
        * np.sum(values * values, axis=1)
    )
    return CommonLatentLlr(joint, naive)


def erasure_received_kl(information: np.ndarray | float, chi: float):
    """Exact received KL for an observed, hypothesis-independent erasure."""
    success = float(chi)
    if not np.isfinite(success) or not 0.0 <= success <= 1.0:
        raise ValueError("chi must lie in [0, 1]")
    values = np.asarray(information, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("information must be finite and non-negative")
    result = success * values
    return float(result) if np.ndim(information) == 0 else result


def hypothesis_dependent_erasure_kl(
    local_information: float, chi_h1: float, chi_h0: float
) -> float:
    """Received KL when packet arrival itself has hypothesis information."""
    information = float(local_information)
    p = float(chi_h1)
    q = float(chi_h0)
    if not np.isfinite(information) or information < 0.0:
        raise ValueError("local_information must be finite and non-negative")
    if not (0.0 < p < 1.0 and 0.0 < q < 1.0):
        raise ValueError("chi_h1 and chi_h0 must lie strictly inside (0,1)")
    bernoulli = p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))
    return float(bernoulli + p * information)


def aggregation_reliability_threshold(
    local_information: Sequence[float], separate_success: Sequence[float]
) -> float:
    """Minimum aggregate-packet success probability preserving expected KL."""
    information = np.asarray(local_information, dtype=float)
    success = np.asarray(separate_success, dtype=float)
    if information.ndim != 1 or information.size == 0 or success.shape != information.shape:
        raise ValueError("paired non-empty information and success vectors are required")
    if np.any(~np.isfinite(information)) or np.any(information < 0.0):
        raise ValueError("local information must be finite and non-negative")
    if np.any(~np.isfinite(success)) or np.any((success < 0.0) | (success > 1.0)):
        raise ValueError("success probabilities must lie in [0, 1]")
    total = float(np.sum(information))
    if total <= 0.0:
        raise ValueError("at least one observation must carry positive information")
    return float(np.dot(information, success) / total)


def evidence_delivery_variance(
    local_information: Sequence[float], chi: float, *, aggregated: bool
) -> float:
    """Variance of delivered KL mass for shared or independent erasures."""
    information = np.asarray(local_information, dtype=float)
    success = float(chi)
    if information.ndim != 1 or information.size == 0:
        raise ValueError("local_information must be a non-empty vector")
    if np.any(~np.isfinite(information)) or np.any(information < 0.0):
        raise ValueError("local information must be finite and non-negative")
    if not np.isfinite(success) or not 0.0 <= success <= 1.0:
        raise ValueError("chi must lie in [0, 1]")
    mass = float(np.sum(information) ** 2 if aggregated else np.sum(information ** 2))
    return success * (1.0 - success) * mass


def allocate_reliability_protection(
    local_information: Sequence[float],
    base_success: Sequence[float],
    marginal_cost: Sequence[float],
    budget: float,
    *,
    minimum_success: Sequence[float] | float = 0.0,
) -> ReliabilityProtectionPlan:
    """Maximize expected received KL under a linear reliability budget.

    Mandatory reliability floors are funded first.  Remaining continuous
    protection is assigned by descending information-per-cost ratio, which is
    the exact solution of this bounded fractional-knapsack problem.
    """
    information = np.asarray(local_information, dtype=float)
    base = np.asarray(base_success, dtype=float)
    cost = np.asarray(marginal_cost, dtype=float)
    total_budget = float(budget)
    floor = np.broadcast_to(np.asarray(minimum_success, dtype=float), base.shape)
    if information.ndim != 1 or information.size == 0:
        raise ValueError("local_information must be a non-empty vector")
    if base.shape != information.shape or cost.shape != information.shape:
        raise ValueError("information, success, and cost vectors must match")
    if np.any(~np.isfinite(information)) or np.any(information < 0.0):
        raise ValueError("information must be finite and non-negative")
    if np.any(~np.isfinite(base)) or np.any((base < 0.0) | (base > 1.0)):
        raise ValueError("base success probabilities must lie in [0,1]")
    if np.any(~np.isfinite(floor)) or np.any((floor < 0.0) | (floor > 1.0)):
        raise ValueError("minimum success probabilities must lie in [0,1]")
    if np.any(~np.isfinite(cost)) or np.any(cost <= 0.0):
        raise ValueError("marginal costs must be finite and positive")
    if not np.isfinite(total_budget) or total_budget < 0.0:
        raise ValueError("budget must be finite and non-negative")
    final = np.maximum(base, floor).copy()
    used = float(np.dot(final - base, cost))
    if used > total_budget + 1e-12:
        raise ValueError("budget cannot satisfy the declared reliability floors")
    remaining = total_budget - used
    order = np.argsort(-(information / cost), kind="stable")
    for index in order:
        capacity = float(1.0 - final[index])
        increase = min(capacity, remaining / float(cost[index]))
        final[index] += increase
        used += increase * float(cost[index])
        remaining -= increase * float(cost[index])
        if remaining <= 1e-12:
            break
    return ReliabilityProtectionPlan(
        final_success=tuple(float(value) for value in final),
        budget_used=float(used),
        expected_received_kl=float(np.dot(information, final)),
    )


def swerling_information(
    path_coefficient: np.ndarray | float,
    power: np.ndarray | float,
    looks: int,
):
    """Forward KL under the declared zero-mean variance-change model."""
    alpha = np.asarray(path_coefficient, dtype=float)
    allocated = np.asarray(power, dtype=float)
    if looks < 1:
        raise ValueError("looks must be positive")
    if np.any(~np.isfinite(alpha)) or np.any(alpha < 0.0):
        raise ValueError("path coefficients must be finite and non-negative")
    if np.any(~np.isfinite(allocated)) or np.any(allocated < 0.0):
        raise ValueError("power must be finite and non-negative")
    result = llr_kld(alpha * allocated, looks)
    return result


def symmetric_complementarity_oracle(
    strong_coefficient: float,
    weak_coefficient: float,
    total_power: float,
    looks: int,
) -> SymmetricComplementarityResult:
    """Exact single-vs-equal-split solution for two symmetric aspect views."""
    a = float(strong_coefficient)
    b = float(weak_coefficient)
    power = float(total_power)
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(power)):
        raise ValueError("coefficients and power must be finite")
    if a <= b or b < 0.0 or power <= 0.0 or looks < 1:
        raise ValueError("require a>b>=0, positive total_power, and positive looks")
    single = float(swerling_information(b, power, looks))
    split = float(
        swerling_information(a, power / 2.0, looks)
        + swerling_information(b, power / 2.0, looks)
    )
    ratio = float("inf") if b == 0.0 else a / b
    return SymmetricComplementarityResult(
        single_robust_information=single,
        equal_split_robust_information=split,
        optimal_robust_information=max(single, split),
        choose_cooperation=bool(split > single),
        low_snr_ratio=ratio,
        low_snr_cooperation_condition=bool(ratio > np.sqrt(3.0)),
    )


def gaussian_phase_coherence(size: int, phase_error_std_rad: float) -> np.ndarray:
    """Coherence matrix for independent zero-mean Gaussian phase errors."""
    sigma = float(phase_error_std_rad)
    if size < 1 or not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("size must be positive and phase uncertainty non-negative")
    coherence = np.full((size, size), np.exp(-(sigma ** 2)), dtype=complex)
    np.fill_diagonal(coherence, 1.0)
    return coherence


def coherent_power_oracle(
    channel: Sequence[complex],
    total_power: float,
    phase_coherence: np.ndarray,
) -> CoherentPowerOracle:
    """Optimize expected coherent power through the principal eigenvector."""
    h = np.asarray(channel, dtype=complex)
    rho = np.asarray(phase_coherence, dtype=complex)
    power = float(total_power)
    if h.ndim != 1 or h.size == 0 or np.any(~np.isfinite(h)):
        raise ValueError("channel must be a finite non-empty vector")
    if rho.shape != (h.size, h.size) or np.any(~np.isfinite(rho)):
        raise ValueError("phase_coherence must be a finite square channel matrix")
    if power <= 0.0 or not np.isfinite(power):
        raise ValueError("total_power must be finite and positive")
    if not np.allclose(rho, rho.conj().T, atol=1e-12):
        raise ValueError("phase_coherence must be Hermitian")
    if not np.allclose(np.real(np.diag(rho)), 1.0, atol=1e-12):
        raise ValueError("phase_coherence must have a unit diagonal")
    if float(np.min(np.linalg.eigvalsh(rho))) < -1e-10:
        raise ValueError("phase_coherence must be positive semidefinite")
    covariance = np.outer(h, h.conj()) * rho
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues[0]) < -1e-10:
        raise ValueError("phase_coherence must induce a positive semidefinite matrix")
    principal = eigenvectors[:, -1]
    optimal = power * max(float(eigenvalues[-1]), 0.0)
    single = power * float(np.max(np.abs(h) ** 2))
    gain = optimal / single if single > 0.0 else 1.0
    weights = np.sqrt(power) * principal
    return CoherentPowerOracle(
        optimal_power=optimal,
        best_single_power=single,
        gain=gain,
        principal_weights=tuple(complex(value) for value in weights),
    )


def lower_tail_detection_summary(
    probabilities: Sequence[float], quantile: float = 0.05
) -> dict[str, float | int]:
    """Report distributional detection endpoints without promoting a sample minimum."""
    values = np.asarray(probabilities, dtype=float)
    level = float(quantile)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("probabilities must be a finite non-empty vector")
    if np.any((values < 0.0) | (values > 1.0)) or not 0.0 < level <= 0.5:
        raise ValueError("probabilities lie in [0,1] and quantile in (0,0.5]")
    cutoff = float(np.quantile(values, level, method="linear"))
    tail = values[values <= cutoff + 1e-15]
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "quantile_level": level,
        "lower_quantile": cutoff,
        "lower_cvar": float(np.mean(tail)),
        "mean": float(np.mean(values)),
    }
