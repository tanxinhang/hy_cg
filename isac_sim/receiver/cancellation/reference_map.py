"""Reference-only protected MAP estimation of shared direct coefficients."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isac_sim.receiver.cancellation.containers import Observation


@dataclass(frozen=True)
class ReferenceMAPEstimate:
    """Coefficient posterior fitted without consuming the sensing CPI."""

    h_hat: np.ndarray
    covariance: np.ndarray
    n_reference: int


def _projected_dictionary(obs: Observation) -> np.ndarray:
    basis = obs.basis_belief
    if basis is None or not basis.shape[1]:
        return np.asarray(obs.X, dtype=complex)
    return obs.X - basis @ (basis.conj().T @ obs.X)


def fit_reference_map(
    observations: list[Observation] | tuple[Observation, ...],
    prior_variance: float | None,
) -> ReferenceMAPEstimate:
    """Fit one coefficient vector from independent protected references."""
    if not observations:
        raise ValueError("at least one reference observation is required")
    width = int(observations[0].X.shape[1])
    precision = np.zeros((width, width), dtype=complex)
    rhs = np.zeros(width, dtype=complex)
    for obs in observations:
        if obs.X.shape[1] != width:
            raise ValueError("direct dictionary width differs across references")
        sigma2 = float(obs.sigma2)
        if sigma2 <= 0.0:
            raise ValueError("noise variance must be positive")
        dictionary = _projected_dictionary(obs)
        precision += dictionary.conj().T @ dictionary / sigma2
        rhs += dictionary.conj().T @ obs.y / sigma2
    if prior_variance is not None and prior_variance > 0.0:
        precision += np.eye(width) / float(prior_variance)
    covariance = np.linalg.pinv(precision, rcond=1e-12)
    h_hat = covariance @ rhs
    return ReferenceMAPEstimate(h_hat, covariance, len(observations))


def apply_reference_map(obs: Observation, estimate: ReferenceMAPEstimate):
    """Return sensing residual and its coefficient-error covariance factor."""
    dictionary = _projected_dictionary(obs)
    residual = np.asarray(obs.y) - dictionary @ estimate.h_hat
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (estimate.covariance + estimate.covariance.conj().T))
    keep = eigenvalues > np.finfo(float).eps * max(
        float(np.max(eigenvalues, initial=0.0)), 1.0)
    factor = dictionary @ (
        eigenvectors[:, keep] * np.sqrt(np.maximum(eigenvalues[keep], 0.0))[None, :]
    )
    return residual, factor


__all__ = ["ReferenceMAPEstimate", "apply_reference_map", "fit_reference_map"]
