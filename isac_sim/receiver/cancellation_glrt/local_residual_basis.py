"""Receiver-known local bases for residual-direction diagnostics."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from isac_sim.receiver.cancellation import EPS, Observation
from isac_sim.receiver.cancellation_glrt.covariance import LowRankCovariance


def receiver_known_local_bases(
    obs: Observation,
    target: int,
) -> Mapping[str, np.ndarray]:
    """Build nested bases without using simulator-truth fields.

    ``X`` contains receiver-estimated direct templates/tangents and ``A`` the
    belief-side target templates/tangents.  The returned matrices therefore do
    not use ``x_direct``, ``s_target``, truth geometry, or true coefficients.
    """
    if obs.A_target_ids is None:
        raise ValueError("A_target_ids is required for a target-local basis")
    ids = np.asarray(obs.A_target_ids)
    if ids.shape != (obs.A.shape[1],):
        raise ValueError("A_target_ids must label every column of A")
    tested = obs.A[:, ids == int(target)]
    if tested.shape[1] == 0:
        raise ValueError(f"target {target} has no belief-side dictionary columns")
    return {
        "direct": np.asarray(obs.X),
        "direct_tested_target": np.concatenate((obs.X, tested), axis=1),
        "direct_all_targets": np.concatenate((obs.X, obs.A), axis=1),
    }


def projection_coverage(direction: np.ndarray, basis: np.ndarray) -> tuple[float, int]:
    """Return energy fraction of ``direction`` captured by ``span(basis)``."""
    vector = np.asarray(direction).reshape(-1)
    matrix = np.asarray(basis)
    if matrix.ndim != 2 or matrix.shape[0] != vector.size:
        raise ValueError("basis and direction dimensions do not agree")
    energy = float(np.vdot(vector, vector).real)
    if energy <= EPS or matrix.shape[1] == 0:
        return 0.0, 0
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0:
        return 0.0, 0
    tol = np.finfo(float).eps * max(matrix.shape) * float(singular[0])
    rank = int(np.sum(singular > tol))
    if rank == 0:
        return 0.0, 0
    captured = float(np.linalg.norm(u[:, :rank].conj().T @ vector) ** 2)
    return float(np.clip(captured / energy, 0.0, 1.0)), rank


def whitened_projection_coverage(
    covariance: LowRankCovariance,
    direction: np.ndarray,
    basis: np.ndarray,
) -> tuple[float, int]:
    """Measure local-basis coverage in the detector's whitened geometry."""
    return projection_coverage(
        covariance.whiten(np.asarray(direction)),
        covariance.whiten_matrix(np.asarray(basis)),
    )
