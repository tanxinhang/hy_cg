"""cancellation 积木：残差功率先验分位数。"""

from __future__ import annotations

import math
import numpy as np

from isac_sim.receiver.cancellation.result import CancellationResult

def residual_power_prior_quantile(
    result: CancellationResult,
    probability: float,
    *,
    samples: int = 8192,
) -> float:
    """Deterministic prior-predictive quantile of TP-UIC residual power.

    Direct-path coefficients use independent unit phases and the fitted-noise
    quadratic form is a weighted sum of unit exponentials.  A fixed random
    stream makes this numerical quadrature reproducible and independent of the
    receiver realization used for evaluation.
    """
    gram = result.residual_direct_gram
    eigenvalues = result.residual_noise_eigenvalues
    if gram is None or eigenvalues is None:
        raise ValueError("residual prior descriptors are unavailable for this arm")
    return _residual_quantile_from_descriptors(
        gram, eigenvalues, probability, samples=samples
    )


def _residual_quantile_from_descriptors(
    gram: np.ndarray,
    eigenvalues: np.ndarray,
    probability: float,
    *,
    samples: int = 8192,
) -> float:
    """Shared deterministic quadrature used by reporting and arm selection."""
    p = float(probability)
    if not np.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    n = int(samples)
    if n < 256:
        raise ValueError("samples must be at least 256")
    gram = np.asarray(gram, dtype=complex)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    rng = np.random.default_rng(0x54505549)
    if gram.shape[0]:
        phases = np.exp(
            2j * math.pi * rng.random((n, int(gram.shape[0])))
        )
        structural = np.real(np.einsum(
            "bi,ij,bj->b", phases.conj(), gram, phases, optimize=True
        ))
    else:
        structural = np.zeros(n, dtype=float)
    if eigenvalues.size:
        fitted_noise = rng.exponential(
            scale=1.0, size=(n, int(eigenvalues.size))
        ) @ eigenvalues
    else:
        fitted_noise = np.zeros(n, dtype=float)
    values = np.maximum(structural + fitted_noise, 0.0)
    return float(np.quantile(values, p, method="higher"))
