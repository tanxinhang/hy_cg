"""Bottleneck-aware sensing/reporting power refinement.

The evaluator is deliberately a black box: each call must replay receiver
cancellation, FBL feasibility, scheduling and fusion.  The optimiser therefore
cannot accidentally optimise a geometry or link-rate proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PowerAOResult:
    sense: np.ndarray
    comm: np.ndarray
    delivered_pd: np.ndarray
    history: tuple[dict, ...]


def smooth_weakest_pd(pd: np.ndarray, tau: float) -> float:
    """Stable soft minimum used only for finite-difference search directions."""
    values = np.asarray(pd, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("delivered PD must be a non-empty finite vector")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    z = -values / float(tau)
    zmax = float(np.max(z))
    return -float(tau) * (zmax + np.log(np.exp(z - zmax).sum()))


def _project_capped_simplex(x: np.ndarray, budget: float, peak: float) -> np.ndarray:
    """Euclidean projection onto ``0<=x_i<=peak, sum(x)<=budget``."""
    out = np.clip(np.asarray(x, dtype=float), 0.0, float(peak))
    if float(np.sum(out)) <= float(budget):
        return out
    lo = float(np.min(x - peak))
    hi = float(np.max(x))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        candidate = np.clip(x - mid, 0.0, float(peak))
        if float(np.sum(candidate)) > float(budget):
            lo = mid
        else:
            hi = mid
    return np.clip(x - hi, 0.0, float(peak))


def marginal_delivered_pd_power_ao(
    sense: np.ndarray,
    comm: np.ndarray,
    evaluate: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    fleet_budget_w: float,
    peak_power_w: float,
    tau: float = 0.02,
    finite_difference_w: float = 0.02,
    initial_trust_w: float = 0.15,
    min_trust_w: float = 1e-3,
    max_rounds: int = 12,
) -> PowerAOResult:
    """Finite-difference projected ascent with exact weakest-PD acceptance."""
    sense = np.asarray(sense, dtype=float).copy()
    comm = np.asarray(comm, dtype=float).copy()
    if sense.shape != comm.shape or sense.ndim != 1:
        raise ValueError("sense and comm must be same-length vectors")
    if np.any(sense < 0.0) or np.any(comm < 0.0):
        raise ValueError("powers must be non-negative")
    if np.any(sense + comm > peak_power_w + 1e-12):
        raise ValueError("initial per-UAV power exceeds peak")
    if float(np.sum(sense + comm)) > fleet_budget_w + 1e-12:
        raise ValueError("initial power exceeds fleet budget")

    x = np.concatenate([sense, comm])
    m = sense.size

    def project(values: np.ndarray) -> np.ndarray:
        # First impose each UAV's joint peak, then the fleet cap. Alternating
        # projections converges rapidly for these two convex sets.
        out = np.maximum(np.asarray(values, dtype=float), 0.0)
        for _ in range(16):
            for j in range(m):
                pair = np.array([out[j], out[m + j]])
                out[[j, m + j]] = _project_capped_simplex(
                    pair, peak_power_w, peak_power_w
                )
            out = _project_capped_simplex(out, fleet_budget_w, peak_power_w)
        return out

    def replay(values: np.ndarray) -> np.ndarray:
        return np.asarray(evaluate(values[:m].copy(), values[m:].copy()), dtype=float)

    pd = replay(x)
    trust = float(initial_trust_w)
    history: list[dict] = []
    for iteration in range(int(max_rounds)):
        base_smooth = smooth_weakest_pd(pd, tau)
        gradient = np.zeros_like(x)
        for k in range(x.size):
            probe = x.copy()
            probe[k] += float(finite_difference_w)
            probe = project(probe)
            distance = float(np.linalg.norm(probe - x))
            if distance > 1e-12:
                gradient[k] = (
                    smooth_weakest_pd(replay(probe), tau) - base_smooth
                ) / distance
        norm = float(np.linalg.norm(gradient))
        if norm <= 1e-14:
            break
        candidate = project(x + trust * gradient / norm)
        candidate_pd = replay(candidate)
        old_worst = float(np.min(pd))
        new_worst = float(np.min(candidate_pd))
        accepted = new_worst > old_worst + 1e-12
        history.append({
            "iteration": iteration,
            "trust_w": trust,
            "old_worst_pd": old_worst,
            "new_worst_pd": new_worst,
            "accepted": accepted,
        })
        if accepted:
            x, pd = candidate, candidate_pd
            trust = min(float(initial_trust_w), trust * 1.25)
        else:
            trust *= 0.5
            if trust < float(min_trust_w):
                break
    # Ensure the caller's mutable model is left at the accepted point.
    pd = replay(x)
    return PowerAOResult(x[:m], x[m:], pd, tuple(history))
