"""Fixed-window evidence accumulation, distinct from TP-UIC reference CPIs."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FixedWindowEvidence:
    cumulative: np.ndarray
    increments: np.ndarray
    converged: bool
    convergence_frame: int | None


def accumulate_llr(
    per_frame_llr,
    *,
    relative_increment_tolerance: float = 0.01,
    stable_frames: int = 3,
) -> FixedWindowEvidence:
    """Accumulate aligned per-frame LLR and report evidence-rate convergence."""
    values = np.asarray(per_frame_llr, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("per_frame_llr must be a non-empty finite vector")
    cumulative = np.cumsum(values)
    running_rate = cumulative / np.arange(1, values.size + 1)
    relative = np.full(values.size, np.inf)
    if values.size > 1:
        relative[1:] = np.abs(np.diff(running_rate)) / np.maximum(
            np.abs(running_rate[:-1]), 1e-12
        )
    frame = None
    width = max(int(stable_frames), 1)
    for end in range(width, values.size + 1):
        if np.all(relative[end - width:end] <= float(relative_increment_tolerance)):
            frame = end
            break
    return FixedWindowEvidence(
        cumulative=cumulative,
        increments=values,
        converged=frame is not None,
        convergence_frame=frame,
    )
