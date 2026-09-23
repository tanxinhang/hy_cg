"""Train/H0-only selection for a scalar residual-covariance correction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from isac_sim.detection.evaluation_split import SplitDataset


@dataclass(frozen=True)
class CovarianceScaleSelection:
    """Frozen scale choice and train-only calibration diagnostics."""

    scale: float
    median_inflation: float
    objective: float


def select_covariance_scale(
    train: SplitDataset,
    candidates: Sequence[float],
    inflation_key: str = "h0_inflation_by_scale",
) -> CovarianceScaleSelection:
    """Select the scale whose median H0 inflation is closest to one.

    Ties prefer the physical, unlearned scale one and then the closest scale to
    one.  Only a dataset explicitly carrying the ``train`` role is accepted.
    """
    if train.role != "train":
        raise ValueError("covariance scale must be fitted on the train split")
    if not train.rows:
        raise ValueError("covariance scale fitting requires train/H0 scenes")
    scales = tuple(float(value) for value in candidates)
    if not scales or any(not np.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("candidate covariance scales must be finite and positive")

    evaluated = []
    for scale in scales:
        values = []
        for row in train.rows:
            mapping: Mapping = row[inflation_key]
            if scale in mapping:
                values.append(float(mapping[scale]))
            elif str(scale) in mapping:
                values.append(float(mapping[str(scale)]))
            else:
                raise ValueError(f"train row lacks covariance scale {scale}")
        median = float(np.median(values))
        objective = abs(float(np.log(max(median, np.finfo(float).tiny))))
        evaluated.append((objective, abs(scale - 1.0), scale, median))
    objective, _, scale, median = min(evaluated)
    return CovarianceScaleSelection(scale, median, objective)


__all__ = ["CovarianceScaleSelection", "select_covariance_scale"]
