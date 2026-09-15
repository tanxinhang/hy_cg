"""Conservative operating-threshold diagnostics for low-RCS detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RcsOperatingPoint:
    """One evaluated RCS point under a fixed, declared experiment scope."""

    rcs_m2: float
    worst_pd: float
    max_pfa: float


@dataclass(frozen=True)
class DetectableRcsBracket:
    """A conservative bracket rather than an unsupported interpolated value."""

    last_failing_rcs_m2: float | None
    first_passing_rcs_m2: float | None
    pd_required: float
    pfa_limit: float
    status: str
    evaluated_points: tuple[RcsOperatingPoint, ...]

    @property
    def is_bracketed(self) -> bool:
        return (
            self.last_failing_rcs_m2 is not None
            and self.first_passing_rcs_m2 is not None
        )


def bracket_minimum_detectable_rcs(
    points: Iterable[RcsOperatingPoint],
    *,
    pd_required: float,
    pfa_limit: float,
) -> DetectableRcsBracket:
    """Bracket minimum detectable RCS from explicitly evaluated points.

    A point passes only when its worst-target detection probability reaches the
    requirement and its maximum false-alarm rate respects the declared limit.
    No monotonicity repair or interpolation is hidden in the result.
    """
    ordered = tuple(sorted(points, key=lambda point: point.rcs_m2))
    if not ordered:
        raise ValueError("at least one RCS operating point is required")
    if not 0.0 <= pd_required <= 1.0 or not 0.0 <= pfa_limit <= 1.0:
        raise ValueError("probability requirements must lie in [0, 1]")
    if any(not math.isfinite(point.rcs_m2) or point.rcs_m2 <= 0.0 for point in ordered):
        raise ValueError("RCS values must be finite and positive")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for point in ordered for value in (point.worst_pd, point.max_pfa)
    ):
        raise ValueError("operating-point probabilities must lie in [0, 1]")
    if len({point.rcs_m2 for point in ordered}) != len(ordered):
        raise ValueError("RCS values must be unique")

    passes = [
        point.worst_pd >= pd_required and point.max_pfa <= pfa_limit
        for point in ordered
    ]
    passing_indices = [index for index, passed in enumerate(passes) if passed]
    if not passing_indices:
        return DetectableRcsBracket(
            last_failing_rcs_m2=ordered[-1].rcs_m2,
            first_passing_rcs_m2=None,
            pd_required=pd_required,
            pfa_limit=pfa_limit,
            status="above_tested_range",
            evaluated_points=ordered,
        )

    first_pass = passing_indices[0]
    lower_failures = [
        point.rcs_m2 for point, passed in zip(ordered[:first_pass], passes[:first_pass])
        if not passed
    ]
    if first_pass == 0:
        status = "at_or_below_tested_range"
        last_fail = None
    else:
        status = "bracketed" if lower_failures else "nonmonotone_or_unresolved"
        last_fail = max(lower_failures, default=None)
    # Flag a later failure: a finite Monte Carlo sweep cannot then support a
    # monotone threshold interpretation without further replication.
    if any(not passed for passed in passes[first_pass + 1:]):
        status = "nonmonotone_or_unresolved"
    return DetectableRcsBracket(
        last_failing_rcs_m2=last_fail,
        first_passing_rcs_m2=ordered[first_pass].rcs_m2,
        pd_required=pd_required,
        pfa_limit=pfa_limit,
        status=status,
        evaluated_points=ordered,
    )
