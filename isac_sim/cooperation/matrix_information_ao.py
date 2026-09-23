"""Exact local protection search and monotone block coordinate ascent."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np


def protection_subsets(
    targets: Sequence[int],
    rank_of: Callable[[frozenset[int]], int],
    max_rank: int,
) -> tuple[frozenset[int], ...]:
    """Enumerate every binary z subset satisfying the protection-DoF budget."""
    ids = tuple(sorted(set(int(q) for q in targets)))
    feasible = []
    for size in range(len(ids) + 1):
        for subset in combinations(ids, size):
            chosen = frozenset(subset)
            if int(rank_of(chosen)) <= int(max_rank):
                feasible.append(chosen)
    return tuple(feasible)


def protection_neighborhood(
    targets: Sequence[int],
    protected: frozenset[int],
    rank_of: Callable[[frozenset[int]], int],
    max_rank: int,
) -> tuple[frozenset[int], ...]:
    """Return feasible marginal add/drop/swap protection decisions."""
    ids = tuple(sorted(set(int(q) for q in targets)))
    incumbent = frozenset(protected)
    candidates: set[frozenset[int]] = set()
    for target in ids:
        trial = incumbent ^ {target}
        if int(rank_of(trial)) <= int(max_rank):
            candidates.add(trial)
    for old in incumbent:
        for new in set(ids) - incumbent:
            trial = (incumbent - {old}) | {new}
            if int(rank_of(trial)) <= int(max_rank):
                candidates.add(frozenset(trial))
    return tuple(sorted(candidates, key=lambda item: (len(item), tuple(item))))


@dataclass(frozen=True)
class MatrixAOResult:
    active: tuple[bool, ...]
    protection: tuple[frozenset[int], ...]
    information: np.ndarray
    history: tuple[float, ...]

    @property
    def objective(self) -> float:
        return float(np.min(self.information))


def activation_neighborhood(
    active: Sequence[bool], *, include_swaps: bool = True
) -> tuple[tuple[bool, ...], ...]:
    """Return non-empty one-flip and optional one-for-one swap decisions.

    This is a marginal decision set: its size is O(M^2), versus O(2^M) for
    exhaustive activation masks.  The incumbent itself is intentionally not
    returned because the caller already has its information value.
    """
    state = tuple(bool(v) for v in active)
    candidates: set[tuple[bool, ...]] = set()
    for index in range(len(state)):
        trial = list(state)
        trial[index] = not trial[index]
        if any(trial):
            candidates.add(tuple(trial))
    if include_swaps:
        enabled = [i for i, value in enumerate(state) if value]
        disabled = [i for i, value in enumerate(state) if not value]
        for old in enabled:
            for new in disabled:
                trial = list(state)
                trial[old], trial[new] = False, True
                candidates.add(tuple(trial))
    return tuple(sorted(candidates))


def monotone_matrix_information_ao(
    initial_active: Sequence[bool],
    initial_protection: Sequence[frozenset[int]],
    active_candidates: Iterable[Sequence[bool]] | Callable[
        [tuple[bool, ...]], Iterable[Sequence[bool]]
    ],
    protection_candidates: Sequence[
        Sequence[frozenset[int]] | Callable[
            [tuple[bool, ...], frozenset[int]], Iterable[frozenset[int]]
        ]
    ],
    evaluate: Callable[[tuple[bool, ...], tuple[frozenset[int], ...]], np.ndarray],
    *,
    max_rounds: int = 20,
    tolerance: float = 1e-12,
) -> MatrixAOResult:
    """Alternate activation and receiver-local z updates.

    Every accepted block update is checked against ``min_q information_q``;
    ties retain the incumbent, making the returned history monotone by design.
    A callable ``active_candidates`` creates a fresh marginal neighborhood at
    each round; a fixed iterable retains the small-problem oracle interface.
    """
    active = tuple(bool(v) for v in initial_active)
    protection = tuple(frozenset(v) for v in initial_protection)
    info = np.asarray(evaluate(active, protection), dtype=float)
    history = [float(np.min(info))]

    for _ in range(int(max_rounds)):
        changed = False
        best = (history[-1], active, info)
        candidates_now = (
            active_candidates(active) if callable(active_candidates)
            else active_candidates
        )
        for candidate in candidates_now:
            candidate = tuple(bool(v) for v in candidate)
            value = np.asarray(evaluate(candidate, protection), dtype=float)
            score = float(np.min(value))
            if score > best[0] + tolerance:
                best = (score, candidate, value)
        if best[1] != active:
            _, active, info = best
            changed = True

        for receiver, subset_source in enumerate(protection_candidates):
            incumbent_score = float(np.min(info))
            best_z, best_info = protection[receiver], info
            subsets = (
                subset_source(active, protection[receiver])
                if callable(subset_source) else subset_source
            )
            for subset in subsets:
                trial_z = list(protection)
                trial_z[receiver] = frozenset(subset)
                value = np.asarray(evaluate(active, tuple(trial_z)), dtype=float)
                score = float(np.min(value))
                if score > incumbent_score + tolerance:
                    incumbent_score, best_z, best_info = score, frozenset(subset), value
            if best_z != protection[receiver]:
                updated = list(protection)
                updated[receiver] = best_z
                protection, info = tuple(updated), best_info
                changed = True
        history.append(float(np.min(info)))
        if not changed:
            break
    return MatrixAOResult(active, protection, info, tuple(history))
