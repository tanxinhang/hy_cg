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


@dataclass(frozen=True)
class MatrixAOResult:
    active: tuple[bool, ...]
    protection: tuple[frozenset[int], ...]
    information: np.ndarray
    history: tuple[float, ...]

    @property
    def objective(self) -> float:
        return float(np.min(self.information))


def monotone_matrix_information_ao(
    initial_active: Sequence[bool],
    initial_protection: Sequence[frozenset[int]],
    active_candidates: Iterable[Sequence[bool]],
    protection_candidates: Sequence[Sequence[frozenset[int]]],
    evaluate: Callable[[tuple[bool, ...], tuple[frozenset[int], ...]], np.ndarray],
    *,
    max_rounds: int = 20,
    tolerance: float = 1e-12,
) -> MatrixAOResult:
    """Alternate exact activation and receiver-local z updates.

    Every accepted block update is checked against ``min_q information_q``;
    ties retain the incumbent, making the returned history monotone by design.
    """
    active = tuple(bool(v) for v in initial_active)
    protection = tuple(frozenset(v) for v in initial_protection)
    info = np.asarray(evaluate(active, protection), dtype=float)
    history = [float(np.min(info))]

    for _ in range(int(max_rounds)):
        changed = False
        best = (history[-1], active, info)
        for candidate in active_candidates:
            candidate = tuple(bool(v) for v in candidate)
            value = np.asarray(evaluate(candidate, protection), dtype=float)
            score = float(np.min(value))
            if score > best[0] + tolerance:
                best = (score, candidate, value)
        if best[1] != active:
            _, active, info = best
            changed = True

        for receiver, subsets in enumerate(protection_candidates):
            incumbent_score = float(np.min(info))
            best_z, best_info = protection[receiver], info
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
