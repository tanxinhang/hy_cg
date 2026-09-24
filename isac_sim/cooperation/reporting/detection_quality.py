"""Detector-consistent quality exchanged between TP-UIC and cooperation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class ReceiverDetectionQuality:
    """Per-receiver, per-target covariance-change information.

    ``eigenvalues[j][q]`` are the non-zero modes of
    ``C0^-1/2 S_q C0^-1/2`` produced by TP-UIC's residual model.  Keeping the
    modes, rather than a residual-power scalar, lets cooperation evaluate the
    same stochastic LLR used by the detector without pretending that joint
    receiver information is an independent bistatic-link score.
    """

    eigenvalues: tuple[tuple[np.ndarray, ...], ...]
    n_looks: int
    p_fa: float
    belief_only: bool = True

    @classmethod
    def from_stochastic_information(
        cls, items, *, p_fa: float, belief_only: bool = True
    ) -> "ReceiverDetectionQuality":
        """Build the cooperation contract from TP-UIC information objects."""
        rows = tuple(tuple(np.asarray(item.eigenvalues, dtype=float).copy()
                           for item in row) for row in items)
        looks = {int(item.n_looks) for row in items for item in row}
        if len(looks) != 1:
            raise ValueError("all receiver information must use the same look count")
        return cls(rows, looks.pop(), float(p_fa), bool(belief_only))

    def __post_init__(self) -> None:
        if not self.eigenvalues or not self.eigenvalues[0]:
            raise ValueError("receiver detection quality cannot be empty")
        q_count = len(self.eigenvalues[0])
        if any(len(row) != q_count for row in self.eigenvalues):
            raise ValueError("every receiver must describe the same targets")
        if int(self.n_looks) <= 0 or not 0.0 < float(self.p_fa) < 1.0:
            raise ValueError("n_looks must be positive and p_fa must lie in (0, 1)")
        for row in self.eigenvalues:
            for modes in row:
                values = np.asarray(modes, dtype=float)
                if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
                    raise ValueError("information modes must be finite non-negative vectors")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.eigenvalues), len(self.eigenvalues[0])

    def local_pd(self) -> np.ndarray:
        """Return detector PD for every receiver/target at the declared PFA."""
        return np.vstack([self._pd(row) for row in self.eigenvalues])

    def fused_pd(
        self, delivery_probability: np.ndarray, receiver_correlation: float = 0.0
    ) -> np.ndarray:
        """Expected PD under independent report erasures.

        Correlation is a sensitivity-only design effect; zero preserves exact
        independent-receiver mode concatenation.
        """
        success = np.asarray(delivery_probability, dtype=float)
        m, q = self.shape
        if success.shape != (m, q):
            raise ValueError(f"delivery_probability must have shape {(m, q)}")
        if np.any(~np.isfinite(success)) or np.any((success < 0.0) | (success > 1.0)):
            raise ValueError("delivery probabilities must lie in [0, 1]")
        if receiver_correlation < 0.0:
            raise ValueError("receiver_correlation must be non-negative")
        out = np.zeros(q, dtype=float)
        for target in range(q):
            for mask in range(1 << m):
                delivered = [j for j in range(m) if mask & (1 << j)]
                probability = np.prod([
                    success[j, target] if j in delivered else 1.0 - success[j, target]
                    for j in range(m)
                ])
                if delivered:
                    modes = np.concatenate([
                        np.asarray(self.eigenvalues[j][target], dtype=float)
                        for j in delivered
                    ]) / (1.0 + receiver_correlation * (len(delivered) - 1))
                    pd = self._pd((modes,))[0]
                else:
                    pd = float(self.p_fa)
                out[target] += float(probability) * float(pd)
        return out

    def _pd(self, targets) -> np.ndarray:
        out = []
        looks = float(self.n_looks)
        for modes in targets:
            lam = np.asarray(modes, dtype=float)
            beta = lam / (1.0 + lam)
            constant = -looks * float(np.sum(np.log1p(lam)))
            mu0 = constant + looks * float(np.sum(beta))
            mu1 = constant + looks * float(np.sum(lam))
            var0 = looks * float(np.sum(beta**2))
            var1 = looks * float(np.sum(lam**2))
            if var0 <= 0.0 or var1 <= 0.0:
                out.append(float(self.p_fa))
                continue
            threshold = mu0 + float(norm.ppf(1.0 - self.p_fa)) * np.sqrt(var0)
            out.append(float(norm.sf((threshold - mu1) / np.sqrt(var1))))
        return np.asarray(out)
