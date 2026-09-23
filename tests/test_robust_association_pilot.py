"""Contracts for robust association candidates, scoring, and search budgets."""
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_robust_association_pilot as pilot  # noqa: E402


def test_greedy_budget_is_linear_and_selects_complementary_receiver():
    receivers = list(range(6))
    utility = lambda s: sum({0: 3.0, 1: 2.0, 2: 1.0}.get(v, 0.0) for v in s)
    selected, seen = pilot._greedy(receivers, 3, utility)
    assert selected == (0, 1, 2)
    assert len(seen) == 6 + 5 + 4


def test_shrinkage_reduces_offdiagonal_covariance():
    h0 = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
    h1 = h0 + np.array([1., 0.5])
    raw = pilot._h0_covariance(h0)
    shrunk = pilot._shrunk_cov(h0, h1, shrink=0.7)
    assert abs(shrunk[0, 1]) < abs(raw[0, 1])
    assert np.all(np.linalg.eigvalsh(shrunk) > 0)


def test_conditional_gain_rejects_a_redundant_receiver():
    # Receiver 1 is nearly a copy of receiver 0; receiver 2 adds an independent
    # mean shift. CIG must prefer the complementary receiver after selecting 0.
    h0 = np.array([
        [-2.0, -2.01, -1.0], [-1.0, -0.99, 1.0],
        [1.0, 1.01, -1.0], [2.0, 1.99, 1.0],
        [-1.5, -1.49, 0.5], [1.5, 1.49, -0.5],
    ])
    h1 = h0 + np.array([1.0, 1.0, 0.8])
    data = {"train_h0": h0, "train_h1": h1}
    redundant = pilot._conditional_gain(data, (0,), 1, shrink=0.05)
    complementary = pilot._conditional_gain(data, (0,), 2, shrink=0.05)
    assert complementary > redundant
