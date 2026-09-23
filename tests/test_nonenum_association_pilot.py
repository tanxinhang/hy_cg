"""Contracts for non-enumerative association search budgets and selection."""
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import run_nonenum_association_pilot as pilot  # noqa: E402


def test_search_budget_distinguishes_partial_states_from_k_subsets():
    seen = {(0,), (1,), (0, 1), (0, 1, 2), (1, 2, 3)}
    budget = pilot._budget_summary(seen, k=3, receiver_count=6)

    assert budget["evaluated_search_states_total"] == 5
    assert budget["evaluated_unique_k_subsets"] == 2
    assert budget["total_possible_k_subsets"] == 20
    assert budget["k_subset_fraction_evaluated"] == pytest.approx(0.1)
