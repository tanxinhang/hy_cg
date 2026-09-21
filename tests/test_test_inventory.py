"""Anti-loss gate for splitting the oversized historical test files.

Splitting a big test file fails *silently*: pytest keeps running, the totals
still look plausible, and a few ``def test_*`` just never made it across.  The
only symptom is that a contract quietly stopped being checked.

``_test_inventory.py`` freezes the names each oversized file owned before any
split.  This module asserts that every one of them is still defined somewhere
under ``tests/`` — so a sloppy split fails loudly, at the moment it happens.

The baseline is provenance, not configuration: it is keyed by the ORIGINAL
filename and is never updated when a file is split.  Deleting a test on purpose
is still allowed, but it must be done by editing this baseline in the same
commit, which keeps the decision reviewable.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from _test_inventory import BASELINE, TOTAL_NAMES

TESTS = Path(__file__).resolve().parent


def _defined_names() -> set[str]:
    """Every ``def test_*`` under ``tests/``, regardless of which file or class."""
    out: set[str] = set()
    for path in sorted(TESTS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test"):
                    out.add(node.name)
    return out


class TestInventoryPreserved(unittest.TestCase):
    def test_no_historical_test_name_was_lost(self) -> None:
        defined = _defined_names()
        missing: dict[str, list[str]] = {}
        for origin, names in BASELINE.items():
            gone = [n for n in names if n not in defined]
            if gone:
                missing[origin] = gone
        self.assertEqual(
            {}, missing,
            "a test disappeared during a split; re-add it or update "
            "tests/_test_inventory.py in the same commit if removal is intended",
        )

    def test_total_test_name_count_did_not_shrink(self) -> None:
        # Duplicated names across files would inflate the set union, so also
        # guard the raw count of definitions.
        count = 0
        for path in sorted(TESTS.glob("*.py")):
            if path.name.startswith("_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            count += sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test")
            )
        self.assertGreaterEqual(
            count, TOTAL_NAMES,
            "fewer test definitions than the frozen baseline; a split dropped one",
        )

    def test_baseline_files_are_the_oversized_ones(self) -> None:
        # Keeps the baseline honest: it exists for the 350-line offenders only.
        self.assertEqual(len(BASELINE), 6)
        self.assertEqual(TOTAL_NAMES, sum(len(v) for v in BASELINE.values()))


if __name__ == "__main__":
    unittest.main()
