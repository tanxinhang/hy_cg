from __future__ import annotations

import unittest

from isac_sim.low_rcs_rescue import (
    RcsOperatingPoint,
    bracket_minimum_detectable_rcs,
)


class MinimumDetectableRcsTests(unittest.TestCase):
    def test_conservative_bracket_uses_first_passing_grid_point(self) -> None:
        result = bracket_minimum_detectable_rcs(
            [
                RcsOperatingPoint(0.05, 0.40, 0.05),
                RcsOperatingPoint(0.10, 0.70, 0.05),
                RcsOperatingPoint(0.20, 0.90, 0.05),
                RcsOperatingPoint(0.50, 0.96, 0.05),
            ],
            pd_required=0.95,
            pfa_limit=0.05,
        )
        self.assertTrue(result.is_bracketed)
        self.assertEqual(result.last_failing_rcs_m2, 0.20)
        self.assertEqual(result.first_passing_rcs_m2, 0.50)
        self.assertEqual(result.status, "bracketed")

    def test_false_alarm_violation_does_not_pass(self) -> None:
        result = bracket_minimum_detectable_rcs(
            [RcsOperatingPoint(0.5, 0.99, 0.051)],
            pd_required=0.95,
            pfa_limit=0.05,
        )
        self.assertEqual(result.status, "above_tested_range")


if __name__ == "__main__":
    unittest.main()
