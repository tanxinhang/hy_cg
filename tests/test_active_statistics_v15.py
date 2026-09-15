from __future__ import annotations

import unittest

from isac_sim.active_statistics import paired_cluster_summary


class ActiveStatisticsTests(unittest.TestCase):
    def test_paired_cluster_summary_reports_distribution_and_weak_subset(self) -> None:
        result = paired_cluster_summary(
            [2.0, 4.0, 1.2, 8.0], [1.0, 3.0, 0.5, 6.0],
            ["a", "a", "b", "b"], bootstrap_samples=200, seed=7,
        )
        self.assertEqual(result["pairs"], 4)
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["weak_pairs"], 2)
        self.assertEqual(result["fraction_delta_gt_epsilon"], 1.0)
        self.assertGreater(result["delta_log1p_mean"], 0.0)
        self.assertLessEqual(
            result["delta_mean_cluster_ci95_low"], result["delta_mean"]
        )
        self.assertGreaterEqual(
            result["delta_mean_cluster_ci95_high"], result["delta_mean"]
        )


if __name__ == "__main__":
    unittest.main()
