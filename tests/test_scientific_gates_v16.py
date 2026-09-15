from __future__ import annotations

import unittest

import numpy as np

from isac_sim.scientific_gates import (
    PhysicalHeadroomDashboard,
    aggregate_partitioned_llrs,
    allocate_reliability_protection,
    aggregation_reliability_threshold,
    coherent_power_oracle,
    common_latent_gaussian_llrs,
    erasure_received_kl,
    evidence_delivery_variance,
    gaussian_phase_coherence,
    hypothesis_dependent_erasure_kl,
    lower_tail_detection_summary,
    swerling_information,
    symmetric_complementarity_oracle,
)


class SufficientStatisticGateTests(unittest.TestCase):
    def test_partitioned_exact_llr_is_samplewise_identical(self) -> None:
        rng = np.random.default_rng(1601)
        llrs = rng.normal(size=(128, 5))
        result = aggregate_partitioned_llrs(llrs, ((0, 3), (1,), (2, 4)))
        self.assertLess(result.max_abs_error, 1e-12)

    def test_duplicate_or_missing_evidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly once"):
            aggregate_partitioned_llrs(np.ones((4, 3)), ((0, 1), (1, 2)))

    def test_common_latent_breaks_naive_marginal_llr_additivity(self) -> None:
        observations = np.asarray([[1.0, -0.5, 2.0], [0.2, 0.3, 0.4]])
        result = common_latent_gaussian_llrs(observations, 0.7)
        self.assertGreater(result.max_abs_gap, 1e-3)


class TransportInformationGateTests(unittest.TestCase):
    def test_observed_erasure_scales_kl_linearly(self) -> None:
        information = np.asarray([0.1, 1.0, 4.0])
        np.testing.assert_allclose(erasure_received_kl(information, 0.37), 0.37 * information)

    def test_aggregate_reliability_threshold_is_information_weighted(self) -> None:
        threshold = aggregation_reliability_threshold((1.0, 3.0), (0.5, 0.9))
        self.assertAlmostEqual(threshold, 0.8)

    def test_aggregation_has_higher_catastrophic_loss_variance(self) -> None:
        separate = evidence_delivery_variance((1.0, 2.0, 3.0), 0.8, aggregated=False)
        aggregate = evidence_delivery_variance((1.0, 2.0, 3.0), 0.8, aggregated=True)
        self.assertGreater(aggregate, separate)

    def test_hypothesis_dependent_arrival_adds_bernoulli_information(self) -> None:
        ordinary = float(erasure_received_kl(2.0, 0.8))
        dependent = hypothesis_dependent_erasure_kl(2.0, 0.8, 0.6)
        self.assertGreater(dependent, ordinary)

    def test_importance_aware_protection_solves_linear_budget(self) -> None:
        plan = allocate_reliability_protection(
            (5.0, 1.0), (0.5, 0.5), (1.0, 1.0), 0.5
        )
        self.assertEqual(plan.final_success, (1.0, 0.5))
        self.assertAlmostEqual(plan.budget_used, 0.5)
        self.assertAlmostEqual(plan.expected_received_kl, 5.5)


class PhysicalHeadroomGateTests(unittest.TestCase):
    def test_identical_view_power_split_cannot_beat_single_tx(self) -> None:
        alpha = 0.04
        total_power = 1.0
        looks = 16
        single = float(swerling_information(alpha, total_power, looks))
        split = 2.0 * float(swerling_information(alpha, total_power / 2.0, looks))
        self.assertLess(split, single)

    def test_low_snr_complementarity_threshold_predicts_both_sides(self) -> None:
        strong = symmetric_complementarity_oracle(2.0, 1.0, 1e-4, 16)
        weak = symmetric_complementarity_oracle(1.5, 1.0, 1e-4, 16)
        self.assertTrue(strong.low_snr_cooperation_condition)
        self.assertTrue(strong.choose_cooperation)
        self.assertFalse(weak.low_snr_cooperation_condition)
        self.assertFalse(weak.choose_cooperation)

    def test_coherent_eigenvalue_oracle_has_correct_limiting_cases(self) -> None:
        channel = np.asarray([1.0 + 0.0j, 0.5 + 0.0j, 0.25 + 0.0j])
        perfect = coherent_power_oracle(channel, 2.0, np.ones((3, 3), dtype=complex))
        random_phase = coherent_power_oracle(channel, 2.0, np.eye(3, dtype=complex))
        self.assertAlmostEqual(perfect.optimal_power, 2.0 * np.sum(np.abs(channel) ** 2))
        self.assertGreater(perfect.gain, 1.0)
        self.assertAlmostEqual(random_phase.gain, 1.0)

    def test_gaussian_phase_coherence_interpolates_between_limits(self) -> None:
        low_error = gaussian_phase_coherence(3, 0.1)
        high_error = gaussian_phase_coherence(3, 5.0)
        self.assertGreater(abs(low_error[0, 1]), abs(high_error[0, 1]))
        np.testing.assert_allclose(np.diag(low_error), np.ones(3))


class RecoveryAndGeneralizationGateTests(unittest.TestCase):
    def test_candidate_and_algorithm_gap_decomposition_is_exact(self) -> None:
        dashboard = PhysicalHeadroomDashboard(
            metric="kl", best_single=2.0, noncoherent_full_oracle=5.0,
            coherent_oracle=7.0, restricted_oracle=4.5, proposed=4.0,
        )
        self.assertAlmostEqual(dashboard.candidate_loss, 0.5)
        self.assertAlmostEqual(dashboard.algorithm_loss, 0.5)
        self.assertAlmostEqual(dashboard.total_noncoherent_gap, 1.0)
        self.assertAlmostEqual(dashboard.decomposition_error, 0.0)
        self.assertAlmostEqual(dashboard.recovery_fraction, 2.0 / 3.0)

    def test_lower_tail_endpoint_is_distinct_from_sample_minimum(self) -> None:
        summary = lower_tail_detection_summary((0.1, 0.4, 0.6, 0.8, 0.9), 0.2)
        self.assertEqual(summary["count"], 5)
        self.assertGreater(summary["lower_quantile"], summary["minimum"])
        self.assertLessEqual(summary["lower_cvar"], summary["lower_quantile"])


if __name__ == "__main__":
    unittest.main()
