"""Canonical objective, soft-statistic moment and waveform contracts.

Split out of ``test_canonical_consistency.py`` (2026-09-21 audit).  The class name
is deliberately unchanged so node ids stay stable; method bodies are verbatim.
Covered by the anti-loss baseline in ``tests/_test_inventory.py``.
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from isac_sim.core.config import Config, apply_preset
from isac_sim.detection.corr import correlation_aware_weights, covariance_matrix
from isac_sim.detection.fusion import (
    calibrated_fused_threshold,
    compute_weights,
    predicted_pd_for_links,
    selection_utility,
    target_alpha,
)
from isac_sim.detection.llr import llr_delta, llr_var0
from isac_sim.sensing.model import (
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from isac_sim.cooperation.reporting import ReportingPlan
from isac_sim.sensing.soft_channel import (
    draw_received_soft_stat,
    draw_received_soft_vector,
    received_moments,
)
from isac_sim.sensing.waveform import (
    full_otfs_kernel,
    otfs_demodulate,
    otfs_modulate,
    sweep_compare_analytic_vs_psf,
    waveform_llr_detection_check,
)


class ObjectiveAndMomentTests(unittest.TestCase):
    def test_otfs_modulation_round_trip_and_integer_impulse(self) -> None:
        rng = np.random.default_rng(1901)
        Xdd = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        reconstructed = otfs_demodulate(otfs_modulate(Xdd), 8, 8)
        np.testing.assert_allclose(reconstructed, Xdd, atol=1e-12)

        kernel = full_otfs_kernel(8, 8, 0.0, 0.0, 30_000.0, 5900)
        self.assertAlmostEqual(float(np.abs(kernel[0, 0]) ** 2), 1.0, places=11)
        self.assertAlmostEqual(float(np.sum(np.abs(kernel) ** 2)), 1.0, places=11)

    def test_otfs_psf_matches_analytic_dd_gain(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        comparison = sweep_compare_analytic_vs_psf(
            cfg, n_samples=16, rng=np.random.default_rng(1902)
        )
        coarse_error = np.abs(
            comparison["eta_c_analytic"] - comparison["eta_c_psf"]
        )
        local_error = np.abs(
            comparison["eta_loc_analytic"] - comparison["eta_loc_psf"]
        )
        self.assertLess(float(np.max(coarse_error)), 1e-3)
        self.assertLess(float(np.max(local_error)), 1e-10)

    def test_waveform_llr_matches_exact_finite_look_mixture(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        result = waveform_llr_detection_check(
            cfg,
            raw_gamma=0.5,
            interference_gamma=1.0,
            report_success=0.90,
            n_trials=20_000,
            rng=np.random.default_rng(1903),
        )
        self.assertGreater(result["leakage_projection"], 0.20)
        self.assertLess(result["gamma_effective"], result["raw_gamma"])
        self.assertLess(
            abs(result["empirical_pd"] - result["exact_mixture_pd"]), 0.015
        )
        self.assertLess(
            abs(result["empirical_pfa"] - result["exact_mixture_pfa"]), 0.010
        )
        self.assertAlmostEqual(result["calibrated_exact_pfa"], 0.05, places=10)
        self.assertLess(
            abs(result["calibrated_empirical_pd"] - result["calibrated_exact_pd"]),
            0.015,
        )
        self.assertLess(
            abs(result["calibrated_empirical_pfa"] - 0.05), 0.010
        )

    def test_multi_report_calibrated_threshold_controls_false_alarm(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 4, 1
        cfg.dd.use_otfs_bin_validity = False
        cfg.detect.fused_calibration_samples = 16_384
        rng = np.random.default_rng(1904)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        plan = ReportingPlan(mode="explicit", f_q=np.array([1]))
        links = [(0, 1), (2, 1), (0, 3)]
        weights = compute_weights(cfg, tables, 0, links, plan=plan, base=base)
        threshold = calibrated_fused_threshold(
            cfg, tables, 0, links, weights, plan=plan, base=base
        )
        weight_vector = np.asarray([weights[link] for link in links])
        samples = np.asarray([
            weight_vector @ draw_received_soft_vector(
                cfg, tables, links, 0, rng, False, plan, base
            )
            for _ in range(12_000)
        ])
        self.assertLess(abs(float(np.mean(samples > threshold)) - 0.05), 0.012)

    def test_gaussian_replacement_exact_mixture_threshold_controls_false_alarm(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 4, 1
        cfg.dd.use_otfs_bin_validity = False
        cfg.detect.fused_calibration_samples = 16_384
        cfg.detect.exact_gaussian_replacement_threshold = True
        rng = np.random.default_rng(2904)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        plan = ReportingPlan(mode="explicit", f_q=np.array([1]))
        links = [(0, 1), (2, 1), (0, 3)]
        weights = compute_weights(cfg, tables, 0, links, plan=plan, base=base)
        threshold = calibrated_fused_threshold(
            cfg, tables, 0, links, weights, plan=plan, base=base
        )
        weight_vector = np.asarray([weights[link] for link in links])
        samples = np.asarray([
            weight_vector @ draw_received_soft_vector(
                cfg, tables, links, 0, rng, False, plan, base
            )
            for _ in range(20_000)
        ])
        self.assertLess(abs(float(np.mean(samples > threshold)) - 0.05), 0.008)

    def test_singleton_true_erasure_threshold_uses_zero_failure_atom(self) -> None:
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        cfg.scale.M, cfg.scale.Q = 3, 1
        cfg.dd.use_otfs_bin_validity = False
        rng = np.random.default_rng(1911)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        plan = ReportingPlan(mode="explicit", f_q=np.array([2]))
        link = (0, 1)
        tables.chi_comm[1, 2] = 0.6
        weights = compute_weights(
            cfg, tables, 0, [link], plan=plan, base=base
        )
        threshold = calibrated_fused_threshold(
            cfg, tables, 0, [link], weights, plan=plan, base=base
        )
        samples = np.asarray([
            weights[link] * draw_received_soft_stat(
                cfg, tables, link, 0, rng, False, plan
            )
            for _ in range(30_000)
        ])
        self.assertGreater(float(np.mean(samples == 0.0)), 0.38)
        self.assertLess(abs(float(np.mean(samples > threshold)) - 0.05), 0.008)

    def test_correlated_joint_sampler_matches_declared_h0_model(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 4, 1
        cfg.corr.enable = True
        cfg.dd.use_otfs_bin_validity = False
        rng = np.random.default_rng(1701)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        links = [(0, 1), (2, 1), (0, 3)]
        plan = ReportingPlan(mode="explicit", f_q=np.array([1]))

        moments = [received_moments(cfg, tables, link, 0, plan) for link in links]
        sigma = np.sqrt([moment.v0 for moment in moments])
        expected = covariance_matrix(cfg, links, sigma, base=base, q=0)
        samples = np.vstack([
            draw_received_soft_vector(
                cfg, tables, links, 0, rng, False, plan, base
            )
            for _ in range(8000)
        ])
        empirical = np.cov(samples, rowvar=False, ddof=0)
        scale = np.maximum(np.abs(expected), 1e-12)
        self.assertLess(float(np.max(np.abs(empirical - expected) / scale)), 0.12)

        weights = correlation_aware_weights(
            cfg,
            links,
            np.ones(len(links)),
            sigma,
            base=base,
            q=0,
        )
        variance = float(weights @ expected @ weights)
        threshold = 1.6448536269514722 * math.sqrt(variance)
        fused = samples @ weights
        self.assertTrue(0.04 <= float(np.mean(fused > threshold)) <= 0.06)

    def test_correlation_mode_retains_singleton_llr_skewness(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 3, 1
        cfg.corr.enable = True
        cfg.dd.use_otfs_bin_validity = False
        rng = np.random.default_rng(1702)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        link = (0, 1)
        plan = ReportingPlan(mode="explicit", f_q=np.array([1]))
        weights = {link: 1.0}
        from isac_sim.detection.fusion import fused_h0_skewness

        self.assertGreater(
            fused_h0_skewness(cfg, tables, 0, [link], weights, plan, base),
            0.0,
        )

    def test_detector_aligned_pd_is_a_probability(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.scale.M, cfg.scale.Q = 4, 1
        rng = np.random.default_rng(17)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
        pd = predicted_pd_for_links(cfg, tables, 0, [(0, 1)])
        self.assertTrue(0.0 <= pd <= 1.0)

    def test_alpha_is_gradient_of_fair_utility_before_clipping(self) -> None:
        cfg = Config()
        cfg.scale.Q = 3
        cfg.selector.alpha_floor = -1e9
        cfg.selector.alpha_cap = 1e9
        D = np.array([0.6, 1.7, 3.5])
        analytic = target_alpha(cfg, D)
        numeric = np.zeros_like(D)
        h = 1e-6
        for q in range(len(D)):
            plus, minus = D.copy(), D.copy()
            plus[q] += h
            minus[q] -= h
            numeric[q] = (selection_utility(cfg, plus) - selection_utility(cfg, minus)) / (2 * h)
        self.assertTrue(np.allclose(analytic, numeric, rtol=2e-4, atol=2e-5))

    def test_llr_channel_sampling_matches_analytic_moments(self) -> None:
        cfg = Config()
        cfg.scale.M, cfg.scale.Q = 2, 1
        cfg.detect.soft_stat_model = "llr"
        cfg.detect.comm_error_model = "erasure"
        cfg.detect.n_looks = 8
        gamma = 0.8
        chi = 0.65
        tables = SimpleNamespace(
            gamma_sense=np.full((2, 2, 1), gamma),
            mu_soft=np.full((2, 2, 1), llr_delta(gamma, cfg.detect.n_looks)),
            var0_q=np.full((2, 2, 1), llr_var0(gamma, cfg.detect.n_looks)),
            sigma0=np.ones((2, 2)),
            chi_comm=np.array([[0.0, chi], [chi, 0.0]]),
        )
        link = (0, 1)
        expected = received_moments(cfg, tables, link, 0)
        rng = np.random.default_rng(123)
        h0 = np.array([
            draw_received_soft_stat(cfg, tables, link, 0, rng, False) for _ in range(50000)
        ])
        h1 = np.array([
            draw_received_soft_stat(cfg, tables, link, 0, rng, True) for _ in range(50000)
        ])
        self.assertAlmostEqual(float(h0.mean()), expected.m0, delta=0.025)
        self.assertAlmostEqual(float(h0.var()), expected.v0, delta=0.04 * max(expected.v0, 1.0))
        self.assertAlmostEqual(float(h1.mean()), expected.m1, delta=0.04 * max(abs(expected.m1), 1.0))
        self.assertAlmostEqual(float(h1.var()), expected.v1, delta=0.06 * max(expected.v1, 1.0))
        self.assertAlmostEqual(float(np.mean(h0 == 0.0)), 1.0 - chi, delta=0.01)

    def test_signed_correlated_weights_recover_closed_form_deflection(self) -> None:
        cfg = Config()
        cfg.corr.enable = True
        links = [(0, 1), (0, 2), (3, 2)]
        delta = np.array([1.0, 0.3, 0.2])
        sigma = np.array([1.0, 1.0, 1.0])
        Sigma = covariance_matrix(cfg, links, sigma)
        w = correlation_aware_weights(cfg, links, delta, sigma)
        observed = float((w @ delta) ** 2 / (w @ Sigma @ w))
        expected = float(delta @ np.linalg.solve(Sigma + 1e-10 * np.eye(3), delta))
        self.assertTrue(math.isclose(observed, expected, rel_tol=1e-8, abs_tol=1e-8))


if __name__ == "__main__":
    unittest.main()
