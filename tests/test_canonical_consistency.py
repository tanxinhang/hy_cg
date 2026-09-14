from __future__ import annotations

import copy
import math
import unittest
from types import SimpleNamespace

import numpy as np

from isac_sim.config import Config, apply_preset, validate_config
from isac_sim.corr import correlation_aware_weights, covariance_matrix
from isac_sim.fusion import predicted_pd_for_links, selection_utility, target_alpha
from isac_sim.llr import llr_delta, llr_var0
from isac_sim.model import build_base_gains, compute_link_tables, generate_geometry
from isac_sim.packetization import packetization_audit
from isac_sim.reporting import ReportingPlan
from isac_sim.reporting import assign_fusion_nodes
from isac_sim.soft_channel import draw_received_soft_stat, received_moments
from isac_sim.selection import select_c2f_adaptive
from isac_sim.simulate import (
    rng_for_detection,
    rng_for_method,
    run_method_on_trial,
    run_one_trial,
    run_simulation,
)


class CanonicalConfigurationTests(unittest.TestCase):
    def test_nearest_target_alias_is_backward_compatible(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.scale.M, cfg.scale.Q = 4, 2
        rng = np.random.default_rng(31)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)

        cfg.fusion.rule = "nearest_target"
        target_plan = assign_fusion_nodes(cfg, base, tables, geom)
        cfg.fusion.rule = "nearest_centroid"
        legacy_plan = assign_fusion_nodes(cfg, base, tables, geom)

        self.assertTrue(np.array_equal(target_plan.f_q, legacy_plan.f_q))

    def test_unknown_fusion_rule_is_rejected(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.fusion.rule = "oracle_truth"
        with self.assertRaisesRegex(ValueError, "fusion.rule"):
            validate_config(cfg)

    def test_packetization_audit_distinguishes_items_from_padded_packets(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        selected = {
            0: [(0, 1), (2, 1), (3, 2)],
            1: [(0, 1)],
        }
        audit = packetization_audit(cfg, selected)
        self.assertEqual(audit["reports"], 4.0)
        self.assertEqual(audit["active_reporters"], 2.0)
        self.assertEqual(audit["max_reports_per_uav"], 3.0)
        self.assertEqual(audit["current_bits"], 2560.0)
        self.assertEqual(audit["item_only_bits"], 640.0)
        self.assertEqual(audit["packed_packets_same_target"], 3.0)
        self.assertEqual(audit["packed_bits_same_target"], 1920.0)
        self.assertEqual(audit["conflict_graph_slots"], 3.0)
        self.assertAlmostEqual(
            audit["conflict_slot_delay_ms_no_interference"], 3.2
        )

        explicit = ReportingPlan(mode="explicit", f_q=np.array([3, 3]))
        congested = packetization_audit(cfg, selected, explicit)
        self.assertEqual(congested["assigned_fusion_uavs"], 1.0)
        self.assertEqual(congested["max_targets_per_fusion_uav"], 2.0)
        self.assertEqual(congested["max_reports_per_fusion_uav"], 4.0)
        self.assertEqual(congested["conflict_graph_slots"], 4.0)

    def test_explicit_paired_reference_is_labelled_without_proposed_alias(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 3
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 4
        cfg.run.num_mc = 2
        cfg.run.verbose = False
        methods = ["proposed_c2f", "proposed_c2f_adaptive_pd", "sense_sinr"]
        summary = run_simulation(
            cfg, methods=methods, paired_reference="proposed_c2f_adaptive_pd"
        )
        self.assertEqual(
            summary["sense_sinr"]["paired_reference_method"],
            "proposed_c2f_adaptive_pd",
        )
        self.assertNotIn("paired_proposed_delta_P_D", summary["sense_sinr"])

    def test_belief_refinement_deflection_excludes_uncaptured_links(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.scale.M, cfg.scale.Q = 4, 1
        cfg.detect.num_false_per_target = 1
        rng = np.random.default_rng(29)
        geom = generate_geometry(cfg, rng)
        belief_base = build_base_gains(cfg, geom, rng)
        belief_tables = compute_link_tables(cfg, belief_base)
        plan = assign_fusion_nodes(cfg, belief_base, belief_tables, geom)

        truth_base = copy.deepcopy(belief_base)
        truth_base.tau[:, :, 0] += 1.0e-4
        truth_tables = compute_link_tables(cfg, truth_base)
        zeros = np.zeros_like(truth_base.tau)
        result = run_method_on_trial(
            cfg,
            belief_base,
            belief_tables,
            "all_neighbor",
            0,
            plan=plan,
            eval_base=truth_base,
            eval_tables=truth_tables,
            belief_dd_std=(zeros, zeros),
        )
        self.assertGreater(sum(len(v) for v in result.selected_links.values()), 0)
        self.assertEqual(result.belief_capture_rate, 0.0)
        self.assertTrue(np.array_equal(result.D_fuse_per_target, np.zeros(1)))

    def test_adaptive_detector_rule_matches_reference_reporting_budget(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 3
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 4
        result = run_one_trial(
            cfg, 0, methods=["proposed_c2f", "proposed_c2f_adaptive_pd"]
        )
        ref = result["proposed_c2f"]
        combined = result["proposed_c2f_adaptive_pd"]
        self.assertEqual(
            sum(len(v) for v in ref.selected_links.values()),
            sum(len(v) for v in combined.selected_links.values()),
        )
        self.assertEqual(ref.overhead_bits, combined.overhead_bits)
        self.assertAlmostEqual(ref.overhead_delay_s, combined.overhead_delay_s)

    def test_adaptive_c2f_respects_fine_and_link_caps(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.refine.shortlist_size = 3
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 3
        rng = np.random.default_rng(23)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        selected, _, stats = select_c2f_adaptive(cfg, base, tables)
        self.assertLessEqual(sum(len(v) for v in selected.values()), 3)
        self.assertTrue(all(len(v) <= 2 for v in selected.values()))
        self.assertLessEqual(stats["fine_eval_c2f"], 2 * cfg.refine.shortlist_size)
        self.assertLessEqual(stats["fine_eval_c2f"], stats["fine_eval_full"])

    def test_paper_preset_is_coherent(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        validate_config(cfg)
        self.assertEqual(cfg.comm.interference_model, "orthogonal")
        self.assertEqual(cfg.comm.mac_model, "serial")
        self.assertEqual(cfg.fusion.mode, "explicit")
        self.assertEqual(cfg.prior.scheduler_rcs, "mean")
        self.assertEqual(cfg.detect.rcs_model, "mean")
        self.assertEqual(cfg.selector.score_mode, "exact_utility")
        self.assertTrue(cfg.refine.enable)
        self.assertEqual(cfg.refine.mode, "window")
        self.assertEqual((cfg.geometry.uav_speed_min, cfg.geometry.uav_speed_max),
                         (30.0, 60.0))
        self.assertEqual((cfg.geometry.target_speed_min, cfg.geometry.target_speed_max),
                         (50.0, 90.0))

    def test_scheduler_uses_mean_not_realized_rcs(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.detect.rcs_model = "swerling1"
        cfg.scale.M, cfg.scale.Q = 3, 2
        rng = np.random.default_rng(7)
        geom = generate_geometry(cfg, rng)
        truth = build_base_gains(cfg, geom, rng)
        scheduler = build_base_gains(cfg, geom, rng, channel=truth, rcs_view="mean")
        non_diag = ~np.eye(cfg.scale.M, dtype=bool)
        expected = np.full_like(scheduler.rcs_fluct, cfg.detect.target_rcs)
        self.assertTrue(np.allclose(scheduler.rcs_fluct[non_diag], expected[non_diag]))
        self.assertFalse(np.allclose(truth.rcs_fluct[non_diag], expected[non_diag]))

    def test_orthogonal_reporting_has_no_payload_multiuser_interference(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.scale.M, cfg.scale.Q = 4, 1
        rng = np.random.default_rng(9)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        all_active = np.ones(cfg.scale.M, dtype=bool)
        one_active = np.array([True, False, False, False])
        t_all = compute_link_tables(cfg, base, active_tx_mask=all_active)
        t_one = compute_link_tables(cfg, base, active_tx_mask=one_active)
        self.assertTrue(np.allclose(t_all.gamma_comm, t_one.gamma_comm))
        self.assertTrue(np.allclose(t_all.gamma_sense, t_one.gamma_sense))

    def test_detector_common_random_numbers_are_method_independent(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        a = rng_for_detection(cfg, 11).normal(size=16)
        b = rng_for_detection(cfg, 11).normal(size=16)
        self.assertTrue(np.array_equal(a, b))
        selector_a = rng_for_method(cfg, 11, "proposed_lagrangian").normal(size=4)
        selector_b = rng_for_method(cfg, 11, "sense_sinr").normal(size=4)
        self.assertFalse(np.array_equal(selector_a, selector_b))


class ObjectiveAndMomentTests(unittest.TestCase):
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
