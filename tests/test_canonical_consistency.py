from __future__ import annotations

import copy
import math
import unittest
from types import SimpleNamespace

import numpy as np

from isac_sim.config import Config, apply_preset, validate_config
from isac_sim.corr import correlation_aware_weights, covariance_matrix
from isac_sim.fusion import (
    calibrated_fused_threshold,
    compute_weights,
    predicted_pd_for_links,
    selection_utility,
    target_alpha,
)
from isac_sim.llr import llr_delta, llr_var0
from isac_sim.model import (
    build_base_gains,
    compute_link_tables,
    generate_geometry,
    rescale_sensing_tables_for_rcs,
)
from isac_sim.packetization import packetization_audit
from isac_sim.reporting import ReportingPlan
from isac_sim.reporting import assign_fusion_nodes, is_local_observation
from isac_sim.soft_channel import (
    draw_received_soft_stat,
    draw_received_soft_vector,
    local_moments,
    received_moments,
)
from isac_sim.waveform import (
    full_otfs_kernel,
    otfs_demodulate,
    otfs_modulate,
    sweep_compare_analytic_vs_psf,
    waveform_llr_detection_check,
)
from isac_sim.selection import feasible_links_for_target, select_c2f_adaptive
from isac_sim.simulate import (
    rng_for_detection,
    rng_for_method,
    run_method_on_trial,
    run_one_trial,
    run_simulation,
    total_overhead_bits,
    total_overhead_delay_s,
)


class CanonicalConfigurationTests(unittest.TestCase):
    def test_small_uav_scenario_presets_are_physically_named_and_valid(self) -> None:
        expected = {
            "small-uav-compact-800m": (800.0, 0.05),
            "small-uav-dense-s1": (1000.0, 0.1),
            "small-uav-nominal-s2": (2000.0, 0.05),
            "small-uav-sparse-s3": (4000.0, 0.02),
        }
        for name, (area, rcs) in expected.items():
            cfg = apply_preset(Config(), name)
            validate_config(cfg)
            self.assertEqual(cfg.geometry.area_xy, area)
            self.assertEqual(cfg.detect.target_rcs, rcs)
            self.assertEqual(
                cfg.radio.radar_tx_gain_dbi + cfg.radio.radar_rx_gain_dbi
                - cfg.radio.radar_system_loss_db,
                0.0,
            )

    def test_net_radar_gain_sweep_override_has_no_invented_tx_rx_split(self) -> None:
        from isac_sim.model import radar_hardware_gain

        cfg = apply_preset(Config(), "small-uav-nominal-s2")
        cfg.radio.radar_net_gain_db = 15.0
        self.assertAlmostEqual(radar_hardware_gain(cfg), 10.0**1.5)
        self.assertEqual(cfg.radio.radar_tx_gain_dbi, 0.0)
        self.assertEqual(cfg.radio.radar_rx_gain_dbi, 0.0)

    def test_explicit_radar_budget_preserves_equivalent_echo_scale(self) -> None:
        historical = apply_preset(Config(), "paper-canonical")
        bridge = apply_preset(Config(), "small-uav-link-budget-bridge")
        for cfg in (historical, bridge):
            cfg.scale.M, cfg.scale.Q = 4, 2
            cfg.dd.use_otfs_bin_validity = False
            cfg.detect.rcs_model = "mean"

        rng = np.random.default_rng(905)
        geom = generate_geometry(historical, rng)
        base_h = build_base_gains(historical, geom, np.random.default_rng(906))
        base_b = build_base_gains(bridge, geom, np.random.default_rng(906))
        tab_h = compute_link_tables(historical, base_h)
        tab_b = compute_link_tables(bridge, base_b)

        # 0.1 m^2 with a 27 dB net radar budget is 50.1187 m^2 effective,
        # only 0.0103 dB from the historical 50 m^2 implicit scale.
        ratio = tab_b.raw_gamma_sense / np.maximum(tab_h.raw_gamma_sense, 1e-300)
        active = tab_h.raw_gamma_sense > 0.0
        np.testing.assert_allclose(ratio[active], 10.0 ** (27.0 / 10.0) / 500.0)

    def test_mean_rcs_scales_bistatic_gain_and_sensing_sinr_linearly(self) -> None:
        low = apply_preset(Config(), "small-uav-compact-800m")
        low.scale.M, low.scale.Q = 4, 2
        low.dd.use_otfs_bin_validity = False
        low.detect.rcs_model = "mean"
        low.detect.rcs_aspect_enable = False
        low.detect.target_rcs = 0.05
        high = copy.deepcopy(low)
        high.detect.target_rcs = 0.10

        geom = generate_geometry(low, np.random.default_rng(1905))
        base_low = build_base_gains(
            low, geom, np.random.default_rng(1906), rcs_view="mean"
        )
        base_high = build_base_gains(
            high,
            geom,
            np.random.default_rng(1907),
            channel=base_low,
            rcs_view="mean",
        )
        tab_low = compute_link_tables(low, base_low)
        tab_high = compute_link_tables(high, base_high)

        active = base_low.target_gain > 0.0
        np.testing.assert_allclose(
            base_high.target_gain[active] / base_low.target_gain[active], 2.0
        )
        np.testing.assert_allclose(
            tab_high.gamma_sense[active] / tab_low.gamma_sense[active], 2.0
        )

    def test_rcs_counterfactual_recomputes_llr_moments_without_mutation(self) -> None:
        cfg = apply_preset(Config(), "small-uav-compact-800m")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.dd.use_otfs_bin_validity = False
        geom = generate_geometry(cfg, np.random.default_rng(1910))
        base = build_base_gains(
            cfg, geom, np.random.default_rng(1911), rcs_view="mean"
        )
        tables = compute_link_tables(cfg, base)
        original_gamma = tables.gamma_sense.copy()

        lower = rescale_sensing_tables_for_rcs(cfg, tables, np.array([0.5, 0.8]))

        np.testing.assert_array_equal(tables.gamma_sense, original_gamma)
        np.testing.assert_allclose(lower.gamma_sense[:, :, 0], original_gamma[:, :, 0] * 0.5)
        np.testing.assert_allclose(lower.gamma_sense[:, :, 1], original_gamma[:, :, 1] * 0.8)
        np.testing.assert_allclose(
            lower.mu_soft, llr_delta(lower.gamma_sense, cfg.detect.n_looks)
        )
        np.testing.assert_allclose(
            lower.var0_q, llr_var0(lower.gamma_sense, cfg.detect.n_looks)
        )

    def test_invalid_radar_system_loss_is_rejected(self) -> None:
        cfg = Config()
        cfg.radio.radar_system_loss_db = -1.0
        with self.assertRaisesRegex(ValueError, "system_loss"):
            validate_config(cfg)

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

    def test_waveform_impairments_reduce_sensing_sinr_only_when_enabled(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.dd.use_otfs_bin_validity = False
        rng = np.random.default_rng(44)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        ideal = compute_link_tables(cfg, base)

        cfg.waveform_impairments.enable = True
        cfg.waveform_impairments.clutter_inr = 0.5
        cfg.waveform_impairments.multipath_inr = 0.25
        cfg.waveform_impairments.unresolved_target_inr = 0.2
        cfg.waveform_impairments.sync_delay_bins = 0.1
        cfg.waveform_impairments.sync_doppler_bins = -0.1
        impaired = compute_link_tables(cfg, base)
        active = ideal.gamma_sense > 0.0
        self.assertTrue(np.all(impaired.gamma_sense[active] < ideal.gamma_sense[active]))
        np.testing.assert_array_equal(impaired.gamma_comm, ideal.gamma_comm)

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

    def test_adaptive_detector_rule_is_reference_budget_independent(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 3
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 4
        rng = np.random.default_rng(17)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        plan = assign_fusion_nodes(cfg, base, tables, geom)
        fine = compute_link_tables(cfg, base, dd_gain=base.eta_fine)

        low = run_method_on_trial(
            cfg, base, tables, "proposed_c2f_adaptive_pd", 0,
            reference_counts={0: 0, 1: 0}, c2f_tables=fine, plan=plan,
        )
        high = run_method_on_trial(
            cfg, base, tables, "proposed_c2f_adaptive_pd", 0,
            reference_counts={0: 99, 1: 99}, c2f_tables=fine, plan=plan,
        )
        self.assertEqual(low.selected_links, high.selected_links)
        self.assertEqual(low.overhead_bits, high.overhead_bits)
        self.assertAlmostEqual(low.overhead_delay_s, high.overhead_delay_s)

    def test_local_fusion_evidence_has_no_reporting_cost(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.scale.M, cfg.scale.Q = 3, 1
        cfg.dd.use_otfs_bin_validity = False
        rng = np.random.default_rng(19)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        plan = ReportingPlan(mode="explicit", f_q=np.array([1]))
        link = (0, 1)

        # Sensing availability is independent of the i-j reporting adjacency.
        base.edge_mask[0, 1] = base.edge_mask[1, 0] = False
        self.assertGreater(tables.gamma_sense[0, 1, 0], 0.0)
        self.assertIn(link, feasible_links_for_target(cfg, base, tables, 0, plan))
        selected = {0: [link]}
        self.assertEqual(total_overhead_bits(cfg, selected, plan), 0.0)
        self.assertEqual(total_overhead_delay_s(cfg, tables, selected, plan), 0.0)
        self.assertEqual(packetization_audit(cfg, selected, plan)["reports"], 0.0)
        self.assertEqual(
            received_moments(cfg, tables, link, 0, plan),
            local_moments(cfg, tables, link, 0),
        )

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

    def test_local_evidence_counterfactual_cap_is_enforced(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 5, 2
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 5
        cfg.selector.max_links_per_target = 4
        cfg.selector.max_total_links = 8
        rng = np.random.default_rng(1703)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        plan = assign_fusion_nodes(cfg, base, tables, geom)

        cfg.selector.max_local_observations_per_target = 0
        selected_zero, _, _ = select_c2f_adaptive(cfg, base, tables, plan)
        self.assertTrue(all(
            not is_local_observation(plan, link, q)
            for q, links in selected_zero.items() for link in links
        ))

        cfg.selector.max_local_observations_per_target = 1
        selected_one, _, _ = select_c2f_adaptive(cfg, base, tables, plan)
        for q, links in selected_one.items():
            local_count = sum(is_local_observation(plan, link, q) for link in links)
            self.assertLessEqual(local_count, 1)

    def test_remote_report_hard_cap_is_enforced(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 5, 2
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 5
        cfg.selector.max_links_per_target = 4
        cfg.selector.max_total_links = 8
        cfg.selector.max_local_observations_per_target = 1
        cfg.selector.max_remote_reports = 2
        rng = np.random.default_rng(1704)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        plan = assign_fusion_nodes(cfg, base, tables, geom)
        selected, _, _ = select_c2f_adaptive(cfg, base, tables, plan)
        remote_count = sum(
            not is_local_observation(plan, link, q)
            for q, links in selected.items() for link in links
        )
        self.assertLessEqual(remote_count, 2)

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
        from isac_sim.fusion import fused_h0_skewness

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
