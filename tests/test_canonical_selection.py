"""Canonical reporting, selection-cap and end-to-end detection contracts.

Split out of ``test_canonical_consistency.py`` (2026-09-21 audit).  Method bodies
are moved verbatim; only the file boundary changed.  Covered by the anti-loss
baseline in ``tests/_test_inventory.py``.
"""
from __future__ import annotations

import copy
import unittest

import numpy as np

from isac_sim.core.config import Config, apply_preset
from isac_sim.sensing.model import (
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from audits.packetization import packetization_audit
from isac_sim.cooperation.reporting import ReportingPlan
from isac_sim.cooperation.reporting import assign_fusion_nodes, is_local_observation
from isac_sim.sensing.soft_channel import (
    local_moments,
    received_moments,
)
from experiments.selection import feasible_links_for_target, select_c2f_adaptive
from experiments.flow.simulate import (
    evaluate_detection,
    rng_for_detection,
    rng_for_method,
    run_method_on_trial,
    run_simulation,
    total_overhead_bits,
    total_overhead_delay_s,
)


class CanonicalSelectionTests(unittest.TestCase):
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

    def test_detection_h1_repetitions_preserve_target_denominator(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.scale.M, cfg.scale.Q = 3, 1
        cfg.detect.num_h1_per_target = 5
        cfg.detect.num_false_per_target = 2
        cfg.dd.use_otfs_bin_validity = False
        rng = np.random.default_rng(3901)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        plan = ReportingPlan(mode="explicit", f_q=np.array([1]))
        got = evaluate_detection(
            cfg,
            tables,
            {0: [(0, 1)]},
            rng_for_detection(cfg, 0),
            "proposed_c2f_adaptive_pd",
            plan,
            base,
            trial_index=0,
        )
        detected, total, _false, total_false, *_rest, per_target = got
        self.assertEqual(total, 5)
        self.assertEqual(total_false, 2)
        self.assertEqual(int(per_target[0]), int(detected))
        self.assertGreaterEqual(detected, 0)
        self.assertLessEqual(detected, total)

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


if __name__ == "__main__":
    unittest.main()
