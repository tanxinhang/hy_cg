"""``target-local-v1`` fusion-node allocation, capacity and weak-target contracts.

Split out of ``test_target_local_v1.py`` (2026-09-21 audit).  Method bodies are
moved verbatim; only the file boundary changed.  Covered by the anti-loss
baseline in ``tests/_test_inventory.py``.
"""
from __future__ import annotations

import unittest

import numpy as np

from isac_sim.core.config import Config, apply_preset
from isac_sim.sensing.model import build_base_gains, compute_link_tables, generate_geometry
from isac_sim.cooperation.reporting import assign_fusion_nodes
from isac_sim.cooperation.reporting import report_dest
from experiments.flow.simulate import run_one_trial, run_simulation


class TargetLocalV1AllocationTests(unittest.TestCase):
    def test_capacitated_assignment_respects_per_uav_target_capacity(self) -> None:
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        cfg.scale.M, cfg.scale.Q = 3, 2
        cfg.fusion.max_targets_per_uav = 1
        cfg.selector.max_links_per_target = 1
        rng = np.random.default_rng(111)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)

        plan = assign_fusion_nodes(cfg, base, tables, geom)

        self.assertEqual(len(set(plan.f_q.tolist())), cfg.scale.Q)

    def test_capacity_constrained_nearest_assignment_respects_target_cap(self) -> None:
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        cfg.fusion.rule = "nearest_target_capacitated"
        cfg.scale.M, cfg.scale.Q = 3, 2
        cfg.fusion.max_targets_per_uav = 1
        rng = np.random.default_rng(113)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)

        plan = assign_fusion_nodes(cfg, base, tables, geom)

        self.assertEqual(len(set(plan.f_q.tolist())), cfg.scale.Q)

    def test_pd_lookahead_assignment_respects_target_cap(self) -> None:
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        cfg.fusion.rule = "capacitated_pd_lookahead"
        cfg.scale.M, cfg.scale.Q = 3, 2
        cfg.fusion.max_targets_per_uav = 1
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_remote_reports = 1
        cfg.selector.max_observations_per_fusion_uav = 2
        cfg.selector.candidate_topk_per_target = 4
        rng = np.random.default_rng(114)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)

        plan = assign_fusion_nodes(cfg, base, tables, geom)

        self.assertEqual(len(set(plan.f_q.tolist())), cfg.scale.Q)

    def test_v11_selection_enforces_receiver_and_fusion_processing_caps(self) -> None:
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.fusion.max_targets_per_uav = 1
        cfg.selector.max_links_per_target = 3
        cfg.selector.max_total_links = 6
        cfg.selector.max_remote_reports = 2
        cfg.selector.max_observations_per_receiver = 1
        cfg.selector.max_observations_per_fusion_uav = 2
        cfg.refine.shortlist_size = 5
        cfg.detect.num_false_per_target = 1

        result = run_one_trial(
            cfg, 0, methods=["proposed_c2f_adaptive_pd"]
        )["proposed_c2f_adaptive_pd"]
        rx_counts = np.zeros(cfg.scale.M, dtype=int)
        fusion_counts = np.zeros(cfg.scale.M, dtype=int)
        for q, links in result.selected_links.items():
            for link in links:
                rx_counts[link[1]] += 1
                fusion_counts[report_dest(result.reporting_plan, link, q)] += 1
        self.assertTrue(np.all(rx_counts <= 1))
        self.assertTrue(np.all(fusion_counts <= 2))

    def test_local_anchor_precedes_remote_allocation(self) -> None:
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        cfg.fusion.rule = "capacitated_pd_lookahead"
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.fusion.max_targets_per_uav = 1
        cfg.selector.require_local_anchor = True
        cfg.selector.max_local_observations_per_target = 1
        cfg.selector.max_remote_reports = 2
        cfg.selector.max_observations_per_receiver = 2
        cfg.selector.max_observations_per_fusion_uav = 2
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 4

        result = run_one_trial(
            cfg, 0, methods=["proposed_c2f_adaptive_pd"]
        )["proposed_c2f_adaptive_pd"]

        self.assertTrue(all(
            any(link[1] == report_dest(result.reporting_plan, link, q) for link in links)
            for q, links in result.selected_links.items()
        ))

    def test_weak_target_is_fixed_before_method_specific_selection(self) -> None:
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.selector.max_local_observations_per_target = 1
        cfg.selector.max_remote_reports = 2
        cfg.selector.max_observations_per_receiver = 2
        cfg.selector.max_observations_per_fusion_uav = 2
        cfg.fusion.max_targets_per_uav = 1
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 3
        results = run_one_trial(
            cfg,
            0,
            methods=["proposed_c2f_adaptive_pd", "sense_sinr_budgeted"],
        )
        proposed = results["proposed_c2f_adaptive_pd"]
        baseline = results["sense_sinr_budgeted"]
        self.assertEqual(proposed.weak_target_index, baseline.weak_target_index)
        self.assertIn(proposed.weak_target_detected, {0, 1})
        self.assertLessEqual(
            sum(
                link[1] != report_dest(baseline.reporting_plan, link, q)
                for q, links in baseline.selected_links.items() for link in links
            ),
            2,
        )

    def test_capacity_activity_metrics_are_exported(self) -> None:
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.selector.max_local_observations_per_target = 1
        cfg.selector.max_remote_reports = 1
        cfg.selector.max_observations_per_receiver = 1
        cfg.selector.max_observations_per_fusion_uav = 2
        cfg.fusion.max_targets_per_uav = 1
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 3
        cfg.run.num_mc = 2
        cfg.run.verbose = False
        summary = run_simulation(
            cfg, methods=["proposed_c2f_adaptive_pd"]
        )["proposed_c2f_adaptive_pd"]
        self.assertIn("P_D_weak", summary)
        self.assertIn("receiver_capacity_binding_rate", summary)
        self.assertGreaterEqual(summary["assignment_capacity_binding_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
