"""``target-local-v1`` joint oracle, bundle master and column-generation contracts.

Split out of ``test_target_local_v1.py`` (2026-09-21 audit).  Method bodies are
moved verbatim; only the file boundary changed.  Covered by the anti-loss
baseline in ``tests/_test_inventory.py``.
"""
from __future__ import annotations

import unittest

import numpy as np

from isac_sim.core.config import Config, apply_preset
from isac_sim.sensing.model import build_base_gains, compute_link_tables, generate_geometry
from isac_sim.detection.oracle import (
    joint_fusion_selection_oracle,
    lexicographic_gap_components,
    lexicographic_objective_vector,
)
from experiments.exploratory.bundle_master import (
    joint_bundle_column_generation,
    joint_bundle_restricted_master,
    rcs_robust_bundle_column_generation,
)
from experiments.flow.simulate import run_one_trial


class TargetLocalV1BundleTests(unittest.TestCase):
    def test_joint_oracle_obeys_lexicographic_hard_budgets(self) -> None:
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        cfg.scale.M, cfg.scale.Q = 3, 2
        cfg.fusion.max_targets_per_uav = 1
        cfg.selector.max_links_per_target = 1
        cfg.selector.max_total_links = 2
        cfg.selector.max_remote_reports = 0
        cfg.selector.max_observations_per_receiver = 1
        cfg.selector.max_observations_per_fusion_uav = 1
        rng = np.random.default_rng(112)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)

        plan, selected, metrics = joint_fusion_selection_oracle(cfg, base, tables)

        self.assertEqual(metrics["remote_reports"], 0.0)
        self.assertIn("total_detection_deficit", metrics)
        self.assertLessEqual(metrics["processing_load"], 2.0)
        self.assertEqual(len(set(plan.f_q.tolist())), cfg.scale.Q)
        self.assertTrue(all(len(links) <= 1 for links in selected.values()))

        candidate = lexicographic_objective_vector(
            cfg, base, tables, plan, selected
        )
        gaps = lexicographic_gap_components(candidate, metrics)
        self.assertEqual(gaps["exact_lexicographic_match"], 1.0)
        self.assertTrue(
            all(value == 0.0 for key, value in gaps.items() if key.startswith("delta_"))
        )

    def test_joint_oracle_enforces_per_target_local_observation_cap(self) -> None:
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        cfg.scale.M, cfg.scale.Q = 3, 1
        cfg.dd.use_otfs_bin_validity = False
        cfg.selector.max_links_per_target = 3
        cfg.selector.max_total_links = 3
        cfg.selector.max_local_observations_per_target = 1
        cfg.selector.max_remote_reports = 3
        cfg.selector.max_observations_per_receiver = 3
        cfg.selector.max_observations_per_fusion_uav = 3
        rng = np.random.default_rng(113)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)

        plan, selected, _ = joint_fusion_selection_oracle(cfg, base, tables)

        for q, links in selected.items():
            local_count = sum(link[1] == int(plan.f_q[q]) for link in links)
            self.assertLessEqual(local_count, 1)

    def test_restricted_bundle_master_matches_joint_oracle_on_full_small_pool(self) -> None:
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        cfg.scale.M, cfg.scale.Q = 3, 2
        cfg.dd.use_otfs_bin_validity = False
        cfg.fusion.max_targets_per_uav = 1
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 3
        cfg.selector.max_local_observations_per_target = 1
        cfg.selector.max_remote_reports = 1
        cfg.selector.max_observations_per_receiver = 2
        cfg.selector.max_observations_per_fusion_uav = 2
        rng = np.random.default_rng(117)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)

        _, _, oracle = joint_fusion_selection_oracle(cfg, base, tables)
        master = joint_bundle_restricted_master(
            cfg, base, tables, shortlist_per_type=20
        )

        for name, value in oracle.items():
            self.assertAlmostEqual(master.objective[name], value, places=8)

    def test_column_generation_matches_oracle_with_exact_small_pricing(self) -> None:
        cfg = apply_preset(Config(), "joint-bundle-v1.2")
        cfg.scale.M, cfg.scale.Q = 3, 2
        cfg.dd.use_otfs_bin_validity = False
        cfg.fusion.max_targets_per_uav = 1
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 3
        cfg.selector.max_local_observations_per_target = 1
        cfg.selector.max_remote_reports = 1
        cfg.selector.max_observations_per_receiver = 2
        cfg.selector.max_observations_per_fusion_uav = 2
        cfg.selector.bundle_shortlist_per_type = 20
        cfg.selector.bundle_exact_pricing_max_candidates = 20
        rng = np.random.default_rng(117)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)

        _, _, oracle = joint_fusion_selection_oracle(cfg, base, tables)
        master = joint_bundle_column_generation(cfg, base, tables)

        for name, value in oracle.items():
            self.assertAlmostEqual(master.objective[name], value, places=8)
        self.assertGreater(master.column_count, cfg.scale.M * cfg.scale.Q)
        self.assertGreaterEqual(master.pricing_iterations, 1)

    def test_rcs_robust_bundle_is_nominal_at_unit_lower_factor(self) -> None:
        cfg = apply_preset(Config(), "joint-bundle-v1.2")
        cfg.scale.M, cfg.scale.Q = 3, 2
        cfg.dd.use_otfs_bin_validity = False
        cfg.fusion.max_targets_per_uav = 1
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 3
        cfg.selector.max_local_observations_per_target = 1
        cfg.selector.max_remote_reports = 1
        cfg.selector.max_observations_per_receiver = 2
        cfg.selector.max_observations_per_fusion_uav = 2
        cfg.selector.bundle_shortlist_per_type = 20
        cfg.selector.bundle_exact_pricing_max_candidates = 20
        cfg.prior.rcs_lower_factor = 1.0
        rng = np.random.default_rng(121)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng, rcs_view="mean")
        tables = compute_link_tables(cfg, base)

        nominal = joint_bundle_column_generation(cfg, base, tables)
        robust = rcs_robust_bundle_column_generation(cfg, base, tables)

        self.assertEqual(robust.plan.f_q.tolist(), nominal.plan.f_q.tolist())
        self.assertEqual(robust.selected, nominal.selected)
        self.assertEqual(robust.objective, nominal.objective)

    def test_bundle_column_generation_honours_allowed_transmitters(self) -> None:
        cfg = apply_preset(Config(), "joint-bundle-v1.2")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.dd.use_otfs_bin_validity = False
        cfg.selector.max_links_per_target = 3
        cfg.selector.max_total_links = 5
        cfg.selector.bundle_shortlist_per_type = 20
        cfg.selector.bundle_exact_pricing_max_candidates = 20
        rng = np.random.default_rng(122)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng, rcs_view="mean")
        tables = compute_link_tables(cfg, base)

        result = rcs_robust_bundle_column_generation(
            cfg, base, tables, allowed_transmitters=(0, 2)
        )

        self.assertTrue(any(result.selected.values()))
        self.assertTrue(all(
            int(link[0]) in {0, 2}
            for links in result.selected.values() for link in links
        ))

    def test_rcs_robust_column_generation_matches_full_lower_endpoint_pool(self) -> None:
        from isac_sim.sensing.model import rescale_sensing_tables_for_rcs

        cfg = apply_preset(Config(), "joint-bundle-v1.2")
        cfg.scale.M, cfg.scale.Q = 3, 2
        cfg.dd.use_otfs_bin_validity = False
        cfg.fusion.max_targets_per_uav = 1
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 3
        cfg.selector.max_local_observations_per_target = 1
        cfg.selector.max_remote_reports = 1
        cfg.selector.max_observations_per_receiver = 2
        cfg.selector.max_observations_per_fusion_uav = 2
        cfg.selector.bundle_shortlist_per_type = 20
        cfg.selector.bundle_exact_pricing_max_candidates = 20
        cfg.prior.rcs_lower_factor = 0.5
        rng = np.random.default_rng(123)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng, rcs_view="mean")
        tables = compute_link_tables(cfg, base)
        lower = rescale_sensing_tables_for_rcs(
            cfg, tables, cfg.prior.rcs_lower_factor
        )

        full = joint_bundle_restricted_master(
            cfg, base, lower, shortlist_per_type=20
        )
        robust = rcs_robust_bundle_column_generation(cfg, base, tables)

        for name, value in full.objective.items():
            self.assertAlmostEqual(robust.objective[name], value, places=8)

    def test_joint_bundle_method_runs_with_its_own_feasible_plan(self) -> None:
        cfg = apply_preset(Config(), "joint-bundle-v1.2")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.dd.use_otfs_bin_validity = False
        cfg.fusion.max_targets_per_uav = 1
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 3
        cfg.selector.max_local_observations_per_target = 1
        cfg.selector.max_remote_reports = 1
        cfg.selector.max_observations_per_receiver = 2
        cfg.selector.max_observations_per_fusion_uav = 2
        cfg.detect.num_false_per_target = 2
        cfg.refine.shortlist_size = 4

        result = run_one_trial(
            cfg, 0, methods=["joint_bundle_cg"]
        )["joint_bundle_cg"]

        self.assertEqual(len(result.reporting_plan.f_q), cfg.scale.Q)
        self.assertEqual(len(set(result.reporting_plan.f_q.tolist())), cfg.scale.Q)
        self.assertLessEqual(
            sum(
                link[1] != int(result.reporting_plan.f_q[q])
                for q, links in result.selected_links.items() for link in links
            ),
            cfg.selector.max_remote_reports,
        )
        self.assertGreater(result.bundle_column_count, 0.0)


if __name__ == "__main__":
    unittest.main()
