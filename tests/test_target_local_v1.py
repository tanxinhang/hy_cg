from __future__ import annotations

import dataclasses
import json
import unittest

import numpy as np

from isac_sim.belief import geometry_robust_base
from isac_sim.config import (
    Config,
    HEADLINE_RELEASE_PRESET,
    PRESETS,
    apply_preset,
    iter_leaf_paths,
    validate_config,
)
from isac_sim.model import build_base_gains, compute_link_tables, generate_geometry
from isac_sim.reporting import assign_fusion_nodes
from isac_sim.reporting import report_dest
from isac_sim.oracle import (
    joint_fusion_selection_oracle,
    lexicographic_gap_components,
    lexicographic_objective_vector,
)
from isac_sim.bundle_master import (
    joint_bundle_column_generation,
    joint_bundle_restricted_master,
    rcs_robust_bundle_column_generation,
)
from isac_sim.report import scalar_summary_row
from isac_sim.selection import (
    DEFAULT_METHODS,
    feasible_links_for_target,
    polish_risk_secondary_pd,
    polish_worst_target_pd,
)
from isac_sim.fusion import predicted_pd_for_links
from isac_sim.simulate import run_simulation
from isac_sim.simulate import rng_for_observation, run_one_trial
from isac_sim.soft_channel import draw_received_soft_vector


class TargetLocalV1ContractTests(unittest.TestCase):
    def test_risk_secondary_polish_preserves_nominal_service(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.selector.max_links_per_target = 3
        cfg.selector.max_total_links = 4
        cfg.selector.max_tx_nodes = None
        rng = np.random.default_rng(1002)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        nominal = compute_link_tables(cfg, base)
        risk = compute_link_tables(
            cfg, base,
            residual_fraction_by_receiver=np.asarray([1e-4, 3e-4, 1e-3, 3e-3]),
        )
        plan = assign_fusion_nodes(cfg, base, nominal, geom)
        selected = {
            q: feasible_links_for_target(cfg, base, nominal, q, plan)[:2]
            for q in range(cfg.scale.Q)
        }

        def pd(tables, schedule):
            return np.asarray([
                predicted_pd_for_links(
                    cfg, tables, q, schedule[q], plan=plan, base=base
                ) for q in range(cfg.scale.Q)
            ])

        nominal_before = pd(nominal, selected)
        risk_before = pd(risk, selected)
        polished, history = polish_risk_secondary_pd(
            cfg, base, nominal, risk, selected, plan=plan,
            min_links_per_target=1, max_rounds=3,
            allow_target_pair_exchange=True,
        )
        nominal_after = pd(nominal, polished)
        risk_after = pd(risk, polished)
        weak_req = float(cfg.detect.weak_pd_required)
        self.assertEqual(
            sum(map(len, polished.values())), sum(map(len, selected.values()))
        )
        self.assertGreaterEqual(
            float(np.min(nominal_after)) + 1e-12,
            float(np.min(nominal_before)),
        )
        self.assertGreaterEqual(
            float(np.sum(np.minimum(nominal_after, weak_req))) + 1e-12,
            float(np.sum(np.minimum(nominal_before, weak_req))),
        )
        self.assertGreaterEqual(
            float(np.min(risk_after)) + 1e-12,
            float(np.min(risk_before)),
        )
        for step in history:
            self.assertGreaterEqual(
                step["nominal_worst_after"] + 1e-12,
                step["nominal_worst_before"],
            )
            self.assertGreaterEqual(
                step["nominal_objective_after"] + 1e-12,
                step["nominal_objective_before"],
            )
            self.assertGreaterEqual(
                step["risk_worst_after"] + 1e-12,
                step["risk_worst_before"],
            )

    def test_worst_target_polish_preserves_budget_and_is_monotone(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.selector.max_links_per_target = 3
        cfg.selector.max_total_links = 4
        cfg.selector.max_tx_nodes = None
        rng = np.random.default_rng(1001)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        plan = assign_fusion_nodes(cfg, base, tables, geom)
        selected = {
            q: feasible_links_for_target(cfg, base, tables, q, plan)[:2]
            for q in range(cfg.scale.Q)
        }
        before = np.asarray([
            predicted_pd_for_links(
                cfg, tables, q, selected[q], plan=plan, base=base
            )
            for q in range(cfg.scale.Q)
        ])

        polished, history = polish_worst_target_pd(
            cfg, base, tables, selected, plan=plan,
            min_links_per_target=1, max_rounds=4,
        )
        after = np.asarray([
            predicted_pd_for_links(
                cfg, tables, q, polished[q], plan=plan, base=base
            )
            for q in range(cfg.scale.Q)
        ])

        self.assertEqual(
            sum(map(len, polished.values())), sum(map(len, selected.values()))
        )
        self.assertTrue(all(len(polished[q]) >= 1 for q in polished))
        self.assertGreaterEqual(float(np.min(after)) + 1e-12, float(np.min(before)))
        for step in history:
            self.assertGreaterEqual(
                step["worst_pd_after"] + 1e-12, step["worst_pd_before"]
            )

    def test_v11_preset_removes_scalar_price_without_mutating_v1(self) -> None:
        v1 = apply_preset(Config(), "target-local-v1")
        v11 = apply_preset(Config(), "capacitated-target-fusion-v1.1")

        self.assertEqual(v1.fusion.rule, "nearest_target")
        self.assertTrue(v1.selector.use_delay_price)
        self.assertEqual(v11.fusion.rule, "capacitated_value")
        v12 = apply_preset(Config(), "joint-bundle-v1.2")
        self.assertEqual(v12.detect.comm_error_model, "erasure")
        self.assertFalse(v12.selector.use_delay_price)
        self.assertFalse(v12.selector.require_local_anchor)
        self.assertFalse(v11.selector.use_delay_price)
        self.assertEqual(v11.selector.lambda_c, 0.0)
        self.assertEqual(v11.detect.pd_required, 0.95)
        self.assertEqual(v11.detect.weak_pd_required, 0.80)
        self.assertEqual(HEADLINE_RELEASE_PRESET, "target-local-v1")

    def test_v1_differs_from_paper_only_by_fusion_rule(self) -> None:
        paper = apply_preset(Config(), "paper-canonical")
        v1 = apply_preset(Config(), "target-local-v1")

        paper_leaves = dict(iter_leaf_paths(paper))
        v1_leaves = dict(iter_leaf_paths(v1))
        changed = {
            key: (paper_leaves[key], v1_leaves[key])
            for key in paper_leaves
            if paper_leaves[key] != v1_leaves[key]
        }

        self.assertEqual(
            changed,
            {"fusion.rule": ("max_in_rate", "nearest_target")},
        )
        validate_config(v1)

    def test_v1_preset_does_not_mutate_paper_preset(self) -> None:
        self.assertNotEqual(id(PRESETS["paper-canonical"]), id(PRESETS["target-local-v1"]))
        self.assertNotIn("fusion.rule", PRESETS["paper-canonical"])
        self.assertEqual(PRESETS["target-local-v1"]["fusion.rule"], "nearest_target")

    def test_waveform_phase1_is_isolated_from_frozen_v1(self) -> None:
        v1 = apply_preset(Config(), "target-local-v1")
        phase1 = apply_preset(Config(), "target-local-waveform-v2-phase1")

        v1_leaves = dict(iter_leaf_paths(v1))
        phase1_leaves = dict(iter_leaf_paths(phase1))
        changed = {
            key: (v1_leaves[key], phase1_leaves[key])
            for key in v1_leaves
            if v1_leaves[key] != phase1_leaves[key]
        }
        self.assertEqual(
            changed,
            {
                "detect.fused_calibration_samples": (8192, 16_384),
                "waveform_impairments.enable": (False, True),
            },
        )
        self.assertFalse(v1.waveform_impairments.enable)
        self.assertNotIn(
            "proposed_c2f_adaptive_pd_calibrated", DEFAULT_METHODS
        )
        validate_config(phase1)

    def test_nearest_target_requires_planning_geometry(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 4, 2
        rng = np.random.default_rng(101)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)

        with self.assertRaisesRegex(ValueError, "requires predicted target geometry"):
            assign_fusion_nodes(cfg, base, tables, None)

    def test_nearest_target_rejects_nonfinite_planning_geometry(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 4, 2
        rng = np.random.default_rng(102)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        bad_geom = dataclasses.replace(geom, p_tgt=geom.p_tgt.copy())
        bad_geom.p_tgt[0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "must be finite"):
            assign_fusion_nodes(cfg, base, tables, bad_geom)

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
        from isac_sim.model import rescale_sensing_tables_for_rcs

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

    def test_joint_bundle_is_worker_count_deterministic(self) -> None:
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
        cfg.run.num_mc = 2
        cfg.run.verbose = False

        cfg.run.workers = 1
        serial = run_simulation(cfg, methods=["joint_bundle_cg"])
        cfg.run.workers = 2
        parallel = run_simulation(cfg, methods=["joint_bundle_cg"])

        def normalized(value: object) -> str:
            return json.dumps(
                value,
                sort_keys=True,
                default=lambda item: item.tolist()
                if isinstance(item, np.ndarray) else float(item),
            )

        self.assertEqual(normalized(serial), normalized(parallel))

    def test_v1_is_worker_count_deterministic(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 3
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 4
        cfg.run.num_mc = 3
        cfg.run.verbose = False
        methods = ["proposed_c2f_adaptive_pd", "sense_sinr"]

        cfg.run.workers = 1
        serial = run_simulation(
            cfg, methods=methods, paired_reference="proposed_c2f_adaptive_pd"
        )
        cfg.run.workers = 2
        parallel = run_simulation(
            cfg, methods=methods, paired_reference="proposed_c2f_adaptive_pd"
        )

        def normalized(summary: object) -> str:
            return json.dumps(
                summary,
                sort_keys=True,
                default=lambda value: value.tolist()
                if isinstance(value, np.ndarray)
                else float(value),
            )

        self.assertEqual(normalized(serial), normalized(parallel))

    def test_scalar_stress_rows_keep_capture_and_paired_evidence(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 3
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 4
        cfg.run.num_mc = 2
        cfg.run.verbose = False
        methods = ["proposed_c2f_adaptive_pd", "sense_sinr"]
        summary = run_simulation(
            cfg, methods=methods, paired_reference="proposed_c2f_adaptive_pd"
        )

        row = scalar_summary_row(summary, "sense_sinr", {"error_level": "test"})

        self.assertIn("belief_capture_rate_mean", row)
        self.assertEqual(
            row["paired_reference_method"], "proposed_c2f_adaptive_pd"
        )
        self.assertIn("paired_reference_delta_ci95_low", row)

    def test_trial_records_and_cluster_interval_are_available(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 4, 2
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 3
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 4
        cfg.run.num_mc = 3
        cfg.run.verbose = False
        records: list[dict[str, object]] = []
        summary = run_simulation(
            cfg,
            methods=["proposed_c2f_adaptive_pd"],
            trial_records=records,
        )
        self.assertEqual(len(records), cfg.run.num_mc)
        self.assertEqual([record["trial"] for record in records], [0, 1, 2])
        self.assertTrue(all("selected_links" in record for record in records))
        result = summary["proposed_c2f_adaptive_pd"]
        self.assertIn("P_D_cluster_ci95_low", result)
        self.assertIn("P_FA_cluster_ci95_low", result)
        self.assertLessEqual(
            result["P_D_cluster_ci95_low"], result["P_D_cluster_ci95_high"]
        )
        self.assertLessEqual(
            result["P_FA_cluster_ci95_low"], result["P_FA_cluster_ci95_high"]
        )
        self.assertEqual(result["detection_runtime_mean_ms"], 0.0)

    def test_runtime_measurement_is_explicitly_opt_in(self) -> None:
        cfg = apply_preset(Config(), "target-local-waveform-v2-phase1")
        cfg.scale.M, cfg.scale.Q = 4, 1
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 3
        cfg.selector.max_links_per_target = 2
        cfg.selector.max_total_links = 2
        cfg.run.record_runtime = True
        result = run_one_trial(
            cfg, 0, methods=["proposed_c2f_adaptive_pd_calibrated"]
        )["proposed_c2f_adaptive_pd_calibrated"]
        self.assertGreater(result.detection_runtime_s, 0.0)

    def test_distributed_bids_match_central_selection_with_less_coordination(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 5, 2
        cfg.detect.num_false_per_target = 1
        cfg.refine.shortlist_size = 5
        cfg.selector.max_links_per_target = 3
        cfg.selector.max_total_links = 6
        cfg.selector.max_local_observations_per_target = 1
        cfg.selector.max_remote_reports = 2
        for trial in range(3):
            results = run_one_trial(
                cfg,
                trial,
                methods=[
                    "proposed_c2f_adaptive_pd",
                    "proposed_c2f_adaptive_pd_distributed",
                ],
            )
            central = results["proposed_c2f_adaptive_pd"]
            distributed = results["proposed_c2f_adaptive_pd_distributed"]

            self.assertEqual(central.selected_links, distributed.selected_links)
            np.testing.assert_allclose(
                central.D_fuse_per_target, distributed.D_fuse_per_target
            )
            self.assertEqual(
                central.selector_score_evaluations,
                distributed.selector_score_evaluations,
            )
            self.assertLess(
                distributed.coordination_messages,
                central.coordination_messages,
            )

    def test_calibrated_replay_changes_only_the_detector_threshold(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 5, 2
        cfg.detect.num_false_per_target = 2
        cfg.refine.shortlist_size = 5
        cfg.selector.max_links_per_target = 3
        cfg.selector.max_total_links = 6
        for trial in range(3):
            results = run_one_trial(
                cfg,
                trial,
                methods=[
                    "proposed_c2f_adaptive_pd",
                    "proposed_c2f_adaptive_pd_calibrated",
                ],
            )
            reference = results["proposed_c2f_adaptive_pd"]
            calibrated = results["proposed_c2f_adaptive_pd_calibrated"]
            self.assertEqual(reference.selected_links, calibrated.selected_links)
            self.assertEqual(reference.overhead_bits, calibrated.overhead_bits)
            self.assertEqual(reference.overhead_delay_s, calibrated.overhead_delay_s)
            self.assertEqual(reference.detected, calibrated.detected)
            self.assertEqual(reference.false_alarm, calibrated.false_alarm)

    def test_keyed_detector_draw_is_stable_when_selection_set_changes(self) -> None:
        cfg = apply_preset(Config(), "capacitated-target-fusion-v1.1")
        cfg.scale.M, cfg.scale.Q = 4, 1
        cfg.dd.use_otfs_bin_validity = False
        cfg.corr.enable = False
        rng = np.random.default_rng(1811)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        plan = assign_fusion_nodes(cfg, base, tables, geom)
        shared = (0, 1)
        short = [shared]
        long = [(2, 1), shared, (0, 3)]

        def keyed(links):
            return {
                link: rng_for_observation(
                    cfg, 7, 0, link, h1=False, draw_index=4
                )
                for link in links
            }

        short_draw = draw_received_soft_vector(
            cfg, tables, short, 0, np.random.default_rng(1), False,
            plan, base, rng_by_link=keyed(short),
        )
        long_draw = draw_received_soft_vector(
            cfg, tables, long, 0, np.random.default_rng(2), False,
            plan, base, rng_by_link=keyed(long),
        )
        self.assertEqual(short_draw[0], long_draw[long.index(shared)])

    def test_geometry_robust_view_is_conservative_and_non_mutating(self) -> None:
        cfg = apply_preset(Config(), "target-local-v1")
        cfg.scale.M, cfg.scale.Q = 4, 2
        rng = np.random.default_rng(1801)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        original = base.target_gain.copy()

        robust = geometry_robust_base(cfg, base)

        self.assertIsNot(robust, base)
        np.testing.assert_array_equal(base.target_gain, original)
        self.assertTrue(np.all(robust.target_gain <= base.target_gain))
        self.assertTrue(np.any(robust.target_gain < base.target_gain))

        cfg.prior.belief_sigma_pos_m = 0.0
        self.assertIs(geometry_robust_base(cfg, base), base)


if __name__ == "__main__":
    unittest.main()
