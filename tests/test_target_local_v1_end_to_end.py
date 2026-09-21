"""``target-local-v1`` polish, determinism and end-to-end method contracts.

Split out of ``test_target_local_v1.py`` (2026-09-21 audit).  Method bodies are
moved verbatim; only the file boundary changed.  Covered by the anti-loss
baseline in ``tests/_test_inventory.py``.
"""
from __future__ import annotations

import json
import unittest

import numpy as np

from isac_sim.core.config import Config, apply_preset
from isac_sim.sensing.model import build_base_gains, compute_link_tables, generate_geometry
from isac_sim.cooperation.reporting import assign_fusion_nodes
from isac_sim.cooperation.primitives import feasible_links_for_target
from experiments.selection import polish_risk_secondary_pd, polish_worst_target_pd
from isac_sim.detection.fusion import predicted_pd_for_links
from experiments.app.report import scalar_summary_row
from experiments.flow.simulate import run_simulation
from experiments.flow.simulate import rng_for_observation, run_one_trial
from isac_sim.sensing.soft_channel import draw_received_soft_vector


class TargetLocalV1EndToEndTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
