from __future__ import annotations

import unittest
from unittest.mock import patch
import json
import tempfile
from pathlib import Path

import numpy as np

from isac_sim.config import Config, apply_overrides, apply_preset, validate_config
from isac_sim.bundle_master import ObservationBundle, solve_restricted_bundle_master
from isac_sim.fusion import compute_weights, fusion_weight_mode_for_method
from isac_sim.llr import llr_h0_offset
from isac_sim.model import build_base_gains, compute_link_tables, generate_geometry
from isac_sim.simulate import run_one_trial
from isac_sim.soft_channel import received_full_llr_moments
from tools.run_fusion_headroom_v14 import _write_artifact_manifest


class ExactLlrFusionTests(unittest.TestCase):
    def test_cpu_budget_replaces_one_target_per_fusion_cap(self) -> None:
        cfg = apply_overrides(Config(), {
            "scale.M": 2,
            "scale.Q": 2,
            "fusion.max_targets_per_uav": -1,
            "fusion.cpu_rate_cycles_per_s": 3.0,
            "fusion.processing_window_s": 1.0,
            "fusion.cpu_fixed_cycles": 1.0,
            "fusion.cpu_per_observation_cycles": 1.0,
            "selector.max_total_links": -1,
        })
        bundles = []
        for q in range(2):
            bundles.extend([
                ObservationBundle(q, 0, (), 0.0, 0, (0, 0)),
                ObservationBundle(q, 0, ((1, 0),), 0.9, 0, (1, 0)),
            ])
        result = solve_restricted_bundle_master(cfg, bundles)
        # One singleton costs c0+c1=2 cycles; two would violate the 3-cycle
        # node budget even though target-count capacity is fully disabled.
        self.assertEqual(sum(len(v) for v in result.selected.values()), 1)
        self.assertLessEqual(result.objective["cpu_cycles"], 3.0)

    def test_exact_llr_offset_recovers_uncentred_formula(self) -> None:
        gamma = 0.37
        looks = 8
        x = 10.5
        centred = gamma / (1.0 + gamma) * (x - looks)
        exact = -looks * np.log1p(gamma) + x * gamma / (1.0 + gamma)
        self.assertAlmostEqual(
            centred + llr_h0_offset(gamma, looks), exact, places=13
        )

    def test_exact_llr_weights_are_unscaled_ones(self) -> None:
        cfg = apply_overrides(Config(), {
            "scale.M": 3,
            "scale.Q": 1,
            "detect.soft_stat_model": "llr",
            "detect.comm_error_model": "erasure",
        })
        rng = np.random.default_rng(3)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        links = [(0, 1), (2, 1)]
        weights = compute_weights(
            cfg, tables, 0, links, mode="exact_llr_sum"
        )
        self.assertEqual(weights, {(0, 1): 1.0, (2, 1): 1.0})
        self.assertEqual(
            fusion_weight_mode_for_method("joint_bundle_cg_exact_llr"),
            "exact_llr_sum",
        )

    def test_full_llr_erasure_moments_include_random_offset(self) -> None:
        cfg = apply_overrides(Config(), {
            "scale.M": 3,
            "scale.Q": 1,
            "detect.soft_stat_model": "llr",
            "detect.comm_error_model": "erasure",
        })
        rng = np.random.default_rng(9)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tables = compute_link_tables(cfg, base)
        moments = received_full_llr_moments(cfg, tables, (0, 1), 0)
        self.assertLessEqual(moments.m0, 0.0)
        self.assertGreaterEqual(moments.gap, 0.0)
        self.assertGreaterEqual(moments.v0, 0.0)

    def test_detector_ablation_uses_identical_bundle_decision(self) -> None:
        cfg = apply_preset(Config(), "small-uav-compact-800m")
        cfg = apply_overrides(cfg, {
            "scale.M": 4,
            "scale.Q": 2,
            "selector.max_links_per_target": 2,
            "selector.max_total_links": 4,
            "selector.max_remote_reports": -1,
            "selector.max_observations_per_receiver": -1,
            "selector.max_observations_per_fusion_uav": -1,
            "fusion.max_targets_per_uav": -1,
            "detect.fused_calibration_samples": 512,
            "detect.comm_error_model": "erasure",
            "run.num_mc": 1,
            "run.verbose": False,
        })
        validate_config(cfg)
        results = run_one_trial(
            cfg, 0, methods=["joint_bundle_cg", "joint_bundle_cg_exact_llr"]
        )
        nominal = results["joint_bundle_cg"]
        exact = results["joint_bundle_cg_exact_llr"]
        self.assertEqual(nominal.selected_links, exact.selected_links)
        np.testing.assert_array_equal(
            nominal.reporting_plan.f_q, exact.reporting_plan.f_q
        )

    def test_bundle_only_roster_builds_refined_value_table(self) -> None:
        cfg = apply_preset(Config(), "small-uav-compact-800m")
        cfg = apply_overrides(cfg, {
            "scale.M": 4,
            "scale.Q": 2,
            "prior.belief_mode": False,
            "dd.use_otfs_bin_validity": False,
            "selector.max_links_per_target": 2,
            "selector.max_total_links": 4,
            "selector.bundle_shortlist_per_type": 2,
            "detect.fused_calibration_samples": 512,
            "detect.comm_error_model": "erasure",
            "detect.num_false_per_target": 1,
            "run.verbose": False,
        })

        with patch(
            "isac_sim.simulate.compute_link_tables", wraps=compute_link_tables
        ) as mocked:
            run_one_trial(cfg, 0, methods=["joint_bundle_cg"])

        self.assertTrue(any(
            call.kwargs.get("dd_gain") is not None
            for call in mocked.call_args_list
        ))

    def test_artifact_manifests_do_not_overwrite_each_other(self) -> None:
        cfg = Config()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            _write_artifact_manifest(out, "main", cfg, {"mc": 200})
            _write_artifact_manifest(out, "detector", cfg, {"mc": 20})

            main = json.loads((out / "main.config.json").read_text())
            detector = json.loads((out / "detector.config.json").read_text())
            self.assertEqual(main["parameters"]["mc"], 200)
            self.assertEqual(detector["parameters"]["mc"], 20)


if __name__ == "__main__":
    unittest.main()
