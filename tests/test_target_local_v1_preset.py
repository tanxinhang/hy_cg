"""``target-local-v1`` preset, waveform-isolation and geometry-view contracts.

Split out of ``test_target_local_v1.py`` (2026-09-21 audit, 825 -> 4 files).
Method bodies are moved verbatim; only the file boundary changed.  Covered by
the anti-loss baseline in ``tests/_test_inventory.py``.
"""
from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from isac_sim.scenario.belief import geometry_robust_base
from isac_sim.core.config import (
    Config,
    HEADLINE_RELEASE_PRESET,
    PRESETS,
    apply_preset,
    iter_leaf_paths,
    validate_config,
)
from isac_sim.sensing.model import build_base_gains, compute_link_tables, generate_geometry
from isac_sim.cooperation.reporting import assign_fusion_nodes
from experiments.methods import DEFAULT_METHODS
from experiments.flow.simulate import run_one_trial


class TargetLocalV1PresetTests(unittest.TestCase):
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
