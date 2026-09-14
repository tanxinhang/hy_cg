from __future__ import annotations

import dataclasses
import json
import unittest

import numpy as np

from isac_sim.config import Config, PRESETS, apply_preset, iter_leaf_paths, validate_config
from isac_sim.model import build_base_gains, compute_link_tables, generate_geometry
from isac_sim.reporting import assign_fusion_nodes
from isac_sim.report import scalar_summary_row
from isac_sim.simulate import run_simulation


class TargetLocalV1ContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
