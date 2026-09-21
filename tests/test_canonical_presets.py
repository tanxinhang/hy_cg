"""Canonical preset, RCS and scheduler-view contracts.

Split out of ``test_canonical_consistency.py`` (2026-09-21 audit, 752 -> 3 files)
because that file was the second-largest in the suite.  Method bodies are
moved verbatim; only the file boundary changed.  See ``tests/_test_inventory.py``
for the anti-loss baseline that guards this move.
"""
from __future__ import annotations

import copy
import unittest

import numpy as np

from isac_sim.core.config import Config, apply_preset, validate_config
from isac_sim.detection.llr import llr_delta, llr_var0
from isac_sim.sensing.model import (
    build_base_gains,
    compute_link_tables,
    generate_geometry,
    rescale_sensing_tables_for_rcs,
)


class CanonicalPresetTests(unittest.TestCase):
    def test_unsupported_tangent_orders_are_rejected_explicitly(self) -> None:
        cfg = apply_preset(Config(), "paper-canonical")
        cfg.cancellation.tangent_order = 2
        with self.assertRaisesRegex(ValueError, "supports only 0 or 1"):
            validate_config(cfg)

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
        from isac_sim.sensing.model import radar_hardware_gain

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

    def test_invalid_run_detection_correlation_and_cpu_values_are_rejected(self) -> None:
        mutations = [
            ("run.num_mc", lambda cfg: setattr(cfg.run, "num_mc", 0)),
            ("Pfa_target", lambda cfg: setattr(cfg.detect, "Pfa_target", 1.5)),
            ("scale.Q", lambda cfg: setattr(cfg.scale, "Q", 0)),
            ("correlation", lambda cfg: setattr(cfg.corr, "rho_tx", -0.1)),
            ("sum", lambda cfg: setattr(cfg.corr, "rho_tx", 0.8)),
            (
                "cpu_rate_cycles_per_s",
                lambda cfg: setattr(cfg.fusion, "cpu_rate_cycles_per_s", float("nan")),
            ),
        ]
        for expected, mutate in mutations:
            cfg = Config()
            mutate(cfg)
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    validate_config(cfg)

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


if __name__ == "__main__":
    unittest.main()
