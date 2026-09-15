from __future__ import annotations

import dataclasses
import itertools
import unittest

import numpy as np

from isac_sim.active_information import (
    ActiveObservation,
    active_candidate_links,
    SensingMode,
    aspect_scenario_factors,
    evaluate_active_detection,
    observation_information_matrix,
    price_active_information_bundle,
    received_information,
)
from isac_sim.active_system import (
    ActiveColumn,
    _make_active_column,
    active_column_cpu_cycles,
    evaluate_active_transport_headroom,
    generate_active_columns,
    screen_fusion_candidates,
    solve_global_active_master,
)
from isac_sim.config import Config, apply_overrides, apply_preset, validate_config
from isac_sim.coherent_oracle import (
    coherent_group_gamma,
    coherent_oracle_information,
)
from isac_sim.llr import llr_delta, llr_jeffreys, llr_kld, llr_reverse_kld
from isac_sim.model import build_base_gains, compute_link_tables, generate_geometry
from isac_sim.reporting import ReportingPlan, is_local_observation
from isac_sim.selection import feasible_links_for_target


class InformationTheoryTests(unittest.TestCase):
    def test_jeffreys_identity_matches_centered_llr_gap(self) -> None:
        gamma = np.asarray([0.0, 0.01, 0.2, 3.0])
        looks = 16
        np.testing.assert_allclose(
            llr_jeffreys(gamma, looks),
            llr_kld(gamma, looks) + llr_reverse_kld(gamma, looks),
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            llr_jeffreys(gamma, looks), llr_delta(gamma, looks),
            rtol=1e-13, atol=1e-13,
        )

    def test_observed_true_erasure_scales_information_exactly(self) -> None:
        gamma = 0.7
        looks = 8
        chi = 0.63
        self.assertAlmostEqual(
            received_information(gamma, looks, chi),
            chi * llr_kld(gamma, looks),
            places=14,
        )


class ActiveObservationPricingTests(unittest.TestCase):
    def _problem(self):
        cfg = apply_preset(Config(), "small-uav-compact-800m")
        cfg = apply_overrides(cfg, {
            "scale.M": 3,
            "scale.Q": 1,
            "prior.belief_mode": False,
            "dd.use_otfs_bin_validity": False,
            "active_sensing.enable": True,
            "active_sensing.mode_names": ("eco", "nominal"),
            "active_sensing.power_scales": (0.5, 1.0),
            "active_sensing.looks": (8, 16),
            "active_sensing.refined": (False, True),
            "active_sensing.energy_budget_per_target": 24.0,
            "active_sensing.max_candidates_per_pair": 2,
            "active_sensing.branch_node_limit": 1000,
            "selector.max_links_per_target": 2,
        })
        validate_config(cfg)
        rng = np.random.default_rng(1501)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng, rcs_view="mean")
        coarse = compute_link_tables(cfg, base)
        refined = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
        return cfg, base, coarse, refined

    def test_aspect_scenarios_rank_views_differently(self) -> None:
        cfg, base, _, _ = self._problem()
        angles = base.aspect_azimuth.copy()
        angles[0, 1, 0] = 0.0
        angles[1, 2, 0] = np.pi / 2.0
        controlled = dataclasses.replace(base, aspect_azimuth=angles)
        factors = aspect_scenario_factors(
            cfg, controlled, 0, [(0, 1), (1, 2)]
        )
        self.assertGreater(factors[0, 0], factors[0, 1])
        self.assertLess(factors[2, 0], factors[2, 1])

    def test_lossless_report_oracle_holds_acquisition_fixed(self) -> None:
        cfg, base, coarse, refined = self._problem()
        cfg.detect.soft_stat_model = "llr"
        cfg.detect.comm_error_model = "erasure"
        observation = ActiveObservation(
            (0, 1), SensingMode("nominal", 1.0, 16, True)
        )
        result = evaluate_active_transport_headroom(
            cfg, base, coarse, refined, 0, 2, (observation,),
            np.ones(cfg.scale.M),
            calibration_samples=512,
            evaluation_samples=1024,
            seed=91,
        )
        self.assertEqual(result.current.observations, result.lossless_report.observations)
        np.testing.assert_allclose(
            result.current.scenario_generated_information,
            result.lossless_report.scenario_information,
        )
        self.assertAlmostEqual(result.lossless_report.network_evidence_loss, 0.0)
        self.assertGreaterEqual(
            result.lossless_report.robust_information,
            result.current.robust_information,
        )

    def test_fusion_screen_keeps_information_and_locality_candidates(self) -> None:
        cfg, base, coarse, refined = self._problem()
        screened = screen_fusion_candidates(
            cfg, base, coarse, refined, information_limit=1
        )
        self.assertEqual(set(screened), {0})
        self.assertGreaterEqual(len(screened[0]), 1)
        self.assertLessEqual(len(screened[0]), 2)
        self.assertTrue(all(0 <= fusion < cfg.scale.M for fusion in screened[0]))

    def test_priced_rescue_bundle_survives_narrow_enumeration(self) -> None:
        cfg, base, coarse, refined = self._problem()
        cfg.detect.soft_stat_model = "llr"
        cfg.detect.comm_error_model = "erasure"
        modes = (
            SensingMode("eco", 0.5, 8, False),
            SensingMode("nominal", 1.0, 16, True),
        )
        priced = price_active_information_bundle(
            cfg, base, coarse, refined, 0, 0, modes=modes,
            max_observations=2,
        )
        self.assertGreater(len(priced.observations), 1)
        columns = generate_active_columns(
            cfg, base, coarse, refined, modes=modes, candidate_limit=1,
            allowed_fusions={0: (0,)}, calibration_samples=64,
            evaluation_samples=128, seed=17,
        )
        self.assertIn(
            tuple(priced.observations),
            {column.observations for column in columns},
        )

    def test_branch_and_bound_matches_complete_small_enumeration(self) -> None:
        cfg, base, coarse, refined = self._problem()
        modes = (
            SensingMode("eco", 0.5, 8, False),
            SensingMode("nominal", 1.0, 16, True),
        )
        result = price_active_information_bundle(
            cfg, base, coarse, refined, 0, 0, modes=modes,
            max_observations=2,
        )
        self.assertTrue(result.exact)
        self.assertAlmostEqual(result.certificate_gap, 0.0)

        plan = ReportingPlan(mode="explicit", f_q=np.zeros(1, dtype=int))
        all_links = feasible_links_for_target(cfg, base, coarse, 0, plan)
        all_values = observation_information_matrix(
            cfg, base, coarse, refined, 0, 0, all_links, modes
        )
        links = list(result.candidate_links)
        lookup = [all_links.index(link) for link in links]
        values = all_values[:, lookup, :]
        best = 0.0
        for choices in itertools.product(range(-1, len(modes)), repeat=len(links)):
            selected = [index for index, mode in enumerate(choices) if mode >= 0]
            if len(selected) > 2:
                continue
            energy = sum(modes[choices[index]].energy for index in selected)
            if energy > cfg.active_sensing.energy_budget_per_target:
                continue
            info = np.zeros(values.shape[0])
            reports = 0
            for index in selected:
                info += values[:, index, choices[index]]
                reports += int(not is_local_observation(plan, links[index], 0))
            score = (
                float(np.min(info))
                - cfg.active_sensing.energy_price * energy
                - cfg.active_sensing.report_price * reports
            )
            best = max(best, score)
        self.assertAlmostEqual(result.objective, best, places=12)

    def test_scenario_union_retains_complementary_pair(self) -> None:
        cfg = Config()
        cfg.active_sensing.max_candidates_per_pair = 2
        cfg.active_sensing.scenario_topk_per_scenario = 0
        cfg.active_sensing.complementary_pair_topk = 1
        links = [(0, 1), (0, 2), (1, 2)]
        values = np.asarray([
            [[10.0], [0.0], [6.0]],
            [[0.0], [10.0], [6.0]],
        ])
        retained, _, scope = active_candidate_links(cfg, links, values)
        self.assertEqual(set(retained), {(0, 1), (0, 2)})
        self.assertEqual(scope, "shortlist_union")

    def test_full_candidate_pool_has_complete_scope(self) -> None:
        cfg, base, coarse, refined = self._problem()
        cfg.active_sensing.candidate_strategy = "full"
        cfg.active_sensing.complete_pool_max_links = 20
        result = price_active_information_bundle(
            cfg, base, coarse, refined, 0, 0,
            modes=(SensingMode("nominal", 1.0, 16, True),),
        )
        self.assertEqual(result.certificate_scope, "complete_pool")
        self.assertEqual(result.retained_candidate_links, result.total_feasible_links)

    def test_active_power_changes_desired_signal_and_interference(self) -> None:
        cfg, base, coarse, _ = self._problem()
        scales = np.ones(cfg.scale.M)
        scales[0] = 2.0
        powered = compute_link_tables(
            cfg, base, sensing_power_scale_by_uav=scales
        )
        # Transmitter 0 receives the desired-power benefit on at least one
        # feasible echo; another transmitter's echo at a receiver exposed to
        # node 0 cannot be modelled as desired-only scaling.
        self.assertTrue(np.any(
            powered.gamma_sense[0] > coarse.gamma_sense[0] + 1e-15
        ))
        ratio = np.divide(
            powered.gamma_sense[1], coarse.gamma_sense[1],
            out=np.ones_like(powered.gamma_sense[1]),
            where=coarse.gamma_sense[1] > 0.0,
        )
        self.assertTrue(np.any(ratio < 1.0 - 1e-10))

    def test_mixed_mode_detector_controls_pfa_and_uses_per_mode_looks(self) -> None:
        cfg, base, coarse, refined = self._problem()
        cfg.detect.comm_error_model = "erasure"
        observations = (
            ActiveObservation((0, 1), SensingMode("short", 0.5, 3, False)),
            ActiveObservation((1, 2), SensingMode("long", 1.0, 19, True)),
        )
        result = evaluate_active_detection(
            cfg, base, coarse, refined, 0, 0, observations,
            calibration_samples=20_000, evaluation_samples=20_000, seed=99,
        )
        self.assertEqual(len(result.scenario_pd), 4)
        self.assertLess(max(abs(value - cfg.detect.Pfa_target)
                            for value in result.scenario_pfa), 0.015)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in result.scenario_pd))

    def test_fusion_change_does_not_resample_identical_physical_evidence(self) -> None:
        cfg, base, coarse, refined = self._problem()
        cfg.detect.comm_error_model = "erasure"
        cfg.detect.soft_stat_model = "llr"
        # With chi=1 everywhere, changing only the fusion destination has no
        # physical or reporting consequence and must therefore be bit-exact.
        reliable_coarse = dataclasses.replace(
            coarse, chi_comm=np.ones_like(coarse.chi_comm)
        )
        reliable_refined = dataclasses.replace(
            refined, chi_comm=np.ones_like(refined.chi_comm)
        )
        observations = (
            ActiveObservation((0, 1), SensingMode("short", 0.5, 3, False)),
            ActiveObservation((1, 2), SensingMode("long", 1.0, 19, True)),
        )
        left = evaluate_active_detection(
            cfg, base, reliable_coarse, reliable_refined, 0, 0, observations,
            calibration_samples=2048, evaluation_samples=4096, seed=77,
        )
        right = evaluate_active_detection(
            cfg, base, reliable_coarse, reliable_refined, 0, 2, observations,
            calibration_samples=2048, evaluation_samples=4096, seed=77,
        )
        self.assertEqual(left, right)

    def test_receiver_local_llr_is_exactly_equivalent_when_reports_are_lossless(self) -> None:
        cfg, base, coarse, refined = self._problem()
        cfg.detect.comm_error_model = "erasure"
        cfg.detect.soft_stat_model = "llr"
        reliable_coarse = dataclasses.replace(
            coarse, chi_comm=np.ones_like(coarse.chi_comm)
        )
        reliable_refined = dataclasses.replace(
            refined, chi_comm=np.ones_like(refined.chi_comm)
        )
        observations = (
            ActiveObservation((0, 1), SensingMode("short", 0.5, 3, False)),
            ActiveObservation((2, 1), SensingMode("long", 1.0, 19, True)),
        )
        direct = evaluate_active_detection(
            cfg, base, reliable_coarse, reliable_refined, 0, 2, observations,
            calibration_samples=2048, evaluation_samples=4096, seed=78,
            transport_mode="direct_llr",
        )
        locally_aggregated = evaluate_active_detection(
            cfg, base, reliable_coarse, reliable_refined, 0, 2, observations,
            calibration_samples=2048, evaluation_samples=4096, seed=78,
            transport_mode="receiver_local_llr",
        )
        self.assertEqual(direct, locally_aggregated)

    def test_receiver_local_llr_collapses_reports_and_final_fusion_inputs(self) -> None:
        cfg, base, coarse, refined = self._problem()
        cfg.detect.comm_error_model = "erasure"
        cfg.detect.soft_stat_model = "llr"
        observations = (
            ActiveObservation((0, 1), SensingMode("nominal", 1.0, 16, True)),
            ActiveObservation((2, 1), SensingMode("nominal", 1.0, 16, True)),
        )
        common = dict(
            cfg=cfg, base=base, envelope_coarse=coarse,
            envelope_refined=refined, q=0, fusion=2,
            observations=observations, reference_scales=np.ones(cfg.scale.M),
            calibration_samples=128, evaluation_samples=256, seed=79,
        )
        direct = _make_active_column(**common, transport_mode="direct_llr")
        locally_aggregated = _make_active_column(
            **common, transport_mode="receiver_local_llr"
        )
        self.assertEqual(direct.remote_reports, 2)
        self.assertEqual(locally_aggregated.remote_reports, 1)
        self.assertEqual(direct.fusion_inputs, 2)
        self.assertEqual(locally_aggregated.fusion_inputs, 1)
        self.assertLess(locally_aggregated.fusion_cpu_cycles, direct.fusion_cpu_cycles)
        self.assertAlmostEqual(locally_aggregated.cpu_cycles, direct.cpu_cycles)
        self.assertGreater(
            sum(locally_aggregated.local_aggregation_cpu_cycles), 0.0
        )

    def test_unknown_transport_mode_is_rejected(self) -> None:
        cfg, base, coarse, refined = self._problem()
        cfg.detect.comm_error_model = "erasure"
        with self.assertRaisesRegex(ValueError, "transport_mode"):
            evaluate_active_detection(
                cfg, base, coarse, refined, 0, 0, (),
                transport_mode="raw_iq",
            )

    def test_coherent_oracle_has_no_singleton_gain_and_bounded_phase_gain(self) -> None:
        singleton = coherent_group_gamma(np.asarray([0.2]), 0.0)
        perfect = coherent_group_gamma(np.asarray([0.2, 0.2]), 0.0)
        uncertain = coherent_group_gamma(np.asarray([0.2, 0.2]), 10.0)
        self.assertAlmostEqual(float(singleton), 0.2)
        self.assertAlmostEqual(float(perfect), 0.8)
        self.assertAlmostEqual(float(uncertain), 0.4, places=12)

    def test_coherent_oracle_information_dominates_direct_for_shared_receiver(self) -> None:
        cfg, base, coarse, refined = self._problem()
        cfg.detect.comm_error_model = "erasure"
        observations = (
            ActiveObservation((0, 1), SensingMode("nominal", 1.0, 16, True)),
            ActiveObservation((2, 1), SensingMode("nominal", 1.0, 16, True)),
        )
        result = coherent_oracle_information(
            cfg, base, coarse, refined, 0, 2, observations,
            phase_error_std_rad=0.0,
        )
        self.assertGreaterEqual(
            result.robust_coherent_information,
            result.robust_direct_information,
        )

    def test_negative_coherent_phase_uncertainty_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "phase_error_std_rad"):
            coherent_group_gamma(np.asarray([0.2, 0.2]), -0.1)

    def test_invalid_active_mode_configuration_is_rejected(self) -> None:
        cfg = Config()
        cfg.active_sensing.looks = (8,)
        with self.assertRaisesRegex(ValueError, "equal non-zero lengths"):
            validate_config(cfg)

    def test_fractional_look_count_is_rejected(self) -> None:
        cfg = Config()
        cfg.active_sensing.looks = (8, 16.5, 32)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            validate_config(cfg)


class GlobalActiveMasterTests(unittest.TestCase):
    @staticmethod
    def _column(q: int, pd: float, energy0: float, energy1: float) -> ActiveColumn:
        return ActiveColumn(
            target=q, fusion=0, observations=(),
            scenario_information=(1.0,), scenario_pd=(pd,), scenario_pfa=(0.05,),
            energy_by_tx=(energy0, energy1), tx_load=(0, 0),
            receiver_load=(0, 0), remote_reports=0, cpu_cycles=0.0,
        )

    def test_global_master_enforces_shared_transmitter_energy(self) -> None:
        cfg = Config()
        cfg.scale.M = 2
        cfg.scale.Q = 2
        cfg.active_sensing.energy_budget_per_uav = 10.0
        cfg.active_sensing.aspect_enable = False
        cfg.detect.pd_required = 0.9
        columns = [
            self._column(0, 0.9, 10.0, 0.0),
            self._column(0, 0.8, 0.0, 5.0),
            self._column(1, 0.9, 10.0, 0.0),
            self._column(1, 0.8, 0.0, 5.0),
        ]
        result = solve_global_active_master(cfg, columns)
        self.assertTrue(result.exact_over_columns)
        self.assertAlmostEqual(result.objective["worst_detection_deficit"], 0.1)
        self.assertAlmostEqual(result.objective["total_detection_deficit"], 0.1)
        self.assertLessEqual(sum(c.energy_by_tx[0] for c in result.columns), 10.0)

    def test_refinement_has_explicit_compute_cost(self) -> None:
        cfg = Config()
        coarse = ActiveObservation((0, 1), SensingMode("coarse", 1.0, 8, False))
        refined = ActiveObservation((0, 1), SensingMode("fine", 1.0, 8, True))
        difference = (
            active_column_cpu_cycles(cfg, (refined,))
            - active_column_cpu_cycles(cfg, (coarse,))
        )
        self.assertAlmostEqual(difference, cfg.active_sensing.dd_refine_cycles)

    def test_cpu_budget_is_charged_to_receiver_and_final_fusion(self) -> None:
        cfg = Config()
        cfg.scale.M = 2
        cfg.scale.Q = 1
        cfg.active_sensing.aspect_enable = False
        cfg.fusion.cpu_rate_cycles_per_s = 3.0
        cfg.fusion.processing_window_s = 1.0
        cfg.detect.pd_required = 0.9
        common = dict(
            target=0, fusion=1, observations=(), scenario_information=(1.0,),
            scenario_pfa=(0.05,), energy_by_tx=(0.0, 0.0), tx_load=(0, 0),
            receiver_load=(0, 0), remote_reports=0,
            scenario_generated_information=(1.0,),
            scenario_network_information_loss=(0.0,),
        )
        overloaded_receiver = ActiveColumn(
            **common, scenario_pd=(0.95,), cpu_cycles=5.0,
            receiver_cpu_cycles=(4.0, 0.0), fusion_cpu_cycles=1.0,
        )
        feasible = ActiveColumn(
            **common, scenario_pd=(0.80,), cpu_cycles=2.0,
            receiver_cpu_cycles=(0.0, 1.0), fusion_cpu_cycles=1.0,
        )
        result = solve_global_active_master(cfg, [overloaded_receiver, feasible])
        self.assertEqual(result.columns, (feasible,))
        self.assertLessEqual(result.objective["max_uav_cpu_cycles"], 3.0)

    def test_network_evidence_loss_precedes_energy_tie_break(self) -> None:
        cfg = Config()
        cfg.scale.M = 2
        cfg.scale.Q = 1
        cfg.active_sensing.aspect_enable = False
        cfg.detect.pd_required = 0.9
        high_loss = self._column(0, 0.9, 0.0, 0.0)
        low_loss = ActiveColumn(
            target=0, fusion=0, observations=(),
            scenario_information=(0.9,), scenario_pd=(0.9,), scenario_pfa=(0.05,),
            energy_by_tx=(1.0, 0.0), tx_load=(0, 0), receiver_load=(0, 0),
            remote_reports=0, cpu_cycles=0.0,
            scenario_generated_information=(1.0,),
            scenario_network_information_loss=(0.1,),
        )
        high_loss = ActiveColumn(
            **{**high_loss.__dict__,
               "scenario_generated_information": (1.0,),
               "scenario_network_information_loss": (0.5,)}
        )
        result = solve_global_active_master(cfg, [high_loss, low_loss])
        self.assertEqual(result.columns, (low_loss,))

    def test_design_target_preserves_detection_margin(self) -> None:
        cfg = Config()
        cfg.scale.M = 2
        cfg.scale.Q = 1
        cfg.active_sensing.aspect_enable = False
        cfg.detect.pd_required = 0.95
        low_energy = self._column(0, 0.951, 0.0, 0.0)
        margin = self._column(0, 0.980, 1.0, 0.0)
        ordinary = solve_global_active_master(cfg, [low_energy, margin])
        protected = solve_global_active_master(
            cfg, [low_energy, margin], pd_design_target=0.97
        )
        self.assertEqual(ordinary.columns, (low_energy,))
        self.assertEqual(protected.columns, (margin,))
        self.assertEqual(protected.objective["pd_design_target"], 0.97)


if __name__ == "__main__":
    unittest.main()
