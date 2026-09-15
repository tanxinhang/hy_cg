from __future__ import annotations

import dataclasses
import itertools
import unittest

import numpy as np

from isac_sim.active_information import (
    SensingMode,
    aspect_scenario_factors,
    observation_information_matrix,
    price_active_information_bundle,
    received_information,
)
from isac_sim.config import Config, apply_overrides, apply_preset, validate_config
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
        rank = np.max(np.min(all_values, axis=0), axis=1)
        keep = np.argsort(-rank, kind="stable")[:2]
        links = [all_links[int(index)] for index in keep]
        values = all_values[:, keep, :]
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


if __name__ == "__main__":
    unittest.main()
