from __future__ import annotations

import unittest
from unittest.mock import patch

from isac_sim.active_system import GlobalActiveMasterResult
from isac_sim.config import Config
from isac_sim.evidence_acquisition import solve_active_evidence_acquisition


class UnifiedEvidenceAcquisitionTests(unittest.TestCase):
    def test_facade_uses_detection_as_operational_layer_and_declares_scope(self) -> None:
        cfg = Config()
        cfg.active_sensing.enable = True
        cfg.detect.soft_stat_model = "llr"
        cfg.detect.comm_error_model = "erasure"
        master = GlobalActiveMasterResult(
            columns=(), objective={"worst_pd": 0.0},
            candidate_column_count=0, exact_over_columns=True,
        )
        with (
            patch("isac_sim.evidence_acquisition.validate_config"),
            patch("isac_sim.evidence_acquisition.configured_sensing_modes", return_value=()),
            patch("isac_sim.evidence_acquisition.generate_active_columns", return_value=[]),
            patch("isac_sim.evidence_acquisition.solve_global_active_master", return_value=master),
        ):
            result = solve_active_evidence_acquisition(cfg, None, None, None)

        self.assertEqual(
            result.scope.operational_metric,
            "worst_scenario_pd_at_calibrated_pfa",
        )
        self.assertEqual(result.scope.global_certificate, "exact_over_generated_columns")
        self.assertEqual(result.scope.power_externality_model, "full_load_envelope")

    def test_facade_rejects_a_surrogate_detector(self) -> None:
        cfg = Config()
        cfg.active_sensing.enable = True
        cfg.detect.soft_stat_model = "gaussian"
        with patch("isac_sim.evidence_acquisition.validate_config"):
            with self.assertRaisesRegex(ValueError, "soft_stat_model='llr'"):
                solve_active_evidence_acquisition(cfg, None, None, None)


if __name__ == "__main__":
    unittest.main()
