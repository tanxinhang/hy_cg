import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_global_pfa_calibration_pilot as pilot  # noqa: E402


def test_conformal_threshold_has_expected_resolution():
    scores = np.arange(64.0)
    assert np.isinf(pilot._conformal_threshold(scores, 0.01))
    assert pilot._conformal_threshold(scores, 0.05) == 61.0


def test_quantization_respects_bit_budget():
    x = np.linspace(0, 20, 101)
    assert len(np.unique(pilot._quantize(x, 2))) == 4
    assert np.allclose(pilot._quantize(x, 16), x)
