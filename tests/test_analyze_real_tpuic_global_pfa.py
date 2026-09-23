"""Contracts for quantization and scene-count auditing of real TP-UIC scores."""
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import analyze_real_tpuic_global_pfa as analysis  # noqa: E402


def test_quantizer_has_requested_number_of_levels():
    values = np.linspace(0.0, 30.0, 301)
    assert len(np.unique(analysis._quantize(values, 2))) == 4
    assert len(np.unique(analysis._quantize(values, 3))) == 8


def test_high_precision_quantizer_is_identity():
    values = np.array([0.2, 3.7, 99.0])
    assert np.array_equal(analysis._quantize(values, 16), values)


def test_log_scale_shrinkage_interpolates_between_no_and_full_correction():
    ratio = np.array([1.0, 4.0, 9.0])
    half = np.power(ratio, 0.5)
    assert np.allclose(half, [1.0, 2.0, 3.0])
    assert np.all(half <= ratio)
