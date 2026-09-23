"""Ensure quantization precedes final calibration and no test leakage occurs."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_quantized_fixed_fusion import quantize, run  # noqa: E402
from _fixed_fusion_common import input_records as _input  # noqa: E402


def test_quantizer_and_fixed_payload():
    assert quantize(0.0, 3, 8.0) == 0.0
    assert quantize(99.0, 3, 8.0) == 8.0
    with pytest.raises(ValueError):
        quantize(-1.0, 3, 8.0)


def test_calibration_ignores_test_scores(tmp_path):
    a = _input(tmp_path, 0)
    b = _input(tmp_path, 1)
    original = run([a, b], clip=64.0)
    path = b / "records.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["split"] == "test":
            row["stat_h0"] = "1000000"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    changed = run([a, b], clip=64.0)
    assert original["threshold"] == changed["threshold"]
    assert original["conformal_order"] == 20
    assert original["total_payload_bits_per_target"] == 6
    assert changed["test_false_alarms"] > original["test_false_alarms"]


def test_insufficient_scene_count_gives_infinite_gate(tmp_path):
    a = _input(tmp_path, 0)
    b = _input(tmp_path, 1)
    assert run([a, b], alpha=0.001)["threshold"] == float("inf")
