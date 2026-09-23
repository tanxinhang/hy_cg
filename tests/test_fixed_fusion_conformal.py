"""Check the split boundary of the fixed post-fusion conformal gate."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_fixed_fusion_conformal import run  # noqa: E402
from _fixed_fusion_common import input_records as _input  # noqa: E402


def test_postfusion_threshold_uses_calibration_scenes_only(tmp_path):
    a = _input(tmp_path, 0)
    b = _input(tmp_path, 1)
    original = run([a, b])
    rows_path = b / "records.csv"
    with rows_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["split"] == "test":
            row["stat_h0"] = str(1000000 + int(row["scene_id"]))
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    changed = run([a, b])
    assert original["conformal_order"] == 20
    assert original["threshold"] == 39.0
    assert changed["threshold"] == original["threshold"]
    assert changed["test_false_alarms"] > original["test_false_alarms"]


def test_postfusion_rejects_reused_receiver(tmp_path):
    a = _input(tmp_path, 0)
    with pytest.raises(ValueError, match="distinct"):
        run([a, a])
