"""Pin the 400 m factorial gate's receiver/detector axes and split isolation."""

from pathlib import Path


def test_dual_axis_gate_keeps_cancellation_and_detection_axes_explicit():
    text = Path("tools/gate_area400_dual_axis.py").read_text(encoding="utf-8")
    assert '"refined_tp_uic"' in text
    assert "DEFAULT_CPI_COUNTS = (1, 2, 4)" in text
    assert 'parser.add_argument("--cpi-counts"' in text
    assert '"oracle_auc_gap"' in text
    assert '"oracle_pd_gap"' in text
    assert "EvaluationPartition.from_records" in text
    assert "partition.calibration.rows" in text
    assert "partition.test.rows" in text
