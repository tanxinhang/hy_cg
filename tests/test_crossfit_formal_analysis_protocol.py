"""Pin scene-paired uncertainty analysis for the formal hierarchical MAP gate."""

from pathlib import Path


def test_formal_analysis_resamples_whole_test_scenes():
    text = Path("tools/analyze_crossfit_formal.py").read_text(encoding="utf-8")
    assert 'counts != {"train": 1, "calibration": 40, "test": 40}' in text
    assert 'test = [r for r in records if r["split"] == "test"]' in text
    assert "sample = [test[i]" in text
    assert '"map_minus_base_auc"' in text
    assert '"map_minus_base_pd"' in text
