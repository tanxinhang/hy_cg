"""Pin held-out-look nuisance fitting in the hierarchical MAP screen."""

from pathlib import Path


def test_crossfit_gate_fits_only_the_opposite_look():
    text = Path("tools/gate_crossfit_hierarchical_map.py").read_text(encoding="utf-8")
    assert "reference = observations[1 - held_index]" in text
    assert "observations[held_index]" in text
    assert "refine_direct_dd_joint(cfg, [reference])" in text
    assert "EvaluationPartition.from_records" in text
