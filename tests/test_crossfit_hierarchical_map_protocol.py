"""Pin held-out-look nuisance fitting in the hierarchical MAP screen."""

from pathlib import Path


def test_crossfit_gate_fits_only_the_opposite_look():
    text = Path("tools/gate_crossfit_hierarchical_map.py").read_text(encoding="utf-8")
    assert "reference = observations[1 - held_index]" in text
    assert "observations[held_index]" in text
    assert "refine_direct_dd_joint(cfg, [reference])" in text
    assert "refine_direct_dd_joint_gn(" in text
    assert "cfg, [reference], max_nfev=gn_max_nfev" in text
    assert 'choices=("grid", "gn")' in text
    assert 'parser.add_argument("--gn-max-nfev", type=int, default=4)' in text
    assert "EvaluationPartition.from_records" in text
    assert "partition.calibration.rows" in text
    assert '"test_pfa"' in text
    assert '"test_pd"' in text
    assert '"oracle_auc_gap"' in text
