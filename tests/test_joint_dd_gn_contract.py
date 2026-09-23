"""Pin the bounded four-evaluation GN replacement for the coarse DD grid."""

from pathlib import Path


def test_gn_solver_keeps_grid_box_and_validated_budget():
    text = Path("isac_sim/receiver/cancellation/joint_dd_gn.py").read_text(
        encoding="utf-8")
    assert "max_nfev: int = 4" in text
    assert "bounds=(lower, upper)" in text
    assert 'method="trf"' in text
    assert "offsets / scale" in text
