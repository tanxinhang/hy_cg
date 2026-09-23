"""Protocol guards for the receiver-target generalization gate."""

from argparse import Namespace

from tools.gate_tpuic_generalization import _evaluate


def test_small_runs_cannot_claim_formal_generalization():
    args = Namespace(calibration_scenes=4, test_scenes=4, p_fa=0.05)
    rows = []
    for scene, split in enumerate(["train"] + ["calibration"] * 4 + ["test"] * 4):
        row = {"scene_id": scene, "split": split, "boost_db": 50.0,
               "receiver": 0, "target": 0}
        for name in ("two_cpi", "crossfit_map", "perfect_two_cpi"):
            row[f"{name}_h0"] = float(scene)
            row[f"{name}_h1"] = float(scene + 1)
        rows.append(row)
    _, _, decision = _evaluate(rows, args)
    assert decision == {"eligible": False, "status": "screen_only"}
