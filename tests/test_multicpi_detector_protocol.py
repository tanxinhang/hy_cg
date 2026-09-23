"""Noncoherent CPI accumulation must preserve one scene-level exchangeable row."""

from isac_sim.detection.evaluation_split import EvaluationPartition


def test_multicpi_scores_are_nested_inside_unique_scene_rows():
    rows = []
    for role, offset in (("train", 0), ("calibration", 10), ("test", 20)):
        rows.append({
            "split": role, "scene_id": offset,
            "one_cpi_h0": 1.0, "two_cpi_h0": 2.0,
        })
    partition = EvaluationPartition.from_records(rows, require_train=True)
    assert len(partition.train) == len(partition.calibration) == len(partition.test) == 1
