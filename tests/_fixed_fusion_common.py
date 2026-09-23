"""Shared fixture writer for fixed-fusion calibration tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def input_records(root: Path, receiver: int, *, test_shift: float = 0.0) -> Path:
    path = root / f"rx{receiver}"
    path.mkdir()
    manifest = {
        "master_seed": 7, "calibration_scenes": 20, "test_scenes": 2,
        "realisations_per_scene_target_receiver": 1,
        "direct_error_scope": "scene",
        "direct_mismatch_covariance_model": "sigma_point",
        "target_glrt_mode": "neighbourhood_max",
        "oracle_direct_residual_in_cres": False,
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fields = ("split", "scene_id", "receiver", "target", "arm",
              "direct_gain_boost_db", "dof_real", "stat_h0", "stat_h1")
    with (path / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scene in range(22):
            split = "calibration" if scene < 20 else "test"
            score = float((scene if split == "calibration" else scene - 20)
                          + receiver + (test_shift if split == "test" else 0))
            writer.writerow({"split": split, "scene_id": scene,
                             "receiver": receiver, "target": 1,
                             "arm": "tp_uic_full", "direct_gain_boost_db": 30,
                             "dof_real": 2, "stat_h0": score,
                             "stat_h1": score + 1})
    return path
