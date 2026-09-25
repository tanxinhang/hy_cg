#!/usr/bin/env python3
"""Run the preregistered Stage-B variance pilot; do not report calibrated P_D."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import gate_crossfit_target_state_map as experiment  # noqa: E402


ARMS = ("nominal", "wls_map", "local_crossfit", "shared_crossfit", "oracle")


def _args(cli, frozen):
    return SimpleNamespace(
        out=cli.out, target=2, master_seed=20261007, area_xy=400.0,
        uavs=6, targets_count=3, min_uav_separation_m=20.0, m_rx=4,
        target_rcs=0.5, prior_sigma_m=150.0, glrt_radius=0.25,
        glrt_points=3, search_sigma=3.0, coarse_step_m=None,
        solver="coarse_to_fine", screen_receivers=2, screen_keep=40,
        verify_receivers=4, verify_keep=12, top_k=3,
        train_scenes=cli.start_scene + cli.scenes,
        calibration_scenes=0, test_scenes=0, bootstrap=0, p_fa=0.05,
        baseline_curve=True, state_gate=frozen,
    )


def _minimum_separation(path):
    with np.load(path) as snapshot:
        positions = np.asarray(snapshot["truth_p_uav"], dtype=float)
    distance = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1)
    distance[np.eye(len(positions), dtype=bool)] = np.inf
    return float(np.min(distance))


def _read(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarise(rows, radii):
    output = {}
    for radius in radii:
        sample = [r for r in rows if float(r["error_radius_m"]) == radius]
        if not sample:
            continue
        item = {"n": len(sample)}
        for arm in ARMS:
            margins = np.asarray([
                float(r[f"{arm}_h1"]) - float(r[f"{arm}_h0"])
                for r in sample])
            item[arm] = {
                "median_h1_minus_h0": float(np.median(margins)),
                "positive_margin_rate": float(np.mean(margins > 0.0)),
            }
        for name, prefix in (("shared_crossfit", "state_error"),
                             ("wls_map", "wls_state_error")):
            errors = np.asarray([
                np.mean([float(r[f"{prefix}_a_m"]),
                         float(r[f"{prefix}_b_m"])]) for r in sample])
            item[name]["median_state_error_m"] = float(np.median(errors))
            item[name]["improvement_rate_over_nominal"] = float(
                np.mean(errors < radius)) if radius > 0 else float(np.mean(errors == 0))
        accepted = [
            str(r[field]).lower() == "true" for r in sample
            for field in ("state_gate_accepted_a", "state_gate_accepted_b")]
        item["shared_crossfit"]["gate_accept_rate"] = float(np.mean(accepted))
        item["median_runtime_s"] = float(np.median([
            float(r["runtime_s"]) for r in sample]))
        item["minimum_uav_separation_m"] = float(min(
            float(r["minimum_uav_separation_m"]) for r in sample))
        output[str(radius)] = item
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--error-m", default="0,50,150,250")
    parser.add_argument("--start-scene", type=int, default=8)
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    cli = parser.parse_args()
    cli.out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(cli.gate.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise SystemExit("gate must have status=FROZEN")
    args = _args(cli, frozen)
    radii = tuple(float(value) for value in cli.error_m.split(","))
    partial = cli.out / f"partial_shard{cli.shard:02d}.csv"
    rows = _read(partial)
    done = {(float(r["error_radius_m"]), int(r["scene_id"])) for r in rows}
    if not cli.summary_only:
        cfg = experiment._cfg(args)
        scenes = [s for s in range(cli.start_scene, cli.start_scene + cli.scenes)
                  if s % cli.shards == cli.shard]
        for radius in radii:
            for scene in scenes:
                if (radius, scene) in done:
                    continue
                row = experiment._scene_row(cfg, args, scene, radius, args.target)
                snapshot = cli.out / "scenes" / f"scene_{scene:05d}.npz"
                row["minimum_uav_separation_m"] = _minimum_separation(snapshot)
                rows.append(row)
                _write(partial, rows)
                print(f"shard={cli.shard} r={radius / 150.0:.2f} scene={scene} done",
                      flush=True)
        _write(cli.out / f"records_shard{cli.shard:02d}.csv", rows)
        if cli.shards > 1:
            return
    else:
        rows = []
        for shard in range(cli.shards):
            completed = cli.out / f"records_shard{shard:02d}.csv"
            rows.extend(_read(completed if completed.exists() else
                              cli.out / f"partial_shard{shard:02d}.csv"))
    if not rows:
        raise SystemExit("no records found")
    rows.sort(key=lambda r: (float(r["error_radius_m"]), int(r["scene_id"])))
    _write(cli.out / "records.csv", rows)
    payload = {
        "scope": "variance pilot only; no calibrated P_D claim",
        "frozen_gate": str(cli.gate), "radii_m": list(radii),
        "start_scene": cli.start_scene, "requested_scenes": cli.scenes,
        "summary": _summarise(rows, radii),
    }
    (cli.out / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
