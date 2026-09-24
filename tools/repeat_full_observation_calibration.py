#!/usr/bin/env python3
"""Add a paired H0 calibration repeat to the frozen full-observation gate."""
from __future__ import annotations

import argparse
import csv
import json
from copy import copy
from pathlib import Path

import numpy as np

from isac_sim.receiver import cancellation as cx
from tools import gate_crossfit_hierarchical_map as single
from tools import gate_full_observation_detector as gate
from tools import run_tpuic_receiver_benchmark as bench


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [{k: (v if k == "split" else float(v)) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = single._cfg(args)
    repeat_path = args.out / "repeat_h0.csv"
    rows = _read(repeat_path) if repeat_path.exists() else []
    if len(rows) != args.repeat_scenes:
        rows = []
        for scene in range(args.start_scene, args.start_scene + args.repeat_scenes):
            truth, belief, base0 = bench._scene(
                cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz")
            row = {"scene_id": scene}
            for boost in args.boosts:
                base = bench._boost_direct(base0, boost)
                pairs = [single._pair(cfg, truth, belief, base, copy(args), scene, look)
                         for look in range(2)]
                reference, held = pairs[1][1], pairs[0][1]
                sources = cx.refine_direct_dd_joint_gn(
                    cfg, [reference], max_nfev=args.gn_max_nfev)
                refined = cx.apply_direct_dd(cfg, held, sources)
                row[f"nominal_{boost:g}_h0"] = gate._score(
                    cfg, refined, args.target, "tp_uic_full")
                row[f"perfect_{boost:g}_h0"] = gate._score(
                    cfg, held, args.target, "perfect_channel")
            rows.append(row)
        _write(repeat_path, rows)
    evaluation = {}
    for boost in args.boosts:
        original = _read(Path(str(args.original_pattern).format(boost=f"{boost:g}")))
        cal = [float(r["nominal_h0"]) for r in original if r["split"] == "calibration"]
        cal += [float(r[f"nominal_{boost:g}_h0"]) for r in rows]
        perfect = [float(r["perfect_h0"]) for r in original
                   if r["split"] == "calibration"]
        perfect += [float(r[f"perfect_{boost:g}_h0"]) for r in rows]
        test = [r for r in original if r["split"] == "test"]
        nt = bench._calibrated_threshold(np.asarray(cal), args.p_fa, "split_conformal")
        pt = bench._calibrated_threshold(np.asarray(perfect), args.p_fa, "split_conformal")
        old_n = np.asarray([float(r["nominal_h0"]) for r in original
                            if r["split"] == "calibration"])
        old_p = np.asarray([float(r["perfect_h0"]) for r in original
                            if r["split"] == "calibration"])
        new_n = np.asarray([float(r[f"nominal_{boost:g}_h0"]) for r in rows])
        new_p = np.asarray([float(r[f"perfect_{boost:g}_h0"]) for r in rows])
        evaluation[f"{boost:g}"] = {
            "nominal": {"threshold": nt, **gate._metrics(test, "nominal", nt)},
            "perfect": {"threshold": pt, **gate._metrics(test, "perfect", pt)},
            "calibration_scenes": len(cal),
            "independent_batch_thresholds": {
                "original_nominal": bench._calibrated_threshold(
                    old_n, args.p_fa, "split_conformal"),
                "original_perfect": bench._calibrated_threshold(
                    old_p, args.p_fa, "split_conformal"),
                "repeat_nominal": bench._calibrated_threshold(
                    new_n, args.p_fa, "split_conformal"),
                "repeat_perfect": bench._calibrated_threshold(
                    new_p, args.p_fa, "split_conformal"),
            },
        }
    payload = {"protocol": vars(args) | {"out": str(args.out)},
               "evaluation": evaluation}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original-pattern", default=(
        "studies/direction3/data/full_observation_detector_formal_"
        "boost{boost}_20261014/records.csv"))
    parser.add_argument("--boosts", type=float, nargs="+", default=[10, 30, 50])
    parser.add_argument("--start-scene", type=int, default=81)
    parser.add_argument("--repeat-scenes", type=int, default=40)
    parser.add_argument("--master-seed", type=int, default=20261014)
    parser.add_argument("--receiver", type=int, default=1)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--gn-max-nfev", type=int, default=4)
    parser.add_argument("--p-fa", type=float, default=.05)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
