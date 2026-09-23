#!/usr/bin/env python3
"""Read-only audit of true-vs-belief DD offsets in a frozen joint experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isac_sim.cooperation.scientific_validation import load_frozen_scenario
from isac_sim.receiver.cancellation.link_offsets import target_link_offset
from tools.run_tpuic_receiver_benchmark import _make_cfg, build_parser


def run(root: Path, *, radius: float = 0.25) -> dict:
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    parser = build_parser()
    argv = ["--out", "unused", "--uavs", "6", "--targets-count", str(len(result["targets"])),
            "--area-xy", "800", "--target-rcs", "0.05"]
    cfg = _make_cfg(parser.parse_args(argv))
    rows = []
    for record in result["records"]:
        scene = int(record["scene_id"])
        truth, belief, _ = load_frozen_scenario(root / "scenes" / f"scene_{scene:05d}.npz")
        believed = belief.as_geometry(truth)
        for receiver in result["receivers"]:
            for target in result["targets"]:
                errors = []
                for transmitter in range(int(cfg.scale.M)):
                    if transmitter == receiver:
                        continue
                    true_k, true_l = target_link_offset(cfg, truth, transmitter, receiver, target)
                    belief_k, belief_l = target_link_offset(cfg, believed, transmitter, receiver, target)
                    errors.append((true_k - belief_k, true_l - belief_l))
                arr = np.asarray(errors)
                rows.append({"scene": scene, "receiver": receiver, "target": target,
                             "split": record["split"],
                             "median_abs_doppler_bins": float(np.median(np.abs(arr[:, 0]))),
                             "median_abs_delay_bins": float(np.median(np.abs(arr[:, 1]))),
                             "all_links_within_square": bool(np.all(np.abs(arr) <= radius)),
                             "fraction_links_within_square": float(np.mean(np.all(np.abs(arr) <= radius, axis=1)))})
    by_target = {}
    for target in result["targets"]:
        group = [r for r in rows if r["target"] == target and r["split"] == "test"]
        by_target[target] = {
            "median_abs_doppler_bins": float(np.median([r["median_abs_doppler_bins"] for r in group])),
            "median_abs_delay_bins": float(np.median([r["median_abs_delay_bins"] for r in group])),
            "mean_link_fraction_within_neighbourhood": float(np.mean([r["fraction_links_within_square"] for r in group])),
            "receiver_scene_fraction_all_links_within_neighbourhood": float(np.mean([r["all_links_within_square"] for r in group])),
        }
    return {"input": str(root), "radius_bins": radius, "test_by_target": by_target}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--radius", type=float, default=0.25)
    args = p.parse_args()
    print(json.dumps(run(args.input, radius=args.radius), indent=2))


if __name__ == "__main__":
    main()
