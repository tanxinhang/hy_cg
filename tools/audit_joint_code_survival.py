#!/usr/bin/env python3
"""Read-only per-target TP-UIC survival audit on frozen joint test scenes."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from isac_sim.cooperation.scientific_validation import load_frozen_scenario
from isac_sim.core.config import apply_overrides
from isac_sim.receiver import cancellation as cx
from tools.run_joint_global_alarm import joint_observation
from tools.run_tpuic_receiver_benchmark import _boost_direct, _make_cfg, build_parser


def main():
    parser = build_parser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    root = Path(args.baseline)
    old = json.loads((root / "result.json").read_text(encoding="utf-8"))
    cfg = _make_cfg(args)
    rows = []
    for record in old["records"]:
        if record["split"] != "test":
            continue
        scene = int(record["scene_id"])
        truth, belief, base0 = load_frozen_scenario(root / "scenes" / f"scene_{scene:05d}.npz")
        base = _boost_direct(base0, float(args.direct_gain_boost_db))
        for budget in (1, 2):
            cfg_b = apply_overrides(cfg, {"cancellation.max_protected_targets": budget})
            for receiver in old["receivers"]:
                _, obs = joint_observation(cfg_b, truth, belief, base,
                                            scene=scene, receiver=receiver)
                for target in old["targets"]:
                    tested = replace(obs, weak_index=target)
                    arm = cx.cancellation_arms(cfg_b, tested,
                                               weak_target=target)["tp_uic_full"]
                    rows.append({"scene": scene, "budget": budget,
                                 "receiver": receiver, "target": target,
                                 "eta_survive_q": float(arm.eta_survive_q),
                                 "direct_suppression_db": float(10 * np.log10(
                                     max(arm.i_in, 1e-30) / max(arm.i_res, 1e-30)))})
        print(f"scene {scene} complete", flush=True)
    summary = {}
    for budget in (1, 2):
        for target in old["targets"]:
            group = [r for r in rows if r["budget"] == budget and r["target"] == target]
            summary[f"budget_{budget}_target_{target}"] = {
                "median_eta_survive_q": float(np.median([r["eta_survive_q"] for r in group])),
                "median_direct_suppression_db": float(np.median(
                    [r["direct_suppression_db"] for r in group])),
            }
    Path(args.result).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                                 encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
