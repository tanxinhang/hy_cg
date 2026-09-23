#!/usr/bin/env python3
"""Same-observation oracle decomposition; diagnostic only, never a deployable detector."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from isac_sim.cooperation.scientific_validation import load_frozen_scenario
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from tools.run_joint_global_alarm import _auc, joint_observation
from tools.run_tpuic_receiver_benchmark import _boost_direct, _make_cfg, build_parser


def main():
    parser = build_parser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    root = Path(args.baseline)
    source = json.loads((root / "result.json").read_text(encoding="utf-8"))
    cfg = _make_cfg(args)
    modes = ("truth_template_belief_cov", "truth_template_truth_cov",
             "perfect_channel_truth_cov")
    rows = []
    for record in source["records"]:
        if record["split"] != "test":
            continue
        scene = int(record["scene_id"])
        truth, belief, base0 = load_frozen_scenario(root / "scenes" / f"scene_{scene:05d}.npz")
        base = _boost_direct(base0, float(args.direct_gain_boost_db))
        scores = np.zeros((2, len(modes), len(source["receivers"]), len(source["targets"])))
        for i, receiver in enumerate(source["receivers"]):
            obs_pair = joint_observation(cfg, truth, belief, base,
                                         scene=scene, receiver=receiver)
            for hypothesis, obs in enumerate(obs_pair):
                for j, target in enumerate(source["targets"]):
                    tested = replace(obs, weak_index=target)
                    arms = cx.cancellation_arms(cfg, tested, weak_target=target)
                    for k, mode in enumerate(modes):
                        arm_name = ("perfect_channel" if mode.startswith("perfect_channel")
                                    else "tp_uic_full")
                        cov_dictionary = ("belief" if mode == "truth_template_belief_cov"
                                          else "truth")
                        model = gl.residual_model(cfg, tested, arm_name, arms,
                                                  dictionary=cov_dictionary)
                        out = gl.target_conditioned_glrt(
                            cfg, tested, arms[arm_name], model, target=target,
                            p_fa=float(cfg.detect.Pfa_target),
                            dictionary="truth", centre_only=True)
                        scores[hypothesis, k, i, j] = float(out.statistic) / (float(out.dof_real) / 2)
        rows.append({"scene": scene, "scores": scores.tolist()})
        print(f"scene {scene} complete", flush=True)
    arr = np.asarray([r["scores"] for r in rows])
    summary = {}
    for k, mode in enumerate(modes):
        pair = arr[:, :, k].sum(axis=2)
        summary[mode] = {}
        for j, target in enumerate(source["targets"]):
            summary[mode][target] = {
                "pair_auc": _auc(pair[:, 0, j], pair[:, 1, j]),
                "pair_median_h1_minus_h0": float(np.median(pair[:, 1, j] - pair[:, 0, j])),
                "receiver_auc": [
                    _auc(arr[:, 0, k, i, j], arr[:, 1, k, i, j])
                    for i in range(len(source["receivers"]))],
            }
    result = {"summary": summary, "rows": rows,
              "warning": "Truth templates and perfect direct-channel cancellation are diagnostic oracles; no calibration or deployable PFA claim."}
    Path(args.result).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
