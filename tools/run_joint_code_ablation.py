#!/usr/bin/env python3
"""Paired, frozen-scene code-contract ablation for joint target detection."""
from __future__ import annotations

import argparse
import json
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np

from isac_sim.cooperation.scientific_validation import load_frozen_scenario
from isac_sim.core.config import apply_overrides
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.receiver.cancellation.link_offsets import target_link_offset
from isac_sim.receiver.joint_observation import joint_observation
from isac_sim.detection.evaluation_split import EvaluationPartition
from tools.run_joint_global_alarm import _variant
from tools.run_quantized_fixed_fusion import quantize
from tools.run_tpuic_receiver_benchmark import (
    _all_arms, _boost_direct, _make_cfg, _run_arm, build_parser,
)


def _geometry_glrt(cfg, obs, result, model, target):
    """Nine belief-only state perturbations, shared across all target links."""
    sources = [s for s in obs.targets_belief if int(s.target) == target]
    pos = float(cfg.prior.belief_sigma_pos_m)
    vel = float(cfg.prior.belief_sigma_vel_mps)
    deltas = [np.zeros(6)]
    for axis, scale in ((0, pos), (1, pos), (3, vel), (4, vel)):
        for sign in (-1, 1):
            d = np.zeros(6)
            d[axis] = sign * scale
            deltas.append(d)
    trials = []
    for delta in deltas:
        shifted = [replace(
            s,
            doppler_bin=float(s.doppler_bin) + float(np.dot(s.jacobian_doppler_state, delta)),
            delay_bin=float(s.delay_bin) + float(np.dot(s.jacobian_delay_state, delta)),
            u=float(np.clip(float(s.u) + float(np.dot(s.jacobian_bearing_state, delta)), -1, 1)),
        ) for s in sources]
        template = cx.target_dictionary(cfg, shifted, tangent_order=0,
                                        covariance_expanded=False)
        trials.append(gl.target_conditioned_glrt(
            cfg, obs, result, model, target=target,
            p_fa=float(cfg.detect.Pfa_target) / len(deltas),
            dictionary="belief", template_override=template, centre_only=True))
    return max(trials, key=lambda item: item.statistic)


def _state_coupled_exact_glrt(cfg, obs, result, model, target,
                              receiver, belief_geom, belief):
    """Seventeen fixed joint-state hypotheses; exact multistatic geometry."""
    sources = [s for s in obs.targets_belief if int(s.target) == target]
    horizontal = (0, 1, 3, 4)
    cov4 = np.asarray(belief.P[target])[np.ix_(horizontal, horizontal)]
    root = np.linalg.cholesky(cov4 + 1e-12 * np.eye(4))
    deltas = [np.zeros(4)] + [root @ np.asarray(signs, dtype=float)
                               for signs in itertools.product((-1, 1), repeat=4)]
    trials = []
    for delta in deltas:
        p_tgt = np.asarray(belief_geom.p_tgt).copy()
        v_tgt = np.asarray(belief_geom.v_tgt).copy()
        p_tgt[target, :2] += delta[:2]
        v_tgt[target, :2] += delta[2:]
        candidate = replace(belief_geom, p_tgt=p_tgt, v_tgt=v_tgt)
        r = p_tgt[target] - np.asarray(candidate.p_uav[receiver])
        u = (float(r[int(cfg.aperture.axis)] / max(np.linalg.norm(r), 1.0))
             if cfg.aperture.enable and int(cfg.aperture.m_rx) > 1 else 0.0)
        shifted = []
        for src in sources:
            k, l = target_link_offset(cfg, candidate, int(src.uav), receiver, target)
            shifted.append(replace(src, doppler_bin=float(k), delay_bin=float(l), u=u))
        template = cx.target_dictionary(cfg, shifted, tangent_order=0,
                                        covariance_expanded=False)
        trials.append(gl.target_conditioned_glrt(
            cfg, obs, result, model, target=target,
            p_fa=float(cfg.detect.Pfa_target) / len(deltas),
            dictionary="belief", template_override=template, centre_only=True))
    return max(trials, key=lambda item: item.statistic)


def run(args):
    baseline = Path(args.baseline)
    source = json.loads((baseline / "result.json").read_text(encoding="utf-8"))
    cfg = _make_cfg(args)
    cfg_two = apply_overrides(cfg, {"cancellation.max_protected_targets": 2})
    receivers = source["receivers"]
    targets = source["targets"]
    variants = tuple(args.variants.split(","))
    allowed = {"covariance_corrected", "geometry_neighbourhood", "protect_two",
               "state_coupled_exact"}
    if not variants or not set(variants) <= allowed or len(set(variants)) != len(variants):
        raise ValueError(f"variants must be drawn from {sorted(allowed)}")
    # A frozen scene is not a complete protocol: silently changed receiver
    # options previously produced a plausible-looking, but unpaired, ablation.
    anchor = next(r for r in source["records"] if r["split"] == "calibration")
    anchor_scene = int(anchor["scene_id"])
    anchor_truth, anchor_belief, anchor_base = load_frozen_scenario(
        baseline / "scenes" / f"scene_{anchor_scene:05d}.npz")
    anchor_obs = joint_observation(
        cfg, anchor_truth, anchor_belief,
        _boost_direct(anchor_base, float(args.direct_gain_boost_db)),
        scene=anchor_scene, receiver=receivers[0])
    for hypothesis, obs in enumerate(anchor_obs):
        tested = replace(obs, weak_index=targets[0])
        out = _run_arm(cfg, tested, "tp_uic_full", _all_arms(cfg, tested))[2]
        replay = float(out.statistic) / (float(out.dof_real) / 2)
        saved = float(anchor["raw_by_hypothesis_receiver_target"][hypothesis][0][0])
        if not np.isclose(replay, saved, rtol=1e-9, atol=1e-9):
            raise ValueError(f"baseline replay failed: replay={replay}, saved={saved}; "
                             "check all receiver parameters before ablation")
    records = {v: [] for v in variants}
    for old in source["records"]:
        if old["split"] == "train":
            continue
        scene = int(old["scene_id"])
        truth, belief, base0 = load_frozen_scenario(baseline / "scenes" / f"scene_{scene:05d}.npz")
        base = _boost_direct(base0, float(args.direct_gain_boost_db))
        arrays = {v: np.zeros((2, len(receivers), len(targets))) for v in variants}
        survival = {v: [] for v in variants}
        for i, receiver in enumerate(receivers):
            obs0, obs1 = joint_observation(cfg, truth, belief, base,
                                            scene=scene, receiver=receiver)
            if "protect_two" in variants:
                obs0_two, obs1_two = joint_observation(cfg_two, truth, belief, base,
                                                        scene=scene, receiver=receiver)
            for hypothesis, obs in enumerate((obs0, obs1)):
                shared = {}
                for label, cfg_s, obs_s in (
                    ("corrected", cfg, obs),
                    ("protect_two", cfg_two,
                     (obs0_two, obs1_two)[hypothesis] if "protect_two" in variants else obs),
                ):
                    if label == "corrected" and not set(variants) & {
                        "covariance_corrected", "geometry_neighbourhood",
                        "state_coupled_exact"
                    }:
                        continue
                    if label == "protect_two" and "protect_two" not in variants:
                        continue
                    reference = replace(obs_s, weak_index=targets[0])
                    arms_s = cx.cancellation_arms(cfg_s, reference,
                                                    weak_target=targets[0])
                    cov_obs_s = replace(reference, targets_belief=[])
                    model_s = gl.residual_model(cfg_s, cov_obs_s, "tp_uic_full",
                                                arms_s, dictionary="belief")
                    shared[label] = (arms_s, model_s)
                for variant in variants:
                    if variant == "protect_two":
                        obs_v = (obs0_two, obs1_two)[hypothesis]
                        cfg_v = cfg_two
                        arms, model = shared["protect_two"]
                    else:
                        obs_v = obs
                        cfg_v = cfg
                        arms, model = shared["corrected"]
                    for j, target in enumerate(targets):
                        tested = replace(obs_v, weak_index=target)
                        result = arms["tp_uic_full"]
                        if variant == "state_coupled_exact":
                            out = _state_coupled_exact_glrt(
                                cfg_v, tested, result, model, target, receiver,
                                belief.as_geometry(truth), belief)
                        elif variant in ("geometry_neighbourhood", "protect_two"):
                            out = _geometry_glrt(cfg_v, tested, result, model, target)
                        else:
                            out = gl.target_neighbourhood_glrt(
                                cfg_v, tested, result, model, target=target,
                                p_fa=float(cfg_v.detect.Pfa_target), dictionary="belief")
                        rank = float(out.dof_real) / 2
                        arrays[variant][hypothesis, i, j] = float(out.statistic) / rank
                        if hypothesis == 1:
                            # Diagnostic only; truth is not used to form a score.
                            s = tested.s_target
                            retained = model.signal_transfer(s)
                            survival[variant].append(float(np.linalg.norm(retained) /
                                                           max(np.linalg.norm(s), 1e-30)))
        for variant in variants:
            raw = arrays[variant]
            reported = np.vectorize(lambda x: quantize(float(x), args.bits, args.clip))(raw)
            records[variant].append({
                "scene_id": scene, "split": old["split"],
                "raw_by_hypothesis_receiver_target": raw.tolist(),
                "reported_by_hypothesis_receiver_target": reported.tolist(),
                "joint_echo_norm_survival": survival[variant],
            })
        print(f"scene {scene} complete", flush=True)
    summary = {}
    for variant in variants:
        rows = records[variant]
        partition = EvaluationPartition.from_records(rows)
        cal, test = partition.calibration, partition.test
        summary[variant] = {
            "raw": _variant(cal, test, field="raw_by_hypothesis_receiver_target",
                            receiver_index=None, alpha=args.p_fa),
            "quantized": _variant(cal, test, field="reported_by_hypothesis_receiver_target",
                                  receiver_index=None, alpha=args.p_fa),
            "test_median_joint_echo_norm_survival": float(np.median(
                [x for r in test.rows for x in r["joint_echo_norm_survival"]])),
        }
    output = {"baseline": str(baseline), "variants": summary,
              "records": records, "limitations": [
                  "Geometry grid uses nine axial 1-sigma state perturbations, not a continuous optimizer.",
                  "Joint-null covariance excludes candidate-target belief factors; partial-null FWER is not claimed."],
              "replay_anchor": {"scene": anchor_scene, "receiver": receivers[0],
                                "target": targets[0], "passed": True},
              "receiver_flags": {"interference_tangent_order": args.interference_tangent_order,
                                 "interference_uncertainty_weighted": args.interference_uncertainty_weighted,
                                 "direct_mismatch_covariance_model": args.direct_mismatch_covariance_model,
                                 "target_glrt_radius_bins": args.target_glrt_radius_bins,
                                 "max_protected_targets_baseline": args.max_protected_targets}}
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main():
    parser = build_parser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--clip", type=float, default=8.0)
    parser.add_argument("--variants", default="covariance_corrected,geometry_neighbourhood,protect_two")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["variants"], indent=2))


if __name__ == "__main__":
    main()
