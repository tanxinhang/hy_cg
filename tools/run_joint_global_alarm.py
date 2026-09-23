#!/usr/bin/env python3
"""True joint-target H0/H1 pilot with a fixed, quantized UAV subset.

One physical observation is drawn per receiver/scene. Every target detector
processes that same observation. H0 removes all candidate echoes, unlike the
receiver-only pair benchmark, which removes only the tested target.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from isac_sim.receiver import cancellation as cx
from isac_sim.sensing.model import radar_hardware_gain
from tools.run_quantized_fixed_fusion import quantize
from tools.run_tpuic_receiver_benchmark import (
    _all_arms, _boost_direct, _make_cfg, _parse_int_selection,
    _run_arm, _scene, build_parser,
)


def _auc(h0: np.ndarray, h1: np.ndarray) -> float:
    from scipy.stats import rankdata
    n = len(h0)
    ranks = rankdata(np.r_[h0, h1], method="average")
    return float((ranks[n:].sum() - n * (n + 1) / 2) / (n * n))


def _variant(cal: list[dict], test: list[dict], *, field: str,
             receiver_index: int | None, alpha: float,
             weights: np.ndarray | None = None) -> dict:
    def score(row: dict, hypothesis: int) -> np.ndarray:
        values = np.asarray(row[field][hypothesis], dtype=float)
        if weights is not None:
            return weights @ values
        if receiver_index is None:
            return np.sum(values, axis=0)
        return values[receiver_index]

    n = len(cal)
    order = math.ceil((n + 1) * (1 - alpha))
    cal_global = np.asarray([np.max(score(row, 0)) for row in cal])
    threshold = (float("inf") if order > n else
                 float(np.sort(cal_global)[order - 1]))
    h0 = np.asarray([score(row, 0) for row in test])
    h1 = np.asarray([score(row, 1) for row in test])
    return {
        "threshold": threshold, "conformal_order": order,
        "marginal_global_pfa_bound": 0 if order > n else (n + 1 - order) / (n + 1),
        "test_global_false_alarms": int(np.sum(np.max(h0, axis=1) > threshold)),
        "test_per_target_detections": np.sum(h1 > threshold, axis=0).astype(int).tolist(),
        "test_per_target_auc": [_auc(h0[:, q], h1[:, q]) for q in range(h0.shape[1])],
    }


def _learn_minimax_weight(train: list[dict], *, field: str,
                          quant_step: float) -> dict:
    """Frozen 5-point grid; H0 covariance shrinkage and worst-target deflection."""
    if len(train) < 3:
        raise ValueError("at least three independent training scenes required")
    h0 = np.asarray([row[field][0] for row in train], dtype=float)
    h1 = np.asarray([row[field][1] for row in train], dtype=float)
    if h0.shape[1] != 2:
        raise ValueError("weight pilot requires exactly two receivers")
    candidates = []
    for w0 in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = np.asarray([w0, 1 - w0])
        per_target = []
        for q in range(h0.shape[2]):
            delta = np.mean(h1[:, :, q] - h0[:, :, q], axis=0)
            cov = np.cov(h0[:, :, q], rowvar=False, ddof=1)
            cov = (0.3 * cov + 0.7 * np.diag(np.diag(cov))
                   + (quant_step ** 2 / 12) * np.eye(2))
            variance = max(float(w @ cov @ w), 1e-9)
            per_target.append(max(float(w @ delta), 0.0) ** 2 / variance)
        candidates.append({"weights": w.tolist(),
                           "worst_target_train_deflection": min(per_target),
                           "per_target_train_deflection": per_target})
    selected = max(candidates, key=lambda v: (v["worst_target_train_deflection"],
                                                -abs(v["weights"][0] - 0.5)))
    return {"selected": selected, "candidates": candidates,
            "covariance": "H0 sample covariance, 70% diagonal shrinkage plus quantization-step^2/12 diagonal floor",
            "selection": "maximin one-sided deflection on independent train scenes only"}


def joint_observation(cfg, truth, belief, base, *, scene: int, receiver: int):
    """Produce a shared full-target H1 and no-target H0 with independent noise."""
    rng = np.random.default_rng([int(cfg.run.seed), scene, receiver, 0, 404])
    direct_rng = np.random.default_rng([int(cfg.run.seed), scene, receiver, 405])
    m = int(cfg.scale.M)
    power = np.full(m, float(cfg.radio.P_default) * float(cfg.radio.rho))
    obs1 = cx.build_observation(
        cfg, truth, belief.as_geometry(truth), base, receiver,
        rng=rng, direct_error_rng=direct_rng,
        sense_power=power, radiated_power=power,
        processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)),
        active_mask=np.ones(m, dtype=bool), include_echo=True, weak_index=0,
    )
    noise = (rng.normal(size=obs1.y.size) + 1j * rng.normal(size=obs1.y.size)) * math.sqrt(obs1.sigma2 / 2)
    obs0 = replace(obs1, y=obs1.x_direct + noise,
                   s_target=np.zeros_like(obs1.s_target),
                   alpha_true=np.zeros_like(obs1.alpha_true))
    return obs0, obs1


def run(args) -> dict:
    if args.train_scenes < 0 or (0 < args.train_scenes < 3):
        raise ValueError("train-scenes must be 0 or at least 3 independent scenes")
    if args.cal_scenes <= 0 or args.test_scenes <= 0:
        raise ValueError("calibration and test require positive independent scene counts")
    if args.bits < 1 or not math.isfinite(args.clip) or args.clip <= 0:
        raise ValueError("positive bit count and finite positive quantizer clip required")
    if args.realisations != 1 or args.direct_error_scope != "scene":
        raise ValueError("joint pilot requires one realization per scene and scene-fixed direct error")
    if args.arms != "tp_uic_full" or args.target_dictionary != "belief":
        raise ValueError("joint pilot freezes tp_uic_full with belief dictionary")
    if args.direct_gain_boost_db.count(",") or args.oracle_direct_residual_in_cres:
        raise ValueError("one boost and no truth-leaking covariance required")
    cfg = _make_cfg(args)
    receivers = _parse_int_selection(args.receivers, int(cfg.scale.M))
    targets = _parse_int_selection(args.targets, int(cfg.scale.Q))
    if len(receivers) < 1 or targets != tuple(range(int(cfg.scale.Q))):
        raise ValueError("fixed receivers and all candidate targets are required")
    if (args.cal_scenes < math.ceil(1 / args.p_fa) - 1
            and not args.allow_underpowered_smoke):
        raise ValueError("too few independent calibration scenes for a finite conformal gate")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for scene in range(args.train_scenes + args.cal_scenes + args.test_scenes):
        split = ("train" if scene < args.train_scenes else
                 "calibration" if scene < args.train_scenes + args.cal_scenes else "test")
        truth, belief, base0 = _scene(cfg, scene, output / "scenes" / f"scene_{scene:05d}.npz")
        base = _boost_direct(base0, float(args.direct_gain_boost_db))
        raw = np.zeros((2, len(receivers), len(targets)))
        reported = np.zeros_like(raw)
        for i, receiver in enumerate(receivers):
            obs0, obs1 = joint_observation(cfg, truth, belief, base,
                                            scene=scene, receiver=receiver)
            for j, target in enumerate(targets):
                for hypothesis, obs in enumerate((obs0, obs1)):
                    tested = replace(obs, weak_index=target)
                    _, _, glrt = _run_arm(cfg, tested, "tp_uic_full",
                                          _all_arms(cfg, tested))
                    rank = float(glrt.dof_real) / 2
                    if rank <= 0:
                        raise ValueError("zero-rank target detector")
                    raw[hypothesis, i, j] = float(glrt.statistic) / rank
                    reported[hypothesis, i, j] = quantize(
                        raw[hypothesis, i, j], args.bits, args.clip)
        scores = np.sum(reported, axis=1)
        records.append({"scene_id": scene, "split": split,
                        "raw_by_hypothesis_receiver_target": raw.tolist(),
                        "reported_by_hypothesis_receiver_target": reported.tolist(),
                        "h0_per_target": scores[0].tolist(),
                        "h1_per_target": scores[1].tolist(),
                        "global_h0": float(np.max(scores[0]))})
    train = [r for r in records if r["split"] == "train"]
    cal = [r for r in records if r["split"] == "calibration"]
    test = [r for r in records if r["split"] == "test"]
    variants = {}
    reported_label = f"{args.bits}bit"
    for label, field in (("unquantized", "raw_by_hypothesis_receiver_target"),
                         (reported_label, "reported_by_hypothesis_receiver_target")):
        variants[f"pair_{label}"] = _variant(
            cal, test, field=field, receiver_index=None, alpha=args.p_fa)
        for i, receiver in enumerate(receivers):
            variants[f"receiver_{receiver}_{label}"] = _variant(
                cal, test, field=field, receiver_index=i, alpha=args.p_fa)
    learned = None
    if train:
        field = "reported_by_hypothesis_receiver_target"
        learned = _learn_minimax_weight(
            train, field=field, quant_step=args.clip / ((1 << args.bits) - 1))
        weights = np.asarray(learned["selected"]["weights"], dtype=float)
        variants[f"trained_minimax_{reported_label}"] = _variant(
            cal, test, field=field, receiver_index=None,
            alpha=args.p_fa, weights=weights)
    main_variant = variants[f"pair_{reported_label}"]
    result = {
        "protocol": "joint_global_alarm_real_tpuic_v3_split_trained_fusion",
        "global_h0": "all candidate target echoes absent in one shared physical observation per receiver/scene",
        "global_h1": "all candidate target echoes present; each target scored on the same observation",
        "receivers": list(receivers), "targets": list(targets),
        "bits_per_uav_per_target_payload": args.bits,
        "total_payload_bits_per_scene": len(receivers) * len(targets) * args.bits,
        "communication_scope": "ideal error-free scalar payload; no header, coding, outage or latency model",
        "quantizer": {"type": "fixed_uniform_nearest", "range": [0, args.clip]},
        "alpha": args.p_fa, "n_train_independent_scenes": len(train),
        "n_cal_independent_scenes": len(cal),
        "n_test_independent_scenes": len(test),
        "weight_learning": learned,
        "variants": variants,
        "conformal_order": main_variant["conformal_order"],
        "threshold": main_variant["threshold"],
        "marginal_global_pfa_bound": main_variant["marginal_global_pfa_bound"],
        "test_global_false_alarms": main_variant["test_global_false_alarms"],
        "test_per_target_detections": main_variant["test_per_target_detections"],
        "records": records,
        "limitations": ["fixed subset, not learned association",
                        "homogeneous target RCS", "pilot PFA 0.05, not paper PFA 0.001"],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser: argparse.ArgumentParser = build_parser()
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--clip", type=float, default=8.0)
    parser.add_argument("--train-scenes", type=int, default=0)
    parser.add_argument("--allow-underpowered-smoke", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
