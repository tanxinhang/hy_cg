#!/usr/bin/env python3
"""Cheap paired screen of four 6-UAV/3-target formation baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isac_sim.cooperation.formation import (
    minimum_uav_separation,
    project_uav_separation,
    target_ring_formation,
)
from isac_sim.sensing.model import Geometry, generate_geometry
from tools import gate_crossfit_target_state_map as gate
from tools import run_tpuic_receiver_benchmark as bench


def _proxy(geom, target_positions):
    """Belief/truth geometric coverage proxy, larger is better."""
    per_target = []
    for target in np.asarray(target_positions):
        vector = target[None, :] - geom.p_uav
        distance = np.maximum(np.linalg.norm(vector, axis=1), 1.0)
        quality = 1.0 / distance ** 4
        order = np.argsort(quality)[::-1][:2]
        if len(order) == 2:
            unit = vector[order] / distance[order, None]
            diversity = 1.0 - float(np.dot(unit[0], unit[1])) ** 2
        else:
            diversity = 0.0
        per_target.append(float(np.sum(quality[order]) * (1.0 + diversity)))
    values = np.asarray(per_target)
    return float(np.min(values)), values


def _with_positions(truth, positions):
    return Geometry(p_uav=np.asarray(positions), v_uav=truth.v_uav.copy(),
                    p_tgt=truth.p_tgt.copy(), v_tgt=truth.v_tgt.copy())


def _optimise(cfg, truth, belief_geom, rng, samples, movement):
    best, best_score = truth.p_uav.copy(), -np.inf
    for _ in range(int(samples)):
        positions = truth.p_uav.copy()
        assignment = rng.permutation(cfg.scale.M)
        for q in range(cfg.scale.Q):
            for slot in range(2):
                uav = int(assignment[2 * q + slot])
                angle = rng.uniform(0.0, 2.0 * np.pi)
                radius = rng.uniform(50.0, 180.0)
                positions[uav, :2] = np.clip(
                    belief_geom.p_tgt[q, :2] + radius * np.array(
                        [np.cos(angle), np.sin(angle)]), 0.0, cfg.geometry.area_xy)
                positions[uav, 2] = np.clip(
                    belief_geom.p_tgt[q, 2] + rng.uniform(-100.0, 100.0),
                    cfg.geometry.h_uav_min, cfg.geometry.h_uav_max)
        motion = positions - truth.p_uav
        norm = np.linalg.norm(motion, axis=1)
        positions = truth.p_uav + motion * np.minimum(
            1.0, movement / np.maximum(norm, 1e-30))[:, None]
        try:
            positions = project_uav_separation(
                cfg, positions, truth.p_uav, movement)
        except RuntimeError:
            continue
        candidate = _with_positions(truth, positions)
        score, _ = _proxy(candidate, belief_geom.p_tgt)
        if score > best_score:
            best_score, best = score, positions.copy()
    return _with_positions(truth, best)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=int, default=20)
    p.add_argument("--samples", type=int, default=250)
    p.add_argument("--movement-m", type=float, default=200.0)
    p.add_argument("--error-m", type=float, default=150.0)
    p.add_argument("--out", type=Path,
                   default=Path("studies/direction4/data/pilot_formation/summary.json"))
    args = p.parse_args()
    defaults = bench.build_parser().parse_args(["--out", str(args.out.parent),
        "--area-xy", "400", "--uavs", "6", "--targets-count", "3",
        "--min-uav-separation-m", "20"])
    defaults.target_rcs = 0.5
    defaults.master_seed = 20261007
    cfg = bench._make_cfg(defaults)
    records = []
    for scene in range(args.scenes):
        truth = generate_geometry(cfg, np.random.default_rng([cfg.run.seed, scene, 101]))
        from isac_sim.scenario.belief import BeliefState
        belief0 = BeliefState.from_truth(
            cfg, truth, np.random.default_rng([cfg.run.seed, scene, 202]))
        belief, _ = gate._belief_for_radius(cfg, truth, belief0, 2,
                                             args.error_m, scene)
        bg = belief.as_geometry(truth)
        single_b = target_ring_formation(cfg, bg, 1, horizontal_radius_m=100.0,
                                          max_movement_m=args.movement_m)
        dual_b = target_ring_formation(cfg, bg, 2, horizontal_radius_m=100.0,
                                        max_movement_m=args.movement_m)
        formations = {
            "random": truth,
            "single": _with_positions(truth, single_b.p_uav),
            "dual": _with_positions(truth, dual_b.p_uav),
            "belief_aware": _optimise(cfg, truth, bg,
                np.random.default_rng([cfg.run.seed, scene, 909]),
                args.samples, args.movement_m),
        }
        for name, formation in formations.items():
            predicted, _ = _proxy(formation, bg.p_tgt)
            realised, per_target = _proxy(formation, truth.p_tgt)
            records.append({"scene": scene, "formation": name,
                "predicted_proxy": predicted, "realised_proxy": realised,
                "per_target": per_target.tolist(),
                "min_separation_m": minimum_uav_separation(formation.p_uav),
                "max_movement_m": float(np.max(np.linalg.norm(
                    formation.p_uav - truth.p_uav, axis=1)))})
    summary = {}
    for name in ("random", "single", "dual", "belief_aware"):
        block = [r for r in records if r["formation"] == name]
        values = np.asarray([r["realised_proxy"] for r in block])
        summary[name] = {"median_realised_proxy": float(np.median(values)),
            "p10_realised_proxy": float(np.quantile(values, 0.1)),
            "min_separation_m": float(min(r["min_separation_m"] for r in block)),
            "max_movement_m": float(max(r["max_movement_m"] for r in block))}
    payload = {"protocol": vars(args) | {"out": str(args.out)},
               "summary": summary, "records": records,
               "warning": "screening proxy only; full receiver-chain validation required"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
