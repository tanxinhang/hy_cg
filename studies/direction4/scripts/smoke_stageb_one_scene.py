#!/usr/bin/env python3
"""Run one Stage-B row and enforce the pre-compute gates.

This smoke test deliberately does not calibrate a threshold or report P_D.  It
checks one paired H1/H0 row before the 8--12-scene variance pilot is launched.
"""
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

from tools import gate_crossfit_target_state_map as gate
from isac_sim.receiver.target_state.objective import ObjectiveEvaluator
from isac_sim.receiver.target_state.views import make_receiver_views
from isac_sim.receiver.target_state.nms import non_maximum_suppression
from isac_sim.receiver.target_state.fit import fit_shared_target_offset


ARMS = ("nominal", "wls_map", "local_crossfit", "shared_crossfit", "oracle")


def _args(cli):
    return SimpleNamespace(
        out=cli.out, target=2, master_seed=20261007, area_xy=400.0,
        uavs=6, targets_count=3, min_uav_separation_m=20.0, m_rx=4,
        target_rcs=0.5, prior_sigma_m=150.0, glrt_radius=0.25,
        glrt_points=3, search_sigma=3.0, coarse_step_m=None,
        solver="coarse_to_fine", screen_receivers=2, screen_keep=40,
        verify_receivers=4, verify_keep=12, top_k=3,
        train_scenes=1, calibration_scenes=0, test_scenes=0,
        bootstrap=0, p_fa=0.05, baseline_curve=True,
    )


def _minimum_separation(snapshot):
    positions = np.asarray(snapshot["truth_p_uav"], dtype=float)
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    distance[np.eye(len(positions), dtype=bool)] = np.inf
    return float(np.min(distance))


def _diagnose_objective(cfg, args, cli, row):
    truth, belief0, base = gate._scene_with_retry(
        cfg, cli.scene, cli.out / "scenes" / f"scene_{cli.scene:05d}.npz")
    belief, _ = gate._belief_for_radius(
        cfg, truth, belief0, args.target, cli.error_m, cli.scene)
    belief_geometry = belief.as_geometry(truth)
    receivers = tuple(range(int(cfg.scale.M)))
    true_delta = np.array([row["true_delta_x_m"], row["true_delta_y_m"]])
    selected = (
        np.array([row["estimated_delta_a_x"], row["estimated_delta_a_y"]]),
        np.array([row["estimated_delta_b_x"], row["estimated_delta_b_y"]]),
    )
    output = []
    for fold in range(2):
        observations = [
            gate._pair(cfg, truth, belief_geometry, base, args,
                       cli.scene, rx, fold)[0]
            for rx in receivers
        ]
        scorers = [gate._setup(cfg, obs, args.target)[2]
                   for obs in observations]
        views = make_receiver_views(receivers, observations, scorers, args.target)
        evaluator = ObjectiveEvaluator(
            cfg, views, args.target, belief_geometry, args.prior_sigma_m)
        wls = fit_shared_target_offset(
            cfg, observations, args.target, belief, receivers=receivers,
            belief_geometry=belief_geometry, scorers=scorers,
            prior_sigma_m=args.prior_sigma_m, search_sigma=args.search_sigma,
            solver="jacobian_wls")
        at_zero = evaluator.objective(np.zeros(2))[0]
        at_truth = evaluator.objective(true_delta)[0]
        at_selected = evaluator.objective(selected[fold])[0]
        span = float(args.search_sigma * args.prior_sigma_m)
        step = float(args.prior_sigma_m / 2.0)
        axis = np.arange(-span, span + 0.5 * step, step)
        points = np.array([(x, y) for x in axis for y in axis], dtype=float)
        values = np.array([evaluator.objective(point)[0] for point in points])
        nearest_index = int(np.argmin(np.linalg.norm(points - true_delta, axis=1)))
        order = np.argsort(values)[::-1]
        rank = int(np.flatnonzero(order == nearest_index)[0]) + 1
        kept, _ = non_maximum_suppression(points, values, 3, step)
        nearest_is_top3_nms = bool(np.any(
            np.linalg.norm(kept - points[nearest_index], axis=1) < 1e-9))
        wls_refined = np.asarray(wls.delta_xy_m, dtype=float).copy()
        wls_refined_value = evaluator.objective(wls_refined)[0]
        for local_step in (step, step / 2.0, step / 4.0, step / 8.0):
            best, best_value = wls_refined.copy(), wls_refined_value
            for dx in (-local_step, 0.0, local_step):
                for dy in (-local_step, 0.0, local_step):
                    candidate = wls_refined + np.array([dx, dy])
                    if np.max(np.abs(candidate)) > span + 1e-9:
                        continue
                    value = evaluator.objective(candidate)[0]
                    if value > best_value:
                        best, best_value = candidate, value
            wls_refined, wls_refined_value = best, best_value
        dense_step = step / 2.0
        dense_axis = np.arange(-span, span + 0.5 * dense_step, dense_step)
        dense_points = np.array(
            [(x, y) for x in dense_axis for y in dense_axis], dtype=float)
        dense_values = np.array(
            [evaluator.objective(point)[0] for point in dense_points])
        dense_best_index = int(np.argmax(dense_values))
        dense_best = dense_points[dense_best_index]
        dense_kept, dense_kept_values = non_maximum_suppression(
            dense_points, dense_values, 2, step)
        dense_peak_gap = (float(dense_kept_values[0] - dense_kept_values[1])
                          if len(dense_kept_values) > 1 else float("inf"))
        zero_gains = evaluator.gains(np.zeros(2), views)
        dense_gains = evaluator.gains(dense_best, views)
        output.append({
            "fold": "AB"[fold],
            "objective_zero": float(at_zero),
            "objective_truth": float(at_truth),
            "objective_selected": float(at_selected),
            "nearest_truth_coarse_point": points[nearest_index].tolist(),
            "nearest_truth_coarse_objective": float(values[nearest_index]),
            "nearest_truth_coarse_rank": rank,
            "nearest_truth_in_top3_nms": nearest_is_top3_nms,
            "wls_seed": np.asarray(wls.delta_xy_m).tolist(),
            "wls_seed_error_m": float(np.linalg.norm(
                np.asarray(wls.delta_xy_m) - true_delta)),
            "wls_refined": wls_refined.tolist(),
            "wls_refined_objective": float(wls_refined_value),
            "wls_refined_error_m": float(np.linalg.norm(
                wls_refined - true_delta)),
            "dense_step_m": dense_step,
            "dense_candidate_count": int(len(dense_points)),
            "dense_best": dense_best.tolist(),
            "dense_best_objective": float(dense_values[dense_best_index]),
            "dense_best_error_m": float(np.linalg.norm(
                dense_best - true_delta)),
            "dense_gain_over_zero": float(
                dense_values[dense_best_index] - at_zero),
            "dense_peak_gap": dense_peak_gap,
            "dense_receiver_support": int(np.sum(dense_gains > zero_gains)),
            "dense_boundary_hit": bool(
                span - np.max(np.abs(dense_best)) <= dense_step + 1e-9),
            "classification": (
                "search_missed_better_truth_peak"
                if at_truth > at_selected + 1e-9
                else "objective_prefers_false_peak"
            ),
        })
    fold_distance = float(np.linalg.norm(
        np.asarray(output[0]["dense_best"]) -
        np.asarray(output[1]["dense_best"])))
    for item in output:
        item["dense_cross_fold_distance_m"] = fold_distance
        item["provisional_gate_inputs_only"] = {
            "positive_gain": item["dense_gain_over_zero"] > 0.0,
            "majority_receiver_support": item["dense_receiver_support"] >= 4,
            "not_boundary": not item["dense_boundary_hit"],
            "cross_fold_distance_le_prior_sigma": fold_distance <= args.prior_sigma_m,
        }
        item["provisional_gate_accepts"] = all(
            item["provisional_gate_inputs_only"].values())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--error-m", type=float, default=150.0)
    parser.add_argument("--out", type=Path, default=Path(
        "studies/direction4/data/stageb_one_scene_r1"))
    parser.add_argument("--diagnose-only", action="store_true")
    cli = parser.parse_args()
    cli.out.mkdir(parents=True, exist_ok=True)
    args = _args(cli)
    cfg = gate._cfg(args)
    summary_path = cli.out / "summary.json"
    if cli.diagnose_only:
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        diagnosis = _diagnose_objective(cfg, args, cli, prior["row"])
        prior["diagnosis"] = diagnosis
        summary_path.write_text(
            json.dumps(prior, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(diagnosis, indent=2, ensure_ascii=False))
        return
    row = gate._scene_row(cfg, args, cli.scene, cli.error_m, args.target)

    snapshot_path = cli.out / "scenes" / f"scene_{cli.scene:05d}.npz"
    with np.load(snapshot_path) as snapshot:
        minimum_separation = _minimum_separation(snapshot)
    required = [f"{arm}_{hyp}" for arm in ARMS for hyp in ("h1", "h0")]
    required += [
        "state_error_a_m", "state_error_b_m",
        "wls_state_error_a_m", "wls_state_error_b_m", "runtime_s",
    ]
    missing = [name for name in required if name not in row]
    shared_error = float(np.mean([
        row["state_error_a_m"], row["state_error_b_m"]]))
    gates = {
        "schema_complete": not missing,
        "minimum_separation_ge_20m": minimum_separation >= 20.0 - 1e-9,
        "shared_state_moves_toward_truth": shared_error < cli.error_m,
        "finite_runtime": bool(np.isfinite(float(row["runtime_s"]))),
    }
    payload = {
        "protocol": {
            "scene": cli.scene, "error_m": cli.error_m,
            "normalized_error": cli.error_m / 150.0,
            "arms": list(ARMS), "shared_solver": "coarse_to_fine",
            "minimum_uav_separation_constraint_m": 20.0,
            "scope": "one-row compute smoke; no calibrated P_D claim",
        },
        "audit": {
            "runtime_s": float(row["runtime_s"]),
            "minimum_uav_separation_m": minimum_separation,
            "shared_state_error_mean_m": shared_error,
            "wls_state_error_mean_m": float(np.mean([
                row["wls_state_error_a_m"], row["wls_state_error_b_m"]])),
            "missing_fields": missing,
            "gates": gates,
            "passed": all(gates.values()),
        },
        "row": row,
    }
    with (cli.out / "record.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["audit"], indent=2, ensure_ascii=False), flush=True)
    if not payload["audit"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
