#!/usr/bin/env python3
"""Calibrate and freeze the Stage-B state-update gate on r=0 scenes only.

This driver intentionally computes no H0/H1 detector threshold and no P_D.  It
uses only deployable state-search diagnostics.  Threshold formulas are fixed in
code before observations are inspected; truth-state errors are recorded for
audit but never enter :func:`freeze_gate`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.direction4.scripts.smoke_stageb_one_scene import _args  # noqa: E402
from isac_sim.receiver.target_state.fit import (  # noqa: E402
    fit_shared_target_offset,
)
from tools import gate_crossfit_target_state_map as gate  # noqa: E402


def _higher_quantile(values, q):
    """Version-stable conservative empirical quantile."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("gate calibration requires finite non-empty values")
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, q, interpolation="higher"))


def freeze_gate(records, quantile=0.95):
    """Return thresholds without consulting truth errors or selected deltas."""
    folds = [fold for record in records for fold in record["folds"]]
    distances = [record["cross_fold_distance_m"] for record in records]
    return {
        "schema_version": 1,
        "status": "FROZEN",
        "calibration_regime": "r=0 only",
        "threshold_formula_preregistered": {
            "gain_over_zero": f"empirical q={quantile:g}, method=higher",
            "receiver_support": "fixed strict majority: >=4 of 6",
            "cross_fold_distance_m": f"empirical q={quantile:g}, method=higher",
            "boundary": "reject every boundary hit",
        },
        "accept_rule": {
            "gain_over_zero_strictly_gt": _higher_quantile(
                [fold["gain_over_zero"] for fold in folds], quantile),
            "receiver_support_ge": 4,
            "cross_fold_distance_m_le": _higher_quantile(distances, quantile),
            "boundary_hit_must_equal": False,
        },
        "calibration_counts": {
            "scenes": len(records),
            "folds": len(folds),
        },
        "forbidden_after_freeze": [
            "changing thresholds from injected truth errors",
            "changing thresholds from test-set AUC, P_D, or oracle gap",
        ],
    }


def calibrate_scene(cfg, args, out, scene, radius_m=0.0):
    """Run the exact confirmatory 37.5 m coarse-to-fine state solver."""
    truth, belief0, base = gate._scene_with_retry(
        cfg, scene, out / "scenes" / f"scene_{scene:05d}.npz")
    belief, _ = gate._belief_for_radius(
        cfg, truth, belief0, args.target, radius_m, scene)
    belief_geometry = belief.as_geometry(truth)
    receivers = tuple(range(int(cfg.scale.M)))
    folds = []
    for look in range(2):
        observations = [
            gate._pair(cfg, truth, belief_geometry, base, args,
                       scene, rx, look)[0]
            for rx in receivers
        ]
        scorers = [gate._setup(cfg, obs, args.target)[2]
                   for obs in observations]
        fit = fit_shared_target_offset(
            cfg, observations, args.target, belief, receivers=receivers,
            belief_geometry=belief_geometry, scorers=scorers,
            prior_sigma_m=args.prior_sigma_m,
            search_sigma=args.search_sigma, solver="coarse_to_fine",
            coarse_step_m=37.5)
        folds.append({
            "fold": "AB"[look],
            "delta": np.asarray(fit.delta_xy_m).tolist(),
            "gain_over_zero": float(fit.gain_over_zero),
            "receiver_support": int(fit.receiver_support),
            "boundary_hit": bool(fit.boundary_hit),
            "audit_state_error_m": float(np.linalg.norm(
                np.asarray(fit.delta_xy_m) - (
                    np.asarray(truth.p_tgt[args.target, :2]) -
                    np.asarray(belief_geometry.p_tgt[args.target, :2])))),
        })
    return {
        "scene_id": scene,
        "normalized_error": float(radius_m / args.prior_sigma_m),
        "cross_fold_distance_m": float(np.linalg.norm(
            np.asarray(folds[0]["delta"]) - np.asarray(folds[1]["delta"]))),
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(
        "studies/direction4/data/stageb_gate_calibration_r0"))
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--quantile", type=float, default=0.95)
    cli = parser.parse_args()
    if cli.scenes < 1:
        raise SystemExit("--scenes must be positive")
    cli.out.mkdir(parents=True, exist_ok=True)
    smoke_cli = SimpleNamespace(out=cli.out)
    args = _args(smoke_cli)
    cfg = gate._cfg(args)
    records = []
    for scene in range(cli.scenes):
        record = calibrate_scene(cfg, args, cli.out, scene)
        records.append(record)
        (cli.out / "calibration_records.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"r=0 calibration scene {scene + 1}/{cli.scenes} complete", flush=True)
    frozen = freeze_gate(records, cli.quantile)
    (cli.out / "frozen_gate.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(frozen, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
