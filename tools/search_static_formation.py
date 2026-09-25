"""Search structured 6-UAV/3-target formations before full receiver validation.

The search score is deliberately only a screening proxy.  Every retained
candidate is exported as a frozen scenario so ``run_matrix_information_chain``
can evaluate it with the unchanged TP-UIC, coordination, reporting and fusion
chain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isac_sim.cooperation.scientific_validation import save_frozen_scenario
from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.scenario.belief import BeliefState
from isac_sim.sensing.model import Geometry, build_base_gains, generate_geometry


def _proxy(cfg, geom: Geometry, base) -> tuple[float, np.ndarray]:
    """Distance/direct-coupling proxy; never reported as detection probability."""
    m, q = cfg.scale.M, cfg.scale.Q
    useful = np.zeros(q)
    noise_floor = 1e-18
    for target in range(q):
        links = []
        for tx in range(m):
            for rx in range(m):
                if tx == rx or not base.valid_dd[tx, rx, target]:
                    continue
                interference = noise_floor + base.direct_gain[tx, rx]
                links.append(base.target_gain[tx, rx, target] / interference)
        # Reporting permits only a small number of useful receiver contributions.
        useful[target] = np.sum(np.sort(np.asarray(links))[-6:])
    scaled = np.log1p(useful / max(float(np.median(useful)), 1e-30))
    return float(np.min(scaled)), scaled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-movement-m", type=float, default=400.0)
    parser.add_argument("--out-dir", default="studies/direction3/data/formation_search_trial0")
    args = parser.parse_args()

    cfg = apply_preset(Config(), "paper-canonical")
    cfg = apply_overrides(cfg, {
        "geometry.area_xy": 400.0,
        "detect.target_rcs": 0.1,
        "scale.M": 6,
        "scale.Q": 3,
        "run.seed": int(args.seed),
        "run.verbose": False,
        "cancellation.enable": True,
        "aperture.enable": True,
        "aperture.m_rx": 4,
        "detect.target_response_model": "swerling2_fast",
    })
    scene_rng = np.random.default_rng([args.seed, args.trial])
    truth = generate_geometry(cfg, scene_rng)
    belief = BeliefState.from_truth(cfg, truth, scene_rng)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    search_rng = np.random.default_rng([args.seed, args.trial, 314159])
    retained: list[tuple[float, np.ndarray, np.ndarray]] = []
    for index in range(int(args.samples) + 1):
        if index == 0:
            positions = truth.p_uav.copy()
        else:
            positions = np.empty_like(truth.p_uav)
            permutation = search_rng.permutation(6)
            for target in range(3):
                # Two UAVs per target, but angles/radii/altitudes are optimized
                # rather than fixed to an opposite-point ring.
                angles = search_rng.uniform(0.0, 2.0 * np.pi, size=2)
                radii = search_rng.uniform(35.0, 220.0, size=2)
                for slot in range(2):
                    uav = int(permutation[2 * target + slot])
                    xy = truth.p_tgt[target, :2] + radii[slot] * np.array([
                        np.cos(angles[slot]), np.sin(angles[slot])
                    ])
                    positions[uav, :2] = np.clip(xy, 0.0, cfg.geometry.area_xy)
                    positions[uav, 2] = np.clip(
                        truth.p_tgt[target, 2] + search_rng.uniform(-180.0, 180.0),
                        cfg.geometry.h_uav_min, cfg.geometry.h_uav_max,
                    )
            displacement = positions - truth.p_uav
            distance = np.linalg.norm(displacement, axis=1)
            scale = np.minimum(
                1.0, float(args.max_movement_m) / np.maximum(distance, 1e-30)
            )
            positions = truth.p_uav + displacement * scale[:, None]
        candidate = Geometry(
            p_uav=positions,
            v_uav=truth.v_uav.copy(),
            p_tgt=truth.p_tgt.copy(),
            v_tgt=truth.v_tgt.copy(),
        )
        channel_rng = np.random.default_rng([args.seed, args.trial, 271828])
        base = build_base_gains(cfg, candidate, channel_rng)
        score, per_target = _proxy(cfg, candidate, base)
        retained.append((score, per_target, positions.copy()))

    retained.sort(key=lambda item: item[0], reverse=True)
    records = []
    # Export baseline plus the five strongest, geometrically distinct candidates.
    selected = [("baseline", retained[-1])]  # overwritten below with true baseline
    baseline_item = next(item for item in retained if np.allclose(item[2], truth.p_uav))
    selected = [("baseline", baseline_item)]
    for rank, item in enumerate(retained[:5], start=1):
        selected.append((f"candidate_{rank}", item))
    for name, (score, per_target, positions) in selected:
        candidate = Geometry(
            p_uav=positions,
            v_uav=truth.v_uav.copy(),
            p_tgt=truth.p_tgt.copy(),
            v_tgt=truth.v_tgt.copy(),
        )
        base = build_base_gains(
            cfg, candidate, np.random.default_rng([args.seed, args.trial, 271828])
        )
        snapshot = out_dir / f"{name}.npz"
        save_frozen_scenario(snapshot, candidate, belief, base)
        records.append({
            "name": name,
            "proxy_worst": score,
            "proxy_per_target": per_target.tolist(),
            "uav_positions_m": positions.tolist(),
            "snapshot": str(snapshot),
        })
    result = {
        "status": "screening_only_until_full_chain_validation",
        "samples": int(args.samples),
        "trial": int(args.trial),
        "max_movement_m": float(args.max_movement_m),
        "targets_m": truth.p_tgt.tolist(),
        "candidates": records,
    }
    (out_dir / "search.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
