"""41 行**配对**消融：只算状态误差，不跑五条臂。

为什么不去跑完整的 Gate
----------------------
一整行 = 五条臂（三条拟合臂 + 两条 GLRT 臂）× 两个假设 ≈ 390 s，其中状态误差
只依赖 ``shared_crossfit`` 那一条臂的两次拟合（≈ 2×15 s）。为了拿 p90 去跑
1.7 h × 2 个求解器是把 90% 的算力花在**与判据无关**的列上。这里复用 Gate 的
场景构造（``_scene_with_retry`` / ``_belief_for_radius`` / ``_pair``），把
``score_fn`` 换成常数 0 —— 检测统计量一个都不算。

口径：**同一 scene_id 上配对**，两折各得一个偏移（``state_error_a/b``）。

用法（分片）：
    python studies/direction4/scripts/ablate_gated_topk_state.py \\
        --out studies/direction4/data/ablate_state --shards 6 --shard 0
合并（等所有分片写完）：
    python studies/direction4/scripts/ablate_gated_topk_state.py \\
        --out studies/direction4/data/ablate_state --merge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from isac_sim.receiver.target_state import (  # noqa: E402
    crossfit_target_state_score,
)
from tools import gate_crossfit_target_state_map as gate  # noqa: E402

SOLVERS = ("coarse_to_fine", "gated_topk")


def _args(out: str, radius: float, solver: str) -> Namespace:
    return Namespace(
        out=Path(out), target=2, error_m=f"{radius:g}", train_scenes=1,
        calibration_scenes=20, test_scenes=20, master_seed=20261007,
        area_xy=400.0, uavs=6, targets_count=3, min_uav_separation_m=20.0,
        m_rx=4, target_rcs=0.5, prior_sigma_m=150.0, search_sigma=3.0,
        coarse_step_m=None, solver=solver, glrt_radius=0.25, glrt_points=3,
        p_fa=0.05, bootstrap=400, shard=0, shards=1, smoke=False,
        summary_only=False, screen_receivers=2, screen_keep=40,
        verify_receivers=4, verify_keep=12, top_k=3,
    )


def _describe(values: np.ndarray) -> str:
    if values.size == 0:
        return "n=0"
    return (f"median {np.median(values):7.1f}  p90 {np.percentile(values, 90):7.1f}"
            f"  p95 {np.percentile(values, 95):7.1f}  max {values.max():7.1f}"
            f"  rmse {float(np.sqrt(np.mean(values ** 2))):7.1f}  n={values.size}")


def run_shard(out: Path, radius: float, shard: int, shards: int) -> None:
    args = _args(str(out), radius, "coarse_to_fine")
    cfg = gate._cfg(args)
    receivers = tuple(range(int(cfg.scale.M)))
    total = args.train_scenes + args.calibration_scenes + args.test_scenes
    for scene in [s for s in range(total) if s % shards == shard]:
        truth, belief0, base = gate._scene_with_retry(
            cfg, scene, out / "scenes" / f"scene_{scene:05d}.npz")
        belief, _ = gate._belief_for_radius(cfg, truth, belief0, args.target,
                                            radius, scene)
        belief_geom = belief.as_geometry(truth)
        true_delta = np.asarray(truth.p_tgt[args.target, :2], float) - np.asarray(
            belief_geom.p_tgt[args.target, :2], float)
        looks = [[gate._pair(cfg, truth, belief_geom, base, args, scene, rx, look)[0]
                  for rx in receivers] for look in range(2)]
        scorers = [[gate._setup(cfg, obs, args.target)[2] for obs in look]
                   for look in looks]
        for solver in SOLVERS:
            args.solver = solver
            result = crossfit_target_state_score(
                cfg, looks[0], looks[1], args.target, belief,
                receivers=receivers, belief_geometry=belief_geom,
                scorers_a=scorers[0], scorers_b=scorers[1],
                score_fn=lambda observations: 0.0,
                prior_sigma_m=args.prior_sigma_m,
                search_sigma=args.search_sigma, solver=solver,
                solver_options=gate._solver_options(args))
            row = {
                "scene_id": scene,
                "role": gate._role(scene, args),
                "solver": solver,
                "true_delta_x_m": float(true_delta[0]),
                "true_delta_y_m": float(true_delta[1]),
                "delta_a": [float(v) for v in result.delta_a_xy_m],
                "delta_b": [float(v) for v in result.delta_b_xy_m],
                "state_error_a_m": float(np.linalg.norm(
                    np.asarray(result.delta_a_xy_m) - true_delta)),
                "state_error_b_m": float(np.linalg.norm(
                    np.asarray(result.delta_b_xy_m) - true_delta)),
                "objective_a": float(result.objective_a),
                "objective_b": float(result.objective_b),
                "evaluations": int(result.evaluations),
                "seconds": float(result.seconds),
            }
            with (out / f"shard_{shard:02d}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        print(f"[shard {shard}] scene {scene} done", flush=True)


def merge(out: Path) -> int:
    rows = []
    for path in sorted(out.glob("shard_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    if not rows:
        raise SystemExit(f"no shard_*.jsonl under {out}")
    # 续跑/重跑会把同一个 (solver, scene) 写两遍 —— 保留**最后**一次。
    dedup: dict[tuple[str, int], dict] = {}
    for row in rows:
        dedup[(str(row["solver"]), int(row["scene_id"]))] = row
    rows = list(dedup.values())
    per = {}
    for row in rows:
        per.setdefault(str(row["solver"]), {}).setdefault(int(row["scene_id"]), []).append(row)
    errors, per_scene = {}, {}
    for solver, scenes in per.items():
        vals = [float(r[k]) for rs in scenes.values() for r in rs
                for k in ("state_error_a_m", "state_error_b_m")]
        errors[solver] = np.asarray(vals)
        per_scene[solver] = {s: float(np.mean([float(r[k]) for r in rs
                                               for k in ("state_error_a_m",
                                                         "state_error_b_m")]))
                             for s, rs in scenes.items()}
    print(f"radius 150 m，{len(per.get('coarse_to_fine', {}))} 个场景，"
          f"每个求解器 {len(next(iter(errors.values())))} 个折样本")
    for solver in SOLVERS:
        if solver in errors:
            print(f"  {solver:15s} {_describe(errors[solver])}")
    if len(per_scene) == len(SOLVERS):
        shared = sorted(set(per_scene[SOLVERS[0]]) & set(per_scene[SOLVERS[1]]))
        diff = np.array([per_scene["gated_topk"][s] - per_scene["coarse_to_fine"][s]
                         for s in shared])
        print(f"\n配对差（{len(shared)} 场景，负 = gated_topk 更准）："
              f" median {np.median(diff):+.1f}  "
              f"改善 {int(np.sum(diff < 0))}/{len(diff)} 场景")
        worst = sorted(shared, key=lambda s: -per_scene["coarse_to_fine"][s])[:8]
        print("  旧求解器最差的 8 个场景（coarse -> gated）：")
        for s in worst:
            print(f"    scene {s:3d}: {per_scene['coarse_to_fine'][s]:8.1f} -> "
                  f"{per_scene['gated_topk'][s]:8.1f}")
    (out / "state_ablation.json").write_text(
        json.dumps({"per_solver": {k: v.tolist() for k, v in errors.items()},
                    "per_scene": {k: {str(s): v for s, v in d.items()}
                                  for k, d in per_scene.items()}},
                   indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'state_ablation.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=150.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        return merge(args.out)
    run_shard(args.out, args.radius, int(args.shard), max(1, int(args.shards)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
