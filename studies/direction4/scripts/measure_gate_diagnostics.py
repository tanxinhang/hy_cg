"""阶段 A 落下的四个诊断量，在"改善/恶化"两类场景上各采几个样本。

阶段 A 的配对消融（``ablate_gated_topk_state.py``）给出：41 个场景里 27 个两
个求解器完全一致、11 个改善、**3 个恶化**。恶化的是不是"低证据峰"？—— 这个
判断决定阶段 D 的门控值不值得做，所以不能靠猜：

- 若恶化场景的 ``gain_over_zero`` / ``peak_gap`` 明显低于改善场景，则门控（
  证据不足就退回先验）能同时保住 11 个改善、砍掉 3 个恶化；
- 若它们一样大，则门控**无效**，得回头改搜索（比如把细化阶段的接收机数降下来
  换更多 seed）。

用法：python studies/direction4/scripts/measure_gate_diagnostics.py \
        --scenes 0,7,8,14,19,34,38,40
"""
from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from isac_sim.receiver.target_state import (  # noqa: E402
    fit_shared_target_offset,
)
from tools import gate_crossfit_target_state_map as gate  # noqa: E402


def _args(out: str, radius: float) -> Namespace:
    return Namespace(
        out=Path(out), target=2, error_m=f"{radius:g}", train_scenes=1,
        calibration_scenes=20, test_scenes=20, master_seed=20261007,
        area_xy=400.0, uavs=6, targets_count=3, min_uav_separation_m=20.0,
        m_rx=4, target_rcs=0.5, prior_sigma_m=150.0, search_sigma=3.0,
        coarse_step_m=None, solver="gated_topk", glrt_radius=0.25,
        glrt_points=3, p_fa=0.05, bootstrap=400, shard=0, shards=1,
        smoke=False, summary_only=False, screen_receivers=2, screen_keep=40,
        verify_receivers=4, verify_keep=12, top_k=3,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("studies/direction4/data/ablate_state"))
    parser.add_argument("--scenes", default="0,7,8,14,19,34,38,40")
    parser.add_argument("--radius", type=float, default=150.0)
    args = parser.parse_args()

    cfg = gate._cfg(_args(str(args.out), args.radius))
    receivers = tuple(range(int(cfg.scale.M)))
    print(f"{'scene':>5} {'solver':15s} {'err':>7} {'J0':>9} {'J_best':>9} "
          f"{'gain':>8} {'gap':>8} {'sup':>3} {'bnd':>3}")
    for scene in (int(s) for s in args.scenes.split(",")):
        ns = _args(str(args.out), args.radius)
        truth, belief0, base = gate._scene_with_retry(
            cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz")
        belief, _ = gate._belief_for_radius(cfg, truth, belief0, ns.target,
                                            args.radius, scene)
        bgeom = belief.as_geometry(truth)
        true_delta = np.asarray(truth.p_tgt[ns.target, :2], float) - np.asarray(
            bgeom.p_tgt[ns.target, :2], float)
        looks = [[gate._pair(cfg, truth, bgeom, base, ns, scene, rx, look)[0]
                  for rx in receivers] for look in range(2)]
        scorers = [[gate._setup(cfg, obs, ns.target)[2] for obs in look]
                   for look in looks]
        for solver in ("coarse_to_fine", "gated_topk"):
            ns.solver = solver
            for fold in (0, 1):
                fit = fit_shared_target_offset(
                    cfg, looks[fold], ns.target, belief, receivers=receivers,
                    belief_geometry=bgeom, scorers=scorers[fold],
                    prior_sigma_m=ns.prior_sigma_m,
                    search_sigma=ns.search_sigma, solver=solver,
                    solver_options=gate._solver_options(ns))
                err = float(np.linalg.norm(fit.delta_xy_m - true_delta))
                print(f"{scene:5d} {solver + '/' + 'AB'[fold]:15s} {err:7.1f} "
                      f"{fit.objective_at_zero:9.2f} {fit.objective:9.2f} "
                      f"{fit.gain_over_zero:8.2f} {fit.peak_gap:8.2f} "
                      f"{fit.receiver_support:3d} "
                      f"{'Y' if fit.boundary_hit else '.':>3}")
        print(f"{'':5} {'(先验不动)':15s} {float(np.linalg.norm(true_delta)):7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
