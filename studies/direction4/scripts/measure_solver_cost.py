"""逐求解器计时：同一批真实场景上 ``coarse_to_fine`` vs ``gated_topk``。

为什么单独测，而不是从 Gate 的 ``runtime_s`` 列里读
-------------------------------------------------
``runtime_s`` 是一整行（五条臂 + 场景构造），且 Gate 1 是在 8 路并发下跑的
（2.9× 争用惩罚）—— 拿它当基线会把"并发噪声"当成"求解器差异"。这里在同一台
机器、同一批场景、串行、交替顺序各跑两遍，只测**拟合**本身。

用法：python studies/direction4/scripts/measure_solver_cost.py --scenes 0,1,2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from _targetstate_common import (  # noqa: E402
    TARGET, make_cfg, make_pair, make_scorers, make_world,
)
from isac_sim.receiver.target_state import fit_shared_target_offset  # noqa: E402

RECEIVERS = tuple(range(6))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default="0,1,2")
    parser.add_argument("--radius-m", type=float, default=150.0)
    parser.add_argument("--search-sigma", type=float, default=3.0)
    args = parser.parse_args()

    for scene in (int(s) for s in args.scenes.split(",")):
        cfg = make_cfg()
        truth, belief, base, bgeom = make_world(cfg, scene=scene,
                                                radius_m=args.radius_m)
        look = [make_pair(cfg, truth, bgeom, base, rx, scene=scene)[0]
                for rx in RECEIVERS]
        scorers = make_scorers(cfg, look)
        kwargs = dict(receivers=RECEIVERS, belief_geometry=bgeom,
                      scorers=scorers, prior_sigma_m=150.0,
                      search_sigma=args.search_sigma)
        print(f"\nscene {scene} (radius {args.radius_m:g} m, "
              f"search_sigma {args.search_sigma:g})")
        rows = {}
        for wave in (("coarse_to_fine", "gated_topk"),
                     ("gated_topk", "coarse_to_fine")):
            for solver in wave:
                started = time.perf_counter()
                fit = fit_shared_target_offset(cfg, look, TARGET, belief,
                                               solver=solver, **kwargs)
                rows.setdefault(solver, []).append(
                    (time.perf_counter() - started, int(fit.evaluations),
                     int(fit.receiver_evaluations), np.asarray(fit.delta_xy_m)))
        for solver, runs in rows.items():
            best = min(runs, key=lambda r: r[0])
            print(f"  {solver:15s} {best[0]:7.2f}s  "
                  f"candidates={best[1]:4d}  receiver_evals={best[2]:5d}  "
                  f"delta={np.round(best[3], 2).tolist()}")
        times = {s: min(r[0] for r in v) for s, v in rows.items()}
        print(f"  加速比 coarse/gated = "
              f"{times['coarse_to_fine'] / times['gated_topk']:.2f}x")
        evals = {s: min(r[2] for r in v) for s, v in rows.items()}
        print(f"  评价次数比 coarse/gated = "
              f"{evals['coarse_to_fine'] / evals['gated_topk']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
