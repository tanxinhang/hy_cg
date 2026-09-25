#!/usr/bin/env python3
"""``--glrt-points`` 3 vs 1：省多少时间、掉多少统计量（README §8 用它否决了这条路）。

held-out 统计量走 :func:`target_neighbourhood_glrt`，它在 ±0.25 格的局部 DD
网格上取最大值（默认 3×3 = 9 次求值）。看起来是 9 倍的开销，像是最大的杠杆。

实测：单次调用 0.95 s → 0.63 s，只有 **1.53×** —— 因为真正贵的是 arm 装配与
白化（固定成本 ~0.59 s），9 个格点里每加一个只多 ~0.04 s。而统计量从 9.6007
掉到 7.2623（**−24.4 %**），且 GLRT 只占一个场景行的 12 % ⇒ 全局只省 4 %，
却要重标所有门限。所以不改。

顺带钉住另一件事：BLAS 线程数（OMP/OPENBLAS/MKL = 1/2/4/16）对单次 GLRT
毫无影响（0.92–1.02 s）⇒ 瓶颈是内存带宽不是核数，8 路并发的 2.9× 惩罚也
因此不能用"少给线程"来换。

用法::

    python studies/direction4/scripts/measure_glrt_grid_cost.py --scene 0
    for T in 1 2 4 16; do OMP_NUM_THREADS=$T ... python ... ; done
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools import gate_crossfit_target_state_map as gate  # noqa: E402
from tools import run_tpuic_receiver_benchmark as bench  # noqa: E402


def _args(points: int) -> object:
    from types import SimpleNamespace
    return SimpleNamespace(
        out=ROOT / "studies/direction4/data/gate1",
        master_seed=20261007, area_xy=400.0, uavs=6, targets_count=3, m_rx=4,
        target_rcs=0.5, prior_sigma_m=150.0, glrt_radius=0.25,
        glrt_points=points, target=2,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--reps", type=int, default=3)
    ns = ap.parse_args()

    results = {}
    for points in (3, 1):
        args = _args(points)
        cfg = gate._cfg(args)
        truth, belief0, base = bench._scene(
            cfg, ns.scene, Path(args.out) / "scenes" / f"scene_{ns.scene:05d}.npz")
        belief, _ = gate._belief_for_radius(cfg, truth, belief0, args.target,
                                            ns.radius, ns.scene)
        belief_geom = belief.as_geometry(truth)
        t0 = time.perf_counter()
        h1, _ = gate._pair(cfg, truth, belief_geom, base, args, ns.scene, 0, 0)
        t_pair = time.perf_counter() - t0
        vals, dt = [], 0.0
        for _ in range(int(ns.reps)):
            t0 = time.perf_counter()
            vals.append(gate._score(cfg, h1, args.target))
            dt = time.perf_counter() - t0
        results[points] = (dt, vals, t_pair)
        print(f"grid_points={points}: pair={t_pair:.1f}s  glrt={dt:.2f}s/call  "
              f"stat={vals[0]:.6f}  spread={max(vals) - min(vals):.2e}")

    d3, d1 = results[3][0], results[1][0]
    print(f"speedup x{d3 / d1:.2f}   "
          f"statistic change {(results[1][1][0] - results[3][1][0]) / results[3][1][0]:+.2%}")


if __name__ == "__main__":
    main()
