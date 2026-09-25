#!/usr/bin/env python3
"""一个场景行的钱花在哪儿：逐臂计时（README §8 的成本模型由本脚本产出）。

为什么要单独测
--------------
Gate 1 的 41 行跑出来中位 1137 s/行，而"5 臂 × 2 假设 × 6 接收机"按单次 GLRT
0.95 s 估只有 ~60 s —— 差 20 倍。不搞清楚这个差，Gate 2 的档数只能拍脑袋定。

结论（半径 150 m，``--search-sigma 3``，串行，H1 一侧）：

    nominal          11.7 s      <- 12 次 held-out GLRT
    oracle           11.6 s
    self_fit         57.5 s      <- 12 次 setup + 2 次搜索 + 12 次 GLRT
    local_crossfit   57.9 s
    shared_crossfit  56.3 s

即**三个拟合臂占 88 %**，场景构造 + ``_pair`` 只有 1.5 s。8 路并发再乘
2.9× 的竞争惩罚 ⇒ 一档 41 行 = 6159 s ≈ 1.7 h。

用法::

    python studies/direction4/scripts/measure_scene_cost.py --scene 0 --radius 150
    python studies/direction4/scripts/measure_scene_cost.py --only nominal   # 并发压测
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools import gate_crossfit_target_state_map as gate  # noqa: E402
from tools import run_tpuic_receiver_benchmark as bench  # noqa: E402


def _args(**kw) -> SimpleNamespace:
    base = dict(
        out=ROOT / "studies/direction4/data/gate1",
        target=2, master_seed=20261007, area_xy=400.0, uavs=6,
        targets_count=3, m_rx=4, target_rcs=0.5, prior_sigma_m=150.0,
        glrt_radius=0.25, glrt_points=3, search_sigma=3.0,
        coarse_step_m=None, solver="coarse_to_fine",
        train_scenes=1, calibration_scenes=20, test_scenes=20, bootstrap=400,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--search-sigma", type=float, default=3.0)
    ap.add_argument("--only", default="",
                    help="只跑某几个臂（逗号分隔），用于并发压测")
    ns = ap.parse_args()
    want = set(ns.only.split(",")) if ns.only else None

    args = _args(search_sigma=ns.search_sigma)
    cfg = gate._cfg(args)
    t0 = time.perf_counter()
    truth, belief0, base = bench._scene(
        cfg, ns.scene, Path(args.out) / "scenes" / f"scene_{ns.scene:05d}.npz")
    t_scene = time.perf_counter() - t0
    belief, _ = gate._belief_for_radius(cfg, truth, belief0, args.target,
                                        ns.radius, ns.scene)
    belief_geom = belief.as_geometry(truth)
    receivers = tuple(range(int(cfg.scale.M)))

    t0 = time.perf_counter()
    pairs = [[gate._pair(cfg, truth, belief_geom, base, args, ns.scene, rx, look)
              for rx in receivers] for look in range(2)]
    looks = [[p[0] for p in look] for look in pairs]
    print(f"scene={t_scene:.1f}s  pairs={time.perf_counter() - t0:.1f}s"
          f"  (H1 一侧，H0 再乘 2)", flush=True)

    def timed(name, fn):
        if want is not None and name not in want:
            return None
        t = time.perf_counter()
        out = fn()
        print(f"{name:16s} {time.perf_counter() - t:8.1f}s", flush=True)
        return out

    timed("nominal", lambda: gate._arm_nominal(cfg, looks, args.target))
    timed("oracle", lambda: gate._arm_oracle(
        cfg, looks, args.target, belief_geom, truth))
    timed("self_fit", lambda: gate._arm_self_fit(
        cfg, looks, args.target, belief, receivers, belief_geom, args))
    timed("local_crossfit", lambda: gate._arm_local_crossfit(
        cfg, looks, args.target, belief, receivers, belief_geom, args))
    res = timed("shared_crossfit", lambda: gate._arm_shared_crossfit(
        cfg, looks, args.target, belief, receivers, belief_geom, args))
    if res is not None:
        print("evaluations:", res[1].get("evaluations"),
              "converged:", res[1].get("converged"), flush=True)


if __name__ == "__main__":
    main()
