#!/usr/bin/env python3
"""量 G(delta) 的相关峰宽 —— 粗网格间距就是由它决定的。

为什么值得单独留一个脚本
------------------------
方案 §6 把粗网格间距定成 2*sigma_p = 300 m，隐含假设是"G 的峰有几
百米宽"。实测半高宽只有 25--75 m，于是粗网格落在峰内的概率≈0，搜索退化成
"在噪声包里挑最大的那个"（Gate-0 五个场景跑飞两个，真值处的目标函数明明高
6.4 倍）。本脚本把那个数字变成可复现的测量；结论本身由
``tests/test_target_state_map.py::test_solver_recovers_offsets_from_many_directions``
钉住。

用法
----
PYTHONPATH=<repo> python studies/direction4/scripts/measure_correlation_peak_width.py \
    --error-m 150 --scenes 1,3 --step 25 --half 600
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from isac_sim.receiver.target_state import (
    shift_belief_geometry,
    shifted_target_sources,
)
from tools import gate_crossfit_target_state_map as gate

TARGET = 2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=Path("studies/direction4/data/peak_width"))
    p.add_argument("--error-m", type=float, default=150.0)
    p.add_argument("--scenes", default="1,3")
    p.add_argument("--step", type=float, default=25.0)
    p.add_argument("--half", type=float, default=600.0)
    p.add_argument("--target-rcs", type=float, default=0.5)
    args = p.parse_args()

    cfg_args = SimpleNamespace(
        out=args.out, target=TARGET, master_seed=20261007, area_xy=400.0,
        uavs=6, targets_count=3, m_rx=4, target_rcs=args.target_rcs,
        prior_sigma_m=150.0, search_sigma=4.0, glrt_radius=0.25,
        glrt_points=3, train_scenes=1, calibration_scenes=2, test_scenes=2,
    )
    cfg = gate._cfg(cfg_args)
    receivers = tuple(range(int(cfg.scale.M)))
    args.out.mkdir(parents=True, exist_ok=True)

    for scene in (int(v) for v in args.scenes.split(",")):
        truth, belief0, base = gate.bench._scene(
            cfg, scene, args.out / f"scene_{scene:05d}.npz")
        belief, _ = gate._belief_for_radius(
            cfg, truth, belief0, TARGET, args.error_m, scene)
        bgeom = belief.as_geometry(truth)
        true_delta = np.asarray(truth.p_tgt[TARGET, :2], float) - np.asarray(
            bgeom.p_tgt[TARGET, :2], float)
        look = [gate._pair(cfg, truth, bgeom, base, cfg_args, scene, rx, 0)[0]
                for rx in receivers]
        scorers = [gate._setup(cfg, obs, TARGET)[2] for obs in look]

        def gain(delta):
            geom = shift_belief_geometry(bgeom, TARGET, np.asarray(delta, float))
            return float(np.sum([
                sc.energy(shifted_target_sources(
                    cfg, geom, int(rx), TARGET, obs.targets_belief),
                    tested_only=True)
                for rx, sc, obs in zip(receivers, scorers, look)]))

        gain(np.zeros(2))
        t0 = time.perf_counter()
        gain(np.zeros(2))
        print(f"scene {scene}: one 6-receiver evaluation "
              f"{(time.perf_counter()-t0)*1000:.1f} ms")
        ts = np.arange(-args.half, args.half + 1e-9, args.step)
        unit = true_delta / np.linalg.norm(true_delta)
        for name, direction in (("radial", unit),
                                ("tangential", np.array([-unit[1], unit[0]]))):
            vals = np.asarray([gain(true_delta + t * direction) for t in ts])
            peak, floor = float(vals.max()), float(np.median(vals))
            half = floor + 0.5 * (peak - floor)
            above = np.flatnonzero(vals >= half)
            fwhm = float(ts[above].max() - ts[above].min()) if above.size else float("nan")
            print(f"  {name:11s} peak={peak:9.2f} median={floor:7.2f} "
                  f"FWHM={fwhm:5.0f} m  contrast={peak/max(floor,1e-9):5.2f}x")


if __name__ == "__main__":
    main()
