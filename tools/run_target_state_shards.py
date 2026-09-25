#!/usr/bin/env python3
"""Shard driver for ``gate_crossfit_target_state_map.py``.

Why this exists
---------------
Measured cost (radius 150 m, ``--search-sigma 3``), not a guess
-------------------------------------------------------------
Serially one scene-row (H1 + H0) is ~390 s, and **88 % of that is the three
fitting arms** (``self_fit`` 57.5 s, ``local_crossfit`` 57.9 s,
``shared_crossfit`` 56.3 s per hypothesis; the two GLRT-only arms are 11.7 s
and 11.6 s).  Scene construction + ``_pair`` is 1.5 s, i.e. negligible.

Running 8 shards costs a **2.9x contention penalty** (1137 s median per row),
so one error radius = 41 rows x 1137 s / 8 = **~6159 s ≈ 1.7 h**.  Ruled out by
measurement: BLAS thread count (0.92--1.02 s per GLRT at 1/2/4/16 threads --
the wall is memory bandwidth, not cores) and ``--glrt-points 1`` (1.53x faster
per GLRT but the statistic drops 24 %, and GLRT is only 12 % of a row).

Correctness argument
--------------------
Scene rows are seeded only by ``scene_id`` (never by shard or radius), so every
shard rebuilds byte-identical scenes and identical H0/H1 pairs.  The threshold
for a (radius, method) key is calibrated from that key's own calibration-split
H0 rows, and the split into train/calibration/test depends only on
``scene_id``.  Therefore the merged summary is independent of how the work was
sharded.

Usage
-----
python tools/run_target_state_shards.py --out studies/direction4/data/gate1 \
    --error-m 0,150,400 --train-scenes 4 --calibration-scenes 10 \
    --test-scenes 20 --workers 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE / "gate_crossfit_target_state_map.py"

# Hard cap on concurrent worker processes (see run_tpuic_receiver_shards.py).
SAFE_MAX_WORKERS = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2,
                        help="每个 worker 的 BLAS 线程数")
    parser.add_argument("--error-m", default="0,150,400")
    parser.add_argument("--train-scenes", type=int, default=4)
    parser.add_argument("--calibration-scenes", type=int, default=20)
    parser.add_argument("--test-scenes", type=int, default=20)
    parser.add_argument("--target-rcs", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--extra", default="")
    args, rest = parser.parse_known_args()

    workers = max(1, min(int(args.workers), SAFE_MAX_WORKERS))
    args.out.mkdir(parents=True, exist_ok=True)
    common = [
        sys.executable, str(GATE), "--out", str(args.out),
        "--error-m", args.error_m,
        "--train-scenes", str(args.train_scenes),
        "--calibration-scenes", str(args.calibration_scenes),
        "--test-scenes", str(args.test_scenes),
        "--target-rcs", str(args.target_rcs),
        "--bootstrap", str(args.bootstrap),
        "--shards", str(workers),
    ] + rest

    # ``tools/*.py`` 运行时 sys.path[0] 是 tools/，不是仓库根 —— 不显式给
    # PYTHONPATH 子进程会 ImportError: isac_sim。
    #
    # BLAS 线程必须钉住：不设上限时 8 个 worker 各抢满核线程，在 16 核机器上
    # 互相踩 —— 实测 17 分钟没有一个场景行完成（单进程串行只要 ~10 min）。
    # 与 run_tpuic_receiver_shards.py 同一口径：workers * threads ≈ 核数。
    env = dict(os.environ, PYTHONPATH=str(HERE.parent))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = str(max(1, int(args.threads)))
    started = time.perf_counter()
    procs = []
    for shard in range(workers):
        cmd = common + ["--shard", str(shard)]
        print("launch:", " ".join(cmd[-6:]), flush=True)
        procs.append(subprocess.Popen(cmd, cwd=str(HERE.parent), env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"shard failed with codes {codes}")
    print(f"shards done in {time.perf_counter() - started:.1f}s")

    # train/calibration/test 的划分只由 scene_id 决定，所以合并这一步**必须**
    # 拿到和分片完全相同的三个数字，否则 role 会错位、门限会拿错折。
    merge = [
        sys.executable, str(GATE), "--out", str(args.out),
        "--error-m", args.error_m, "--shards", str(workers),
        "--train-scenes", str(args.train_scenes),
        "--calibration-scenes", str(args.calibration_scenes),
        "--test-scenes", str(args.test_scenes),
        "--target-rcs", str(args.target_rcs), "--summary-only",
    ] + rest
    subprocess.run(merge, cwd=str(HERE.parent), env=env, check=True)


if __name__ == "__main__":
    main()
