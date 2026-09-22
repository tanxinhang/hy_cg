#!/usr/bin/env python3
"""
Shard driver for ``run_tpuic_receiver_benchmark.py``.

Why this exists
---------------
One benchmark record (scene x boost x receiver x target x realisation) costs
~8 s wall on this machine, so the "formal" grid quoted in the benchmark
docstring (50+50 scenes x 64 realisations x 6 receivers x 3 targets x 5
boosts) is ~5.8e5 records, i.e. ~40 days serially.  It is not runnable.

The benchmark itself is left untouched -- it is the fixed exam.  This driver
only splits the work along an axis the benchmark already treats as
independent: the per-(receiver, target) summary key.

Correctness argument
--------------------
``_summarise`` groups by ``(direct_gain_boost_db, receiver, target, arm)``.
The threshold for a key is calibrated from that key's own calibration-split
H0 samples.  Therefore two shards with disjoint (receiver, target) sets
produce disjoint key sets, and

    summary(union of shard records) == union of summary(shard records)

bit for bit.  Scenes are seeded only by ``scene_id`` (not by shard), so every
shard rebuilds byte-identical scenes; the split into calibration/test is
therefore identical too.  No result depends on how the work was sharded.

Usage
-----
python tools/run_tpuic_receiver_shards.py --out studies/direction3/data/tpuic_receiver_v1 \
    --cal-scenes 12 --test-scenes 24 --realisations 8 \
    --receivers 0,1,2 --targets 0,1 \
    --direct-gain-boost-db 0,20,40 --workers 6
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "run_tpuic_receiver_benchmark.py"

# Hard cap on concurrent worker processes.  18 processes on a 16-core box was
# tried and abandoned: too many concurrent interpreters is fragile and hard to
# kill cleanly.  Throughput above ~8 workers is marginal anyway because the
# per-record cost is dominated by BLAS-bound work on a single 16384-dim vector.
SAFE_MAX_WORKERS = 8


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _typed(rows: list[dict]) -> list[dict]:
    """Restore the dtypes the benchmark wrote, so ``_summarise`` agrees."""
    out = []
    for row in rows:
        new = {}
        for key, value in row.items():
            if key in ("arm", "split", "selected_base_arm"):
                new[key] = value
            elif key in ("receiver", "target", "scene_id", "realisation",
                         "dof_real", "protect_dim", "n_coefficients",
                         "tested_protected", "n_protected_targets"):
                new[key] = int(value)
            else:
                new[key] = float(value)
        out.append(new)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--cal-scenes", type=int, default=12)
    p.add_argument("--test-scenes", type=int, default=24)
    p.add_argument("--realisations", type=int, default=8)
    p.add_argument("--receivers", default="0,1,2")
    p.add_argument("--targets", default="0,1")
    p.add_argument("--direct-gain-boost-db", default="0,20,40")
    p.add_argument("--p-fa", type=float, default=0.05)
    p.add_argument("--arms", default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--threads", type=int, default=2,
                   help="BLAS threads per worker process")
    p.add_argument("--master-seed", type=int, default=20260922)
    p.add_argument("--uavs", type=int, default=6)
    p.add_argument("--targets-count", type=int, default=3)
    p.add_argument("--target-rcs", type=float, default=0.05)
    p.add_argument("--area-xy", type=float, default=800.0)
    p.add_argument("--m-rx", type=int, default=4)
    p.add_argument("--direct-dd-sigma", type=float, default=0.10)
    p.add_argument("--belief-pos-sigma", type=float, default=20.0)
    p.add_argument("--belief-vel-sigma", type=float, default=3.0)
    p.add_argument("--max-protected-targets", type=int, default=3)
    p.add_argument("--python", default=sys.executable)
    return p


def main() -> None:
    args = build_parser().parse_args()

    receivers = [v.strip() for v in args.receivers.split(",") if v.strip()]
    targets = [v.strip() for v in args.targets.split(",") if v.strip()]
    requested = int(args.workers)
    workers = max(1, min(requested, SAFE_MAX_WORKERS))
    if workers != requested:
        print(f"clamped --workers {requested} -> {workers} "
              f"(SAFE_MAX_WORKERS={SAFE_MAX_WORKERS})", flush=True)

    # A shard owns exactly one target and a striped subset of receivers, so its
    # key set is a genuine cross product and is disjoint from every other
    # shard's.  (Taking the union of receivers AND targets per shard would
    # silently re-expand into a cross product and duplicate keys.)
    per_target = max(1, workers // max(1, len(targets)))
    shards: list[tuple[str, list[str]]] = []
    for t in targets:
        for i in range(per_target):
            chunk = receivers[i::per_target]
            if chunk:
                shards.append((t, chunk))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_root = out_dir / "shards"

    common = [
        args.python, "-u", str(BENCH),
        "--cal-scenes", str(args.cal_scenes),
        "--test-scenes", str(args.test_scenes),
        "--realisations", str(args.realisations),
        "--p-fa", str(args.p_fa),
        "--master-seed", str(args.master_seed),
        "--uavs", str(args.uavs),
        "--targets-count", str(args.targets_count),
        "--target-rcs", str(args.target_rcs),
        "--area-xy", str(args.area_xy),
        "--m-rx", str(args.m_rx),
        "--direct-gain-boost-db", str(args.direct_gain_boost_db),
        "--direct-dd-sigma", str(args.direct_dd_sigma),
        "--belief-pos-sigma", str(args.belief_pos_sigma),
        "--belief-vel-sigma", str(args.belief_vel_sigma),
        "--max-protected-targets", str(args.max_protected_targets),
    ]
    if args.arms:
        common += ["--arms", args.arms]

    env_prefix = {
        "OMP_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
    }

    import os
    env = dict(os.environ)
    env.update(env_prefix)
    env["PYTHONPATH"] = str(HERE.parent)

    procs = []
    t0 = time.perf_counter()
    for k, (t, chunk) in enumerate(shards):
        rx = ",".join(chunk)
        log = out_dir / f"shard_{k:02d}.log"
        handle = log.open("w", encoding="utf-8")
        cmd = common + [
            "--out", str(shards_root / f"shard_{k:02d}"),
            "--receivers", rx,
            "--targets", t,
        ]
        procs.append((k, (t, chunk), log,
                      subprocess.Popen(cmd, stdout=handle,
                                       stderr=subprocess.STDOUT, env=env)))
        print(f"[shard {k:02d}] target={t} rx={{{rx}}} -> {log.name}",
              flush=True)

    codes = []
    for k, shard, log, proc in procs:
        code = proc.wait()
        codes.append(code)
        print(f"[shard {k:02d}] exit={code}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"all shards finished in {elapsed / 60.0:.1f} min", flush=True)

    if any(c != 0 for c in codes):
        raise SystemExit("one or more shards failed; see shard_*.log")

    # Merge.  Shard key sets are disjoint, so the union of per-shard summaries
    # equals the summary of the union.
    rows: list[dict] = []
    for k, shard, log, _proc in procs:
        rows.extend(_typed(_read_csv(
            shards_root / f"shard_{k:02d}" / "records.csv")))
    print(f"merged {len(rows)} records", flush=True)

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent))  # so the merge step can import isac_sim
    import run_tpuic_receiver_benchmark as bench

    summaries, scene_rows = bench._summarise(rows, float(args.p_fa))
    bench._write_csv(out_dir / "records.csv", rows)
    bench._write_csv(out_dir / "summary.csv", summaries)
    bench._write_csv(out_dir / "scene_summary.csv", scene_rows)

    print("=" * 108)
    print(
        f"{'boost':>7} {'rx':>3} {'q':>3} {'arm':<20} "
        f"{'PFA':>7} {'PD':>7} {'AUC':>7} {'kappa_s':>9} {'eta_q':>8} {'xi':>7}"
    )
    for s in summaries:
        print(
            f"{s['direct_gain_boost_db']:7.1f} "
            f"{s['receiver']:3d} {s['target']:3d} "
            f"{s['arm']:<20} "
            f"{s['empirical_pfa']:7.3f} "
            f"{s['empirical_pd']:7.3f} "
            f"{s['auc']:7.3f} "
            f"{s['median_kappa_structural_db']:9.2f} "
            f"{s['median_eta_survive_q']:8.3f} "
            f"{s['median_conflict_index']:7.3f}"
        )
    print("=" * 108)
    print(f"summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
