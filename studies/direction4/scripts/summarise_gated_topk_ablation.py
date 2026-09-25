"""配对汇总：``gated_topk`` vs ``coarse_to_fine`` 的状态误差 / 目标函数 / 运行时间。

用法（两个目录各放一份 ``records*.csv``，或直接用 Gate 1 的 records.csv 当基线）：

    python studies/direction4/scripts/summarise_gated_topk_ablation.py \\
        --base studies/direction4/data/gate1 \\
        --new  studies/direction4/data/ablate_gt_smoke

口径：
- 状态误差取 ``state_error_a_m`` / ``state_error_b_m``（两折各一个偏移，共 2n 个
  样本）；报中位 + P90 + 最大 + n（铁律：不报裸 se）。
- **配对**只在同一 ``scene_id`` 上做：跨 run 一律用同法名边际差。
- ``objective_a/b`` 是训练折上的目标函数值：**更高 = 搜索器找到了更好的峰**，
  与状态误差无关，用来区分"搜索更准"和"只是换了个峰"。
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

FIELDS = ("state_error_a_m", "state_error_b_m")


def _load(root: Path) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(root)):
        if name.startswith("records") and name.endswith(".csv"):
            with (root / name).open("r", newline="", encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
    return rows


def _state_errors(rows) -> np.ndarray:
    out = []
    for row in rows:
        for key in FIELDS:
            if row.get(key):
                out.append(float(row[key]))
    return np.asarray(out, dtype=float)


def _per_scene(rows) -> dict[int, float]:
    """一个场景一个样本：两折状态误差的平均（同一场景两折不是独立的）。"""
    out = {}
    for row in rows:
        vals = [float(row[k]) for k in FIELDS if row.get(k)]
        if vals:
            out[int(row["scene_id"])] = float(np.mean(vals))
    return out


def _describe(values: np.ndarray) -> str:
    if values.size == 0:
        return "n=0"
    return (f"median {np.median(values):8.1f}  p90 {np.percentile(values, 90):8.1f}"
            f"  max {values.max():8.1f}  mean {values.mean():8.1f}  n={values.size}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True,
                        help="coarse_to_fine 的记录目录")
    parser.add_argument("--new", type=Path, required=True,
                        help="gated_topk 的记录目录")
    args = parser.parse_args()

    base_rows, new_rows = _load(args.base), _load(args.new)
    base, new = _state_errors(base_rows), _state_errors(new_rows)
    print(f"base (coarse_to_fine) {args.base}: {_describe(base)}")
    print(f"new  (gated_topk)     {args.new }: {_describe(new)}")

    base_scene, new_scene = _per_scene(base_rows), _per_scene(new_rows)
    shared = sorted(set(base_scene) & set(new_scene))
    if shared:
        diff = np.array([new_scene[s] - base_scene[s] for s in shared])
        print(f"\n配对（{len(shared)} 个共有场景，负 = gated_topk 更准）：")
        print(f"  {_describe(diff)}")
        print(f"  改善的场景数 {int(np.sum(diff < 0))} / {len(diff)}")
        for s in shared:
            flag = "-" if new_scene[s] < base_scene[s] else "+"
            print(f"    scene {s:3d}: {base_scene[s]:8.1f} -> {new_scene[s]:8.1f} "
                  f" {flag}{abs(new_scene[s] - base_scene[s]):7.1f}")

    for label, rows in (("base", base_rows), ("new", new_rows)):
        runtime = np.array([float(r["runtime_s"]) for r in rows if r.get("runtime_s")])
        obj = np.array([float(r[k]) for r in rows for k in ("objective_a",
                        "objective_b") if r.get(k)])
        if runtime.size:
            print(f"\n{label} runtime_s: median {np.median(runtime):.0f} "
                  f"n={runtime.size}")
        if obj.size:
            print(f"{label} objective: median {np.median(obj):.3f} n={obj.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
