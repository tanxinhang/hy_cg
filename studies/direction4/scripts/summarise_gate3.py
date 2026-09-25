#!/usr/bin/env python3
"""Gate 3 的跨目标汇总：``P_D^worst = min_q P_D,q``（方案 §12 Gate 3 的终点）。

为什么需要单独一个脚本
----------------------
``gate_crossfit_target_state_map.py`` 一次只跑一个 ``--target``，每个目标各出
一份 ``summary.json``。而 Gate 3 的判定口径是"三个目标里**最差**的那个"——
只报单目标的 P_D 会掩盖掉最弱那一环，这正是多目标场景要问的问题。

所以本脚本把三份 summary 按 (误差半径, 方法) 对齐，逐格取**跨目标最小值**：

    AUC_worst   = min_q AUC_q
    P_D_worst   = min_q P_D,q
    P_FA_worst  = max_q P_FA,q      <- 误警是"越小越好"，取最大
并标出每格的**瓶颈目标**（哪个 q 拖后腿），这一列比数字本身更有用。

用法::

    python studies/direction4/scripts/summarise_gate3.py \
        studies/direction4/data/gate3_t0/summary.json \
        studies/direction4/data/gate3_t1/summary.json \
        studies/direction4/data/gate3_t2/summary.json
    python studies/direction4/scripts/summarise_gate3.py --glob 'data/gate3_t*/summary.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

ARMS = ("nominal", "self_fit", "local_crossfit", "shared_crossfit", "oracle")


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else data.get("by_method", [])


def _target_of(path: str) -> str:
    """从目录名里认目标号（``gate3_t1`` / ``gate3_smoke_t1``）；认不出就用文件名。"""
    head = os.path.basename(os.path.dirname(os.path.abspath(path)))
    for token in reversed(head.replace("gate3_smoke_t", "gate3_t").split("_")):
        if token.startswith("t") and token[1:].isdigit():
            return token[1:]
    return head


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summaries", nargs="*", help="各目标的 summary.json")
    ap.add_argument("--glob", default="", help="例如 'data/gate3_t*/summary.json'")
    ap.add_argument("--json-out", default="", help="把结果也写成 json")
    ns = ap.parse_args()

    paths = list(ns.summaries)
    if ns.glob:
        paths += sorted(glob.glob(ns.glob))
    if not paths:
        raise SystemExit("没有输入：给若干 summary.json 或 --glob")

    cells: dict[tuple[float, str], list[tuple[str, dict]]] = defaultdict(list)
    for path in paths:
        target = _target_of(path)
        for entry in _load(path):
            key = (float(entry["error_radius_m"]), str(entry["method"]))
            cells[key].append((target, entry))

    radii = sorted({r for r, _ in cells})
    rows = []
    for radius in radii:
        for method in ARMS:
            items = cells.get((radius, method))
            if not items:
                continue
            auc = min(float(e.get("test_auc", float("nan"))) for _, e in items)
            pd = min(float(e.get("test_pd", float("nan"))) for _, e in items)
            pfa = max(float(e.get("empirical_pfa", float("nan"))) for _, e in items)
            worst_auc = min(items, key=lambda it: float(it[1].get("test_auc", 1e9)))[0]
            worst_pd = min(items, key=lambda it: float(it[1].get("test_pd", 1e9)))[0]
            rows.append({
                "error_radius_m": radius, "method": method,
                "n_targets": len(items),
                "auc_worst": auc, "worst_auc_target": worst_auc,
                "pd_worst": pd, "worst_pd_target": worst_pd,
                "pfa_worst": pfa,
            })

    print(f"{'radius':>7} {'method':<16} {'AUC_worst':>10} {'worst_q':>8} "
          f"{'P_D_worst':>10} {'worst_q':>8} {'P_FA_max':>9}")
    for r in rows:
        print(f"{r['error_radius_m']:>7g} {r['method']:<16} {r['auc_worst']:>10.4f} "
              f"{r['worst_auc_target']:>8} {r['pd_worst']:>10.3f} "
              f"{r['worst_pd_target']:>8} {r['pfa_worst']:>9.3f}")

    # 判据（方案 §13 口径搬到最差目标上）
    for radius in radii:
        base = {r["method"]: r for r in rows if r["error_radius_m"] == radius}
        if "nominal" not in base or "shared_crossfit" not in base:
            continue
        d_auc = base["shared_crossfit"]["auc_worst"] - base["nominal"]["auc_worst"]
        d_pd = base["shared_crossfit"]["pd_worst"] - base["nominal"]["pd_worst"]
        ok = (d_auc >= 0.05 or d_pd >= 0.10)
        print(f"\nradius={radius:g}: ΔAUC_worst={d_auc:+.4f}  "
              f"ΔP_D_worst={d_pd:+.3f}  -> {'过' if ok else '不过'}")

    if ns.json_out:
        with open(ns.json_out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        print(f"\nwrote {ns.json_out}")


if __name__ == "__main__":
    sys.exit(main())
