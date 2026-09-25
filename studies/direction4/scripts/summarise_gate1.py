#!/usr/bin/env python3
"""把 ``summary.json`` 打印成人能读的表，并按 §6 的判据给出通过/不通过。

用法：python studies/direction4/scripts/summarise_gate1.py data/gate1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

COLS = ("method", "threshold", "test_auc", "test_pd", "empirical_pfa",
        "oracle_auc_gap", "delta_auc_vs_nominal", "delta_pd_vs_nominal")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:8.4f}" if abs(v) < 1e6 else f"{v:8.3g}"
    return f"{str(v)[:8]:>8}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--csvs", action="store_true", help="同时列出 records.csv 的行数")
    args = p.parse_args()

    payload = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    proto = payload["protocol"]
    print("protocol:", json.dumps(proto, ensure_ascii=False))
    rows = payload["evaluation"]
    by_radius: dict[float, dict[str, dict]] = {}
    for r in rows:
        by_radius.setdefault(float(r["error_radius_m"]), {})[r["method"]] = r

    for radius in sorted(by_radius):
        block = by_radius[radius]
        print(f"\n=== error radius {radius:g} m "
              f"(n_cal={next(iter(block.values()))['n_calibration']}, "
              f"n_test={next(iter(block.values()))['n_test']}) ===")
        print("  ".join(f"{c[:12]:>12}" for c in ("method", *COLS[1:])))
        for name in ("nominal", "self_fit", "local_crossfit", "shared_crossfit",
                     "oracle"):
            if name not in block:
                continue
            r = block[name]
            print("  ".join(_fmt(r.get(c, "")).rjust(12) for c in COLS)
                  .replace(_fmt(name), f"{name:>12}"))
        for name in ("self_fit", "local_crossfit", "shared_crossfit"):
            r = block.get(name)
            if not r or "delta_auc_ci95" not in r:
                continue
            lo, hi = r["delta_auc_ci95"]
            pdl, pdh = r["delta_pd_ci95"]
            print(f"  {name:15s} ΔAUC 95%CI [{lo:+.4f},{hi:+.4f}]  "
                  f"ΔP_D 95%CI [{pdl:+.4f},{pdh:+.4f}]  "
                  f"{'CI 不含 0' if lo * hi > 0 else 'CI 跨 0 -> 分辨不出'}")
        shared = block.get("shared_crossfit", {})
        if "state_rmse_m" in shared:
            base = block.get("nominal", {}).get("state_rmse_m", float("nan"))
            print(f"  state: shared 中位 {shared.get('state_median_m', float('nan')):.1f} m, "
                  f"RMSE {shared['state_rmse_m']:.1f} m, "
                  f"p95 {shared['state_rmse_p95_m']:.1f} m | "
                  f"先验不动的基线 {base:.1f} m")
        nom = block.get("nominal", {})
        if nom:
            lo, hi = nom.get("pfa_ci95", (0.0, 0.0))
            print(f"  P_FA (设计值 {proto.get('p_fa')}): "
                  + ", ".join(
                      f"{n}={block[n]['empirical_pfa']:.3f}"
                      f"[{block[n].get('pfa_ci95', [0, 0])[0]:.3f},"
                      f"{block[n].get('pfa_ci95', [0, 0])[1]:.3f}]"
                      for n in ("nominal", "self_fit", "local_crossfit",
                                "shared_crossfit", "oracle") if n in block))

    # §6 判据
    print("\n=== 判据 ===")
    for radius in sorted(by_radius):
        block = by_radius[radius]
        nom = block.get("nominal", {})
        shared = block.get("shared_crossfit", {})
        local = block.get("local_crossfit", {})
        if not nom or not shared:
            continue
        d_auc = shared.get("delta_auc_vs_nominal")
        d_pd = shared.get("delta_pd_vs_nominal")
        ci = shared.get("delta_auc_ci95", (0.0, 0.0))
        gap = nom.get("oracle_auc_gap")
        closed = None
        if gap not in (None, 0.0) and gap:
            closed = 1.0 - shared.get("oracle_auc_gap", 0.0) / gap
        gate = (d_auc is not None and d_auc >= 0.05) or (
            d_pd is not None and d_pd >= 0.10)
        print(f"  radius {radius:>6g}: ΔAUC={d_auc:+.4f} ΔP_D={d_pd:+.4f} "
              f"({'过' if gate else '未过'} ΔAUC≥0.05 或 ΔP_D≥0.10)"
              if d_auc is not None else f"  radius {radius:>6g}: (缺数据)")
        if ci[0] * ci[1] <= 0:
            print(f"      ⚠ ΔAUC 的 95% CI 跨 0 —— n_test=20 只能分辨 ≥0.10，"
                  f"结论是'分辨不出'不是'没有'")
        if local.get("delta_auc_vs_nominal") is not None:
            ok = shared.get("delta_auc_vs_nominal", 0) >= local.get(
                "delta_auc_vs_nominal", 0)
            print(f"      shared ≥ local: {'是' if ok else '否'}")
        if closed is not None:
            print(f"      oracle AUC 缺口闭合 {closed * 100:.0f}%")

    if args.csvs:
        with (args.run / "records.csv").open(encoding="utf-8") as fh:
            n = sum(1 for _ in csv.DictReader(fh))
        print(f"\nrecords.csv rows = {n}")
    return None


if __name__ == "__main__":
    sys.exit(main())
