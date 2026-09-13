#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the re-run paper results into a single human-readable summary.

Reads every CSV under ``results/`` (produced by ``tools/rerun_paper.py``) and
writes ``results/README.md`` with:

* the main-comparison table and the derived headline numbers
  (all-neighbor retention, overhead/delay reduction, Top-K Deflection delay
  saving) that the paper quotes in the abstract and Section 4;
* the component-ablation table (Table II);
* the DD-domain / C2F refinement comparison (fine-grid evaluation saving);
* the lambda / comm / robustness sweep summaries.

Usage:
    python tools/summarize_results.py [--results results]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def mode_csv(res: Path, subdir: str, mode: str | None = None) -> Path:
    """Path of a mode's CSV.

    ``rerun_paper.py`` passes ``--out <results>/<subdir>`` and the CLI then
    writes to ``<out>/<mode>/<mode>.csv``, so the full path is
    ``<results>/<subdir>/<mode>/<mode>.csv``.
    """
    mode = subdir if mode is None else mode
    return res / subdir / mode / f"{mode}.csv"


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by_method(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        if "method" in r:
            out[r["method"]] = r
    return out


def fnum(row: Dict[str, str], key: str, nd: int = 4) -> str:
    if key not in row or row[key] == "":
        return "-"
    try:
        return f"{float(row[key]):.{nd}f}"
    except ValueError:
        return row[key]


def pct(a: float, b: float) -> str:
    if b == 0:
        return "-"
    return f"{100.0 * a / b:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    res = args.results

    lines: List[str] = []
    add = lines.append

    add("# 论文数据重跑汇总（isac_sim）\n")
    add("> 由 `tools/rerun_paper.py` 以论文参数（M=15, Q=10, UAV 30-60 m/s, "
        "目标 50-90 m/s, MC=1000, seed=2026）生成，`tools/summarize_results.py` 归纳。\n")

    # ------------------------------------------------------------------
    # Main comparison
    # ------------------------------------------------------------------
    main_rows = load_rows(mode_csv(res, "main"))
    if main_rows:
        m = by_method(main_rows)
        add("## 1. 主对比（Fig. 2）\n")
        add("| Method | P_D | Bits (kbit) | Delay (ms) | Worst P_D |")
        add("|---|---|---|---|---|")
        order = ["proposed_lagrangian", "topk_deflection", "sense_sinr",
                 "single_best", "nearest", "shortest_bistatic", "random",
                 "all_neighbor"]
        for name in order:
            if name not in m:
                continue
            r = m[name]
            add(f"| {name} | {fnum(r, 'P_D')} | "
                f"{fnum(r, 'B_mean_bits', 2) if 'B_mean_bits' in r else '-'} | "
                f"{fnum(r, 'T_mean_ms', 2) if 'T_mean_ms' in r else '-'} | "
                f"{fnum(r, 'actual_worst_target_P_D')} |")

        if "proposed_lagrangian" in m and "all_neighbor" in m:
            p, a = m["proposed_lagrangian"], m["all_neighbor"]
            try:
                pd_p, pd_a = float(p["P_D"]), float(a["P_D"])
                bb_p, bb_a = float(p["B_mean_bits"]), float(a["B_mean_bits"])
                tt_p, tt_a = float(p["T_mean_ms"]), float(a["T_mean_ms"])
                add("\n**派生数字（摘要口径）：**\n")
                add(f"- 保留 all-neighbor 检测概率：`{pct(pd_p, pd_a)}`（论文 94.8%）")
                add(f"- 降低信令开销：`{pct(bb_a - bb_p, bb_a)}`（论文 92.1%）")
                add(f"- 降低交换时延：`{pct(tt_a - tt_p, tt_a)}`（论文 94.8%）")
                add(f"- 时延绝对值：all-neighbor `{tt_a:.1f}` ms → proposed `{tt_p:.1f}` ms")
            except (KeyError, ValueError):
                pass
            if "topk_deflection" in m:
                tk = m["topk_deflection"]
                try:
                    tt_tk = float(tk["T_mean_ms"])
                    add(f"- 比 Top-K Deflection 时延低：`{pct(tt_tk - tt_p, tt_tk)}`"
                        f"（论文 29%）")
                except (KeyError, ValueError):
                    pass
        add("")

    # Control run under the conservative full-concurrent interference model.
    ctrl_rows = load_rows(mode_csv(res, "main_full_concurrent", "main"))
    if main_rows and ctrl_rows:
        m_new, m_old = by_method(main_rows), by_method(ctrl_rows)
        if "proposed_lagrangian" in m_new and "proposed_lagrangian" in m_old:
            add("### 1.1 干扰模型对照（active_set vs full_concurrent）\n")
            add("| 干扰模型 | proposed P_D | proposed 时延 (ms) | 保留率 |")
            add("|---|---|---|---|")
            for tag, m in [("active_set（新，活跃集并发）", m_new),
                           ("full_concurrent（保守全并发）", m_old)]:
                p, a = m["proposed_lagrangian"], m["all_neighbor"]
                try:
                    pp, pa = float(p["P_D"]), float(a["P_D"])
                    tt = float(p["T_mean_ms"])
                    add(f"| {tag} | {pp:.4f} | {tt:.2f} | {pp / pa:.1%} |")
                except (KeyError, ValueError):
                    continue
            add("")

    # ------------------------------------------------------------------
    # C2F / DD refinement
    # ------------------------------------------------------------------
    c2f_rows = load_rows(mode_csv(res, "c2f"))
    if c2f_rows:
        m = by_method(c2f_rows)
        add("## 2. C2F DD 精化（Fig. 2 / §4 DD 消融）\n")
        add("| Method | P_D | Fine-grid eval | T (ms) | Bits (kbit) |")
        add("|---|---|---|---|---|")
        for name in ["proposed_lagrangian", "proposed_c2f", "proposed_c2f_full", "all_neighbor"]:
            if name not in m:
                continue
            r = m[name]
            add(f"| {name} | {fnum(r, 'P_D')} | "
                f"{fnum(r, 'fine_eval_c2f_mean', 1) if 'fine_eval_c2f_mean' in r else '-'} | "
                f"{fnum(r, 'T_mean_ms', 2)} | {fnum(r, 'B_mean_bits', 0)} |")
        if {"proposed_lagrangian", "proposed_c2f", "proposed_c2f_full"} <= set(m):
            s0, sc, sf = m["proposed_lagrangian"], m["proposed_c2f"], m["proposed_c2f_full"]
            try:
                fc = float(sc["fine_eval_c2f_mean"])
                ff = float(sf["fine_eval_full_mean"])
                p0, pc, pf = (float(x["P_D"]) for x in (s0, sc, sf))
                add("")
                add(f"- **C2F 精化增益**：P_D `{p0:.4f}` → `{pc:.4f}`"
                    f"（{pc - p0:+.4f}，论文 0.6262→0.6648）")
                add(f"- **C2F 匹配全量精化**：`{pc:.4f}` vs `{pf:.4f}`"
                    f"（差 {abs(pc - pf):.4f}），但 fine-grid 评估只用 `{fc:.0f}` / `{ff:.0f}`")
                add(f"- **fine-grid 评估减少**：`{pct(ff - fc, ff)}`"
                    f"（`{fc:.0f}` vs `{ff:.0f}`，论文 41.1%）")
            except (KeyError, ValueError):
                pass
        add("")

    # ------------------------------------------------------------------
    # Ablation (Table II)
    # ------------------------------------------------------------------
    fair_rows = load_rows(mode_csv(res, "fair-ablation"))
    if fair_rows:
        add("## 3. 组件消融（Table II，固定预算）\n")
        add("| Variant | P_D | Bits (kbit) | Delay (ms) | Worst P_D |")
        add("|---|---|---|---|---|")
        for r in fair_rows:
            if r.get("method") != "proposed_lagrangian":
                continue
            add(f"| {r.get('variant', '-')} | {fnum(r, 'P_D')} | "
                f"{fnum(r, 'B_mean_bits', 2)} | {fnum(r, 'T_mean_ms', 2)} | "
                f"{fnum(r, 'actual_worst_target_P_D')} |")
        add("")

    dd_rows = load_rows(mode_csv(res, "dd-ablation"))
    if dd_rows:
        add("## 4. DD 机制消融\n")
        add("| Variant | Method | P_D |")
        add("|---|---|---|")
        for r in dd_rows:
            add(f"| {r.get('variant', '-')} | {r.get('method', '-')} | {fnum(r, 'P_D')} |")
        add("")

    # ------------------------------------------------------------------
    # Sweeps
    # ------------------------------------------------------------------
    for label, path, xkey in [
        ("λ_c 扫描（Fig. 3）", mode_csv(res, "lambda-sweep"), "lambda_cost"),
        ("R_min 扫描（Fig. 5）", mode_csv(res, "comm-sweep"), "R_min_mbps"),
    ]:
        rows = load_rows(path)
        if rows:
            add(f"## {label}\n")
            add("| x | Method | P_D | T (ms) | links |")
            add("|---|---|---|---|---|")
            for r in rows:
                add(f"| {fnum(r, xkey, 4)} | {r.get('method', '-')} | "
                    f"{fnum(r, 'P_D')} | {fnum(r, 'T_mean_ms', 2)} | "
                    f"{fnum(r, 'selected_links_mean', 1)} |")
            add("")

    for axis in ["comm_model", "error_sigma", "residual_direct"]:
        rows = load_rows(res / f"robustness_{axis}" / "robustness" / "robustness.csv")
        if rows:
            add(f"## 鲁棒性（Fig. 6，axis = {axis}）\n")
            add("| condition | Method | P_D | T (ms) |")
            add("|---|---|---|---|")
            for r in rows:
                add(f"| {r.get('condition_value', '-')} | {r.get('method', '-')} | "
                    f"{fnum(r, 'P_D')} | {fnum(r, 'T_mean_ms', 2)} |")
            add("")

    out_path = res / "README.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
