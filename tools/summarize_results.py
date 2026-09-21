#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the re-run paper results into a single human-readable summary.

Reads every CSV under ``results/`` (produced by ``tools/rerun_paper.py``) and
writes ``results/README.md`` with:

* the canonical main-comparison table, paired confidence intervals, and derived
  resource reductions used by the paper;
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
    add("> 由 `tools/rerun_paper.py` 以 canonical 论文参数生成，"
        "`tools/summarize_results.py` 归纳。主对比使用 MC=1000、seed=2026；"
        "辅助审计的试验数以各目录 `config.json` 为准。\n")

    # ------------------------------------------------------------------
    # Main comparison
    # ------------------------------------------------------------------
    main_rows = load_rows(mode_csv(res, "main"))
    if main_rows:
        m = by_method(main_rows)
        add("## 1. 主对比（Fig. 2）\n")
        add("| Method | P_D | Proposed - method (95% CI) | Bits (kbit) | Delay (ms) | Worst P_D |")
        add("|---|---|---|---|---|---|")
        order = ["proposed_c2f", "proposed_lagrangian", "global_topk_deflection",
                 "cost_aware_greedy", "exact_marginal_greedy",
                 "topk_deflection", "sense_sinr",
                 "single_best", "nearest", "shortest_bistatic", "random",
                 "all_neighbor"]
        for name in order:
            if name not in m:
                continue
            r = m[name]
            paired = "-"
            if r.get("paired_proposed_delta_P_D", "") != "":
                paired = (
                    f"{float(r['paired_proposed_delta_P_D']):+.4f} "
                    f"[{float(r['paired_proposed_delta_ci95_low']):+.4f}, "
                    f"{float(r['paired_proposed_delta_ci95_high']):+.4f}]"
                )
            add(f"| {name} | {fnum(r, 'P_D')} | {paired} | "
                f"{float(r['B_mean_bits']) / 1000.0:.2f} | "
                f"{fnum(r, 'T_mean_ms', 2) if 'T_mean_ms' in r else '-'} | "
                f"{fnum(r, 'actual_worst_target_P_D')} |")

        proposed_name = "proposed_c2f" if "proposed_c2f" in m else "proposed_lagrangian"
        if proposed_name in m and "all_neighbor" in m:
            p, a = m[proposed_name], m["all_neighbor"]
            try:
                pd_p, pd_a = float(p["P_D"]), float(a["P_D"])
                bb_p, bb_a = float(p["B_mean_bits"]), float(a["B_mean_bits"])
                tt_p, tt_a = float(p["T_mean_ms"]), float(a["T_mean_ms"])
                add("\n**派生数字（摘要口径）：**\n")
                add(f"- 保留 all-neighbor 检测概率：`{pct(pd_p, pd_a)}`")
                add(f"- 降低信令开销：`{pct(bb_a - bb_p, bb_a)}`")
                add(f"- 降低交换时延：`{pct(tt_a - tt_p, tt_a)}`")
                add(f"- 时延绝对值：all-neighbor `{tt_a:.1f}` ms → proposed `{tt_p:.1f}` ms")
            except (KeyError, ValueError):
                pass
            if "topk_deflection" in m:
                tk = m["topk_deflection"]
                try:
                    tt_tk = float(tk["T_mean_ms"])
                    add(f"- 比 Top-K Deflection 时延低：`{pct(tt_tk - tt_p, tt_tk)}`")
                except (KeyError, ValueError):
                    pass
        add("")

    # Control run under the conservative full-concurrent interference model.
    ctrl_rows = load_rows(mode_csv(res, "main_full_concurrent", "main"))
    if main_rows and ctrl_rows:
        m_new, m_old = by_method(main_rows), by_method(ctrl_rows)
        if "proposed_lagrangian" in m_new and "proposed_lagrangian" in m_old:
            add("### 1.1 干扰模型对照（orthogonal canonical vs full-concurrent ablation）\n")
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
                    f"（{pc - p0:+.4f}，论文 0.9481→0.9572）")
                add(f"- **C2F 匹配全量精化**：`{pc:.4f}` vs `{pf:.4f}`"
                    f"（差 {abs(pc - pf):.4f}），但 fine-grid 评估只用 `{fc:.0f}` / `{ff:.0f}`")
                add(f"- **fine-grid 评估减少**：`{pct(ff - fc, ff)}`"
                    f"（`{fc:.0f}` vs `{ff:.0f}`，论文 44.6%）")
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

    for axis in ["comm_model", "error_sigma", "residual_direct", "direct_cancellation"]:
        rows = load_rows(res / f"robustness_{axis}" / "robustness" / "robustness.csv")
        if rows:
            add(f"## 鲁棒性（Fig. 6，axis = {axis}）\n")
            add("| condition | Method | P_D | T (ms) |")
            add("|---|---|---|---|")
            for r in rows:
                add(f"| {r.get('condition_value', '-')} | {r.get('method', '-')} | "
                    f"{fnum(r, 'P_D')} | {fnum(r, 'T_mean_ms', 2)} |")
            add("")

    # ------------------------------------------------------------------
    # Theoretical refinements (new experiments)
    # ------------------------------------------------------------------
    # NOTE: every one of these must go through ``mode_csv``.  ``rerun_paper.py``
    # passes ``--out <results>/<subdir>`` and the CLI then writes to
    # ``<out>/<mode>/<mode>.csv``, i.e. ``<results>/<subdir>/<mode>/<mode>.csv``.
    # Reading ``<results>/<mode>/<mode>.csv`` instead silently picks up leftovers
    # from ad-hoc manual runs and reports a stale model as if it were current.
    add("## 5. 结构审计（有限实例，不构成保证）\n")

    bm_rows = load_rows(mode_csv(res, "belief-mismatch"))
    if bm_rows:
        add("### 5.1 belief 失配（truth vs belief）\n")
        add("| belief σ_p (m) | Method | P_D | capture |")
        add("|---|---|---|---|")
        for r in bm_rows:
            add(f"| {fnum(r, 'belief_sigma_pos_m', 0)} | {r.get('method', '-')} | "
                f"{fnum(r, 'P_D')} | {fnum(r, 'belief_capture_rate_mean', 3)} |")
        add("")

    fbl_rows = load_rows(mode_csv(res, "fbl-sweep"))
    if fbl_rows:
        add("### 5.2 有限块长可靠性（FBL）\n")
        add("| n_block | χ_mean | P_D | T (ms) |")
        add("|---|---|---|---|")
        for r in fbl_rows:
            add(f"| {fnum(r, 'n_block', 0)} | {fnum(r, 'selected_chi_mean')} | "
                f"{fnum(r, 'P_D')} | {fnum(r, 'T_mean_ms', 3)} |")
        add("")

    corr_rows = load_rows(mode_csv(res, "correlation-ablation"))
    if corr_rows:
        add("### 5.3 相关感知融合消融\n")
        add("| corr | P_D | D_mean | links |")
        add("|---|---|---|---|")
        for r in corr_rows:
            add(f"| {r.get('corr_enable', '-')} | {fnum(r, 'P_D')} | "
                f"{fnum(r, 'D_mean', 3)} | {fnum(r, 'selected_links_mean', 1)} |")
        add("")

    sub_rows = load_rows(mode_csv(res, "submodularity"))
    if sub_rows:
        add("### 5.4 有限实例的边际收益与曲率审计\n")
        add("| mono. viol. | submod. viol. | curvature | matroid reference |")
        add("|---|---|---|---|")
        for r in sub_rows:
            add(f"| {fnum(r, 'monotone_violation_rate')} | "
                f"{fnum(r, 'submodularity_violation_rate')} | "
                f"{fnum(r, 'curvature')} | {fnum(r, 'matroid_reference_bound')} |")
        add("")

    gap_rows = load_rows(mode_csv(res, "same-objective-gap"))
    if gap_rows:
        gaps = []
        for r in gap_rows:
            try:
                gaps.append(float(r["gap"]))
            except (KeyError, ValueError):
                continue
        if gaps:
            import statistics
            add("### 5.5 同目标 greedy-vs-oracle 间隙\n")
            add(f"- mean gap：`{statistics.mean(gaps):.4f}`，"
                f"median `{statistics.median(gaps):.4f}`，max `{max(gaps):.4f}`\n")

    ic_rows = load_rows(mode_csv(res, "interference-consistency"))
    if ic_rows:
        add("### 5.6 通信/感知干扰耦合与直射对消预算\n")
        add("**干扰记账（同口径 vs 解耦残差）**\n")
        add("| variant | near-far (dB) | I_comm/N0 (dB) | I_sense/N0 (dB, model) | "
            "I_comm/I_sense (dB) | γ^s (dB) | P_D | links |")
        add("|---|---|---|---|---|---|---|---|")
        for r in ic_rows:
            if r.get("group") != "bookkeeping" or r.get("method") != "proposed_lagrangian":
                continue
            add(f"| {r.get('variant', '-')} | {fnum(r, 'near_far_db', 1)} | "
                f"{fnum(r, 'comm_inr_db', 1)} | {fnum(r, 'sense_inr_table_db', 1)} | "
                f"{fnum(r, 'inr_ratio_db', 1)} | {fnum(r, 'sense_sinr_db', 1)} | "
                f"{fnum(r, 'P_D')} | {fnum(r, 'selected_links_mean', 1)} |")
        add("")
        add("> 不同口径的结果不可直接比较（论文用 active_set）。")
        add("> ⚠️ 本节原本还有一张「直射对消扫描」表（P_D 对 κ_dc 的曲线）。")
        add("> 它建立在被删除的 `interference.direct_cancellation_db` 上——那个常数")
        add("> 没有接收机实现支撑却撑着整个 SINR 分母，所以那张曲线是记账不是性能。")
        add("> 对消深度现在只有一个来源：TP-UIC 的**实测**残余。\n")

    # ------------------------------------------------------------------
    # Legacy vs corrected model, same operating point
    # ------------------------------------------------------------------
    im_rows = load_rows(res / "isac_models" / "isac_models.csv")
    if im_rows:
        add("## 6. 旧模型 vs 修正模型（同一运行口径，MC=1000）\n")
        add("> 由 `tools/compare_isac_models.py` 生成；所有模型都固定论文工作点")
        add("> （active_set + 论文运动学），因此差异只来自干扰记账与 SINR 保护项。\n")
        add("| model | method | P_D | links | T (ms) | bits | chi |")
        add("|---|---|---|---|---|---|---|")
        for r in im_rows:
            if r.get("method") not in ("proposed_lagrangian", "all_neighbor"):
                continue
            add(f"| {r.get('model', '-')} | {r.get('method', '-')} | "
                f"{fnum(r, 'P_D')} | {fnum(r, 'selected_links_mean', 1)} | "
                f"{fnum(r, 'T_mean_ms', 2)} | {fnum(r, 'B_mean_bits', 0)} | "
                f"{fnum(r, 'selected_chi_mean', 3)} |")
        add("")
        for name in ("legacy", "coupled + guard fix"):
            row = next((r for r in im_rows
                        if r.get("model") == name
                        and r.get("method") == "proposed_lagrangian"), None)
            if row:
                add(f"- **{name}**：proposed $P_D$ = `{fnum(row, 'P_D')}`，"
                    f"时延 `{fnum(row, 'T_mean_ms', 2)}` ms，链路 `{fnum(row, 'selected_links_mean', 1)}`")
        add("")

    out_path = res / "README.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
