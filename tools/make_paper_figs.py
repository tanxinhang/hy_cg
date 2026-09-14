"""Rebuild the paper's figure files from the freshly generated results.

The paper includes four figures:

    fig2_main_comparison.pdf  <- results/main/main/main.csv
    fig3_lambda_tradeoff.pdf  <- results/lambda-sweep/lambda-sweep/figs/lambda_paper.png
    fig5_comm_sweep.pdf       <- results/comm-sweep/comm-sweep/figs/comm_sweep_paper.png
    fig6_robustness.pdf       <- built here from the four robustness axes

``fig1`` (the framework diagram) is hand-drawn and is left untouched.

The main-comparison panel is re-rendered from its CSV so a ``--no-plots``
release run is sufficient.  Sweep panels are converted from the PNGs produced
by their experiment plotters.

Usage::

    python tools/make_paper_figs.py            # write into the paper's figs/
    python tools/make_paper_figs.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

ROOT = Path(__file__).resolve().parent.parent
PAPER_FIGS = ROOT / "Conference-LaTeX-template_10-17-19" / "figs"

METHODS = ["proposed_c2f", "global_topk_deflection", "cost_aware_greedy",
           "exact_marginal_greedy", "sense_sinr", "all_neighbor"]
STYLE = {
    "proposed_lagrangian": ("#1f77b4", "o", "Proposed"),
    "proposed_c2f": ("#1f77b4", "o", "Proposed C2F"),
    "topk_deflection": ("#ff7f0e", "s", "Top-K deflection"),
    "global_topk_deflection": ("#ff7f0e", "s", "Global Top-K"),
    "cost_aware_greedy": ("#9467bd", "D", "Cost-aware greedy"),
    "exact_marginal_greedy": ("#8c564b", "P", "Exact-marginal greedy"),
    "sense_sinr": ("#2ca02c", "^", "Sensing-SINR"),
    "single_best": ("#9467bd", "D", "Single-best"),
    "all_neighbor": ("#d62728", "v", "All-neighbour"),
}


def rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        print(f"  [skip] missing {path.relative_to(ROOT)}")
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def f(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def png_to_pdf(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"  [skip] missing {src.relative_to(ROOT)}")
        return False
    img = plt.imread(src)
    h, w = img.shape[0], img.shape[1]
    fig = plt.figure(figsize=(7.0, 7.0 * h / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(dst, dpi=300)
    plt.close(fig)
    return True


def build_fig2(out: Path, results_root: Path) -> bool:
    """Render the three-panel headline comparison from the released CSV."""
    rs = rows(results_root / "main" / "main" / "main.csv")
    by_method = {r["method"]: r for r in rs}
    active = [m for m in METHODS if m in by_method]
    if not active:
        print("  [skip] no main comparison data")
        return False

    labels = [STYLE[m][2] for m in active]
    x = list(range(len(active)))
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]}
    )
    panels = [
        ("P_D", r"Detection probability $P_D$", "P_D_ci95_half_width"),
        ("T_mean_ms", "Serial reporting delay (ms)", None),
        ("active_target_ratio_mean", "Active-target ratio", None),
    ]
    for panel_label, ax, (metric, ylabel, error_metric) in zip("abc", axes, panels):
        values = [f(by_method[m], metric) for m in active]
        colors = [STYLE[m][0] for m in active]
        kwargs: Dict[str, Any] = {}
        if error_metric is not None:
            kwargs.update(yerr=[f(by_method[m], error_metric) for m in active],
                          capsize=3)
        ax.bar(x, values, color=colors, **kwargs)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=7)
        ax.text(-0.16, 1.03, panel_label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom")
        if metric == "P_D":
            ax.set_ylim(0.35, 0.95)
        elif metric == "T_mean_ms":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return True


def build_fig6(out: Path, results_root: Path) -> bool:
    """Four-panel robustness figure, one panel per axis."""
    axes_spec = [
        ("comm_model", "condition_value", "Communication error model", False),
        ("error_sigma", "condition_value", "Soft-error scale", True),
        ("residual_direct", "condition_value", "Residual direct factor", True),
        ("direct_cancellation", "condition_value",
         r"Direct-path cancellation $\kappa_{\rm dc}$ (dB)", True),
    ]
    data = {}
    for axis, xkey, xlabel, numeric in axes_spec:
        rs = rows(results_root / f"robustness_{axis}" / "robustness" /
                  "robustness.csv")
        if rs:
            data[axis] = (rs, xkey, xlabel, numeric)
    if not data:
        print("  [skip] no robustness data")
        return False

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, (axis, *_rest) in zip(axs.ravel(), axes_spec):
        if axis not in data:
            ax.axis("off")
            continue
        rs, xkey, xlabel, numeric = data[axis]
        xs = []
        for r in rs:
            v = r[xkey]
            xs.append(float(v) if numeric else v)
        order = sorted(set(xs), key=(lambda v: v) if numeric else (lambda v: str(v)))
        xpos = {v: k for k, v in enumerate(order)}
        for m in METHODS:
            pts = [(xpos[float(r[xkey]) if numeric else r[xkey]], f(r, "P_D"))
                   for r in rs if r.get("method") == m]
            if not pts:
                continue
            pts.sort()
            color, marker, label = STYLE[m]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker,
                    color=color, label=label, markersize=3.5, linewidth=1.2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{v:g}" if numeric else str(v) for v in order],
                           fontsize=7, rotation=20 if not numeric else 0)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(r"$P_D$", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_title(axis.replace("_", " "), fontsize=8)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7,
                   frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--results", type=Path, default=ROOT / "results_release",
                    help="validated rerun root (default: results_release)")
    args = ap.parse_args()

    if not PAPER_FIGS.exists():
        raise SystemExit(f"paper figs dir not found: {PAPER_FIGS}")

    # Keep the previous figures so nothing is silently destroyed.
    backup = ROOT / "archive" / "paper_figs_pre_correction"
    if not backup.exists() and not args.dry_run:
        backup.mkdir(parents=True, exist_ok=True)
        for p in PAPER_FIGS.glob("*.pdf"):
            shutil.copy2(p, backup / p.name)
        print(f"backed up previous figures -> {backup.relative_to(ROOT)}")

    results_root = args.results.resolve()
    main_dst = PAPER_FIGS / "fig2_main_comparison.pdf"
    if args.dry_run:
        print(f"  would write {main_dst.name} from main/main/main.csv")
    elif build_fig2(main_dst, results_root):
        print(f"  wrote {main_dst.name}  <- main/main/main.csv")

    jobs = [
        (results_root / "lambda-sweep" / "lambda-sweep" / "figs" / "lambda_paper.png",
         PAPER_FIGS / "fig3_lambda_tradeoff.pdf"),
        (results_root / "comm-sweep" / "comm-sweep" / "figs" / "comm_sweep_paper.png",
         PAPER_FIGS / "fig5_comm_sweep.pdf"),
    ]
    for src, dst in jobs:
        if args.dry_run:
            print(f"  would write {dst.name} from {src.relative_to(ROOT)}")
            continue
        if png_to_pdf(src, dst):
            print(f"  wrote {dst.name}  <- {src.relative_to(ROOT)}")

    dst6 = PAPER_FIGS / "fig6_robustness.pdf"
    if args.dry_run:
        print(f"  would build {dst6.name} from the four robustness axes")
    elif build_fig6(dst6, results_root):
        print(f"  wrote {dst6.name}  <- 4 robustness axes")

    # The old vertical robustness variant is superseded.
    old_vert = PAPER_FIGS / "fig6_robustness_vertical.pdf"
    if old_vert.exists() and not args.dry_run:
        print(f"  note: {old_vert.name} is stale and no longer referenced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
