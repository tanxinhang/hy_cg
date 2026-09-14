"""Build the target-local fusion V1 paper figures from released CSV/JSON data.

The script is intentionally self-contained and deterministic.  It reads only
the V1 release tree and writes SVG (editable master), PDF (LaTeX), and PNG
(visual QA) outputs into the conference manuscript's ``figs`` directory.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results_target_local_v1"
OUT = ROOT / "Conference-LaTeX-template_10-17-19" / "figs"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#7A5195"
GREY = "#6B7280"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.transparent": False,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def val(row: dict[str, str], key: str) -> float:
    return float(row[key])


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="bottom")
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_main() -> None:
    rows = read_csv(DATA / "main" / "main.csv")
    keys = ["proposed_c2f_adaptive_pd", "exact_marginal_greedy", "sense_sinr"]
    names = ["V1 C2F", "Exact\nmarginal", "Sensing\nSINR"]
    colors = [BLUE, ORANGE, GREEN]
    data = {r["method"]: r for r in rows}

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.18),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.05]})
    x = range(3)

    pd = [val(data[k], "P_D") for k in keys]
    lo = [val(data[k], "P_D_ci95_low") for k in keys]
    hi = [val(data[k], "P_D_ci95_high") for k in keys]
    axes[0].bar(x, pd, color=colors, width=0.68,
                yerr=[[p - l for p, l in zip(pd, lo)],
                      [h - p for p, h in zip(pd, hi)]], capsize=3,
                error_kw={"elinewidth": 0.8, "capthick": 0.8})
    axes[0].set_xticks(list(x), names)
    axes[0].set_ylabel(r"Mean detection probability, $P_D$")
    y0 = min(lo) - 0.006
    y1 = max(hi) + 0.006
    axes[0].set_ylim(max(0.0, y0), min(1.0, y1))
    axes[0].axhline(0.95, color=GREY, linestyle="--", linewidth=0.8)
    for i, y in enumerate(pd):
        axes[0].text(i, y + 0.0021, f"{y:.4f}", ha="center", fontsize=6.8)
    panel(axes[0], "a")

    worst = [val(data[k], "actual_worst_target_P_D") for k in keys]
    axes[1].bar(x, worst, color=colors, width=0.68)
    axes[1].set_xticks(list(x), names)
    axes[1].set_ylabel(r"Worst-target $P_D$")
    axes[1].set_ylim(max(0.0, min(worst) - 0.012), min(1.0, max(worst) + 0.012))
    for i, y in enumerate(worst):
        axes[1].text(i, y + 0.0015, f"{y:.3f}", ha="center", fontsize=6.8)
    panel(axes[1], "b")

    refs = ["Exact-marginal", "Sensing-SINR"]
    baseline_rows = [data["exact_marginal_greedy"], data["sense_sinr"]]
    gains = [val(row, "paired_reference_delta_P_D") for row in baseline_rows]
    lows = [val(row, "paired_reference_delta_ci95_low") for row in baseline_rows]
    highs = [val(row, "paired_reference_delta_ci95_high") for row in baseline_rows]
    yy = [1, 0]
    axes[2].errorbar(gains, yy,
                     xerr=[[g - l for g, l in zip(gains, lows)],
                           [h - g for g, h in zip(gains, highs)]],
                     fmt="o", color=BLUE, markersize=5, capsize=3)
    axes[2].axvline(0, color=GREY, linewidth=0.8)
    axes[2].set_yticks(yy, refs)
    axes[2].set_xlabel(r"Paired V1 gain in $P_D$ (95% CI)")
    span_lo = min(lows + [0.0])
    span_hi = max(highs + [0.0])
    pad = max(0.002, 0.12 * (span_hi - span_lo))
    axes[2].set_xlim(span_lo - pad, span_hi + pad)
    axes[2].xaxis.set_major_locator(MaxNLocator(4))
    for g, y in zip(gains, yy):
        axes[2].text(g + 0.00025, y + 0.10, f"{g:+.4f}", fontsize=6.8)
    panel(axes[2], "c")

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.27, top=0.91, wspace=0.47)
    save(fig, "fig2_v1_main_comparison")


def build_fusion_ablation() -> None:
    rows = read_csv(DATA / "overview" / "fusion-rule" / "fusion-rule.csv")
    keys = ["max_in_rate", "max_min_rate", "nearest_target"]
    names = ["Max-in-rate", "Max-min-rate", "Nearest-target"]
    colors = [GREY, ORANGE, BLUE]
    data = {r["fusion_rule"]: r for r in rows}
    packets = {}
    for key in keys:
        with (DATA / "overview" / "packetization" / f"{key}.json").open(encoding="utf-8") as h:
            packets[key] = json.load(h)["summary"]

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.22))
    x = range(3)

    pd = [val(data[k], "P_D") for k in keys]
    lo = [val(data[k], "P_D_ci95_low") for k in keys]
    hi = [val(data[k], "P_D_ci95_high") for k in keys]
    axes[0].bar(x, pd, color=colors, width=0.68,
                yerr=[[p-l for p, l in zip(pd, lo)], [h-p for p, h in zip(pd, hi)]],
                capsize=3, error_kw={"elinewidth": 0.8, "capthick": 0.8})
    axes[0].set_xticks(list(x), names, rotation=18, ha="right")
    axes[0].set_ylabel(r"Detection probability, $P_D$")
    axes[0].set_ylim(0.78, 1.00)
    panel(axes[0], "a")

    reports = [val(data[k], "selected_links_mean") for k in keys]
    delays = [val(data[k], "T_mean_ms") for k in keys]
    w = 0.36
    axes[1].bar([i-w/2 for i in x], reports, width=w, color=PURPLE, label="Reports")
    axes[1].bar([i+w/2 for i in x], delays, width=w, color=GREEN, label="Delay (ms)")
    axes[1].set_xticks(list(x), names, rotation=18, ha="right")
    axes[1].set_ylabel("Mean count / ms")
    axes[1].set_ylim(0, 19)
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    panel(axes[1], "b")

    assigned = [packets[k]["assigned_fusion_uavs_mean"] for k in keys]
    slots = [packets[k]["conflict_graph_slots_mean"] for k in keys]
    axes[2].bar([i-w/2 for i in x], assigned, width=w, color=BLUE,
                label="Fusion UAVs")
    axes[2].bar([i+w/2 for i in x], slots, width=w, color=RED,
                label="Conflict slots")
    axes[2].set_xticks(list(x), names, rotation=18, ha="right")
    axes[2].set_ylabel("Mean count")
    axes[2].set_ylim(0, 18)
    axes[2].legend(frameon=False, ncol=1, loc="upper right")
    panel(axes[2], "c")

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.28, top=0.92, wspace=0.38)
    save(fig, "fig3_v1_fusion_ablation")


def dual_axis(ax: plt.Axes, x: list[float], primary: list[float], secondary: list[float],
              xlabel: str, secondary_label: str, secondary_color: str = ORANGE,
              xlabels: list[str] | None = None) -> None:
    ax.plot(x, primary, "o-", color=BLUE, label=r"$P_D$")
    ax.set_ylabel(r"$P_D$", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_ylim(min(primary) - 0.025, 1.005)
    ax.set_xlabel(xlabel)
    if xlabels is not None:
        ax.set_xticks(x, xlabels)
    twin = ax.twinx()
    twin.spines["right"].set_visible(True)
    twin.plot(x, secondary, "s--", color=secondary_color, label=secondary_label,
              markersize=4)
    twin.set_ylabel(secondary_label, color=secondary_color)
    twin.tick_params(axis="y", colors=secondary_color)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def build_operating_envelope() -> None:
    pred_all = read_csv(DATA / "prediction-stress" / "prediction_stress.csv")
    pred = [r for r in pred_all if r["method"] == "proposed_c2f_adaptive_pd"]
    comm = read_csv(DATA / "overview" / "communication" / "communication.csv")
    geom = read_csv(DATA / "overview" / "geometry" / "geometry.csv")
    lamb = read_csv(DATA / "overview" / "lambda" / "lambda.csv")
    block = read_csv(DATA / "overview" / "blocklength" / "blocklength.csv")
    scale = read_csv(DATA / "overview" / "scale" / "scale.csv")

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.45))
    specs = [
        (pred, "belief_sigma_pos_m", "belief_capture_rate_mean", "Position error std. (m)", "Belief capture"),
        (comm, "R_min_mbps", "feasible_target_ratio_mean", r"$R_{\min}$ (Mbit/s)", "Feasible targets"),
        (geom, "area_xy_m", "selected_links_mean", "Square side (m)", "Reports"),
        (lamb, "lambda_c", "selected_links_mean", r"Communication price $\lambda_c$", "Reports"),
        (block, "n_block", "T_mean_ms", "Blocklength (uses)", "Delay (ms)"),
        (scale, "M", "selected_links_mean", "Number of UAVs", "Reports"),
    ]
    for letter, ax, (rows, xkey, skey, xlabel, slabel) in zip("abcdef", axes.ravel(), specs):
        xx = [val(r, xkey) for r in rows]
        yy = [val(r, "P_D") for r in rows]
        ss = [val(r, skey) for r in rows]
        dual_axis(ax, xx, yy, ss, xlabel, slabel)
        panel(ax, letter)
        if xkey in {"n_block", "area_xy_m"}:
            ax.tick_params(axis="x", rotation=15)

    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.96,
                        hspace=0.56, wspace=0.66)
    save(fig, "fig4_v1_operating_envelope")


def main() -> None:
    build_main()
    build_fusion_ablation()
    build_operating_envelope()
    print(f"Wrote V1 figures to {OUT}")


if __name__ == "__main__":
    main()
