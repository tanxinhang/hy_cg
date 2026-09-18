"""Two-operating-point main comparison figure for the low-RCS scenario band.

Reads the released 4 km / 50 m^2 main table and the 600 m / RCS 0.1 m^2 rerun,
both produced by ``tools/rerun_target_local_v1.py --suite main`` (same preset,
same seed, same report regime, MC=1000), and plots

  (a) mean detection probability with 95% Wilson intervals,
  (b) communication cost (selected payload bits, log scale),
  (c) paired V1-minus-baseline differences with 95% confidence intervals.

The point of the figure is the change of the *comparison*, not of the absolute
level: against exact-marginal greedy V1 stays ahead at both points, while
against Sensing-SINR the difference goes from unresolved (4 km) to resolved and
negative (600 m).  Interval half widths are drawn from ``main.csv`` only; no
number here is recomputed.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHODS = [("proposed_c2f_adaptive_pd", "V1 (adaptive C2F)", "#1f4e9c"),
           ("exact_marginal_greedy", "exact-marginal greedy", "#d97706"),
           ("sense_sinr", "Sensing-SINR", "#2e8b57")]
POINTS = [("released 4 km / 50 m$^2$", None),
          ("new 600 m / 0.1 m$^2$", None)]


def load(path):
    return {r["method"]: r for r in csv.DictReader(path.open(encoding="utf8"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--released", type=Path,
                        default=Path("results_target_local_v1/main/main.csv"))
    parser.add_argument("--new", type=Path,
                        default=Path("results_lowrcs_500_800_main/main/main.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("results_lowrcs_500_800_main/main_comparison.png"))
    args = parser.parse_args()

    tables = {"released\n4 km / 50 m$^2$": load(args.released),
              "new\n600 m / 0.1 m$^2$": load(args.new)}
    figs, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    width = 0.26
    labels = list(tables)
    xs = range(len(labels))

    for index, (method, name, color) in enumerate(METHODS):
        offsets = [x + (index - 1) * width for x in xs]
        point = [tables[lab][method] for lab in labels]

        def value(row, key):
            return float(row[key])

        # (a) mean P_D with CI
        axes[0].bar(offsets, [value(r, "P_D") for r in point], width,
                    color=color, label=name,
                    yerr=[[value(r, "P_D") - value(r, "P_D_ci95_low") for r in point],
                          [value(r, "P_D_ci95_high") - value(r, "P_D") for r in point]],
                    capsize=3)
        # (b) selected payload
        axes[1].bar(offsets, [value(r, "B_mean_bits") / 1e3 for r in point],
                    width, color=color, label=name)
        # (c) paired difference against the V1 reference
        if method == "proposed_c2f_adaptive_pd":
            continue
        axes[2].bar(offsets, [value(r, "paired_reference_delta_P_D") for r in point],
                    width, color=color, label=name,
                    yerr=[[value(r, "paired_reference_delta_P_D")
                           - value(r, "paired_reference_delta_ci95_low") for r in point],
                          [value(r, "paired_reference_delta_ci95_high")
                           - value(r, "paired_reference_delta_P_D") for r in point]],
                    capsize=3)

    axes[0].axhline(0.95, ls="--", lw=1.0, color="0.35")
    axes[0].text(0.98, 0.93, "released requirement 0.95", ha="right", va="top",
                 fontsize=7.5, color="0.35")
    axes[0].set_ylim(0.0, 1.20)
    axes[0].set_ylabel("mean $P_D$ (MC=1000)")
    axes[0].set_title("(a) detection probability", fontsize=10)

    axes[1].set_yscale("log")
    axes[1].set_ylim(0.1, 60.0)
    axes[1].set_ylabel("selected payload (kbit, log scale)")
    axes[1].set_title("(b) communication cost", fontsize=10)
    for index, x in enumerate(xs):
        ratio = (float(tables[labels[index]]["sense_sinr"]["B_mean_bits"])
                 / float(tables[labels[index]]["proposed_c2f_adaptive_pd"]["B_mean_bits"]))
        axes[1].text(0.25 + 0.5 * index, 0.04,
                     f"{100 * (1 - 1 / ratio):.1f}% below\nSensing-SINR",
                     ha="center", va="bottom", fontsize=8,
                     transform=axes[1].transAxes)

    axes[2].axhline(0.0, color="0.3", lw=1.0)
    axes[2].set_ylim(-0.055, 0.065)
    axes[2].set_ylabel("paired $P_D$ difference vs V1")
    axes[2].set_title("(c) paired differences", fontsize=10)
    axes[2].text(0.5, 0.03, "shifted from unresolved (4 km) to resolved negative (600 m)",
                 transform=axes[2].transAxes, ha="center", va="bottom", fontsize=8,
                 color="#b91c1c")

    for ax in axes:
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(fontsize=8, loc="upper left")

    figs.suptitle("Prior-assisted confirmation at two operating points "
                  "(same preset, seed 2026, caps and report regime)", fontsize=10.5)
    figs.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figs.savefig(args.out, dpi=190)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
