"""Deployment-side vs RCS map for the 500-800 m low-RCS scenario.

Three series per RCS: the released ``base`` cell, the best already-implemented
software cell, and the ``gain15`` hardware reference axis.  Statistics are the
two that the sweep's protocol fixes: mean target P_D (solid) and worst-target
P_D averaged over trials (dashed).  Reads ``map.json`` produced by
``tools/analyse_lowrcs_500_800.py``.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SOFTWARE = ["looks64", "looks128", "maxmin", "maxmin_looks64", "maxmin_looks128"]
COLORS = {0.05: "#1f4e79", 0.1: "#c0504d", 0.2: "#4f8a3b"}
MARKERS = {"base": "o", "software": "s", "gain15": "^"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path,
                        default=Path("results/v1_lowrcs_500_800/map.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/v1_lowrcs_500_800/map.png"))
    args = parser.parse_args()
    entries = json.loads(args.map.read_text(encoding="utf8"))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for rcs in sorted({e["rcs_m2"] for e in entries}):
        rows = sorted([e for e in entries if e["rcs_m2"] == rcs],
                      key=lambda e: e["area_m"])
        area = [e["area_m"] for e in rows]
        base = [e["cells"]["base"] for e in rows]
        software = [max((e["cells"][k]["mean_target"], k) for k in SOFTWARE
                        if k in e["cells"])[0] for e in rows]
        software_worst = [e["cells"][max(
            (e["cells"][k]["mean_target"], k) for k in SOFTWARE
            if k in e["cells"])[1]]["worst_target"] for e in rows]
        gain = [e["cells"]["gain15"]["mean_target"] for e in rows]
        for ax, values, key, label in (
                (axes[0], software, "software", "best software cell"),
                (axes[1], gain, "gain15", "+15 dB hardware gain")):
            ax.plot(area, [e["mean_target"] for e in base],
                    marker=MARKERS["base"], color=COLORS[rcs], lw=1.4,
                    ls="--", label=f"base, σ={rcs:g}")
            ax.plot(area, values, marker=MARKERS[key], color=COLORS[rcs],
                    lw=1.8, label=f"{label}, σ={rcs:g}")
        axes[0].plot(area, software_worst, marker="x", color=COLORS[rcs],
                     lw=1.0, ls=":", label=f"worst target, σ={rcs:g}")

    for ax in axes:
        ax.axhline(0.80, color="k", lw=1.0, ls="-.")
        ax.axhline(0.05, color="grey", lw=0.9, ls=":")
        ax.set_xlabel("deployment side (m)")
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_xlim(470, 830)
    axes[0].set_ylabel("mean target P_D")
    axes[0].annotate("requirement 0.80", (505, 0.815), fontsize=8)
    axes[0].annotate("P_FA floor", (505, 0.065), fontsize=8, color="grey")
    axes[0].set_title("software knobs", fontsize=10)
    axes[1].set_title("hardware reference axis", fontsize=10)
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Low-RCS scenario: 500-800 m x RCS 0.05-0.2 m² "
                 "(seed 10917, MC=100)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
