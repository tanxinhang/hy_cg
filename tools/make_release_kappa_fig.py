"""Release-gauge kappa sweep figure (600 m / RCS 0.1 / MC=1000).

Three panels, all from ``run_isac_sim --mode main`` CSVs:

  (a) P_D vs kappa for the three paper methods      -> where is the knee
  (b) communication overhead (bits / delay) vs kappa -> the efficiency story
  (c) proposed's paired advantage vs kappa           -> does kappa eat the story

Data source (do not hand-edit the numbers; re-run the sweep and re-plot):
  results_lowrcs_500_800_main/main/main.csv   (kappa=40)
  results_lowrcs_main_kappa60/main/main.csv   (kappa=60)
  results_lowrcs_main_kappa80/main/main.csv   (kappa=80)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    40: ROOT / "results_lowrcs_500_800_main" / "main" / "main.csv",
    60: ROOT / "results_lowrcs_main_kappa60" / "main" / "main.csv",
    80: ROOT / "results_lowrcs_main_kappa80" / "main" / "main.csv",
}

# Negative control: tx_penalty=0.2 with the gate OFF.  Same kappa as the k=40
# baseline, so it is drawn as a hollow marker at kappa=40.  See
# HARDWARE_FREE_CLOSURE.md sec. 7.3 -- it is strictly harmful.
NEG_CONTROL = ROOT / "results_lowrcs_main_sparse" / "main" / "main.csv"

KAPPAS = (40, 60, 80)

# The release method.  NOTE: in 19-method runs the harness designates
# ``proposed_c2f`` as the paired reference, so the CSV ``paired_*_delta``
# columns do NOT refer to this method.  Panel (c) therefore uses marginal
# differences computed from P_D directly.
RELEASE_METHOD = "proposed_c2f_adaptive_pd"

# method key -> (display label, colour, marker)
SERIES = {
    "proposed_c2f_adaptive_pd": ("proposed (V1)", "#1D9E75", "o"),
    "exact_marginal_greedy": ("exact_marginal", "#BA7517", "s"),
    "sense_sinr": ("sense_sinr", "#888888", "^"),
}


def _f(row, col):
    try:
        return float(row.get(col, ""))
    except (TypeError, ValueError):
        return float("nan")


def load():
    """Return ``{kappa: {method: row}}`` for the kappas whose CSV exists."""
    out = {}
    for k, path in RUNS.items():
        if not path.exists():
            continue
        with open(path, newline="") as fh:
            out[k] = {r["method"]: r for r in csv.DictReader(fh)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results_release_kappa600.png")
    args = ap.parse_args()

    data = load()
    if not data:
        raise SystemExit("no main.csv found; run the kappa sweep first")
    ks = [k for k in KAPPAS if k in data]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    ax0, ax1, ax2 = axes

    # ---- (a) P_D vs kappa -------------------------------------------------
    for method, (label, colour, marker) in SERIES.items():
        xs, ys, los, his = [], [], [], []
        for k in ks:
            if method not in data[k]:
                continue
            r = data[k][method]
            xs.append(k)
            ys.append(_f(r, "P_D"))
            los.append(_f(r, "P_D_ci95_low"))
            his.append(_f(r, "P_D_ci95_high"))
        ax0.errorbar(xs, ys, yerr=[np.subtract(ys, los), np.subtract(his, ys)],
                     label=label, color=colour, marker=marker, markersize=5,
                     linewidth=1.6, capsize=2.5)
    ax0.axhline(0.95, color="#A32D2D", linestyle="--", linewidth=1.0)
    ax0.text(79.5, 0.9565, "0.95 requirement", fontsize=7.5,
             color="#A32D2D", ha="right")
    # knee annotation
    if 60 in data and "proposed_c2f_adaptive_pd" in data[60]:
        p40 = _f(data[40]["proposed_c2f_adaptive_pd"], "P_D")
        p60 = _f(data[60]["proposed_c2f_adaptive_pd"], "P_D")
        p80 = _f(data[80]["proposed_c2f_adaptive_pd"], "P_D")
        ax0.annotate("", xy=(60, p60), xytext=(40, p40),
                     arrowprops=dict(arrowstyle="->", color="#1D9E75",
                                     lw=1.0, ls=":"))
        ax0.text(44, (p40 + p60) / 2 + 0.008,
                 f"+{p60 - p40:.3f}", fontsize=8, color="#1D9E75")
        ax0.text(60.5, p80 - 0.028, f"60→80 only +{p80 - p60:.3f}",
                 fontsize=8, color="#666666")
    ax0.set_xlabel(r"$\kappa_{dc}$ (dB)")
    ax0.set_ylabel(r"pooled $P_D$")
    ax0.set_title("(a) detection: knee at 60 dB", fontsize=9.5, loc="left")
    ax0.set_xticks(ks)
    ax0.set_xlim(35, 85)
    ax0.set_ylim(0.60, 0.99)
    ax0.grid(alpha=0.25, linewidth=0.5)
    ax0.legend(fontsize=8, frameon=False, loc="lower right")

    # ---- (b) overhead vs kappa -------------------------------------------
    method = "proposed_c2f_adaptive_pd"
    bits = [_f(data[k][method], "B_mean_bits") for k in ks]
    ax1.plot(ks, bits, marker="o", color="#1D9E75", linewidth=1.8,
             label="proposed bits")
    ax1b = ax1.twinx()
    obs = [_f(data[k][method], "selected_observations_mean") for k in ks]
    ax1b.plot(ks, obs, marker="s", color="#534AB7", linewidth=1.4,
              linestyle="--", label="observations")
    for k, b in zip(ks, bits):
        ax1.annotate(f"{b:.0f}", (k, b), textcoords="offset points",
                     xytext=(0, 7), fontsize=8, ha="center", color="#1D9E75")
    if len(bits) > 1:
        ax1.annotate("", xy=(60, bits[1]), xytext=(40, bits[0]),
                     arrowprops=dict(arrowstyle="->", color="#A32D2D",
                                     lw=1.0, ls=":"))
        ax1.text(41, (bits[0] + bits[1]) / 2,
                 f"−{100 * (1 - bits[1] / bits[0]):.0f}%", fontsize=8,
                 color="#A32D2D")
    ax1.set_xlabel(r"$\kappa_{dc}$ (dB)")
    ax1.set_ylabel("overhead (bit)", color="#1D9E75")
    ax1b.set_ylabel("observations / trial", color="#534AB7")
    ax1.set_title("(b) overhead collapses with kappa", fontsize=9.5,
                  loc="left")
    ax1.set_xticks(ks)
    ax1.grid(alpha=0.25, linewidth=0.5)

    # ---- (c) marginal advantage of the release method ---------------------
    # Marginal (not paired) differences: the paired columns in 19-method runs
    # reference ``proposed_c2f``, mixing them with the 3-method baseline would
    # be a gauge error.  See HARDWARE_FREE_CLOSURE.md sec. 6.
    for ref, label, colour in (("exact_marginal_greedy", "vs exact_marginal",
                                "#1D9E75"),
                               ("sense_sinr", "vs sense_sinr", "#A32D2D")):
        xs, ys = [], []
        for k in ks:
            d = data[k]
            if RELEASE_METHOD in d and ref in d:
                xs.append(k)
                ys.append(_f(d[RELEASE_METHOD], "P_D") - _f(d[ref], "P_D"))
        ax2.plot(xs, ys, label=label, color=colour, marker="o", markersize=5,
                 linewidth=1.6)
        for x, y in zip(xs, ys):
            ax2.annotate(f"{y:+.3f}", (x, y), textcoords="offset points",
                         xytext=(7, -4), fontsize=7.5, color=colour)

    # negative control: penalty without gate, at kappa=40
    if NEG_CONTROL.exists():
        with open(NEG_CONTROL, newline="") as fh:
            neg = {r["method"]: r for r in csv.DictReader(fh)}
        if RELEASE_METHOD in neg and "sense_sinr" in neg:
            y = _f(neg[RELEASE_METHOD], "P_D") - _f(neg["sense_sinr"], "P_D")
            ax2.plot([40], [y], marker="o", markersize=8, color="#A32D2D",
                     markerfacecolor="none", markeredgewidth=1.6)
            ax2.annotate("gate off\n" + f"{y:+.3f}", (40, y),
                         textcoords="offset points", xytext=(10, -14),
                         fontsize=7.5, color="#A32D2D")

    ax2.axhline(0.0, color="#333333", linewidth=1.0)
    ax2.text(ks[0] + 0.6, 0.0048, "proposed better ↑", fontsize=7.5,
             color="#1D9E75")
    ax2.text(ks[0] + 0.6, -0.0072, "proposed worse ↓", fontsize=7.5,
             color="#A32D2D")
    ax2.set_xlabel(r"$\kappa_{dc}$ (dB)")
    ax2.set_ylabel(r"marginal $\Delta P_D$")
    ax2.set_title("(c) kappa does not eat the advantage", fontsize=9.5,
                  loc="left")
    ax2.set_xticks(ks)
    ax2.set_xlim(36, 84)
    ax2.set_ylim(-0.075, 0.06)
    ax2.grid(alpha=0.25, linewidth=0.5)
    ax2.legend(fontsize=8, frameon=False, loc="lower left")

    fig.suptitle("600 m / RCS 0.1 / $G^{\\rm hw}=0$ dB / MC=1000 "
                 "(release gauge)", fontsize=10.5, y=1.01)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
