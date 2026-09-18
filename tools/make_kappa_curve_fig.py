"""Plot the direct-path-cancellation (kappa_dc) curve at 600 m / RCS 0.1.

Question the figure answers: "can we buy proposed's performance with an
interference-cancellation algorithm?"  The answer has two halves, so the figure
has two panels:

  (a) worst-target P_D against kappa_dc for four levers.  Shows that the
      receiver's cancellation depth saturates around 60 dB (the remaining floor
      is the self/multi-UAV residual, which kappa_dc cannot touch) while the
      hardware-gain lever is flat from the start.
  (b) the two differences that decide which family to spend on.  Below the
      noise/interference crossover the hardware lever beats the algorithm; above
      it the algorithm wins and keeps winning.

All numbers come from ``results_v1_lowrcs_kappa600_k*/summary.json``, written by
``tools/audit_v1_lowrcs_sweep.py --override interference.direct_cancellation_db=K``.
That sweep's cell 口径 is cap8 + detector_pd at MC=100, so this figure must not
be quoted alongside the MC=1000 main-comparison numbers without saying so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KAPPAS = (40, 50, 60, 80)
CELL = "area600_rcs0.1"
NOTE = "compact geometry, 600 m / RCS 0.1 m$^2$, MC=100 (cap8 + detector_pd)"

LEVERS = (
    ("base", "proposed, released", "#4d4d4d", "o", "-"),
    ("maxmin", "max-min reallocation", "#1D9E75", "s", "-"),
    ("looks128", "L=128 (8x CPI)", "#378ADD", "^", "-"),
    ("gain15", "$G_{hw}$ +15 dB", "#BA7517", "D", "-"),
)


def load(out_dir: Path):
    curves = {}
    for kappa in KAPPAS:
        path = out_dir / f"results_v1_lowrcs_kappa600_k{kappa}" / "summary.json"
        if not path.exists():
            raise SystemExit(
                f"missing {path}; run tools/audit_v1_lowrcs_sweep.py with "
                f"--override interference.direct_cancellation_db={kappa}")
        data = json.loads(path.read_text(encoding="utf8"))
        if CELL not in data:
            raise SystemExit(f"{path} has no cell {CELL!r} (has {sorted(data)})")
        curves[kappa] = data[CELL]
    return curves


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results_lowrcs_kappa600_kcurve.png")
    args = ap.parse_args()

    curves = load(args.dir)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    for key, label, color, marker, style in LEVERS:
        ys = [curves[k][key]["worst_pd"] for k in KAPPAS]
        ax0.plot(KAPPAS, ys, marker=marker, color=color, linestyle=style,
                 linewidth=1.8, markersize=6, label=label)
        ax0.annotate(f"{ys[-1]:.2f}", (KAPPAS[-1], ys[-1]), textcoords="offset points",
                     xytext=(6, -3), fontsize=8, color=color)
    ax0.axhline(0.95, color="#A32D2D", linestyle="--", linewidth=1.2)
    ax0.annotate("0.95 requirement", (67, 0.895), fontsize=8, color="#A32D2D")
    ax0.set_xlabel(r"direct-path cancellation depth $\kappa_{dc}$ (dB)")
    ax0.set_ylabel("worst-target $P_D$")
    ax0.set_ylim(0.0, 1.05)
    ax0.set_xticks(KAPPAS)
    ax0.set_xticklabels(
        [f"{k}\nrinr {curves[k]['base']['rinr_median']:.2f}" for k in KAPPAS],
        fontsize=9)
    ax0.grid(alpha=0.25, linewidth=0.5)
    ax0.legend(fontsize=8, frameon=False, loc="lower right", labelspacing=0.35)
    ax0.set_title("(a) what each lever buys", fontsize=10)

    algo = [curves[k]["maxmin"]["worst_pd"] - curves[k]["base"]["worst_pd"]
            for k in KAPPAS]
    hw = [curves[k]["gain15"]["worst_pd"] - curves[k]["base"]["worst_pd"]
          for k in KAPPAS]
    swap = [curves[k]["maxmin"]["worst_pd"] - curves[k]["gain15"]["worst_pd"]
            for k in KAPPAS]
    ax1.plot(KAPPAS, algo, marker="s", color="#1D9E75", linewidth=1.8,
             markersize=6, label="algorithm over baseline")
    ax1.plot(KAPPAS, hw, marker="D", color="#BA7517", linewidth=1.8,
             markersize=6, label="hardware over baseline")
    ax1.plot(KAPPAS, swap, marker="o", color="#7F77DD", linewidth=1.8,
             markersize=6, linestyle="--", label="algorithm $-$ hardware")
    ax1.axhline(0.0, color="#888780", linewidth=0.8)
    ax1.annotate("hardware leads", (42.5, -0.33), fontsize=8, color="#BA7517")
    ax1.annotate("algorithm leads", (64, 0.40), fontsize=8, color="#1D9E75")
    ax1.set_xlabel(r"direct-path cancellation depth $\kappa_{dc}$ (dB)")
    ax1.set_ylabel("worst-target $P_D$ difference")
    ax1.set_xticks(KAPPAS)
    ax1.grid(alpha=0.25, linewidth=0.5)
    ax1.legend(fontsize=8, frameon=False, loc="lower right", labelspacing=0.35)
    ax1.set_title("(b) which family to spend on", fontsize=10)

    fig.suptitle("Interference cancellation is a physical knob; the algorithm "
                 "complements it", fontsize=11)
    fig.text(0.5, 0.005, NOTE, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
