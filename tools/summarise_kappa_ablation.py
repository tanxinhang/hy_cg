"""Tabulate the kappa_dc (direct-path cancellation) ablation.

Answers the question "is kappa=40 dB an extra gain?" by separating two things
that are easy to conflate:

* **feasibility** -- what the sensing task does with no cancellation at all
  (kappa=0).  If P_D sits on top of P_FA there, kappa is not a performance
  knob but a prerequisite.
* **headroom** -- how much of the residual direct path is still above the
  processed echo at each kappa (from ``tools/probe_kappa_necessity.py``), i.e.
  whether the calibrated value is conservative or aggressive for *this*
  geometry.

Arms live in ``results_kappa_ablation/<tag>/main/``.  All arms share seed, MC,
preset and method, so differences are reported as paired per-trial deltas.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM_DIR = ROOT / "results_kappa_ablation"
COORD_DIR = ROOT / "results_coordwire"

# kappa -> (coord arm tag or None, external dir or None)
ARM_LABEL = {
    "k0": "kappa=0 (no cancellation)",
    "k10": "kappa=10",
    "k20": "kappa=20",
    "k30": "kappa=30",
    "k50": "kappa=50",
    "A_off": "kappa=40 (release default)",
    "J_coord_cap3": "kappa=40 + coordination, max_tx=3",
    "k0_coord_cap3": "kappa=0 + coordination, max_tx=3",
    "k50_coord_cap3": "kappa=50 + coordination, max_tx=3",
    "k60_coord_cap3": "kappa=60 + coordination, max_tx=3",
}

# Order for display: plain sweep first, then the coordination crosses.
ARM_ORDER = ["k0", "k10", "k20", "k30", "A_off", "k50",
             "k0_coord_cap3", "J_coord_cap3", "k50_coord_cap3", "k60_coord_cap3"]

METRIC_COLUMNS = {
    "P_D": "P_D",
    "P_FA": "P_FA",
    "bits": "B_mean_bits",
    "obs": "selected_observations_mean",
    "worst_pd": "actual_worst_target_P_D",
}


def arm_path(tag: str) -> Path:
    # The coordination arms were written before this ablation existed and live
    # at the top level as results_coordwire_<tag>/main/.
    if tag.startswith("A_") or tag.startswith("J_"):
        return ROOT / f"results_coordwire_{tag}" / "main"
    return ARM_DIR / tag / "main"


def read_main(arm_dir: Path) -> dict[str, float]:
    path = arm_dir / "main.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        return {}
    row = rows[0]
    out = {}
    for key, col in METRIC_COLUMNS.items():
        if col in row:
            try:
                out[key] = float(row[col])
            except (TypeError, ValueError):
                out[key] = math.nan
    return out


def read_trials(arm_dir: Path) -> list[dict[str, str]]:
    path = arm_dir / "trials.csv"
    if not path.exists():
        return []
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


def paired_pd_diff(base_rows, arm_rows):
    n = min(len(base_rows), len(arm_rows))
    diffs = []
    for i in range(n):
        try:
            db = float(base_rows[i]["detected"])
            da = float(arm_rows[i]["detected"])
            tot = float(base_rows[i]["total_targets"])
        except (KeyError, TypeError, ValueError):
            continue
        if tot <= 0:
            continue
        diffs.append((da - db) / tot)
    if not diffs:
        return math.nan, math.nan, 0
    mean = sum(diffs) / len(diffs)
    if len(diffs) > 1:
        var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
        se = math.sqrt(var / len(diffs))
    else:
        se = math.nan
    return mean, 1.96 * se, len(diffs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_kappa_ablation/summary.csv")
    args = ap.parse_args()

    base_tag = "A_off"
    base_rows = read_trials(arm_path(base_tag))

    rows = []
    for tag in ARM_ORDER:
        d = arm_path(tag)
        m = read_main(d)
        if not m:
            continue
        delta, hw, n = (math.nan, math.nan, 0)
        if tag != base_tag:
            delta, hw, n = paired_pd_diff(base_rows, read_trials(d))
        rows.append({
            "arm": tag,
            "label": ARM_LABEL.get(tag, tag),
            **{k: m.get(k, math.nan) for k in METRIC_COLUMNS},
            "paired_delta_P_D": delta,
            "paired_ci95_halfwidth": hw,
            "n_paired_trials": n,
        })

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"arms: {len(rows)}  ->  {out}")
    print()
    hdr = (f"{'arm':18s} {'P_D':>7s} {'worst':>7s} {'P_FA':>6s} "
           f"{'bits':>9s} {'obs':>6s} {'dP_D(paired)':>20s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["arm"] == base_tag:
            ds = "  (base)"
        elif math.isnan(r["paired_delta_P_D"]):
            ds = "  n/a"
        else:
            ds = f"{r['paired_delta_P_D']:+.4f}+/-{r['paired_ci95_halfwidth']:.4f}"
        print(f"{r['arm']:18s} {r['P_D']:7.4f} {r['worst_pd']:7.4f} {r['P_FA']:6.4f} "
              f"{r['bits']:9.1f} {r['obs']:6.1f} {ds:>20s}")

    # Feasibility verdict: is kappa=0 distinguishable from chance?
    k0 = next((r for r in rows if r["arm"] == "k0"), None)
    if k0:
        print()
        gap = k0["P_D"] - k0["P_FA"]
        print(f"feasibility check @ kappa=0: P_D {k0['P_D']:.4f} vs P_FA "
              f"{k0['P_FA']:.4f}  (gap {gap:+.4f})")
        if abs(gap) < 0.01:
            print("  -> indistinguishable from chance: kappa is a feasibility "
                  "prerequisite, NOT a performance gain.")
        else:
            print("  -> above chance; kappa is not strictly required for "
                  "detection at this geometry.")


if __name__ == "__main__":
    main()
