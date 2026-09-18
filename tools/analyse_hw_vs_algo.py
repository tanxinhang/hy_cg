"""Hardware gain vs algorithmic interference handling -- the closing argument.

The question this answers
-------------------------
An earlier probe arm ("gain15") showed that simply *assuming* +15 dB of radar
net hardware gain lifts the worst-target P_D a long way.  That is a hardware
assumption, not a contribution: it must not be the thing that closes the gap.
This script measures the alternative -- can the **algorithmic** paths (direct-path
cancellation ``kappa_dc`` and illuminator coordination/tx_penalty) reach the same
level with the hardware held at the release default of **0 dB**?

What it reads
-------------
One coordination-experiment CSV per (geometry, RCS, kappa_dc, G_hw) cell, each
carrying the three paired arms ``uncoordinated`` / ``mask_only`` / ``sparse``.
The runner stamps every row with ``g_hw_db`` and ``kappa_dc`` so cells can be
pooled by label instead of by directory name.

What it reports
---------------
1.  Every cell as (G_hw, kappa) -> worst-P_D per arm, min/mean/std across seeds.
    The paper-relevant statistic is the WORST seed, not the mean.
2.  The **equivalent hardware gain** of the algorithm: interpolate the G_hw at
    which the *uncoordinated* arm reaches the level the *coordinated* arm
    already reaches at G_hw = 0.  This converts an algorithmic improvement into
    the dB of hardware it replaces, which is the only honest way to claim
    "the algorithm removes the need for the hardware assumption".
3.  A PNG with worst-P_D against G_hw, one curve per algorithmic setting.

Usage
-----
    python tools/analyse_hw_vs_algo.py --glob "results_coord_600m_*/*.csv"
"""

from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ARMS = ("uncoordinated", "mask_only", "sparse")
PD_REQ = 0.95

# Fallback labels for CSVs written before the per-row g_hw_db / kappa_dc stamps
# existed; keyed by the output directory name.  Keeping these lets an old run be
# pooled with a new one instead of silently dropping its rows.
LEGACY_CELLS = {
    "results_coordination_600m_rcs0.1": (0.0, 40.0),
    "results_coordination_800m_rcs0.05": (0.0, 40.0),
}


def _cell_key(row, path):
    """(G_hw dB, kappa dB) for one CSV row, falling back to the directory name."""
    try:
        return (float(row["g_hw_db"]), float(row["kappa_dc"]))
    except (KeyError, TypeError, ValueError):
        pass
    stem = Path(path).parent.name
    return LEGACY_CELLS.get(stem, (float("nan"), float("nan")))


def load(paths):
    """Return {(g_hw, kappa): {arm: {seed: worst_pd}}} pooled over the cells."""
    cells = {}
    meta = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = _cell_key(row, path)
                arm = row["arm"]
                if arm not in ARMS:
                    continue
                bucket = cells.setdefault(key, {a: {} for a in ARMS})
                bucket[arm][int(row["seed"])] = float(row["worst_pd"])
                meta.setdefault(key, {
                    "area_m": float(row["area_m"]), "rcs_m2": float(row["rcs_m2"]),
                    "scenario": row.get("scenario", "?"),
                    "looks": int(row.get("looks", 16)),
                    "penalty": float(row.get("penalty", 0.0)),
                    "sources": [],
                })
                if path not in meta[key]["sources"]:
                    meta[key]["sources"].append(path)
    return cells, meta


def equivalent_gain(hw_grid, hw_worst, target, algo_label):
    """G_hw at which the hardware-only curve first reaches ``target``.

    Linear interpolation in dB between the bracketing grid points.  Returns
    ``None`` when the hardware curve never reaches ``target`` inside the grid,
    which is the informative outcome: it means no amount of *modelled* hardware
    gain in the swept range buys what the algorithm bought for free.
    """
    order = np.argsort(hw_grid)
    xs = np.asarray(hw_grid, dtype=float)[order]
    ys = np.asarray(hw_worst, dtype=float)[order]
    for i in range(len(xs) - 1):
        lo, hi = ys[i], ys[i + 1]
        if lo < target <= hi and hi > lo:
            frac = (target - lo) / (hi - lo)
            return xs[i] + frac * (xs[i + 1] - xs[i])
        if lo >= target:
            return xs[i]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", default=["results_coord_600m_*/*.csv"])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results_hw_vs_algo")
    ap.add_argument("--fig", type=Path,
                    default=ROOT / "results_hw_vs_algo.png")
    args = ap.parse_args()

    paths = []
    for pattern in args.glob:
        paths.extend(sorted(_glob.glob(str(ROOT / pattern))))
    if not paths:
        raise SystemExit(f"no CSVs matched {args.glob}")

    cells, meta = load(paths)
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("hardware gain vs algorithmic interference handling")
    print("=" * 92)
    print(f"{len(paths)} CSV(s), {len(cells)} cell(s)")
    geoms = {(m["area_m"], m["rcs_m2"], m["scenario"], m["looks"])
             for m in meta.values()}
    for g in sorted(geoms):
        print(f"  geometry: {g[0]:.0f} m / RCS {g[1]} m^2 / {g[2]} / L={g[3]}")
    print()

    # ---- table 1: every cell -------------------------------------------------
    print(f"{'G_hw':>7}{'kappa':>7} | "
          + "".join(f"{a:>26}" for a in ARMS))
    print(f"{'(dB)':>7}{'(dB)':>7} | "
          + "".join(f"{'worst/mean/std':>26}" for _ in ARMS))
    print("-" * 92)
    table1 = []
    for key in sorted(cells):
        g, k = key
        line = f"{g:>7.1f}{k:>7.1f} |"
        for arm in ARMS:
            v = np.asarray(list(cells[key][arm].values()), dtype=float)
            if v.size:
                line += f"{v.min():>12.4f}/{v.mean():.4f}/{v.std():.4f}"
            else:
                line += f"{'-':>26}"
            table1.append(dict(g_hw_db=g, kappa_dc=k, arm=arm,
                               n_seeds=int(v.size),
                               worst_min=float(v.min()) if v.size else None,
                               worst_mean=float(v.mean()) if v.size else None,
                               worst_std=float(v.std()) if v.size else None))
        print(line)
    print()

    # ---- table 2: the equivalent hardware gain ------------------------------
    print("=" * 92)
    print("equivalent hardware gain: what +X dB of G_hw the algorithm replaces")
    print("=" * 92)
    print(f"  criterion: reach worst-target P_D >= {PD_REQ} on the WORST seed")
    print()

    rows = []
    for kappa in sorted({k for _, k in cells}):
        hw = sorted(g for g, k in cells if k == kappa)
        if len(hw) < 2:
            continue
        for arm in ("mask_only", "sparse"):
            g0 = cells.get((0.0, kappa), {}).get(arm, {})
            if not g0:
                continue
            algo_level = min(g0.values())
            hw_levels = [min(cells[(g, kappa)]["uncoordinated"].values())
                         for g in hw]
            equiv = equivalent_gain(hw, hw_levels, algo_level, arm)
            rows.append(dict(kappa_dc=kappa, algo_arm=arm,
                             algo_worst_pd=algo_level, g_hw_equivalent_db=equiv))
            if equiv is None:
                verdict = (f"the hardware sweep never reaches {algo_level:.4f} "
                           f"within {min(hw):+.0f}..{max(hw):+.0f} dB")
            else:
                verdict = f"~{equiv:+.1f} dB of hardware gain"
            print(f"  kappa_dc={kappa:.0f} dB, {arm:>9}: "
                  f"worst {algo_level:.4f} at G_hw=0  <=>  {verdict}")
        # hardware-only ladder for context
        ladder = "  ".join(f"{g:+.0f}:{min(cells[(g, kappa)]['uncoordinated'].values()):.3f}"
                           for g in hw)
        print(f"    hardware-only ladder (uncoordinated) at kappa={kappa:.0f}: {ladder}")
        print()

    (args.out / "hw_vs_algo.json").write_text(
        json.dumps({"cells": table1, "equivalence": rows},
                   indent=2, default=str), encoding="utf8")
    with (args.out / "hw_vs_algo.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table1[0].keys()))
        w.writeheader()
        w.writerows(table1)
    print(f"wrote {args.out / 'hw_vs_algo.csv'} and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
