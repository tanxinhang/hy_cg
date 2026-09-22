#!/usr/bin/env python3
"""
Analysis for ``run_tpuic_receiver_benchmark.py`` output (``summary.csv``).

Why a separate analysis
-----------------------
The benchmark reports one row per ``(boost, receiver, target, arm)``.  The
experiment's real question is not "what is P_D for receiver 3?" but "does the
arm ordering depend on the interference-target conflict index xi?".  That
requires a second pass over the per-key summaries, which this script does.

Reporting rules followed here
-----------------------------
* Absolute values are reported as median + [P10, P90] + n, never as a bare
  mean (naive standard errors on P_D understate the spread badly).
* Arm comparisons are PAIRED within a key (same boost / receiver / target),
  so scene geometry and noise draw cancel.  Never compare one arm's value
  against a different arm's pooled value.
* AUC is the threshold-free primary endpoint; P_D at the empirically
  calibrated threshold is secondary, because the calibration sample size is
  finite and the threshold is itself estimated.

Usage
-----
python tools/analyze_tpuic_receiver.py --summary <out>/summary.csv \
    --records <out>/records.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ARM_ORDER = (
    "no_ic",
    "plain_ls",
    "ridge_ls",
    "protected_ls",
    "tp_uic_stage1",
    "tp_uic_full",
    "detection_aware_tpuic",
    "perfect_channel",
)

XI_BINS = ((0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.01))
XI_LABELS = ("xi<0.05", "0.05-0.2", "0.2-0.5", "xi>=0.5")


def _load(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key == "arm":
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def _arm_sort_key(arm: str):
    return ARM_ORDER.index(arm) if arm in ARM_ORDER else 99


def _describe(values: list[float]) -> str:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return "     n/a"
    med = float(np.median(x))
    lo = float(np.quantile(x, 0.10))
    hi = float(np.quantile(x, 0.90))
    return f"{med:7.3f} [{lo:5.2f},{hi:5.2f}] n={x.size:3d}"


def _xi_bin(xi: float) -> int:
    for i, (lo, hi) in enumerate(XI_BINS):
        if lo <= xi < hi:
            return i
    return len(XI_BINS) - 1


def report(summary: list[dict], records: list[dict] | None) -> None:
    boosts = sorted({float(r["direct_gain_boost_db"]) for r in summary})
    arms = sorted({str(r["arm"]) for r in summary}, key=_arm_sort_key)

    key_index: dict[tuple, dict] = {}
    for r in summary:
        key_index[(float(r["direct_gain_boost_db"]), int(r["receiver"]),
                   int(r["target"]), str(r["arm"]))] = r
    keys = sorted({(float(r["direct_gain_boost_db"]), int(r["receiver"]),
                    int(r["target"])) for r in summary})

    print("=" * 96)
    print("A. Endpoint by arm and direct-INR boost (median [P10,P90] over "
          "(receiver,target) keys)")
    print("=" * 96)
    for boost in boosts:
        print(f"\n-- direct_gain_boost = {boost:+.0f} dB --")
        print(f"{'arm':<18}{'AUC':>26}{'P_D':>26}{'P_FA':>26}")
        for arm in arms:
            auc, pd, pfa = [], [], []
            for k in keys:
                if k[0] != boost:
                    continue
                row = key_index.get((k[0], k[1], k[2], arm))
                if row is None:
                    continue
                auc.append(float(row["auc"]))
                pd.append(float(row["empirical_pd"]))
                pfa.append(float(row["empirical_pfa"]))
            print(f"{arm:<18}{_describe(auc):>26}"
                  f"{_describe(pd):>26}{_describe(pfa):>26}")

    print()
    print("=" * 96)
    print("B. PAIRED difference vs plain_ls, same (boost,receiver,target) key")
    print("=" * 96)
    print(f"{'boost':>6} {'arm':<18}{'dAUC':>26}{'dP_D':>26}")
    for boost in boosts:
        for arm in arms:
            if arm == "plain_ls":
                continue
            da, dp = [], []
            for k in keys:
                if k[0] != boost:
                    continue
                base = key_index.get((k[0], k[1], k[2], "plain_ls"))
                other = key_index.get((k[0], k[1], k[2], arm))
                if base is None or other is None:
                    continue
                if not np.isfinite(base["auc"]) or not np.isfinite(other["auc"]):
                    continue
                da.append(float(other["auc"]) - float(base["auc"]))
                dp.append(float(other["empirical_pd"])
                          - float(base["empirical_pd"]))
            print(f"{boost:>6.0f} {arm:<18}{_describe(da):>26}"
                  f"{_describe(dp):>26}")

    print()
    print("=" * 96)
    print("C. Mechanism metrics (median [P10,P90] over keys)")
    print("=" * 96)
    print(f"{'boost':>6} {'arm':<18}{'kappa_structural_dB':>26}"
          f"{'eta_survive_q':>26}{'identifiable_frac':>24}")
    for boost in boosts:
        for arm in arms:
            ks, et, idf = [], [], []
            for k in keys:
                if k[0] != boost:
                    continue
                row = key_index.get((k[0], k[1], k[2], arm))
                if row is None:
                    continue
                ks.append(float(row["median_kappa_structural_db"]))
                et.append(float(row["median_eta_survive_q"]))
                idf.append(float(row["median_identifiable_fraction"]))
            print(f"{boost:>6.0f} {arm:<18}{_describe(ks):>26}"
                  f"{_describe(et):>26}{_describe(idf):>24}")

    print()
    print("=" * 96)
    print("D. AUC vs interference-target conflict index xi (keys binned by xi)")
    print("=" * 96)
    for boost in boosts:
        print(f"\n-- direct_gain_boost = {boost:+.0f} dB --")
        header = f"{'arm':<18}" + "".join(f"{lab:>22}" for lab in XI_LABELS)
        print(header)
        for arm in arms:
            cells = []
            for b in range(len(XI_BINS)):
                vals = []
                for k in keys:
                    if k[0] != boost:
                        continue
                    row = key_index.get((k[0], k[1], k[2], arm))
                    if row is None:
                        continue
                    if _xi_bin(float(row["median_conflict_index"])) != b:
                        continue
                    vals.append(float(row["auc"]))
                x = np.asarray([v for v in vals if np.isfinite(v)])
                cells.append(
                    f"{np.median(x):7.3f} (n={x.size:2d})" if x.size else
                    f"{'--':>7} (n= 0)"
                )
            print(f"{arm:<18}" + "".join(f"{c:>22}" for c in cells))

    if records:
        print()
        print("=" * 96)
        print("E. Direct INR actually realised (median over realisations)")
        print("=" * 96)
        by_boost = defaultdict(list)
        for r in records:
            by_boost[float(r["direct_gain_boost_db"])].append(
                float(r["direct_inr_db"]))
        for boost in boosts:
            x = np.asarray([v for v in by_boost[boost] if np.isfinite(v)])
            if x.size:
                print(f"  boost {boost:+.0f} dB -> INR median "
                      f"{np.median(x):7.2f} dB  "
                      f"[P10 {np.quantile(x, 0.10):6.2f}, "
                      f"P90 {np.quantile(x, 0.90):6.2f}] n={x.size}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--records", default=None)
    args = p.parse_args()

    summary = _load(Path(args.summary))
    records = _load(Path(args.records)) if args.records else None
    if not summary:
        raise SystemExit("empty summary")
    report(summary, records)


if __name__ == "__main__":
    main()
