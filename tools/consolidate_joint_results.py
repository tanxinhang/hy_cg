"""Consolidate joint-experiment CSVs; later files replace duplicate rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    indexed = {}
    fields = None
    for path in args.inputs:
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                fields = fields or list(row)
                indexed[(int(row["trial"]), row["arm"])] = row
    rows = [indexed[key] for key in sorted(indexed)]
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "joint_tpuic_coordination_audited.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    arms = sorted({row["arm"] for row in rows})
    summary = {"sources": [os.path.abspath(p) for p in args.inputs], "arms": {}}
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        summary["arms"][arm] = {
            "n": len(sub),
            **{key: float(np.mean([float(r[key]) for r in sub])) for key in (
                "p_d", "p_fa", "truth_worst_pd", "truth_objective", "n_tx"
            )},
        }
    summary["paired"] = {}
    for receiver in ("constant", "tpuic"):
        for variant in ("naive", "robust"):
            a, b = receiver + "_serial_" + variant, receiver + "_joint_" + variant
            av = {int(r["trial"]): r for r in rows if r["arm"] == a}
            bv = {int(r["trial"]): r for r in rows if r["arm"] == b}
            ids = sorted(set(av) & set(bv))
            block = {}
            for key in ("p_d", "p_fa", "truth_worst_pd", "truth_objective", "n_tx"):
                delta = np.asarray([
                    float(bv[i][key]) - float(av[i][key]) for i in ids
                ])
                se = float(np.std(delta, ddof=1) / math.sqrt(len(delta)))
                mean = float(np.mean(delta))
                block[key] = {
                    "mean": mean, "se": se,
                    "ci95": [mean - 1.96 * se, mean + 1.96 * se],
                }
            summary["paired"][receiver + "_" + variant] = block
    summary["audit"] = {
        "max_joint_tx": max(
            int(r["n_tx"]) for r in rows if "_joint_" in r["arm"]
        ),
        "strict_gain_pass": all(
            all((not h.get("accepted")) or float(h["gain"]) >= 1e-8
                for h in json.loads(r["history_json"]))
            for r in rows
        ),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print("wrote", os.path.abspath(csv_path))
    print(json.dumps(summary["paired"]["tpuic_robust"], indent=2))
    print("audit", summary["audit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
