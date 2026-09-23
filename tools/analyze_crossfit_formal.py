#!/usr/bin/env python3
"""Paired uncertainty analysis for formal cross-fitted MAP receiver gates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.run_tpuic_receiver_benchmark import _auc, _wilson


NAMES = ("two_cpi", "crossfit_map", "perfect_two_cpi")


def _metric(rows, name, threshold):
    h0 = np.asarray([float(r[f"{name}_h0"]) for r in rows])
    h1 = np.asarray([float(r[f"{name}_h1"]) for r in rows])
    return {
        "auc": float(_auc(h0, h1)),
        "pfa": float(np.mean(h0 > threshold)),
        "pd": float(np.mean(h1 > threshold)),
    }


def analyze(directory: Path, bootstrap: int, seed: int):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "records.csv").open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    counts = {split: sum(r["split"] == split for r in records)
              for split in ("train", "calibration", "test")}
    if counts != {"train": 1, "calibration": 40, "test": 40}:
        raise RuntimeError(f"unexpected split counts: {counts}")
    thresholds = {r["receiver"]: float(r["threshold"])
                  for r in summary["evaluation"]}
    test = [r for r in records if r["split"] == "test"]
    point = {name: _metric(test, name, thresholds[name]) for name in NAMES}
    rng = np.random.default_rng(seed)
    deltas = {"map_minus_base_auc": [], "map_minus_base_pd": [],
              "oracle_minus_map_auc": [], "oracle_minus_map_pd": []}
    for _ in range(bootstrap):
        sample = [test[i] for i in rng.integers(0, len(test), len(test))]
        values = {name: _metric(sample, name, thresholds[name]) for name in NAMES}
        deltas["map_minus_base_auc"].append(
            values["crossfit_map"]["auc"] - values["two_cpi"]["auc"])
        deltas["map_minus_base_pd"].append(
            values["crossfit_map"]["pd"] - values["two_cpi"]["pd"])
        deltas["oracle_minus_map_auc"].append(
            values["perfect_two_cpi"]["auc"] - values["crossfit_map"]["auc"])
        deltas["oracle_minus_map_pd"].append(
            values["perfect_two_cpi"]["pd"] - values["crossfit_map"]["pd"])
    intervals = {key: np.quantile(value, [0.025, 0.975]).tolist()
                 for key, value in deltas.items()}
    binomial = {name: {
        "pfa_wilson95": _wilson(value["pfa"], len(test)),
        "pd_wilson95": _wilson(value["pd"], len(test)),
    } for name, value in point.items()}
    return {"directory": str(directory), "split_counts": counts,
            "point": point, "paired_bootstrap95": intervals,
            "binomial95": binomial}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = [analyze(path, args.bootstrap, args.seed + index)
               for index, path in enumerate(args.directories)]
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
