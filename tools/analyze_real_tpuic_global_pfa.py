#!/usr/bin/env python3
"""Analyze global PFA control using real TP-UIC benchmark statistics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def _quantize(x, bits, clip=30.0):
    if bits >= 16:
        return x
    levels = 2 ** bits
    return np.round(np.clip(x, 0, clip) * (levels - 1) / clip) * clip / (levels - 1)


def _matrix(rows, field, receivers):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(int(row["scene_id"]), int(row["realisation"]))][int(row["receiver"])] = float(row[field])
    return np.asarray([[values[r] for r in receivers] for _, values in sorted(grouped.items())
                       if all(r in values for r in receivers)])


def _auc(h0, h1):
    ranks = rankdata(np.r_[h0, h1], method="average")
    n0, n1 = len(h0), len(h1)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def analyze(records, alpha=0.10, bits=(2, 3, 16), injected_scale=None):
    with Path(records).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    receivers = sorted({int(r["receiver"]) for r in rows})
    cal_rows = [r for r in rows if r["split"] == "calibration"]
    test_scenes = sorted({int(r["scene_id"]) for r in rows if r["split"] == "test"})
    cut = len(test_scenes) // 2
    drift_ids, final_ids = set(test_scenes[:cut]), set(test_scenes[cut:])
    drift_rows = [r for r in rows if int(r["scene_id"]) in drift_ids]
    final_rows = [r for r in rows if int(r["scene_id"]) in final_ids]
    cal0 = _matrix(cal_rows, "stat_h0", receivers)
    drift0 = _matrix(drift_rows, "stat_h0", receivers)
    final0 = _matrix(final_rows, "stat_h0", receivers)
    final1 = _matrix(final_rows, "stat_h1", receivers)
    if injected_scale is not None:
        injected_scale = np.asarray(injected_scale, dtype=float)
        if injected_scale.shape != (len(receivers),):
            raise ValueError("injected_scale must contain one value per receiver")
        drift0 = drift0 * injected_scale
        final0 = final0 * injected_scale
        final1 = final1 * injected_scale
    scale = np.maximum(np.median(cal0, axis=0), 1e-12)
    ratio = np.maximum(np.median(drift0, axis=0) / scale, 1.0)
    rows_out = []
    for bit in bits:
        clip = max(1.0, 1.1 * float(np.max(cal0 / scale)))
        quant = lambda x: _quantize(x, bit, clip=clip)
        cal = quant(cal0 / scale)
        h0 = quant(final0 / scale)
        h1 = quant(final1 / scale)
        adapted0, adapted1 = quant(final0 / scale / ratio), quant(final1 / scale / ratio)
        partial = {
            lam: (
                quant(final0 / scale / np.power(ratio, lam)),
                quant(final1 / scale / np.power(ratio, lam)),
            )
            for lam in (0.25, 0.50, 0.75)
        }
        for fusion, aggregate in (("max", lambda x: np.max(x, axis=1)),
                                  ("sum", lambda x: np.sum(x, axis=1))):
            threshold = float(np.quantile(aggregate(cal), 1.0 - alpha, method="higher"))
            methods = {
                "frozen_empirical": (h0, h1, threshold),
                "global_worst_scale": (h0, h1, threshold * float(np.max(ratio))),
                "per_uav_scale_adaptive": (adapted0, adapted1, threshold),
            }
            for lam, (partial0, partial1) in partial.items():
                methods[f"log_shrink_{lam:.2f}"] = (partial0, partial1, threshold)
            for name, (x0, x1, t) in methods.items():
                s0, s1 = aggregate(x0), aggregate(x1)
                rows_out.append({
                    "bits_per_uav": bit, "fusion": fusion, "method": name,
                    "threshold": t, "pfa": float(np.mean(s0 > t)),
                    "pd": float(np.mean(s1 > t)),
                    "false_alarms": int(np.sum(s0 > t)), "tests": len(s0),
                })
    corr = np.corrcoef(cal0, rowvar=False)
    continuous_auc = {}
    for name, aggregate in (("max", lambda x: np.max(x, axis=1)),
                            ("sum", lambda x: np.sum(x, axis=1))):
        continuous_auc[name] = _auc(
            aggregate(_matrix([r for r in rows if r["split"] == "test"], "stat_h0", receivers) / scale),
            aggregate(_matrix([r for r in rows if r["split"] == "test"], "stat_h1", receivers) / scale),
        )
    return {
        "protocol": "real_tpuic_global_pfa_screen_v1",
        "input": str(Path(records).resolve()), "target_pfa": alpha,
        "injected_scale": None if injected_scale is None else injected_scale.tolist(),
        "receivers": receivers, "sample_counts": {
            "calibration": len(cal0), "drift_estimation": len(drift0), "final_test": len(final0)},
        "estimated_positive_scale_ratios": ratio.tolist(),
        "mean_abs_calibration_correlation": float(np.mean(np.abs(corr[~np.eye(len(receivers), dtype=bool)]))),
        "continuous_test_auc": continuous_auc,
        "results": rows_out,
        "warning": f"Only {len(final0)} final H0 samples; PFA resolution is {1/len(final0):.4g}. Screening only.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--injected-scale", default=None,
                   help="comma-separated controlled scale drift, one per receiver")
    args = p.parse_args()
    injected = None if args.injected_scale is None else [
        float(v) for v in args.injected_scale.split(",") if v.strip()
    ]
    result = analyze(args.records, args.alpha, injected_scale=injected)
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
