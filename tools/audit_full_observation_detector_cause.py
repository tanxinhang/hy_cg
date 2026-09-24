#!/usr/bin/env python3
"""Audit calibration-tail and receiver effects without changing the detector."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, spearmanr

from tools import run_tpuic_receiver_benchmark as bench


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{k: (v if k == "split" else float(v)) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def _q(values):
    return {str(q): float(np.quantile(values, q))
            for q in (0, .25, .5, .75, .9, .95, 1)}


def _bootstrap(rows, name, threshold, seed=20261015, draws=10000):
    rng = np.random.default_rng(seed)
    h0 = np.asarray([r[f"{name}_h0"] for r in rows])
    h1 = np.asarray([r[f"{name}_h1"] for r in rows])
    ids = rng.integers(0, len(rows), size=(draws, len(rows)))
    auc = np.asarray([bench._auc(h0[i], h1[i]) for i in ids])
    pd = np.mean(h1[ids] > threshold, axis=1)
    return {"auc_ci95": np.quantile(auc, [.025, .975]).tolist(),
            "pd_ci95_fixed_threshold": np.quantile(pd, [.025, .975]).tolist()}


def _threshold_sensitivity(c0, t0, t1, pfa, seed=20261016, draws=10000):
    rng = np.random.default_rng(seed)
    thresholds, pfas, pds = [], [], []
    for sample in rng.integers(0, len(c0), size=(draws, len(c0))):
        threshold = bench._calibrated_threshold(c0[sample], pfa, "split_conformal")
        thresholds.append(threshold)
        pfas.append(np.mean(t0 > threshold))
        pds.append(np.mean(t1 > threshold))
    return {"threshold_bootstrap_ci95": np.quantile(thresholds, [.025, .975]).tolist(),
            "test_pfa_bootstrap_ci95": np.quantile(pfas, [.025, .975]).tolist(),
            "test_pd_bootstrap_ci95": np.quantile(pds, [.025, .975]).tolist()}


def _arm(cal, test, name, pfa):
    c0 = np.asarray([r[f"{name}_h0"] for r in cal])
    t0 = np.asarray([r[f"{name}_h0"] for r in test])
    t1 = np.asarray([r[f"{name}_h1"] for r in test])
    threshold = bench._calibrated_threshold(c0, pfa, "split_conformal")
    order = np.argsort(c0)
    return {
        "threshold": threshold,
        "threshold_rank": int(np.searchsorted(np.sort(c0), threshold) + 1),
        "threshold_scene": int(cal[int(order[-2])]["scene_id"]),
        "calibration_h0": _q(c0), "test_h0": _q(t0), "test_h1": _q(t1),
        "calibration_test_h0_ks": {"statistic": float(ks_2samp(c0, t0).statistic),
                                   "pvalue": float(ks_2samp(c0, t0).pvalue)},
        "test_h0_exceedances": int(np.sum(t0 > threshold)),
        "test_h1_exceedances": int(np.sum(t1 > threshold)),
        "test_h0_max_margin": float(np.max(t0) - threshold),
        "auc": float(bench._auc(t0, t1)),
        **_bootstrap(test, name, threshold),
        **_threshold_sensitivity(c0, t0, t1, pfa),
    }


def run(records: Path, out: Path, pfa: float):
    rows = _rows(records)
    cal = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "test"]
    arms = {name: _arm(cal, test, name, pfa) for name in ("nominal", "perfect")}
    n0 = np.asarray([r["nominal_h0"] for r in test])
    p0 = np.asarray([r["perfect_h0"] for r in test])
    n1 = np.asarray([r["nominal_h1"] for r in test])
    p1 = np.asarray([r["perfect_h1"] for r in test])
    nt, pt = arms["nominal"]["threshold"], arms["perfect"]["threshold"]
    payload = {
        "records": str(records), "calibration_scenes": len(cal),
        "test_scenes": len(test), "target_pfa": pfa, "arms": arms,
        "paired_receiver_effect": {
            "h0_spearman": float(spearmanr(n0, p0).statistic),
            "h1_spearman": float(spearmanr(n1, p1).statistic),
            "nominal_minus_perfect_h0": _q(n0 - p0),
            "nominal_minus_perfect_h1": _q(n1 - p1),
            "nominal_at_perfect_threshold": {
                "pfa": float(np.mean(n0 > pt)), "pd": float(np.mean(n1 > pt))},
            "perfect_at_nominal_threshold": {
                "pfa": float(np.mean(p0 > nt)), "pd": float(np.mean(p1 > nt))},
        },
        "interpretation_guards": {
            "probability_zero_of_40_at_p05": float(.95 ** len(test)),
            "note": "KS p-values and bootstrap intervals are diagnostics, not new gates."},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pfa", type=float, default=.05)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
