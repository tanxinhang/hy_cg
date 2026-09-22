#!/usr/bin/env python3
"""Screen global false-alarm calibration under correlated residual mismatch."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2


def _draw(rng, n, m, rho, scale, mean=0.0):
    common = rng.standard_normal((n, 1))
    local = rng.standard_normal((n, m))
    x = math.sqrt(rho) * common + math.sqrt(1.0 - rho) * local + mean
    return x * x * np.asarray(scale)[None, :]


def _quantize(x, bits, clip=20.0):
    if bits >= 16:
        return x
    levels = 2 ** bits
    return np.round(np.clip(x, 0.0, clip) * (levels - 1) / clip) * clip / (levels - 1)


def _conformal_threshold(scores, alpha):
    n = len(scores)
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def simulate(seed=20260922, repeats=300, n_cal=64, n_test=2000, m=6,
             rho=0.65, alpha=0.05, mismatch=0.20, signal_mean=0.8,
             bits=(2, 3, 16)):
    rng = np.random.default_rng(seed)
    base_scale = np.linspace(0.7, 1.5, m)
    rows = []
    for bit in bits:
        totals = {name: [0, 0, 0] for name in
                  ("naive_independence", "empirical_max", "bounded_robust")}
        for _ in range(repeats):
            cal = _draw(rng, n_cal, m, rho, base_scale)
            # Median normalization emulates locally estimated residual scale.
            denom = np.maximum(np.median(cal, axis=0) / chi2.ppf(0.5, 1), 1e-12)
            cal_q = _quantize(cal / denom, bit)
            h0 = _quantize(
                _draw(rng, n_test, m, rho, base_scale * (1.0 + mismatch)) / denom,
                bit,
            )
            h1 = _quantize(
                _draw(rng, n_test, m, rho, base_scale * (1.0 + mismatch), signal_mean) / denom,
                bit,
            )
            cal_score, h0_score, h1_score = map(
                lambda a: np.max(a, axis=1), (cal_q, h0, h1)
            )
            thresholds = {
                "naive_independence": float(chi2.ppf((1.0 - alpha) ** (1.0 / m), 1)),
                "empirical_max": _conformal_threshold(cal_score, alpha),
                "bounded_robust": (1.0 + mismatch) * _conformal_threshold(cal_score, alpha),
            }
            for name, threshold in thresholds.items():
                totals[name][0] += int(np.sum(h0_score > threshold))
                totals[name][1] += int(np.sum(h1_score > threshold))
                totals[name][2] += n_test
        for name, (false_alarms, detections, total) in totals.items():
            rows.append({
                "bits_per_uav": bit,
                "method": name,
                "pfa": false_alarms / total,
                "pd": detections / total,
                "pfa_error": false_alarms / total - alpha,
            })
    return {
        "protocol": "global_pfa_calibration_screen_v1",
        "settings": {
            "seed": seed, "repeats": repeats, "n_cal": n_cal,
            "n_test_per_repeat": n_test, "uavs": m, "rho": rho,
            "target_pfa": alpha, "positive_scale_mismatch": mismatch,
            "signal_mean": signal_mean,
        },
        "results": rows,
        "finite_sample_note": {
            "minimum_alpha_for_finite_conformal_threshold": 1.0 / (n_cal + 1),
            "minimum_calibration_samples_for_alpha_1e-3": 999,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=300)
    p.add_argument("--n-cal", type=int, default=64)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--mismatch", type=float, default=0.20)
    args = p.parse_args()
    result = simulate(repeats=args.repeats, n_cal=args.n_cal,
                      n_test=args.n_test, alpha=args.alpha, mismatch=args.mismatch)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
