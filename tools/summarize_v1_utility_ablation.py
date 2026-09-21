#!/usr/bin/env python3
"""Summarize the V1 utility-term necessity ablation.

Four configurations share one common-random-number scenario stream
(MC=100, seed 2026, preset ``target-local-v1``, Gaussian replacement):

=====================  ==========================================
variant                override
=====================  ==========================================
``full``               released config (soft-min + quadratic deficit)
``no_deficit_penalty`` ``selector.mu_deficit = 0``
``no_softmin``         ``selector.use_softmin_alpha = False``
``neither``            both disabled
=====================  ==========================================

The question is whether either utility term is *individually necessary*. For
each variant this script reports aggregate detection and cost, then trial-paired
differences against ``full`` using the same trial-cluster bootstrap interval
(``_paired_cluster_interval``) as the paper's paired contrasts.

A paired difference whose interval contains zero means the controlled ablation
found no detectable effect at this sample size; it does **not** establish
equivalence, and it is not evidence that the term is useless.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.flow.sweeps import _paired_cluster_interval  # noqa: E402

DEFAULT_DIR = ROOT / "results_v1_utility_ablation"
VARIANT_ORDER = ["full", "no_deficit_penalty", "no_softmin", "neither"]
INTERVAL_SEED = 2026 + 9001


def _trial_rows(path: Path) -> Dict[int, Dict[str, float]]:
    """Map trial index -> per-trial endpoints for a single-method run."""
    out: Dict[int, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            total = float(row.get("total_targets") or 0.0)
            detected = float(row.get("detected") or 0.0)
            out[int(row["trial"])] = {
                "pd": detected / total if total > 0 else 0.0,
                "worst_target_pd": float(row.get("weak_target_detected") or 0.0),
                "reports": float(row.get("remote_reports") or 0.0),
                "observations": float(row.get("selected_observations") or 0.0),
                "bits": float(row.get("overhead_bits") or 0.0),
                "delay_ms": float(row.get("overhead_delay_ms") or 0.0),
            }
    return out


def _load(variant_dir: Path) -> Optional[Dict[int, Dict[str, float]]]:
    trials = variant_dir / "main" / "trials.csv"
    if not trials.exists():
        return None
    return _trial_rows(trials)


def _mean(rows: Dict[int, Dict[str, float]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([r[key] for r in rows.values()]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data: Dict[str, Dict[int, Dict[str, float]]] = {}
    for name in VARIANT_ORDER:
        rows = _load(args.dir / name)
        if rows is None:
            print(f"missing {args.dir / name / 'main' / 'trials.csv'}", file=sys.stderr)
        else:
            data[name] = rows

    if "full" not in data:
        raise SystemExit("the 'full' variant is required as the paired reference")

    print("=" * 78)
    print("V1 utility-term necessity ablation  (MC=100, seed 2026, paired vs 'full')")
    print("=" * 78)
    header = f"{'variant':<20}{'P_D':>9}{'worst':>9}{'obs':>8}{'reports':>9}{'bits':>10}{'delay_ms':>10}"
    print(header)
    print("-" * 78)
    for name in VARIANT_ORDER:
        if name not in data:
            continue
        rows = data[name]
        print(
            f"{name:<20}"
            f"{_mean(rows, 'pd'):>9.4f}"
            f"{_mean(rows, 'worst_target_pd'):>9.4f}"
            f"{_mean(rows, 'observations'):>8.3f}"
            f"{_mean(rows, 'reports'):>9.3f}"
            f"{_mean(rows, 'bits'):>10.1f}"
            f"{_mean(rows, 'delay_ms'):>10.4f}"
        )

    payload: Dict[str, object] = {
        "dir": str(args.dir),
        "mc": len(data["full"]),
        "variants": {},
        "paired_vs_full": {},
    }

    for name in VARIANT_ORDER:
        if name not in data:
            continue
        rows = data[name]
        payload["variants"][name] = {  # type: ignore[index]
            "P_D": _mean(rows, "pd"),
            "worst_target_PD": _mean(rows, "worst_target_pd"),
            "selected_observations": _mean(rows, "observations"),
            "remote_reports": _mean(rows, "reports"),
            "overhead_bits": _mean(rows, "bits"),
            "overhead_delay_ms": _mean(rows, "delay_ms"),
        }

    print()
    print("Trial-paired differences vs 'full' (95% trial-cluster bootstrap):")
    print("-" * 78)
    full = data["full"]
    for name in VARIANT_ORDER:
        if name not in data or name == "full":
            continue
        rows = data[name]
        shared = sorted(set(full) & set(rows))
        entry = {}
        for key, label in (
            ("pd", "P_D"),
            ("worst_target_pd", "worst-target P_D"),
            ("reports", "remote reports"),
            ("observations", "observations"),
        ):
            deltas = np.asarray(
                [rows[t][key] - full[t][key] for t in shared], dtype=float
            )
            low, high = _paired_cluster_interval(deltas, INTERVAL_SEED)
            point = float(np.mean(deltas)) if deltas.size else 0.0
            crosses = low <= 0.0 <= high
            entry[key] = {
                "delta": point,
                "ci95_low": low,
                "ci95_high": high,
                "resolved": not crosses,
            }
            print(
                f"{name:<20}{label:<18}"
                f"{point:>+9.4f}  [{low:>+8.4f}, {high:>+8.4f}]"
                f"  {'resolved' if not crosses else 'unresolved (CI covers 0)'}"
            )
        payload["paired_vs_full"][name] = entry  # type: ignore[index]
        print()

    payload["interpretation"] = (
        "'resolved' means the paired interval excludes zero at this sample size. "
        "'unresolved' means no detectable effect was found here; it is not "
        "evidence of equivalence and does not by itself justify deleting a term."
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
