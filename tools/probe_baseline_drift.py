#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full cell-level diff between the frozen legacy baseline and a fresh run.

``tools/parity_check.py --baseline`` reports only the single worst field, which
is enough to fail a gate but not enough to attribute a drift.  This probe dumps
every differing cell, with an order-of-magnitude classification, so a drift can
be traced to the commit that introduced it.

Usage
-----
    python tools/probe_baseline_drift.py --current /tmp/drift_head
    python tools/probe_baseline_drift.py --current results_x --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / ".workbuddy" / "baseline" / "main" / "main.csv"
SKIP_KEYS = {"method", "variant", "experiment", "robustness_type", "comm_error_model"}

# Relative bands used to label a cell.  ``bit`` means the two floats agree to
# full double precision after parsing; those are almost always string
# round-tripping, not model changes.
BANDS: Sequence[Tuple[str, float]] = (
    ("bit", 0.0),
    ("tiny", 1e-9),
    ("small", 1e-3),
    ("moderate", 0.05),
    ("large", 0.5),
    ("order", float("inf")),
)


def _band(rel: float) -> str:
    for name, cap in BANDS:
        if rel <= cap:
            return name
    return "order"


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def load(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return {r["method"]: r for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--current", required=True,
                    help="either a run output dir (uses <dir>/main/main.csv) or a .csv")
    ap.add_argument("--json", default=None, help="also dump the full diff as JSON")
    ap.add_argument("--max-rows", type=int, default=200)
    args = ap.parse_args()

    base_path = Path(args.baseline)
    cur_path = Path(args.current)
    if cur_path.is_dir():
        cur_path = cur_path / "main" / "main.csv"
    if not cur_path.exists():
        raise SystemExit(f"current csv not found: {cur_path}")

    base = load(base_path)
    cur = load(cur_path)

    missing = [m for m in base if m not in cur]
    extra = [m for m in cur if m not in base]

    records: List[Dict[str, object]] = []
    for method in sorted(base):
        if method not in cur:
            continue
        shared = set(base[method]) & set(cur[method])
        for key in sorted(shared):
            va, vb = base[method][key], cur[method][key]
            if key in SKIP_KEYS or va == "" or vb == "":
                continue
            try:
                fa, fb = float(va), float(vb)
            except ValueError:
                if va != vb:
                    records.append({"method": method, "field": key,
                                    "baseline": va, "current": vb,
                                    "abs": math.inf, "rel": math.inf,
                                    "band": "nonnumeric"})
                continue
            if math.isnan(fa) or math.isnan(fb):
                if math.isnan(fa) != math.isnan(fb):
                    records.append({"method": method, "field": key,
                                    "baseline": fa, "current": fb,
                                    "abs": math.inf, "rel": math.inf,
                                    "band": "nan-mismatch"})
                continue
            diff = abs(fa - fb)
            if diff == 0.0:
                continue
            rel = _rel(fa, fb)
            records.append({"method": method, "field": key,
                            "baseline": fa, "current": fb,
                            "abs": diff, "rel": rel, "band": _band(rel)})

    # ---- report -----------------------------------------------------------
    total_cells = sum(len(set(base[m]) & set(cur[m])) for m in base if m in cur)
    print(f"baseline : {base_path}")
    print(f"current  : {cur_path}")
    print(f"methods  : {len(base)} baseline / {len(cur)} current"
          f"  (missing={missing or 'none'}, extra={extra or 'none'})")
    print(f"cells    : {total_cells} compared, {len(records)} differing")
    print()

    if records:
        by_band: Dict[str, int] = {}
        for r in records:
            by_band[str(r["band"])] = by_band.get(str(r["band"]), 0) + 1
        print("band histogram:")
        for name, _ in BANDS:
            if name in by_band:
                print(f"    {name:14s} {by_band[name]:5d}")
        for name in ("nonnumeric", "nan-mismatch"):
            if name in by_band:
                print(f"    {name:14s} {by_band[name]:5d}")
        print()

        records.sort(key=lambda r: -float(r["rel"]) if math.isfinite(float(r["rel"]))
                     else -math.inf)
        print(f"{'method':22s} {'field':30s} {'baseline':>16s} {'current':>16s}"
              f" {'rel':>10s} band")
        for r in records[: args.max_rows]:
            bl, cu = r["baseline"], r["current"]
            bs = f"{bl:>16.6g}" if isinstance(bl, float) else f"{str(bl):>16s}"
            cs = f"{cu:>16.6g}" if isinstance(cu, float) else f"{str(cu):>16s}"
            rr = r["rel"]
            rs = f"{float(rr):>10.3g}" if math.isfinite(float(rr)) else f"{'inf':>10s}"
            print(f"{r['method']:22s} {r['field']:30s} {bs} {cs} {rs} {r['band']}")
        if len(records) > args.max_rows:
            print(f"    ... {len(records) - args.max_rows} more")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"baseline": str(base_path), "current": str(cur_path),
                        "n_differing": len(records), "missing": missing,
                        "extra": extra, "records": records}, indent=1),
            encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 1 if records else 0


if __name__ == "__main__":
    raise SystemExit(main())
