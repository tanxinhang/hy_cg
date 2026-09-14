"""Compare the legacy decoupled interference model against the coupled ISAC one.

The point of the comparison is not "which number is bigger" but to make the
*modelling* difference explicit:

* the legacy model gives the communication receiver a full interference term and
  the sensing receiver a set of decoupled residual floors that the reporting
  schedule cannot influence;
* the coupled model builds both receivers from one field over the same active
  transmitters, and the sensing side differs only by the direct-path
  cancellation ``kappa_dc`` that the receiver actually achieves.

Usage::

    python tools/compare_isac_models.py --mc 200
    python tools/compare_isac_models.py --mc 200 --out results/isac_models
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim.config import PRESETS, Config, apply_overrides  # noqa: E402
from isac_sim.simulate import run_simulation  # noqa: E402

# The paper's operating point (mirrors tools/rerun_paper.py PAPER_SET).  Every
# model below is evaluated on top of this set, so the comparison isolates the
# interference bookkeeping and the SINR guard instead of also varying the
# orthogonal knobs (kinematics, active-set vs full-concurrent interference).
PAPER_SET: Dict[str, Any] = {
    "geometry.uav_speed_min": 30,
    "geometry.target_speed_min": 50,
    "geometry.target_speed_max": 90,
    "comm.interference_model": "active_set",
}

MODELS: List[Dict[str, Any]] = [
    ("legacy", {"interference.coupling": "legacy", "radio.eps_mode": "legacy"}),
    ("legacy + guard fix", {"interference.coupling": "legacy",
                            "radio.eps_mode": "noise_relative"}),
    ("coupled", {"interference.coupling": "shared_spectrum",
                 "radio.eps_mode": "legacy"}),
    ("coupled + guard fix", PRESETS["isac-consistent"]),
    ("coupled, strict", PRESETS["isac-consistent-strict"]),
]

METHODS = ["proposed_lagrangian", "topk_deflection", "sense_sinr", "single_best",
           "nearest", "shortest_bistatic", "random", "all_neighbor"]

FIELDS = ["P_D", "P_D_ci95_low", "P_D_ci95_high", "D_mean", "selected_links_mean",
          "B_mean_bits", "T_mean_ms", "selected_chi_mean",
          "selected_rate_mean_mbps", "worst_target_satisfied_prob",
          "actual_worst_target_P_D"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    print("fixed across all models: "
          + ", ".join(f"{k}={v}" for k, v in PAPER_SET.items()))
    for name, overrides in MODELS:
        cfg = apply_overrides(Config(), PAPER_SET)
        cfg = apply_overrides(cfg, overrides)
        cfg.run.num_mc = int(args.mc)
        cfg.run.seed = int(args.seed)
        cfg.run.verbose = False
        cfg.run.num_false_per_target = cfg.detect.num_false_per_target
        summary = run_simulation(cfg, methods=METHODS)
        for method in METHODS:
            s = summary.get(method)
            if s is None:
                continue
            rows.append({"model": name, "method": method,
                         **{f: s.get(f) for f in FIELDS}})
        p = summary["proposed_lagrangian"]
        a = summary["all_neighbor"]
        print(f"{name:22s} proposed P_D={p['P_D']:.4f}  D={p['D_mean']:7.2f}  "
              f"links={p['selected_links_mean']:6.1f}  T={p['T_mean_ms']:7.2f}ms  | "
              f"allN P_D={a['P_D']:.4f}")

    out_dir = args.out
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "isac_models.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", path)

    # Compact markdown for the paper/notes.
    print()
    print("| model | method | P_D | links | T (ms) | bits | chi |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %.4f | %.1f | %.2f | %.0f | %.3f |" % (
            r["model"], r["method"], r["P_D"], r["selected_links_mean"],
            r["T_mean_ms"], r["B_mean_bits"], r["selected_chi_mean"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
