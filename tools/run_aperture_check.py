"""P1-2 in production: what does a receive array do to the *released* numbers?

The angular probe measured the array inside the bypass observation model.  This
measures it where a paper number actually comes from: ``build_base_gains`` ->
``dd_collision_count`` -> ``gamma_sense`` -> selection -> ``evaluate_detection``.

Reported, per aperture:

* collision statistics -- how many co-bin targets still mask each other;
* median ``rinr`` and the mean sensing SINR over valid links;
* end-to-end ``P_D`` / ``P_FA`` under the released selector, so the gain (or the
  absence of one) is stated on the quantity the paper claims.

The comparison is paired: the same geometry and the same RNG seed for every
aperture, so the only thing that moves between rows is ``m_rx``.  ``m_rx = 1``
is the released DD-only model and must reproduce it bit for bit.

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/run_aperture_check.py --trials 20 --out results_aperture_check
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from isac_sim.config import Config, apply_overrides, apply_preset
from isac_sim.model import (build_base_gains, compute_link_tables,
                            generate_geometry)
from isac_sim.simulate import evaluate_detection

M_LIST = (1, 4, 8, 16)


def build_cfg(area: float, rcs: float, seed: int, m_rx: int) -> Config:
    cfg = apply_preset(Config(), "paper-canonical")
    cfg = apply_overrides(cfg, {
        "geometry.area_xy": float(area),
        "detect.target_rcs": float(rcs),
        "run.seed": int(seed),
        "run.verbose": False,
    })
    if int(m_rx) > 1:
        cfg.aperture.enable = True
        cfg.aperture.m_rx = int(m_rx)
    return cfg


def _select(cfg, base, tables):
    from isac_sim.selection import select_lagrangian
    return select_lagrangian(cfg, base, tables, plan=None)[0]


def run_trial(cfg: Config, index: int) -> Dict[str, float]:
    rng = np.random.default_rng([cfg.run.seed, int(index)])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    tables = compute_link_tables(cfg, base)

    m = cfg.scale.M
    off = ~np.eye(m, dtype=bool)[:, :, None]
    valid = np.asarray(base.valid_dd, dtype=bool) & off
    cc = np.asarray(base.dd_collision_count, dtype=float)[valid]
    gamma = np.asarray(tables.gamma_sense, dtype=float)[valid]

    selected = _select(cfg, base, tables)
    det_rng = np.random.default_rng([cfg.run.seed, 10 ** 6 + int(index)])
    got = evaluate_detection(cfg, tables, selected, det_rng, "proposed_c2f",
                             None, base, trial_index=int(index))
    detected, total_targets, fa_active, total_fa_active = got[0], got[1], got[2], got[3]

    return {
        "trial": int(index),
        "m_rx": int(cfg.aperture.m_rx),
        "coll_mean": float(cc.mean()),
        "coll_max": float(cc.max()),
        "coll_frac_gt1": float((cc > 1.0 + 1e-9).mean()),
        # ``rinr`` is per (receiver, ?) -- 2-D, not per (i,j,q) -- so it is
        # summarised on its own terms rather than through the (i,j,q) mask.
        "rinr_median": float(np.median(np.asarray(tables.rinr))),
        "gamma_mean": float(gamma.mean()),
        "p_d": float(detected) / max(int(total_targets), 1),
        "p_fa": float(fa_active) / max(int(total_fa_active), 1),
        "n_selected": int(sum(len(v) for v in selected.values())),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results_aperture_check")
    args = ap.parse_args(argv)

    rows: List[Dict[str, float]] = []
    t0 = time.time()
    for m in M_LIST:
        cfg = build_cfg(args.area, args.rcs, args.seed, m)
        for t in range(int(args.trials)):
            rows.append(run_trial(cfg, t))
        print("  m_rx=%-3d done  [%.0f s]" % (m, time.time() - t0), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "aperture_check.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("  %-6s %10s %10s %10s %10s %8s %8s" % (
        "m_rx", "coll mean", "coll max", "coll>1", "rinr med", "P_D", "P_FA"))
    for m in M_LIST:
        sub = [r for r in rows if r["m_rx"] == m]
        if not sub:
            continue
        print("  %-6d %10.3f %10.2f %10.3f %10.3f %8.3f %8.3f" % (
            m,
            float(np.mean([r["coll_mean"] for r in sub])),
            float(np.mean([r["coll_max"] for r in sub])),
            float(np.mean([r["coll_frac_gt1"] for r in sub])),
            float(np.median([r["rinr_median"] for r in sub])),
            float(np.mean([r["p_d"] for r in sub])),
            float(np.mean([r["p_fa"] for r in sub])),
        ))
    print()
    print("  m_rx = 1 is the released DD-only model; every row is the same")
    print("  geometry and seed, so only the aperture moves between them.")
    print("wrote %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
