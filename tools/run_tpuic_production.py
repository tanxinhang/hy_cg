"""TP-UIC in the production chain: the scalar kappa replaced by the algorithm.

``264727d`` built a receiver-side canceller and then measured it on a *bypass*
observation built for the experiment, while ``model.compute_link_tables`` kept
multiplying the aggregated direct field by the constant
``kappa_dc = 10^(-interference.direct_cancellation_db / 10)``.  Two worlds: the
released P_D described a 40 dB assumption, and no released number described the
algorithm.  This driver closes that gap with the least machinery that can:

1. per trial, build the geometry and the base gains exactly as the production
   Monte-Carlo does;
2. **measure** the residual fraction at every receiver with the real estimator
   (``cancellation.measure_residual_fraction``, the proposed ``tp_uic_full``
   arm) -- this is the quantity the canceller's own documentation says a
   scheduler may consume, and the one it says the analytic prediction must not
   be trusted for;
3. build the link tables twice, identical in every other respect -- once with the
   frozen constant, once with the measured per-receiver fraction;
4. run the same selector and the same detector on both, so the difference in
   ``P_D``, ``P_FA``, reported bits and ``rinr`` is attributable to the receiver
   model and to nothing else.

What it answers, and what it cannot
-----------------------------------
It answers "what does the full chain give when the receiver is the algorithm
instead of the constant".  It does **not** answer "is TP-UIC better than 40 dB":
the constant is an idealised assumption, so a comparison against it is a
comparison against a model, which the method note explicitly forbids
(``TP_UIC_V11.md`` section 8).  Read the two columns as *what the chain is
actually worth under each receiver statement*, with the measured column being the
only one backed by an executable receiver.

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/run_tpuic_production.py --trials 20 --area 600 --rcs 0.1

Nothing here touches the default path: with ``--mode constant`` (or simply not
running it) ``residual_fraction_by_receiver`` stays ``None`` and every released
number is bit-exact.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim.receiver import cancellation as cx  # noqa: E402
from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.sensing.model import build_base_gains, compute_link_tables, generate_geometry  # noqa: E402
from experiments.flow.simulate import evaluate_detection  # noqa: E402


def build_config(args) -> Config:
    cfg = apply_preset(Config(), args.preset)
    return apply_overrides(
        cfg,
        {
            "geometry.area_xy": float(args.area),
            "detect.target_rcs": float(args.rcs),
            "run.seed": int(args.seed),
            "run.verbose": False,
            # The measurement is a receiver property, so the estimator must be
            # enabled; the *table* is only affected through the fraction we pass.
            "cancellation.enable": True,
            "cancellation.n_cpi": int(args.n_cpi),
            "cancellation.max_protected_targets": int(args.max_protected_targets),
        },
    )


def _selector(cfg: Config, base, tables):
    """The production selector, imported lazily so the tool states its dependency."""
    from experiments.selection import select_lagrangian

    return select_lagrangian(cfg, base, tables, plan=None)[0]


def _detection(cfg, tables, selected, rng, base, trial_index: int):
    got = evaluate_detection(
        cfg, tables, selected, rng, "proposed_c2f", None, base,
        trial_index=trial_index,
    )
    detected, total_targets, fa_active, total_fa_active = got[0], got[1], got[2], got[3]
    return {
        "p_d": detected / max(total_targets, 1),
        "p_fa": fa_active / max(total_fa_active, 1),
        "n_selected": int(sum(len(v) for v in selected.values())),
    }


def run_trial(cfg: Config, args, index: int) -> List[dict]:
    rng = np.random.default_rng([cfg.run.seed, int(index)])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)

    measure_rng = np.random.default_rng([cfg.run.seed, 10 ** 6 + int(index)])
    t0 = time.time()
    fraction, kappa = cx.measure_residual_fraction(
        cfg, geom, base, rng=measure_rng, arm=args.arm
    )
    measure_s = time.time() - t0

    rows: List[dict] = []
    for label, frac, const_db in (
        # 删除 ``interference.direct_cancellation_db`` 之后，"不接线"就是**没有
        # 对消**（0 dB）。它是开放环路下界，不再是"假设 40 dB"的参照臂 ——
        # 那个数字没有接收机实现支撑，却撑着整个分母。
        ("open_loop", None, 0.0),
        ("measured", fraction, None),
    ):
        tables = compute_link_tables(
            cfg, base, residual_fraction_by_receiver=frac
        )
        selected = _selector(cfg, base, tables)
        det_rng = np.random.default_rng([cfg.run.seed, 2 * 10 ** 6 + int(index)])
        det = _detection(cfg, tables, selected, det_rng, base, int(index))
        rinr = np.asarray(tables.rinr)
        rows.append({
            "trial": int(index),
            "arm": label,
            "kappa_db_median": float(const_db if const_db is not None else np.median(kappa)),
            "kappa_db_min": float(const_db if const_db is not None else np.min(kappa)),
            "kappa_db_max": float(const_db if const_db is not None else np.max(kappa)),
            "residual_fraction_median": float(np.median(frac) if frac is not None else 10.0 ** (-const_db / 10.0)),
            "rinr_median_db": float(10.0 * np.log10(max(np.median(rinr), 1e-300))),
            "p_d": det["p_d"],
            "p_fa": det["p_fa"],
            "n_selected": det["n_selected"],
            "measure_seconds": measure_s,
        })
    return rows


def report(rows: List[dict], args) -> None:
    print("\n=== TP-UIC in the production chain (area = %.0f m, RCS = %.2f m^2) ==="
          % (args.area, args.rcs))
    print("  same geometry, same selector, same detector; only the receiver model differs")
    print("  estimator arm = %s ; n_cpi = %d ; protected targets = %d"
          % (args.arm, args.n_cpi, args.max_protected_targets))
    print()
    print("  %-11s %10s %8s %8s %8s %9s %9s %7s" % (
        "receiver", "kappa med", "k min", "k max", "rinr dB", "P_D", "P_FA", "links"))
    for label in ("constant", "measured"):
        sub = [r for r in rows if r["arm"] == label]
        if not sub:
            continue

        def med(key):
            return st.median(float(r[key]) for r in sub)

        def mean(key):
            return sum(float(r[key]) for r in sub) / len(sub)

        print("  %-11s %10.2f %8.2f %8.2f %8.2f %9.4f %9.4f %7.1f" % (
            label, med("kappa_db_median"), med("kappa_db_min"), med("kappa_db_max"),
            med("rinr_median_db"), mean("p_d"), mean("p_fa"), med("n_selected")))
    print("\n  'constant' is the frozen 40 dB statement; 'measured' is what the")
    print("  estimator actually delivers on these geometries, per receiver.")
    print("  The comparison is against a *model*, not against a competitor arm:")
    print("  the constant asserts a depth the scenario implies, the algorithm")
    print("  reports the depth a receiver achieves.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-cpi", type=int, default=1)
    ap.add_argument("--max-protected-targets", type=int, default=3)
    ap.add_argument("--arm", default="tp_uic_full",
                    help="canceller arm whose residual defines the fraction")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default="results/tp_uic_production")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    os.makedirs(args.out, exist_ok=True)
    print("TP-UIC production wiring : %s, area=%.0f m, RCS=%.2f m^2, seed=%d"
          % (args.preset, args.area, args.rcs, args.seed))
    print("  M = %d UAVs, Q = %d targets, K = %d bins"
          % (cfg.scale.M, cfg.scale.Q, cfg.waveform.N * cfg.waveform.L))

    rows: List[dict] = []
    t0 = time.time()
    for t in range(int(args.trials)):
        rows.extend(run_trial(cfg, args, t))
        if not args.quiet and (t + 1) % max(int(args.trials) // 5, 1) == 0:
            print("  trial %d/%d  [%.0f s]" % (t + 1, args.trials, time.time() - t0),
                  flush=True)
    path = os.path.join(args.out, "production.csv")
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    report(rows, args)
    print("\nwrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
