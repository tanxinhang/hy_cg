"""How many outer alternation epochs does the power/topology optimiser need?

``optimize_power_joint`` is called with ``rounds=2`` by the low-RCS sweep, but
that number is hard-coded rather than derived.  The published
``objective_trace`` records every accepted move, so the last accepted delta can
be compared with the running average: if the final move is still larger than
the mean, the loop was truncated before convergence.

This probe re-runs the optimiser on one frozen scene with increasing epoch
budgets and reports, for each budget, the achieved objective, the number of
accepted coordinate moves and the table-build cost.  The first budget whose
objective stops improving is the empirical convergence epoch count.

The optimiser contract requires orthogonal serial reporting, so the scene is
built in that regime.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import config  # noqa: E402
from isac_sim.belief import BeliefState  # noqa: E402
from isac_sim.config import apply_overrides, validate_config  # noqa: E402
from isac_sim.joint_polish import improve_joint_selection  # noqa: E402
from isac_sim.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from isac_sim.power_joint import optimize_power_joint  # noqa: E402
from isac_sim.reporting import assign_fusion_nodes  # noqa: E402
from isac_sim.selection import select_c2f_adaptive  # noqa: E402

SEED = 10919


def build(area, rcs, trial=0):
    cfg = apply_overrides(
        config(SEED, 8),
        {"geometry.area_xy": float(area), "detect.target_rcs": float(rcs),
         "comm.interference_model": "orthogonal"})
    validate_config(cfg)
    rng = np.random.default_rng([SEED, trial])
    truth = generate_geometry(cfg, rng)
    truth_base = build_base_gains(cfg, truth, rng)
    belief = BeliefState.from_truth(cfg, truth, rng)
    geom = belief.as_geometry(truth)
    base = build_base_gains(cfg, geom, rng, channel=truth_base,
                            rcs_view="mean")
    coarse = compute_link_tables(cfg, base)
    fine = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
    plan = assign_fusion_nodes(cfg, base, coarse, geom)
    return cfg, base, coarse, fine, plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", type=float, default=400.0)
    ap.add_argument("--rcs", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    args = ap.parse_args()

    cfg, base, coarse, fine, plan = build(args.area, args.rcs)
    selected, _d, _s = select_c2f_adaptive(cfg, base, coarse, plan)
    start = improve_joint_selection(cfg, base, coarse, fine, selected, plan).selected

    print(f"scene {args.area:g} m, RCS {args.rcs:g}, orthogonal serial")
    print(f"{'rounds':>7}{'objective':>13}{'gain_vs_start':>15}"
          f"{'accepts':>9}{'builds':>8}{'trace':>7}{'seconds':>9}{'last_delta':>12}")

    baseline = None
    prev = None
    for R in args.rounds:
        begin = time.perf_counter()
        power = optimize_power_joint(cfg, base, start, plan,
                                     grid=(0.2, 0.5, 0.8, 0.95), rounds=R)
        seconds = time.perf_counter() - begin
        trace = [float(v) for v in power.objective_trace]
        if baseline is None:
            baseline = trace[0]
        last = (trace[-1] - trace[-2]) if len(trace) > 1 else 0.0
        mean_delta = ((trace[-1] - trace[0]) / max(len(trace) - 1, 1))
        print(f"{R:>7}{trace[-1]:>13.5f}{trace[-1] - baseline:>15.5f}"
              f"{power.power_accepts:>9}{power.table_builds:>8}"
              f"{len(trace):>7}{seconds:>9.1f}{last:>12.5f}"
              f"   (mean delta {mean_delta:.5f}"
              f"{', last > mean' if last > mean_delta else ', last <= mean'})")
        if prev is not None:
            print(f"{'':>7}{'':>13}   marginal gain vs rounds={args.rounds[args.rounds.index(R)-1]}: "
                  f"{trace[-1] - prev:+.5f}")
        prev = trace[-1]


if __name__ == "__main__":
    main()
