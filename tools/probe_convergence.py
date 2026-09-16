"""How many schedule rounds does each ISAC time-sharing regime need?

Three operational questions, all measured rather than asserted.

1. **Fusion / scheduling convergence** -- the released selector commits exactly
   one observation per round, so the number of rounds equals the number of
   committed links.  ``select_c2f_adaptive`` returns the coarse rollout length
   (``coarse_rounds``) and the replayed fine length (``fine_rounds``), and the
   optional ``trajectory`` hook records the post-commit utility so the round at
   which the rollout has effectively converged can be read off directly.

2. **Centralised vs distributed negotiation** -- ``distributed_bids`` makes every
   target submit one bid per round and only then commit, which is the actual
   multi-round negotiation protocol.  It is implemented but unreferenced by any
   experiment, so its round count has never been measured.

3. **Time-sharing regime** -- ``comm.interference_model`` is the scheduling
   hypothesis, not a mere interference bookkeeping choice:

   * ``orthogonal``      payloads are time/frequency separated -> *sensing then
                         reporting*, two disjoint phases.
   * ``active_set``      only the elected reporters radiate during reporting.
   * ``full_concurrent`` every UAV radiates throughout -> *sensing and reporting
                         at once*, the genuinely concurrent ISAC premise.

   The same geometry is evaluated under all three so the detection cost of
   concurrency is a measured quantity.

The outer power/topology alternation (``optimize_power_joint``) is also traced:
its ``objective_trace`` shows how many alternation epochs actually change the
decision.  That optimiser requires ``orthogonal`` serial reporting.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import config  # noqa: E402
from isac_sim.belief import BeliefState, belief_dd_std_bins  # noqa: E402
from isac_sim.config import apply_overrides, validate_config  # noqa: E402
from isac_sim.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from isac_sim.reporting import assign_fusion_nodes  # noqa: E402
from isac_sim.selection import select_c2f_adaptive  # noqa: E402
from isac_sim.simulate import remote_report_count, run_method_on_trial  # noqa: E402

SEED = 10919
REPORT_CAP = 8
MODELS = ("orthogonal", "active_set", "full_concurrent")
POWER_GRID = (0.2, 0.5, 0.8, 0.95)
POWER_ROUNDS = 2


def build_scene(area, rcs, trial):
    """One frozen scene shared by every regime (paired comparison)."""
    base_cfg = apply_overrides(
        config(SEED, REPORT_CAP),
        {"geometry.area_xy": float(area), "detect.target_rcs": float(rcs)})
    validate_config(base_cfg)
    rng = np.random.default_rng([SEED, trial])
    truth = generate_geometry(base_cfg, rng)
    truth_base = build_base_gains(base_cfg, truth, rng)
    belief = BeliefState.from_truth(base_cfg, truth, rng)
    geom = belief.as_geometry(truth)
    base = build_base_gains(base_cfg, geom, rng, channel=truth_base,
                            rcs_view="mean")
    coarse = compute_link_tables(base_cfg, base)
    fine = compute_link_tables(base_cfg, base, dd_gain=base.eta_fine)
    std = belief_dd_std_bins(base_cfg, geom, belief)
    plan = assign_fusion_nodes(base_cfg, base, coarse, geom)
    return dict(base_cfg=base_cfg, truth_base=truth_base, base=base,
                coarse=coarse, fine=fine, std=std, plan=plan)


def rounds_to_fraction(trajectory, fraction=0.99):
    """First round whose *accumulated gain* reaches ``fraction`` of the total.

    The rollout utility increases monotonically by construction (every commit
    strictly improves the score), so the crossing is well defined and the
    denominator is positive even when the utility itself is negative (the
    latency price makes it so).
    """
    if not trajectory:
        return None
    first, final = trajectory[0]["utility"], trajectory[-1]["utility"]
    span = final - first
    if span <= 0:
        return None
    target = first + fraction * span
    for step in trajectory:
        if step["utility"] >= target:
            return int(step["round"])
    return int(trajectory[-1]["round"])


def run_model(scene, model, channel="erasure"):
    cfg = apply_overrides(scene["base_cfg"],
                          {"comm.interference_model": model,
                           "detect.comm_error_model": channel})
    validate_config(cfg)
    base, coarse, fine = scene["base"], scene["coarse"], scene["fine"]
    plan, std = scene["plan"], scene["std"]
    truth_base, truth_tables = scene["truth_base"], None
    truth_tables = compute_link_tables(cfg, truth_base)

    # --- centralised: one commit per round -------------------------------
    traj = []
    begin = time.perf_counter()
    chosen, d, stats = select_c2f_adaptive(cfg, base, coarse, plan,
                                           trajectory=traj)
    select_s = time.perf_counter() - begin

    # --- distributed: one bid per target per round -----------------------
    traj_d = []
    begin = time.perf_counter()
    chosen_d, d_d, stats_d = select_c2f_adaptive(cfg, base, coarse, plan,
                                                 distributed_bids=True,
                                                 trajectory=traj_d)
    select_d_s = time.perf_counter() - begin

    out = {
        "model": model,
        "comm_error_model": channel,
        "coarse_rounds": stats["coarse_rounds"],
        "fine_rounds": stats["fine_rounds"],
        "utility_first": traj[0]["utility"] if traj else None,
        "utility_final": traj[-1]["utility"] if traj else None,
        "rounds_to_99pct": rounds_to_fraction(traj, 0.99),
        "rounds_to_999pct": rounds_to_fraction(traj, 0.999),
        "rounds_to_90pct": rounds_to_fraction(traj, 0.90),
        "score_evaluations": stats["selector_score_evaluations"],
        "coordination_messages": stats["coordination_messages"],
        "fine_eval_full": stats["fine_eval_full"],
        "fine_eval_c2f": stats["fine_eval_c2f"],
        "select_seconds": select_s,
        "dist_coarse_rounds": stats_d["coarse_rounds"],
        "dist_bid_rounds": stats_d["bid_rounds"],
        "dist_rounds_to_99pct": rounds_to_fraction(traj_d, 0.99),
        "dist_score_evaluations": stats_d["selector_score_evaluations"],
        "dist_coordination_messages": stats_d["coordination_messages"],
        "dist_select_seconds": select_d_s,
    }
    for tag, (sel, dd, st) in (("central", (chosen, d, stats)),
                               ("distributed", (chosen_d, d_d, stats_d))):
        res = run_method_on_trial(
            cfg, base, coarse, "proposed_c2f_adaptive_pd", 0,
            c2f_tables=fine, plan=plan, eval_base=truth_base,
            eval_tables=truth_tables, belief_dd_std=std,
            cached_adaptive_pd=(sel, dd, st))
        out[f"{tag}_pd"] = float(res.detected / res.total_targets)
        per = np.asarray(res.detected_per_target, dtype=float)
        # Target-level mean of this single trial.  Named explicitly: it is NOT
        # a worst-target statistic, which would need a cross-trial average per
        # target before taking the minimum.
        out[f"{tag}_mean_target_pd"] = float(per.mean())
        out[f"{tag}_reports"] = float(remote_report_count(sel, plan))
        out[f"{tag}_observations"] = float(sum(map(len, sel.values())))
        out[f"{tag}_delay_ms"] = float(res.overhead_delay_s * 1000.0)
        out[f"{tag}_bits"] = float(res.overhead_bits)
    return out, traj, traj_d


def run_power_trace(scene):
    """Outer alternation trace.  The optimiser contract is orthogonal/serial."""
    from isac_sim.joint_polish import improve_joint_selection
    from isac_sim.power_joint import optimize_power_joint

    cfg = apply_overrides(scene["base_cfg"],
                          {"comm.interference_model": "orthogonal"})
    validate_config(cfg)
    base, coarse, fine, plan = (scene["base"], scene["coarse"],
                                scene["fine"], scene["plan"])
    try:
        selected, _d, _s = select_c2f_adaptive(cfg, base, coarse, plan)
        start = improve_joint_selection(cfg, base, coarse, fine,
                                        selected, plan).selected
        begin = time.perf_counter()
        power = optimize_power_joint(cfg, base, start, plan,
                                     grid=POWER_GRID, rounds=POWER_ROUNDS)
        seconds = time.perf_counter() - begin
    except ValueError as exc:
        return {"model": "orthogonal", "power_status": f"unsupported: {exc}"}
    trace = [float(v) for v in power.objective_trace]
    deltas = [trace[i + 1] - trace[i] for i in range(len(trace) - 1)]
    return {
        "model": "orthogonal", "power_status": "ok",
        "power_seconds": seconds,
        "power_table_builds": float(power.table_builds),
        "power_accepts": float(power.power_accepts),
        "power_trace_len": float(len(trace)),
        "power_trace": json.dumps(trace),
        "power_deltas": json.dumps(deltas),
        "power_gain": (trace[-1] - trace[0]) if trace else None,
        "power_epochs_with_change": (
            float(sum(1 for i, d in enumerate(deltas)
                      if abs(d) > 1e-12 and i >= 1)) if deltas else 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=6)
    ap.add_argument("--areas", type=float, nargs="+", default=[400.0])
    ap.add_argument("--rcs", type=float, nargs="+", default=[0.05])
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS))
    ap.add_argument("--channel", default="gaussian_replacement",
                    help="audit baseline uses gaussian_replacement; "
                         "target-local-v1 defaults to erasure")
    ap.add_argument("--power", action="store_true",
                    help="also trace the outer power/topology alternation")
    ap.add_argument("--out", type=Path, default=Path("results_v1_convergence"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)

    rows, traj_rows, power_rows = [], [], []
    for area in args.areas:
        for rcs in args.rcs:
            for trial in range(args.mc):
                scene = build_scene(area, rcs, trial)
                for model in args.models:
                    out, traj, traj_d = run_model(scene, model, args.channel)
                    out.update({"area_m": float(area), "rcs_m2": float(rcs),
                                "trial": int(trial)})
                    rows.append(out)
                    for tag, tr in (("central", traj), ("distributed", traj_d)):
                        for step in tr:
                            traj_rows.append(dict(step, area_m=float(area),
                                                  rcs_m2=float(rcs),
                                                  trial=int(trial),
                                                  model=model, tag=tag))
                if args.power and trial == 0:
                    power_rows.append(dict(run_power_trace(scene),
                                           area_m=float(area),
                                           rcs_m2=float(rcs), trial=int(trial)))

    with (args.out / "convergence.csv").open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if traj_rows:
        with (args.out / "trajectory.csv").open("w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(traj_rows[0].keys()))
            w.writeheader()
            w.writerows(traj_rows)
    if power_rows:
        with (args.out / "power_trace.csv").open("w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(power_rows[0].keys()))
            w.writeheader()
            w.writerows(power_rows)

    # --- console summary -------------------------------------------------
    print(f"mc={args.mc}  cells={len(rows)}")
    for model in args.models:
        group = [r for r in rows if r["model"] == model]
        if not group:
            continue
        mean = (lambda k: float(np.mean([r[k] for r in group])))
        safe = lambda k: float(np.mean([r[k] for r in group
                                        if r.get(k) is not None])) \
            if any(r.get(k) is not None for r in group) else float("nan")
        print(f"\n[{model}]")
        print(f"  coarse_rounds  {mean('coarse_rounds'):6.1f}   "
              f"fine_rounds {mean('fine_rounds'):6.1f}")
        print(f"  rounds@90/99/99.9%  "
              f"{safe('rounds_to_90pct'):.1f} / "
              f"{safe('rounds_to_99pct'):.1f} / "
              f"{safe('rounds_to_999pct'):.1f}")
        print(f"  distributed  rounds {mean('dist_coarse_rounds'):5.1f}  "
              f"bid_rounds {mean('dist_bid_rounds'):5.1f}  "
              f"rounds@99% {safe('dist_rounds_to_99pct'):.1f}")
        print(f"  messages  central {mean('coordination_messages'):.0f}  "
              f"distributed {mean('dist_coordination_messages'):.0f}")
        print(f"  P_D  central {mean('central_pd'):.3f}  "
              f"distributed {mean('distributed_pd'):.3f}")
        print(f"  time  central {mean('select_seconds'):.2f}s  "
              f"distributed {mean('dist_select_seconds'):.2f}s")
    for row in power_rows:
        if row.get("power_status") != "ok":
            print(f"\n[power trace] {row['power_status']}")
        else:
            print(f"\n[power trace] builds={row['power_table_builds']:.0f} "
                  f"accepts={row['power_accepts']:.0f} "
                  f"seconds={row['power_seconds']:.1f} "
                  f"trace_len={row['power_trace_len']:.0f} "
                  f"gain={row['power_gain']:.4f}")
            print(f"  deltas = {row['power_deltas']}")


if __name__ == "__main__":
    main()
