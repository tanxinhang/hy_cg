"""Software-knob sweep for the 400-600 m low-RCS operating region.

Operational question
--------------------
The next main scenario is a 400-600 m deployment against targets of mean RCS
0.05-0.2 m^2.  ``detect.weak_pd_required`` is 0.80, while the released V1
configuration reaches only 0.44-0.69 (mean target) at 600 m.  This script
measures how much of that gap the *already implemented* knobs close.

Design
------
Everything is frozen to the joint V1 audit protocol
(``tools.audit_v1_exact_budget.config``: report cap 8, Gaussian replacement
failure model, ``score_mode='detector_pd'``) except the declared cell
overrides.  Geometry, truth/belief gains, coarse and refined tables, DD bins
and the fusion plan are built **once per (area, rcs, trial)** and shared by
every cell; only selection and the Monte-Carlo decision re-execute.  All cells
are therefore paired by trial.

Knob families
-------------
``looks``     ``detect.n_looks`` - OTFS frames per CPI.  A physical CPI length
              rather than a tuning knob: the single-link deflection is
              ``L * gamma^2`` (``isac_sim/llr.py``).
``fusion``    ``fusion.rule`` and ``corr.enable`` - destination rule and
              correlation-aware combining.
``budget``    report / observation caps - a scenario input, not an algorithm.
``power``     ``isac_sim.power_joint`` per-UAV sensing/report split at a fixed
              1 W per UAV and 15 W per fleet, warm-started two ways.
``maxmin``    A coverage-first selection instead of utility maximisation:
              ``balanced_select`` keeps adding the highest-sensing-SINR
              observation **among the targets that currently have the fewest**.
              Since the requirement is a weak-target requirement, this is the
              natural candidate and is unreachable by any override of the
              utility-maximising selector.
``radar``     ``radio.radar_net_gain_db`` - non-algorithmic residual axis
              (antenna gain / EIRP / system loss), desired echo only.
``radio``     ``radio.P_default`` - brute-force power, which under
              ``interference.coupling='shared_spectrum'`` also raises the
              direct-path interference field.

Module-level state is required by the Windows multiprocessing spawn start
method.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_balanced_budget import balanced_select  # noqa: E402
from tools.audit_v1_exact_budget import config  # noqa: E402
from isac_sim.belief import BeliefState, belief_dd_std_bins  # noqa: E402
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
from isac_sim.simulate import remote_report_count, run_method_on_trial  # noqa: E402

SEED = 10919
REPORT_CAP = 8
POWER_GRID = (0.2, 0.5, 0.8, 0.95)
POWER_ROUNDS = 2
STAT_KEYS = ["fine_eval_full", "fine_eval_c2f", "selector_score_evaluations",
             "coordination_messages", "bid_rounds"]
ZERO_STATS = {key: 0.0 for key in STAT_KEYS}

# label -> (overrides, family, selection, power_warm_start)
# selection: "utility" (released selector) | "maxmin" (balanced_select)
# power_warm_start: None | "utility" | "maxmin"
CELLS = {
    "base": ({}, "baseline", "utility", None),
    "looks64": ({"detect.n_looks": 64}, "looks", "utility", None),
    "corr": ({"corr.enable": True}, "fusion", "utility", None),
    "capacitated": ({"fusion.rule": "nearest_target_capacitated"},
                    "fusion", "utility", None),
    "budget": ({"selector.max_remote_reports": 16,
                "selector.max_total_links": 90,
                "selector.max_links_per_target": 9}, "budget", "utility", None),
    "maxmin": ({}, "maxmin", "maxmin", None),
    "maxmin_looks64": ({"detect.n_looks": 64}, "maxmin", "maxmin", None),
    "power": ({}, "power", "utility", "utility"),
    "maxmin_power": ({}, "power", "maxmin", "maxmin"),
    "gain15": ({"radio.radar_net_gain_db": 15.0}, "radar", "utility", None),
    "power2": ({"radio.P_default": 2.0}, "radio", "utility", None),
    # The audit baseline (``target-local-v1``) runs ``interference_model=
    # 'orthogonal'``, where the sensing field equals P_sense so rho scales echo
    # and interference together.  The paper runs ``active_set``, where the field
    # is P_sense + P_comm = P_default and rho cancels out.  These cells measure
    # whether the rho lever that appears under the paper model survives into
    # P_D.  ``orth_rho80`` is the control: the audit operating point.
    "orth_rho80": ({}, "rho", "utility", None),
    "as_rho41": ({"comm.interference_model": "active_set",
                  "radio.rho": 0.41}, "rho", "utility", None),
    "as_rho80": ({"comm.interference_model": "active_set",
                  "radio.rho": 0.80}, "rho", "utility", None),
    "as_rho95": ({"comm.interference_model": "active_set",
                  "radio.rho": 0.95}, "rho", "utility", None),
    "as_rho95_looks64": ({"comm.interference_model": "active_set",
                          "radio.rho": 0.95, "detect.n_looks": 64},
                         "rho", "utility", None),
    "as_rho95_maxmin": ({"comm.interference_model": "active_set",
                         "radio.rho": 0.95}, "rho", "maxmin", None),
}
LABELS = list(CELLS)
METRICS = ["pd", "worst_pd", "weak_pd", "pfa", "reports", "observations",
           "bits", "delay_ms", "n_looks", "rho_mean", "rho_std", "sensing_w",
           "power_seconds"]
FIELDS = (["area_m", "rcs_m2", "trial", "label", "family"] + METRICS
          + ["per_target"])


def _per_target_matrix(group):
    return np.array([[float(v) for v in row["per_target"].split(";")]
                     for row in group], dtype=float)


def _eval_cell(cfg_eval, chosen, d, stats, dest, tables, fine, truth_base,
               base, std, trial, label, family, power_seconds):
    truth_tables = compute_link_tables(cfg_eval, truth_base)
    result = run_method_on_trial(
        cfg_eval, base, tables, "proposed_c2f_adaptive_pd", trial,
        c2f_tables=fine, plan=dest, eval_base=truth_base,
        eval_tables=truth_tables, belief_dd_std=std,
        cached_adaptive_pd=(chosen, d, stats))
    per_target = np.asarray(result.detected_per_target, dtype=float)
    rho = np.array(cfg_eval.radio.rho_by_uav
                   if cfg_eval.radio.rho_by_uav is not None
                   else [cfg_eval.radio.rho] * cfg_eval.scale.M)
    return dict(
        label=label, family=family,
        pd=float(result.detected / result.total_targets),
        worst_pd=float(per_target.min()),
        weak_pd=float(per_target[int(result.weak_target_index)]),
        # Aggregated over trials this reproduces simulate.summarize's
        # actual_worst_target_P_D (mean per target, then min over targets).
        per_target=";".join(str(int(v)) for v in per_target),
        pfa=float(result.false_alarm_overall / result.total_false_overall),
        reports=float(remote_report_count(chosen, dest)),
        observations=float(sum(map(len, chosen.values()))),
        bits=float(result.overhead_bits),
        delay_ms=float(result.overhead_delay_s * 1000.0),
        n_looks=int(cfg_eval.detect.n_looks),
        rho_mean=float(rho.mean()), rho_std=float(rho.std()),
        sensing_w=float((rho * cfg_eval.radio.P_default).sum()),
        power_seconds=float(power_seconds))


def job(spec):
    area, rcs, trial, labels = spec
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
    zero_d = np.zeros(base_cfg.scale.Q, dtype=float)

    rows = []
    for label in labels:
        overrides, family, selection, warm = CELLS[label]
        cfg = apply_overrides(base_cfg, overrides)
        validate_config(cfg)
        seconds = 0.0
        if warm is None:
            if selection == "maxmin":
                chosen = balanced_select(cfg, base, coarse, plan)
                d, stats = zero_d, dict(ZERO_STATS)
            else:
                chosen, d, stats = select_c2f_adaptive(cfg, base, coarse, plan)
            cfg_eval, dest, tables, polished = cfg, plan, coarse, fine
        else:
            if warm == "maxmin":
                start_set = balanced_select(cfg, base, coarse, plan)
            else:
                selected, _d, _s = select_c2f_adaptive(cfg, base, coarse, plan)
                start_set = improve_joint_selection(
                    cfg, base, coarse, fine, selected, plan).selected
            begin = time.perf_counter()
            power = optimize_power_joint(cfg, base, start_set, plan,
                                         grid=POWER_GRID, rounds=POWER_ROUNDS)
            seconds = time.perf_counter() - begin
            state = power.joint
            chosen, dest = state.selected, state.plan
            d, stats = zero_d, dict(ZERO_STATS)
            cfg_eval, tables, polished = state.cfg, state.coarse, state.fine
        row = _eval_cell(cfg_eval, chosen, d, stats, dest, tables, polished,
                         truth_base, base, std, trial, label, family, seconds)
        row.update({"area_m": float(area), "rcs_m2": float(rcs),
                    "trial": int(trial)})
        rows.append(row)
    return rows


def summarize(rows, areas, rcs_values, labels, baseline="base"):
    groups = {}
    for row in rows:
        groups.setdefault((row["area_m"], row["rcs_m2"], row["label"]),
                          []).append(row)
    out = {}
    for area in areas:
        for rcs in rcs_values:
            cells = {}
            for label in labels:
                group = sorted(groups.get((area, rcs, label), []),
                               key=lambda r: r["trial"])
                if not group:
                    continue
                entry = {m: float(np.mean([r[m] for r in group]))
                         for m in METRICS}
                entry["n_trials"] = len(group)
                target_means = _per_target_matrix(group).mean(axis=0)
                entry["mean_target_pd"] = float(target_means.mean())
                entry["worst_target_pd"] = float(target_means.min())
                entry["weak_ci95"] = _mean_ci([r["weak_pd"] for r in group])
                cells[label] = entry
            base_rows = sorted(groups.get((area, rcs, baseline), []),
                               key=lambda r: r["trial"])
            if base_rows:
                base_targets = _per_target_matrix(base_rows).mean(axis=0)
                for label in labels:
                    if label == baseline or label not in cells:
                        continue
                    group = sorted(groups[(area, rcs, label)],
                                   key=lambda r: r["trial"])
                    deltas = [r["pd"] - b["pd"]
                              for r, b in zip(group, base_rows)]
                    target_deltas = (_per_target_matrix(group).mean(axis=0)
                                     - base_targets)
                    cells[label]["minus_base"] = {
                        "pd_mean": float(np.mean(deltas)),
                        "pd_ci95": _mean_ci(deltas),
                        "worst_target_delta": float(target_deltas.min()),
                        "mean_target_delta": float(target_deltas.mean()),
                    }
            out[f"area{area:g}_rcs{rcs:g}"] = cells
    return out


def _mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or values.std() == 0:
        return None
    from scipy.stats import t as student_t
    half = (student_t.ppf((1 + confidence) / 2, len(values) - 1)
            * values.std(ddof=1) / np.sqrt(len(values)))
    return [float(values.mean() - half), float(values.mean() + half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", type=int, default=100)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--areas", type=float, nargs="+",
                        default=[400.0, 600.0])
    parser.add_argument("--rcs", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2])
    parser.add_argument("--cells", nargs="+", default=LABELS, choices=LABELS)
    parser.add_argument("--out", type=Path,
                        default=Path("results_v1_lowrcs_sweep"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    if (args.out / "protocol.json").exists():
        raise SystemExit("Use a fresh output directory")

    protocol = {
        "question": ("Which already-implemented software knob reaches the "
                     "configured weak-target requirement "
                     "(detect.weak_pd_required=0.80) at 400-600 m with mean RCS "
                     "0.05-0.2 m^2?"),
        "seed": SEED, "mc": args.mc, "areas": args.areas, "rcs": args.rcs,
        "report_cap": REPORT_CAP, "power_grid": POWER_GRID,
        "power_rounds": POWER_ROUNDS,
        "cells": {k: CELLS[k][0] for k in args.cells},
        "families": {k: CELLS[k][1] for k in args.cells},
        "selections": {k: CELLS[k][2] for k in args.cells},
        "power_warm_start": {k: CELLS[k][3] for k in args.cells},
        "base_config": asdict(apply_overrides(config(SEED, REPORT_CAP),
                                              {"geometry.area_xy": 600.0})),
        "scope": (
            "Geometry, truth/belief gains, coarse and refined tables, DD bins "
            "and the fusion plan are built once per (area, rcs, trial) and "
            "shared by every cell; only selection and the Monte-Carlo decision "
            "re-execute, so cells are paired by trial. Max-min cells call "
            "tools.audit_v1_balanced_budget.balanced_select with an exact-zero "
            "cached D and zeroed C2F statistics because that selector never "
            "evaluates the scored C2F frontier; the detector is unchanged. The "
            "n_looks axis is a physical CPI length, the budget axis is a "
            "scenario input, the radar/radio axes are non-algorithmic, and only "
            "the fusion, power and max-min axes are algorithm changes. "
            "Exploratory screening, not an equivalence or non-inferiority "
            "test; power_seconds is wall-clock under a shared worker pool."),
        "hashes": {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in list(Path("isac_sim").glob("*.py"))
                   + [Path(__file__),
                      Path("tools/audit_v1_exact_budget.py"),
                      Path("tools/audit_v1_balanced_budget.py")]},
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2),
                                            encoding="utf8")

    labels = list(args.cells)
    specs = [(area, rcs, trial, labels)
             for area in args.areas for rcs in args.rcs
             for trial in range(args.mc)]
    rows = []
    with (args.out / "trials.csv").open("w", newline="",
                                        encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        handle.flush()
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(job, spec) for spec in specs]
            for index, future in enumerate(as_completed(futures), 1):
                batch = future.result()
                rows.extend(batch)
                writer.writerows(batch)
                handle.flush()
                if index % 20 == 0:
                    print(f"{index}/{len(specs)}", flush=True)

    summary = summarize(rows, args.areas, args.rcs, labels)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2),
                                           encoding="utf8")
    print("Finished: " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
