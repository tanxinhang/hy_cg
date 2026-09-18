"""Software-knob sweep for the 500-800 m low-RCS operating region.

Operational question
--------------------
The main scenario is a 500-800 m deployment against targets of mean RCS
0.05-0.2 m^2 (small-UAV class).  A 4 km cell against 0.05-0.2 m^2 targets is
not observable at all (P_D ~ 0.083-0.202, i.e. at the P_FA floor), so the
deployment side is shortened until the region is merely *hard* rather than
hopeless.  ``detect.weak_pd_required`` is 0.80.  This script measures how much
of the resulting gap the *already implemented* knobs close.

Default seed is 10917, not the historical ``SEED`` 10919: 10917 is the seed of
the released 4 km / 600 m RCS-budget table (``tools/audit_v1_rcs_joint.py``,
``tools/audit_v1_rcs_600m.py``), so the ``base`` cell at ``--areas 600``
reproduces that table's ``rcs*_k8`` ``v1`` rows bit-for-bit and doubles as a
cross-tool parity check.  The module-level ``SEED`` is left at 10919 because
``tools/probe_lowrcs_shortfall.py`` imports it.

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
DEFAULT_SEED = 10917
SCENARIO_AREAS = (500.0, 600.0, 700.0, 800.0)
REPORT_CAP = 8

# A deployment side length alone does not define a scenario: with the released
# 4 km vertical geometry (UAV 800-1200 m, target 700-1500 m) the slant range at a
# 500-800 m footprint is dominated by altitude, so the horizontal axis buys only
# ~2.7 dB from 500 to 800 m instead of the geometric 8.2 dB.  ``compact`` adopts
# the vertical geometry of the existing ``small-uav-compact-800m`` preset
# (h_uav 200-500, h_target 200-500, comm_range 1000) and extends it by that
# preset's own ratio comm_range = 1.25 * area_xy, which reproduces the preset
# exactly at 800 m.
SCENARIOS = {
    "paper-vertical": {},
    "compact-small-uav": {
        "geometry.h_uav_min": 200.0,
        "geometry.h_uav_max": 500.0,
        "geometry.h_target_min": 200.0,
        "geometry.h_target_max": 500.0,
    },
}
COMM_RANGE_RATIO = 1.25  # anchored on small-uav-compact-800m: 1000 / 800
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
    "looks128": ({"detect.n_looks": 128}, "looks", "utility", None),
    "corr": ({"corr.enable": True}, "fusion", "utility", None),
    "capacitated": ({"fusion.rule": "nearest_target_capacitated"},
                    "fusion", "utility", None),
    "budget": ({"selector.max_remote_reports": 16,
                "selector.max_total_links": 90,
                "selector.max_links_per_target": 9}, "budget", "utility", None),
    "maxmin": ({}, "maxmin", "maxmin", None),
    "maxmin_looks64": ({"detect.n_looks": 64}, "maxmin", "maxmin", None),
    "maxmin_looks128": ({"detect.n_looks": 128}, "maxmin", "maxmin", None),
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
           "power_seconds", "rinr_median"]
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
    # ``rinr`` is the noise-limited / interference-limited coordinate that every
    # lever ranking is conditional on; recording it lets a kappa_dc sweep state
    # the regime each row was measured in instead of asserting one.
    return dict(
        label=label, family=family,
        pd=float(result.detected / result.total_targets),
        worst_pd=float(per_target.min()),
        weak_pd=float(per_target[int(result.weak_target_index)]),
        rinr_median=float(np.median(truth_tables.rinr)),
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


def scenario_overrides(scenario, area):
    """Resolve a named scenario into dotted-path overrides for one area."""
    overrides = dict(SCENARIOS[scenario])
    if scenario == "compact-small-uav":
        overrides["geometry.comm_range"] = COMM_RANGE_RATIO * float(area)
    return overrides


def parse_phys_override(pairs):
    """Parse ``KEY=VALUE`` strings into a typed dotted-path override dict.

    ``isac_sim.cli.parse_overrides`` keeps every value as ``str``; the sweep
    passes these straight into ``apply_overrides`` with no further coercion, so
    the values are typed here (bool / int / float / str) instead.
    """
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--override expects KEY=VALUE, got {pair!r}")
        key, _, raw = pair.partition("=")
        key, raw = key.strip(), raw.strip()
        low = raw.lower()
        if low in ("true", "false"):
            value = low == "true"
        elif low in ("none", "null"):
            value = None
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        out[key] = value
    return out


def job(spec):
    seed, scenario, area, rcs, trial, labels, override = spec
    base_cfg = apply_overrides(
        config(seed, REPORT_CAP),
        {"geometry.area_xy": float(area), "detect.target_rcs": float(rcs),
         **scenario_overrides(scenario, area), **override})
    validate_config(base_cfg)

    rng = np.random.default_rng([seed, trial])
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
                entry = {m: float(np.mean([float(r.get(m, np.nan))
                                           for r in group]))
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS),
                        default="paper-vertical")
    parser.add_argument("--areas", type=float, nargs="+",
                        default=list(SCENARIO_AREAS))
    parser.add_argument("--rcs", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2])
    parser.add_argument("--cells", nargs="+", default=LABELS, choices=LABELS)
    parser.add_argument(
        "--override", nargs="*", default=[], metavar="KEY=VALUE",
        help="Global PHYSICAL override applied to the shared base config "
             "(before any cell override), so selection and evaluation see the "
             "same channel.  Use this for scenario-level quantities such as "
             "interference.direct_cancellation_db; a cell-level override would "
             "leave the selection-stage link tables on the old value.")
    parser.add_argument("--out", type=Path,
                        default=Path("results_v1_lowrcs_sweep"))
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild summary.json from an existing trials.csv.")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)

    if args.report_only:
        rows = list(csv.DictReader((args.out / "trials.csv").open(
            encoding="utf8")))
        for row in rows:
            for key in ("area_m", "rcs_m2", "trial"):
                row[key] = float(row[key])
            for key in METRICS:
                row[key] = float(row[key])
        labels = [c for c in LABELS
                  if any(r["label"] == c for r in rows)]
        summary = summarize(rows, args.areas, args.rcs, labels)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf8")
        print("Rebuilt summary for " + str(args.out), flush=True)
        return

    if (args.out / "protocol.json").exists():
        raise SystemExit("Use a fresh output directory")

    protocol = {
        "question": ("Which already-implemented software knob reaches the "
                     "configured weak-target requirement "
                     "(detect.weak_pd_required=0.80) at 500-800 m with mean RCS "
                     "0.05-0.2 m^2?"),
        "seed": args.seed, "mc": args.mc, "areas": args.areas, "rcs": args.rcs,
        "scenario": args.scenario,
        "phys_override": parse_phys_override(args.override),
        "scenario_overrides": {f"area{a:g}": scenario_overrides(
            args.scenario, a) for a in args.areas},
        "report_cap": REPORT_CAP, "power_grid": POWER_GRID,
        "power_rounds": POWER_ROUNDS,
        "cells": {k: CELLS[k][0] for k in args.cells},
        "families": {k: CELLS[k][1] for k in args.cells},
        "selections": {k: CELLS[k][2] for k in args.cells},
        "power_warm_start": {k: CELLS[k][3] for k in args.cells},
        "base_config": asdict(apply_overrides(
            config(args.seed, REPORT_CAP),
            {"geometry.area_xy": 600.0,
             **scenario_overrides(args.scenario, 600.0),
             **parse_phys_override(args.override)})),
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
            "test; power_seconds is wall-clock under a shared worker pool. "
            "With the default seed 10917 the 'base' cell at area 600 "
            "reproduces tools/audit_v1_rcs_600m.py's rcs*_k8 'v1' rows."),
        "hashes": {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in list(Path("isac_sim").glob("*.py"))
                   + [Path(__file__),
                      Path("tools/audit_v1_exact_budget.py"),
                      Path("tools/audit_v1_balanced_budget.py")]},
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2),
                                            encoding="utf8")

    labels = list(args.cells)
    override = parse_phys_override(args.override)
    specs = [(args.seed, args.scenario, area, rcs, trial, labels, override)
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
