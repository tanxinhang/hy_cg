"""Direction 2 -- attribution under the *real* closed loop (belief mode).

Why this script exists (and why ``diag_fusion2_attribution.py`` is not enough)
-----------------------------------------------------------------------------
``paper-canonical`` sets ``prior.belief_mode=True``.  In belief mode the
production loop carries **two worlds** (``experiments/flow/simulate.py:1184``):

    base_truth / tables_truth    <- what the detector sees (truth)
    base_belief / tables_belief  <- what the scheduler sees (belief)

and a selected link only counts at detection time if its belief-guided DD
window actually captured the true bin (``truth_captured_links``).  The first
attribution script built a single table, i.e. it silently ran in the
"perfect prior" world -- where the scheduler already picks the true best links
and therefore *adding links is mechanically worth zero*.  Every "fusion is
saturated" conclusion from that script is conditional on a world without
belief error.  This script re-runs the scan in the world the release actually
describes.

Arms (all paired: same geometry, same belief draw, same residual fraction)
--------------------------------------------------------------------------
1  baseline            max_links_per_target = 6
2  links12             max_links_per_target = 12   (can extra links cover a
                                                    mis-ranked belief list?)
3  links9              max_links_per_target = 9
4  links3              max_links_per_target = 3    (scarce budget)
5  robust              scheduler sees geometry_robust_base(belief)
6  robust_links12      both

Pre-registered criteria: |dP_D| >= 0.10 counts (MC=20 resolution); P_FA spread
<= 0.01; every arm shares geometry / belief / residual / detection stream.

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY studies/direction2/scripts/diag_fusion2_belief.py --trials 20
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.receiver import cancellation as cx  # noqa: E402
from isac_sim.scenario.belief import (  # noqa: E402
    BeliefState,
    belief_dd_std_bins,
    geometry_robust_base,
    truth_captured_links,
)
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from experiments.flow.simulate import _build_plan, evaluate_detection  # noqa: E402
from experiments.selection import select_lagrangian  # noqa: E402


ARMS: dict[str, dict] = {
    "baseline": {},
    "links12": {"selector.max_links_per_target": 12},
    "links9": {"selector.max_links_per_target": 9},
    "links3": {"selector.max_links_per_target": 3},
    "robust": {"__robust__": True},
    "robust_links12": {"__robust__": True, "selector.max_links_per_target": 12},
}

# The four arms that survived the n=20 screen, re-run at a larger MC so that
# effects of order 0.05 become resolvable (paired sd ~= 0.122, so se <= 0.017
# needs n >= 53; n=40 is the compromise).  ``exact_thr`` is here because the
# n=20 run showed P_FA drifting from 0.0486 (baseline) to 0.0600
# (robust_links12): the Cornish-Fisher fallback threshold is mis-calibrated for
# the larger fused sets, and the exact mixture calibration already exists.
ARMS_FOCUS: dict[str, dict] = {
    "baseline": {},
    "links12": {"selector.max_links_per_target": 12},
    "robust": {"__robust__": True},
    "exact_thr": {"detect.exact_gaussian_replacement_threshold": True},
}

# ---- model version vs delta-sensitive version ----------------------------
# The two numbers direction 2 was asked to report.  The delta gate is only
# visible under ``residual_accounting=structural`` (direction 1: under the
# production ``measured`` accounting, moving delta shifts kappa by 0.17 dB),
# so both versions run in the structural口径:
#   *_model  : delta = 0    -> dictionary is complete, kappa ~68 dB is an
#              UPPER BOUND, not an attainable operating point
#   *_delta3e-3 : delta = 3e-3 DD bins (an *uncalibrated* sweep value, known to
#              exceed the 1.615e-3 requirement by 1.86x) -> report as
#              "if delta = X then P_D = Y", never as "the real version".
ARMS_TWOVERSION: dict[str, dict] = {
    "baseline": {},
    "links12_measured": {"selector.max_links_per_target": 12},
    "links12_model": {"selector.max_links_per_target": 12,
                      "__accounting__": "structural", "__delta__": 0.0},
    "links12_delta3e3": {"selector.max_links_per_target": 12,
                         "__accounting__": "structural", "__delta__": 3e-3},
    "robust_model": {"__robust__": True,
                     "__accounting__": "structural", "__delta__": 0.0},
    "robust_delta3e3": {"__robust__": True,
                        "__accounting__": "structural", "__delta__": 3e-3},
}


def build_config(args, **extra) -> Config:
    cfg = apply_preset(Config(), args.preset)
    ov = {
        "geometry.area_xy": float(args.area),
        "detect.target_rcs": float(args.rcs),
        "scale.M": int(args.m),
        "scale.Q": int(args.q),
        "run.seed": int(args.seed),
        "run.verbose": False,
        "cancellation.enable": True,
        "aperture.enable": True,
        "aperture.m_rx": int(args.m_rx),
    }
    for key, value in extra.items():
        if key == "__robust__":
            continue
        if key == "__accounting__":
            ov["cancellation.residual_accounting"] = value
            continue
        if key == "__delta__":
            ov["cancellation.direct_estimation_sigma_delay_bins"] = float(value)
            continue
        ov[key] = value
    return apply_overrides(cfg, ov)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--m-rx", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--arm", default="tp_uic_full")
    ap.add_argument("--profile", default="all",
                    choices=("all", "focus", "twoversion"))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.profile == "all":
        arms = ARMS
    elif args.profile == "focus":
        arms = ARMS_FOCUS
    else:
        arms = ARMS_TWOVERSION
    if not args.out:
        args.out = {
            "all": "studies/direction2/data/diag_belief",
            "focus": "studies/direction2/data/diag_belief_focus",
            "twoversion": "studies/direction2/data/diag_belief_twoversion",
        }[args.profile]

    cfg_base = build_config(args)
    arm_cfgs = {name: build_config(args, **ov) for name, ov in arms.items()}
    os.makedirs(args.out, exist_ok=True)

    rows = []
    t0 = time.time()
    for trial in range(int(args.trials)):
        rng = np.random.default_rng([cfg_base.run.seed, int(trial)])
        geom = generate_geometry(cfg_base, rng)

        # --- truth world (what the detector sees) -------------------------
        base_truth = build_base_gains(cfg_base, geom, rng)

        # --- belief world (what the scheduler sees) ------------------------
        belief = BeliefState.from_truth(cfg_base, geom, rng)
        geom_belief = belief.as_geometry(geom)
        base_belief = build_base_gains(
            cfg_base, geom_belief, rng,
            channel=base_truth, rcs_view=cfg_base.prior.scheduler_rcs.lower(),
        )
        belief_dd_std = belief_dd_std_bins(cfg_base, geom_belief, belief)

        for name, cfg in arm_cfgs.items():
            # --- receiver measurement, under THIS arm's receiver model -----
            # (the accounting / delta gates change what the receiver reports,
            # so the fraction cannot be measured once per trial)
            ctx = cx.ReceiverContext.from_trial(cfg, geom, geom, base_truth, arm=args.arm)
            meas = cx.measure_receiver_context(
                ctx, rng=np.random.default_rng([cfg_base.run.seed, 10 ** 6 + int(trial)])
            )
            frac = np.asarray(meas.fraction, dtype=float)
            retention = np.clip(np.asarray(meas.eta_survive, dtype=float), 0.0, 1.0)
            kappa_db = float(np.median(-10.0 * np.log10(np.maximum(frac, 1e-300))))

            tables_truth = compute_link_tables(
                cfg, base_truth,
                residual_fraction_by_receiver=frac,
                target_retention_by_receiver=retention,
            )
            sel_base = base_belief
            if arms[name].get("__robust__"):
                sel_base = geometry_robust_base(cfg, base_belief)
            sel_tables = compute_link_tables(
                cfg, sel_base,
                residual_fraction_by_receiver=frac,
                target_retention_by_receiver=retention,
            )
            plan = _build_plan(cfg, sel_base, sel_tables, geom_belief)
            selected = select_lagrangian(cfg, sel_base, sel_tables, plan)[0]
            # Exactly what the production loop does (simulate.py:976): in
            # belief mode only the links whose belief-guided DD window really
            # captured the true bin reach the detector.  This is the mechanism
            # that turns a mis-ranked belief list into lost detections, and it
            # is absent from the single-world attribution script.
            captured = truth_captured_links(
                cfg, base_truth, sel_base, selected, belief_dd_std
            )
            det_rng = np.random.default_rng([cfg_base.run.seed, 4 * 10 ** 6 + int(trial)])
            got = evaluate_detection(
                cfg, tables_truth, captured, det_rng, "proposed_c2f", plan,
                base_truth, trial_index=int(trial),
            )
            rows.append({
                "trial": int(trial), "arm": name,
                "p_d": got[0] / max(got[1], 1),
                "p_fa": got[2] / max(got[3], 1),
                "n_selected": int(sum(len(v) for v in selected.values())),
                "n_captured": int(sum(len(v) for v in captured.values())),
                "kappa_db": kappa_db,
                "belief_capture_rate": (
                    float(sum(len(v) for v in captured.values()))
                    / max(int(sum(len(v) for v in selected.values())), 1)
                ),
            })
        print("  trial %d/%d [%.0f s]" % (trial + 1, args.trials, time.time() - t0),
              flush=True)

    keys = ["trial", "arm", "p_d", "p_fa", "n_selected", "n_captured",
            "belief_capture_rate", "kappa_db"]
    path = os.path.join(args.out, "belief.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def col(name, key):
        return np.array([r[key] for r in rows if r["arm"] == name], dtype=float)

    base_pd = float(np.mean(col("baseline", "p_d")))
    base_p_fa = float(np.mean(col("baseline", "p_fa")))
    print("\n=== direction 2 attribution, BELIEF mode (paired, n=%d) ===" % int(args.trials))
    print("%-16s %8s %8s %8s %8s %9s %9s %9s"
          % ("arm", "P_D", "P_FA", "links", "capt", "dP_D", "se", "dP_FA"))
    summary = {"arms": {}, "config": {
        "preset": args.preset, "area": args.area, "rcs": args.rcs,
        "seed": args.seed, "trials": int(args.trials)}}
    for name in arm_cfgs:
        pd_ = float(np.mean(col(name, "p_d")))
        p_fa_ = float(np.mean(col(name, "p_fa")))
        d = col(name, "p_d") - col("baseline", "p_d")
        n = d.size
        se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        summary["arms"][name] = {
            "p_d": pd_, "p_fa": p_fa_,
            "n_selected": float(np.mean(col(name, "n_selected"))),
            "n_captured": float(np.mean(col(name, "n_captured"))),
            "belief_capture_rate": float(np.mean(col(name, "belief_capture_rate"))),
            "delta_p_d": pd_ - base_pd, "se": se,
            "delta_p_fa": p_fa_ - base_p_fa,
            "kappa_db_median": float(np.median(col(name, "kappa_db"))),
            "resolvable": bool(abs(pd_ - base_pd) >= 0.10),
        }
        print("%-16s %8.4f %8.4f %8.2f %8.2f %+9.4f %9.4f %+9.4f"
              % (name, pd_, summary["arms"][name]["p_fa"],
                 summary["arms"][name]["n_selected"],
                 summary["arms"][name]["n_captured"], pd_ - base_pd, se,
                 summary["arms"][name]["delta_p_fa"]))

    fas = [summary["arms"][n]["p_fa"] for n in arm_cfgs]
    summary["honesty"] = {"p_fa_spread": float(max(fas) - min(fas)),
                          "pass": bool(max(fas) - min(fas) <= 0.01)}
    print("\nP_FA spread = %.4f" % summary["honesty"]["p_fa_spread"])

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print("wrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
