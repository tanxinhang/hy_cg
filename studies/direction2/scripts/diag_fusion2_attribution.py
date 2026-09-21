"""Direction 2 (cooperative fusion) -- attribution scan of the four levers.

Pre-registered before running (see studies/direction2/docs/AUDIT_DIRECTION2_ATTRIBUTION.md):

* every arm is PAIRED with the baseline: same geometry, same RCS draw, same
  measured residual fraction, same target retention, same detection stream tag;
* MC resolution at n=20 is ~0.10 in P_D (direction 1, measured twice), so only
  |delta P_D| >= 0.10 counts as "a lever with room";
* honesty: P_FA spread across arms <= 0.01, otherwise the gain is a threshold
  artefact;
* the scan only flips existing config keys / method labels. It adds NO new
  capability, so every arm is a **free** variant (no release-number impact).

Two profiles, because the first pass found that ``paper-canonical`` *already*
sets ``soft_stat_model=llr``, ``rcs_model=mean``, ``score_mode=exact_utility``
and ``stop_at_D_min=False``.  Turning those "on" is a no-op, so the informative
direction is ``--profile ablate``: switch the good setting OFF and see how much
falls out (that measures the space already captured), and squeeze the per-target
link budget (``--profile gain``) because a fusion change can only matter where
the budget is actually scarce.

gain arms
---------
1  baseline                 paper-canonical as released
2  links12                  selector.max_links_per_target=12
3  stop_on                  selector.stop_at_D_min=True  (early exit restored)
4  corr_on                  corr.enable=True
5  exact_thr                detect.exact_gaussian_replacement_threshold=True
6  pd_robust                method=proposed_c2f_adaptive_pd_robust

ablate arms (reverse controls: how much is each good setting worth?)
--------------------------------------------------------------------
1  baseline                 as released
2  links3                   max_links_per_target=3  (scarce budget)
3  links2                   max_links_per_target=2
4  links1                   max_links_per_target=1
5  score_first_order        selector.score_mode=first_order
6  rcs_iid                  detect.rcs_model=iid
7  soft_gaussian            detect.soft_stat_model=gaussian
8  no_target_priority       selector.use_target_priority=False
9  corr_on                  corr.enable=True

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY studies/direction2/scripts/diag_fusion2_attribution.py --trials 20
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
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from experiments.flow.simulate import evaluate_detection  # noqa: E402
from tools.run_tpuic_production import _selector  # noqa: E402


def _det(cfg, tables, selected, rng, base, trial_index: int, method: str):
    """Same detector as production, with the method label made an argument."""
    got = evaluate_detection(
        cfg, tables, selected, rng, method, None, base, trial_index=trial_index
    )
    detected, total_targets, fa_active, total_fa_active = got[0], got[1], got[2], got[3]
    return {
        "p_d": detected / max(total_targets, 1),
        "p_fa": fa_active / max(total_fa_active, 1),
        "n_selected": int(sum(len(v) for v in selected.values())),
    }


ARMS_GAIN: dict[str, dict] = {
    "baseline": {},
    "links12": {"selector.max_links_per_target": 12},
    "stop_on": {"selector.stop_at_D_min": True},
    "corr_on": {"corr.enable": True},
    "exact_thr": {"detect.exact_gaussian_replacement_threshold": True},
    "pd_robust": {},
}

ARMS_ABLATE: dict[str, dict] = {
    "baseline": {},
    "links3": {"selector.max_links_per_target": 3},
    "links2": {"selector.max_links_per_target": 2},
    "links1": {"selector.max_links_per_target": 1},
    "score_first_order": {"selector.score_mode": "first_order"},
    "rcs_iid": {"detect.rcs_model": "iid"},
    "soft_gaussian": {"detect.soft_stat_model": "gaussian"},
    "no_target_priority": {"selector.use_target_priority": False},
    "corr_on": {"corr.enable": True},
}

ARMS: dict[str, dict] = ARMS_GAIN

# method label override (the detector side, not the selector)
ARM_METHOD: dict[str, str] = {"pd_robust": "proposed_c2f_adaptive_pd_robust"}


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
    ov.update(extra)
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
    ap.add_argument("--profile", default="gain", choices=("gain", "ablate"))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if not args.out:
        args.out = "studies/direction2/data/diag_attribution_%s" % args.profile

    arms = ARMS_GAIN if args.profile == "gain" else ARMS_ABLATE
    cfg_base = build_config(args)
    arm_cfgs = {name: build_config(args, **ov) for name, ov in arms.items()}
    for name in ARM_METHOD:
        if name in arms:
            arm_cfgs[name] = build_config(args)  # method-level arm, same config
    os.makedirs(args.out, exist_ok=True)

    rows = []
    t0 = time.time()
    for trial in range(int(args.trials)):
        rng = np.random.default_rng([cfg_base.run.seed, int(trial)])
        geom = generate_geometry(cfg_base, rng)
        base = build_base_gains(cfg_base, geom, rng)

        # One measurement per trial: every arm shares the residual fraction and
        # the target retention, so the arms differ only in fusion / scheduling.
        ctx = cx.ReceiverContext.from_trial(cfg_base, geom, geom, base, arm=args.arm)
        meas = cx.measure_receiver_context(
            ctx, rng=np.random.default_rng([cfg_base.run.seed, 10 ** 6 + int(trial)])
        )
        frac = np.asarray(meas.fraction, dtype=float)
        retention = np.clip(np.asarray(meas.eta_survive, dtype=float), 0.0, 1.0)

        for name, cfg in arm_cfgs.items():
            tables = compute_link_tables(
                cfg, base,
                residual_fraction_by_receiver=frac,
                target_retention_by_receiver=retention,
            )
            selected = _selector(cfg, base, tables)
            method = ARM_METHOD.get(name, "proposed_c2f")
            det_rng = np.random.default_rng([cfg_base.run.seed, 4 * 10 ** 6 + int(trial)])
            det = _det(cfg, tables, selected, det_rng, base, int(trial), method)
            rows.append({
                "trial": int(trial), "arm": name,
                "p_d": det["p_d"], "p_fa": det["p_fa"],
                "n_selected": det["n_selected"],
            })
        print("  trial %d/%d [%.0f s]" % (trial + 1, args.trials, time.time() - t0),
              flush=True)

    keys = ["trial", "arm", "p_d", "p_fa", "n_selected"]
    path = os.path.join(args.out, "attribution.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def col(name, key):
        return np.array([r[key] for r in rows if r["arm"] == name], dtype=float)

    base_pd = float(np.mean(col("baseline", "p_d")))
    print("\n=== direction 2 attribution (paired, n=%d) ===" % int(args.trials))
    print("%-18s %8s %8s %8s %9s %9s" % ("arm", "P_D", "P_FA", "links", "dP_D", "se"))
    summary = {"arms": {}, "config": {
        "preset": args.preset, "area": args.area, "rcs": args.rcs,
        "seed": args.seed, "trials": int(args.trials), "arm": args.arm}}
    for name in arm_cfgs:
        pd_ = float(np.mean(col(name, "p_d")))
        d = col(name, "p_d") - col("baseline", "p_d")
        n = d.size
        se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        summary["arms"][name] = {
            "p_d": pd_, "p_fa": float(np.mean(col(name, "p_fa"))),
            "n_selected": float(np.mean(col(name, "n_selected"))),
            "delta_p_d": pd_ - base_pd, "se": se,
            # pre-registered: only |delta| >= 0.10 is resolvable at n=20
            "resolvable": bool(abs(pd_ - base_pd) >= 0.10),
        }
        print("%-18s %8.4f %8.4f %8.2f %+9.4f %9.4f"
              % (name, pd_, summary["arms"][name]["p_fa"],
                 summary["arms"][name]["n_selected"], pd_ - base_pd, se))

    fas = [summary["arms"][n]["p_fa"] for n in arm_cfgs]
    summary["honesty"] = {"p_fa_spread": float(max(fas) - min(fas)),
                          "pass": bool(max(fas) - min(fas) <= 0.01)}
    print("\nP_FA spread = %.4f (<=0.01 => not a threshold artefact)"
          % summary["honesty"]["p_fa_spread"])

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print("wrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
