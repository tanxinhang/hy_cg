"""One trial, one receiver: truth/belief -> TP-UIC -> link tables -> C2F -> detection.

Why this file exists
--------------------
``tools/run_tpuic_production.py`` was the first end-to-end wiring of the
canceller, but it ran ``select_lagrangian`` while every released low-RCS number
comes from ``proposed_c2f_adaptive_pd`` (``select_c2f_adaptive`` under
``score_mode="detector_pd"``).  The two differ by ~0.10 in P_D on the same
scenario, so anything measured on that tool could not be quoted as "the proposed
method with TP-UIC".  It was a production-*like* pipeline, not the production
pipeline.

This driver runs the real one, and it runs the receiver in the *same* trial as
the selector:

    generate_geometry -> (truth)
    BeliefState.from_truth -> geom_belief      (one belief, shared)
        |
        +-- ReceiverContext(truth, belief) -> TP-UIC -> (I_res, eta_surv)
        |
        +-- compute_link_tables(frac, retention) -> select_c2f_adaptive
                                                  -> evaluate_detection

Three arms, identical in every respect except the receiver model, so the
differences are attributable to the receiver and to nothing else:

* ``constant``             -- the frozen ``kappa_dc``, no echo survival.  This is
  the released assumption and the arm every published number describes.
* ``measured_denominator`` -- measured ``I_res`` only.  The half-closed loop:
  it prices the interference left behind but still credits the target with
  energy the canceller took.  This was the state of the wiring before the
  numerator bridge existed.
* ``measured_full``        -- measured ``I_res`` **and** measured ``eta_surv``.
  The affine map the receiver actually applies, on both halves of the SINR.

Because all three share the geometry, the belief, the selector and the detector
draws, the arm differences are *paired*, which is the only comparison the
project's own rules allow across runs (a paired reference taken from a different
method reverses conclusions).

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/run_receiver_closed_loop.py --trials 20 --area 600 --rcs 0.1

Nothing here touches the default path: every arm is opt-in through this file.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys
import time
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim import cancellation as cx  # noqa: E402
from isac_sim.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.model import build_base_gains, compute_link_tables, generate_geometry  # noqa: E402
from isac_sim.selection import select_c2f_adaptive  # noqa: E402
from isac_sim.simulate import _build_plan, evaluate_detection, rng_for_detection  # noqa: E402

METHOD = "proposed_c2f_adaptive_pd"

ARMS = (
    ("constant", False, False),
    ("measured_denominator", True, False),
    ("measured_full", True, True),
)


def build_config(args) -> Config:
    cfg = apply_preset(Config(), args.preset)
    return apply_overrides(
        cfg,
        {
            "geometry.area_xy": float(args.area),
            "detect.target_rcs": float(args.rcs),
            "run.seed": int(args.seed),
            "run.verbose": False,
            "cancellation.enable": True,
            "cancellation.n_cpi": int(args.n_cpi),
            "cancellation.max_protected_targets": int(args.max_protected_targets),
            # ``proposed_c2f_adaptive_pd`` is the refined path; without this the
            # adaptive selector is never constructed and the tool would silently
            # measure a different method while labelling it the proposed one.
            "refine.enable": True,
        },
    )


def _belief_state(cfg: Config, geom, rng):
    """The belief the production scheduler sees -- the receiver gets the same one."""
    from isac_sim.belief import BeliefState

    return BeliefState.from_truth(cfg, geom, rng)


def run_trial(cfg: Config, args, index: int) -> List[dict]:
    rng = np.random.default_rng([cfg.run.seed, int(index)])
    geom = generate_geometry(cfg, rng)
    base_truth = build_base_gains(cfg, geom, rng)

    # ---- One belief, two consumers -------------------------------------
    from isac_sim.belief import belief_dd_std_bins, truth_captured_links

    belief = _belief_state(cfg, geom, rng)
    geom_belief = belief.as_geometry(geom)
    base_belief = build_base_gains(
        cfg, geom_belief, rng, channel=base_truth,
        rcs_view=cfg.prior.scheduler_rcs.lower(),
    )
    belief_dd_std = belief_dd_std_bins(cfg, geom_belief, belief)

    m = int(cfg.scale.M)
    q = int(cfg.scale.Q)
    receivers = range(min(m, int(args.max_receivers))) if args.max_receivers > 0 else range(m)

    # ---- The receiver, measured on the trial's own truth/belief --------
    ctx = cx.ReceiverContext.from_trial(
        cfg, geom, geom_belief, base_truth, arm=args.arm
    )
    t0 = time.time()
    meas = cx.measure_receiver_context(
        ctx, rng=np.random.default_rng([cfg.run.seed, 10 ** 6 + int(index)]),
        receivers=receivers,
    )
    measure_s = time.time() - t0

    # Receivers that were never measured degrade to the *frozen* fraction, not
    # to 1.0.  ``fraction = 1`` means "no cancellation at all", which is 40 dB
    # worse than the released assumption and would make a partial sweep look
    # like a catastrophic receiver instead of an incomplete measurement.
    frozen_fraction = 10.0 ** (-float(cfg.interference.direct_cancellation_db) / 10.0)
    fraction = meas.fraction.copy()
    # Two survival figures, ~1.5 dB apart on this scenario, and which one is
    # bridged must be stated: see ``ReceiverMeasurement.as_retention``.
    retention = meas.as_retention(per_target=q, source=args.retention).copy()
    retention_alt = meas.as_retention(
        per_target=q, source=("q" if args.retention == "field" else "field")
    )
    if len(receivers) < m:
        keep = np.zeros(m, dtype=bool)
        keep[list(receivers)] = True
        fraction = np.where(keep, fraction, frozen_fraction)
        retention = np.where(keep[:, None], retention, 1.0)

    rows: List[dict] = []
    for label, use_fraction, use_retention in ARMS:
        frac = fraction if use_fraction else None
        eta = retention if use_retention else None
        tables_sched = compute_link_tables(
            cfg, base_belief, residual_fraction_by_receiver=frac,
            target_retention_by_receiver=eta,
        )
        # The evaluation table is the refined one built on the *belief* gains,
        # exactly as ``run_method_on_trial`` does (``tables_eval = c2f_tables``
        # when refinement is on) -- the detector is a receiver capability and
        # every method is evaluated on refined statistics.  It carries the same
        # receiver model as the selector: giving the detector the frozen
        # constant while giving the selector the algorithm would be exactly the
        # two-worlds failure this driver exists to remove.
        tables_eval = compute_link_tables(
            cfg, base_belief, dd_gain=base_belief.eta_fine,
            residual_fraction_by_receiver=frac,
            target_retention_by_receiver=eta,
        )
        plan = _build_plan(cfg, base_belief, tables_sched, geom_belief)

        # The fine stage rebuilds its own table, and the default builder would
        # rebuild it with the frozen constant -- i.e. the coarse stage would run
        # in the algorithm's world and the fine stage in the assumption's.  That
        # is the exact two-worlds failure this driver exists to remove, and it
        # is invisible without this injection: the call succeeds and the numbers
        # look plausible.
        def refined_builder(cfg_, base_, dd_gain=None, **kw):
            return compute_link_tables(
                cfg_, base_, dd_gain=dd_gain,
                residual_fraction_by_receiver=frac,
                target_retention_by_receiver=eta, **kw
            )

        sel_cfg = apply_overrides(cfg, {"selector.score_mode": "detector_pd"})
        selected = select_c2f_adaptive(
            sel_cfg, base_belief, tables_sched, plan=plan,
            refined_table_builder=refined_builder,
        )[0]

        # Belief mode: a link only carries target evidence if the belief-guided
        # search window captures the true delay-Doppler bin.  Dropping this
        # would credit the detector with observations the receiver cannot form.
        selected_eval = truth_captured_links(
            cfg, base_truth, base_belief, selected, belief_dd_std
        )
        det_rng = rng_for_detection(cfg, int(index))
        got = evaluate_detection(
            cfg, tables_eval, selected_eval, det_rng, METHOD, plan, base_truth,
            trial_index=int(index),
        )
        detected, total_targets, fa_active, total_fa_active = got[0], got[1], got[2], got[3]
        # The receiver figures are reported over the *measured* subset only: an
        # unmeasured receiver carries the frozen fraction and a 0 dB depth, and
        # folding those into the median would report a receiver nobody ran.
        measured = list(receivers)
        rows.append({
            "trial": int(index),
            "arm": label,
            "p_d": detected / max(total_targets, 1),
            "p_fa": fa_active / max(total_fa_active, 1),
            "n_selected": int(sum(len(v) for v in selected.values())),
            "n_captured": int(sum(len(v) for v in selected_eval.values())),
            "kappa_db_median": float(np.median(meas.kappa_db[measured])) if use_fraction
                               else float(cfg.interference.direct_cancellation_db),
            "fraction_median": float(np.median(meas.fraction[measured])) if use_fraction
                               else frozen_fraction,
            "eta_survive_median": float(np.median(retention[measured][:, 0])) if use_retention else 1.0,
            "eta_survive_alt_median": float(np.median(retention_alt[measured][:, 0])),
            "eta_protect_median": float(np.median(meas.eta_protect[measured])) if use_fraction else float("nan"),
            "rinr_median_db": float(10.0 * np.log10(max(float(np.median(tables_eval.rinr)), 1e-300))),
            "measure_seconds": measure_s,
        })
    return rows


def _paired_delta(rows: Sequence[dict], a: str, b: str, key: str = "p_d"):
    """Paired difference ``b - a`` on the same trial, with a standard error.

    Pairing is mandatory here: the arms share geometry, belief, selector and
    detection draws, so a difference of marginals would mix the receiver effect
    with the scenario effect.
    """
    va = {r["trial"]: float(r[key]) for r in rows if r["arm"] == a}
    vb = {r["trial"]: float(r[key]) for r in rows if r["arm"] == b}
    trials = sorted(set(va) & set(vb))
    if len(trials) < 2:
        return float("nan"), float("nan"), 0
    diffs = [vb[t] - va[t] for t in trials]
    mean = sum(diffs) / len(diffs)
    var = sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1)
    return mean, math.sqrt(var / len(diffs)), len(trials)


def report(rows: List[dict], args) -> None:
    print("\n=== Receiver-closed loop : %s (area = %.0f m, RCS = %.2f m^2) ==="
          % (METHOD, args.area, args.rcs))
    print("  one geometry, one belief, one selector, one detector draw per trial")
    print("  estimator arm = %s ; n_cpi = %d ; protected targets = %d ; trials = %d"
          % (args.arm, args.n_cpi, args.max_protected_targets, args.trials))
    print("  numerator bridge = eta_survive[%s] (see --retention); the other"
          % args.retention)
    print("  figure is reported in the CSV as eta_survive_alt_median")
    print()
    print("  %-21s %10s %10s %9s %9s %8s" % (
        "receiver", "kappa med", "eta_surv", "P_D", "P_FA", "links"))
    for label, _, _ in ARMS:
        sub = [r for r in rows if r["arm"] == label]
        if not sub:
            continue
        mean = lambda k: sum(float(r[k]) for r in sub) / len(sub)
        print("  %-21s %10.2f %10.4f %9.4f %9.4f %8.1f" % (
            label,
            st.median(float(r["kappa_db_median"]) for r in sub),
            st.median(float(r["eta_survive_median"]) for r in sub),
            mean("p_d"), mean("p_fa"),
            st.median(float(r["n_selected"]) for r in sub)))

    print("\n  paired differences on the same trial (mean +/- 1 s.e.):")
    for a, b in (("constant", "measured_denominator"),
                 ("constant", "measured_full"),
                 ("measured_denominator", "measured_full")):
        d, se, n = _paired_delta(rows, a, b)
        if n:
            print("    %-21s -> %-21s  dP_D = %+.4f +/- %.4f  (n = %d)"
                  % (a, b, d, se, n))
    print("\n  The last row is the price of the numerator bridge: how much the")
    print("  half-closed loop over-states P_D by crediting the target with echo")
    print("  energy the canceller actually removed.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-cpi", type=int, default=1)
    ap.add_argument("--max-protected-targets", type=int, default=3)
    ap.add_argument("--arm", default="tp_uic_full")
    ap.add_argument("--retention", default="field", choices=("field", "q"),
                    help="which echo-survival figure the numerator bridges: the "
                         "whole-field ratio (conservative) or the tested target's "
                         "own survival (optimistic); they differ by ~1.5 dB")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--max-receivers", type=int, default=0,
                    help="measure only the first N receivers (0 = all)")
    ap.add_argument("--out", default="results_receiver_closed_loop")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    os.makedirs(args.out, exist_ok=True)
    print("Receiver-closed loop : %s, area=%.0f m, RCS=%.2f m^2, seed=%d"
          % (args.preset, args.area, args.rcs, args.seed))
    print("  M = %d UAVs, Q = %d targets, K = %d bins, method = %s"
          % (cfg.scale.M, cfg.scale.Q, cfg.waveform.N * cfg.waveform.L, METHOD))

    rows: List[dict] = []
    t0 = time.time()
    for t in range(int(args.trials)):
        rows.extend(run_trial(cfg, args, t))
        if not args.quiet and (t + 1) % max(int(args.trials) // 5, 1) == 0:
            print("  trial %d/%d  [%.0f s]" % (t + 1, args.trials, time.time() - t0),
                  flush=True)
    path = os.path.join(args.out, "closed_loop.csv")
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
