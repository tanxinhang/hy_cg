# RETIRED PREMISE (2026-09-20): this script swept / read the config field
# `interference.direct_cancellation_db` (kappa_dc).
# That field was DELETED: it asserted a fixed 40 dB direct-path cancellation with no
# receiver implementation behind it while propping up the whole SINR denominator.
# Direct-path cancellation is now only ever a MEASURED TP-UIC residual.  Running this
# script as-is will fail on the missing attribute -- kept as historical evidence only.
"""Formal experiment: un-coordinated vs sparse-illumination detection (real MC).

Locked scenario: 500 m footprint, RCS 0.2 m^2, V1 release 口径 (``target-local-v1``).

Everything up to now was measured with the analytic surrogate
``predicted_pd_for_links``. This runner replaces it with the implemented detector
(``simulate.evaluate_detection``): exact LLR statistics, the Cornish-Fisher
calibrated threshold, packet mixture, and Monte-Carlo detection counts. The
surrogate is still ~24 % biased low in the deep low-SNR regime, so the numbers
here are the ones that may be quoted.

Three arms, paired inside each trial:
  1. ``uncoordinated`` -- every UAV radiates during the sensing observation.
  2. ``mask_only``     -- same selection, but only the illuminators it uses radiate.
     Isolates "silencing" from "re-selection".
  3. ``sparse``        -- selection *under* the coordination口径: the illuminator
     mask is fed back into both stages of the release selector until it repeats
     (:func:`isac_sim.experiments.flow.sweeps.coordination.select_with_coordination`, which closes F1),
     with ``selector.tx_penalty`` charged per newly woken radiator. Only the
     illuminators of the resulting schedule radiate.

Arm 3 is the only arm that measures coordination rather than silencing: arms 1 and 2
select on un-gated tables on purpose, so their schedule is the un-coordinated one.

What changed on 2026-09-17 (do not compare against earlier runs of this file):
  * the gate now removes the muted illuminator's *echo*, not only its interference
    (model.py, finding A7). Before that, a muted-but-used illuminator was still
    credited with its observation, so every coordinated number here was inflated.
  * the mask is built by ``coordination.illuminator_mask`` only, and
    ``select_c2f_adaptive`` consumes it in its coarse *and* fine stage -- the
    previous arm used the post-hoc reporter-mask convention (findings A8/F1).
  * wiring the price into the coarse rollout was measured to be unnecessary
    (``tools/probe_price_stage.py``: matched-#TX gap +0.02 < 0.05), so the price
    stays fine-only.

Pairing is exact: ``evaluate_detection`` keys every physical draw on
``trial_index`` (``simulate.py:182-189``), independent of the passed generator, so
the same ``(q,i,j)`` observation sees identical primitives across arms and the
false-alarm streams are shared. Differences are therefore attributable to the
schedule, not to RNG.

The question is a worst-case question, so the reported statistic is the
**worst-target P_D per deployment seed**, plus its distribution across seeds --
not a single pooled average, which cannot separate a mean shift from a variance
compression.

Usage::

    python tools/run_coordination_experiment.py --mc 5 --seeds 2      # smoke
    python tools/run_coordination_experiment.py --mc 50 --seeds 4     # formal
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isac_sim.core.config import (  # noqa: E402
    apply_overrides,
    apply_preset,
    default_config,
    validate_config,
)
from experiments.coordination import illuminator_mask, select_with_coordination
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
    radar_hardware_gain,
)
from isac_sim.cooperation.reporting import assign_fusion_nodes  # noqa: E402
from experiments.selection import select_c2f_adaptive
from experiments.flow.simulate import evaluate_detection  # noqa: E402
# The 500-800 m / RCS 0.05-0.2 scene band needs a second geometry; reuse the
# sweep's authoritative definition instead of duplicating the block here.
from tools.audit_v1_lowrcs_sweep import SCENARIOS, scenario_overrides  # noqa: E402

AREA = 500.0
RCS = 0.2
PD_REQ = 0.95
METHOD = "proposed_c2f_adaptive_pd"   # the release headline selector / weight mode
ARMS = ("uncoordinated", "mask_only", "sparse")


def make_cfg(penalty: float, looks: int = 16, area: float = AREA,
             rcs: float = RCS, scenario: str = "locked-500m",
             hardware_gain_db: float | None = None,
             kappa_dc: float | None = None):
    """Build the reference (penalty=0) or coordinated config.

    ``scenario="locked-500m"`` reproduces the original locked scenario exactly
    (the ``target-local-v1`` vertical geometry at the 500 m footprint).  Any
    other value must be a key of the sweep's ``SCENARIOS`` and adds that
    block's geometry overrides, so the 500-800 m band can be measured with the
    same coordination machinery instead of a copied script.

    ``hardware_gain_db`` sets ``radio.radar_net_gain_db`` explicitly.  Leaving
    it ``None`` keeps the release default, which resolves to 0 dB (0 dBi Tx +
    0 dBi Rx - 0 dB loss, see :func:`isac_sim.sensing.model.radar_hardware_gain`).  It
    exists so an "algorithm-only" arm can be paired against a "hardware-only"
    arm at the same geometry/RCS:  raising it is a hardware assumption, not an
    algorithmic improvement, and must never be the thing that closes the gap.

    ``kappa_dc`` overrides ``interference.direct_cancellation_db`` (the preset
    value is 40 dB).  It is the direct-path cancellation depth, i.e. the
    residual self-interference floor once an active cancellation stage is in
    place; like the mask path it attacks the *denominator* of SINR.
    """
    cfg = default_config()
    cfg = apply_preset(cfg, "target-local-v1")
    overrides = {
        "geometry.area_xy": float(area),
        "detect.target_rcs": float(rcs),
        "interference.sense_gate_by_active_tx": True,
        "detect.n_looks": int(looks),
        "selector.tx_penalty": float(penalty),
    }
    if hardware_gain_db is not None:
        overrides["radio.radar_net_gain_db"] = float(hardware_gain_db)
    if kappa_dc is not None:
        overrides["interference.direct_cancellation_db"] = float(kappa_dc)
    if scenario != "locked-500m":
        if scenario not in SCENARIOS:
            raise SystemExit(
                f"--scenario must be 'locked-500m' or one of "
                f"{sorted(SCENARIOS)}, got {scenario!r}")
        overrides.update(scenario_overrides(scenario, area))
    cfg = apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def run_trial(cfg_ref, cfg_sparse, seed: int, trial: int, rounds: int):
    """Return {arm: dict} for one deployment/trial, all arms paired."""
    n_uav, n_tgt = cfg_ref.scale.M, cfg_ref.scale.Q
    geom = generate_geometry(cfg_ref, np.random.default_rng([seed, trial]))
    base = build_base_gains(cfg_ref, geom, np.random.default_rng([seed, trial, 1]))
    t0 = compute_link_tables(cfg_ref, base, active_tx_mask=None)
    plan0 = assign_fusion_nodes(cfg_ref, base, t0, geom=geom)

    # Arms 1-2: the un-coordinated schedule (selection never sees a mask).
    sel_ref, _, _ = select_c2f_adaptive(cfg_ref, base, t0, plan=plan0)
    # Arm 3: coordination closed into the selector itself.
    coord = select_with_coordination(
        cfg_sparse, geom, rounds=rounds, seed=seed, base=base
    )
    sel_sparse = coord.selected

    out = {}
    for arm in ARMS:
        selected = sel_ref if arm in ("uncoordinated", "mask_only") else sel_sparse
        coordinate = arm != "uncoordinated"
        mask = illuminator_mask(selected, n_uav) if coordinate else None
        # Evaluation 口径 must match the release pipeline: ``simulate.py`` feeds the
        # detector the FULL fine DD table (``c2f_tables``, dd_gain=eta_fine), not the
        # coarse ``dd_frac_loss`` table the selector's shortlist stage uses.
        tables = compute_link_tables(
            cfg_ref, base, dd_gain=base.eta_fine, active_tx_mask=mask
        )
        plan = assign_fusion_nodes(cfg_ref, base, tables, geom=geom)
        det = evaluate_detection(
            cfg_ref, tables, selected, np.random.default_rng([seed, trial, 99]),
            METHOD, plan, base, trial_index=trial,
        )
        detected, total_tgt, fa_active, tot_fa_active, fa_over, tot_fa_overall, per_tgt = det
        out[arm] = {
            "per_target": np.asarray(per_tgt, dtype=float),
            "n_tgt": int(total_tgt),
            "fa_active": int(fa_active),
            "tot_fa_active": int(tot_fa_active),
            "fa_overall": int(fa_over),
            "tot_fa_overall": int(tot_fa_overall),
            "n_tx": int(n_uav if mask is None else mask.sum()),
            "n_links": int(sum(len(v) for v in selected.values())),
            "rounds": int(len(coord.history)) if arm == "sparse" else 0,
            "converged": bool(coord.converged) if arm == "sparse" else False,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=50, help="trials per seed")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--penalty", type=float, default=0.2)
    ap.add_argument("--looks", type=int, default=16, help="CPI frames per observation")
    ap.add_argument("--rounds", type=int, default=4,
                    help="round budget for the coordination fixed point (arm 3)")
    ap.add_argument("--area", type=float, default=AREA,
                    help="deployment side length in metres")
    ap.add_argument("--rcs", type=float, default=RCS, help="mean target RCS in m^2")
    ap.add_argument("--scenario", default="locked-500m",
                    help="'locked-500m' (default, original geometry) or a sweep "
                         "scenario name such as 'compact-small-uav'")
    ap.add_argument("--hardware-gain-db", type=float, default=None,
                    help="explicit radio.radar_net_gain_db in dB. Omit to keep the "
                         "release default, which is 0 dB (0 dBi Tx + 0 dBi Rx - "
                         "0 dB loss). Any positive value is a hardware assumption "
                         "and is reported as such; the point of the flag is to let "
                         "an algorithm-only arm be paired against a hardware-only "
                         "arm at identical geometry/RCS.")
    ap.add_argument("--kappa-dc", type=float, default=None,
                    help="interference.direct_cancellation_db in dB. Omit to keep "
                         "the preset value (40 dB).")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "coordination_experiment")
    args = ap.parse_args()

    hw_kw = {"hardware_gain_db": args.hardware_gain_db, "kappa_dc": args.kappa_dc}
    cfg_ref = make_cfg(0.0, args.looks, args.area, args.rcs, args.scenario, **hw_kw)
    cfg_sparse = make_cfg(args.penalty, args.looks, args.area, args.rcs,
                          args.scenario, **hw_kw)
    n_tgt = cfg_ref.scale.Q
    seeds = [20260917, 101, 202, 303][: args.seeds]

    hw_eff = radar_hardware_gain(cfg_ref)
    print("=" * 100)
    print(f"coordination experiment (REAL MC) | {args.area:.0f} m | RCS {args.rcs} m^2 | "
          f"scenario {args.scenario} | method {METHOD} | {args.mc} trials x "
          f"{len(seeds)} seeds | penalty {args.penalty} | looks {args.looks}")
    print(f"  G_hw = {10.0 * np.log10(hw_eff):+.2f} dB "
          f"(radar_net_gain_db={cfg_ref.radio.radar_net_gain_db!r}); "
          f"kappa_dc = {cfg_ref.interference.direct_cancellation_db:.1f} dB; "
          f"gate = {cfg_ref.interference.sense_gate_by_active_tx}")
    print("=" * 100)

    rows = []
    per_seed = {arm: {"pd": [], "worst": [], "n_tx": [], "n_links": [],
                      "fa_a": [0, 0], "fa_o": [0, 0]} for arm in ARMS}
    coord_rounds: list[int] = []
    coord_converged: list[bool] = []

    for seed in seeds:
        agg = {arm: {"det": np.zeros(n_tgt), "n": 0, "n_tx": [], "n_links": [],
                     "fa_a": 0, "tfa_a": 0, "fa_o": 0, "tfa_o": 0} for arm in ARMS}
        for trial in range(args.mc):
            res = run_trial(cfg_ref, cfg_sparse, seed, trial, args.rounds)
            coord_rounds.append(res["sparse"]["rounds"])
            coord_converged.append(res["sparse"]["converged"])
            for arm in ARMS:
                r = res[arm]
                agg[arm]["det"] += r["per_target"]
                agg[arm]["n"] += 1
                agg[arm]["n_tx"].append(r["n_tx"])
                agg[arm]["n_links"].append(r["n_links"])
                agg[arm]["fa_a"] += r["fa_active"]
                agg[arm]["tfa_a"] += r["tot_fa_active"]
                agg[arm]["fa_o"] += r["fa_overall"]
                agg[arm]["tfa_o"] += r["tot_fa_overall"]
        print(f"\nseed {seed}")
        print(f"  {'arm':>15}{'worst P_D':>11}{'mean P_D':>10}{'P_FA(act)':>11}"
              f"{'#TX':>6}{'#links':>8}")
        for arm in ARMS:
            a = agg[arm]
            pd = a["det"] / max(a["n"], 1)
            worst = float(pd.min())
            fa_a = a["fa_a"] / max(a["tfa_a"], 1)
            per_seed[arm]["pd"].append(pd)
            per_seed[arm]["worst"].append(worst)
            per_seed[arm]["n_tx"].append(float(np.mean(a["n_tx"])))
            per_seed[arm]["n_links"].append(float(np.mean(a["n_links"])))
            print(f"  {arm:>15}{worst:>11.4f}{pd.mean():>10.4f}{fa_a:>11.4f}"
                  f"{np.mean(a['n_tx']):>6.1f}{np.mean(a['n_links']):>8.1f}")
            rows.append(dict(seed=seed, arm=arm,
                             area_m=float(args.area), rcs_m2=float(args.rcs),
                             scenario=args.scenario, looks=int(args.looks),
                             penalty=float(args.penalty),
                             g_hw_db=float(10.0 * np.log10(radar_hardware_gain(cfg_ref))),
                             kappa_dc=float(cfg_ref.interference.direct_cancellation_db),
                             worst_pd=worst, mean_pd=float(pd.mean()),
                             pfa_active=fa_a, n_tx=float(np.mean(a["n_tx"])),
                             n_links=float(np.mean(a["n_links"])),
                             per_target=";".join(f"{v:.4f}" for v in pd)))
    print(f"\n  coordination fixed point: rounds {np.mean(coord_rounds):.2f} mean / "
          f"{max(coord_rounds)} max; converged on {sum(coord_converged)}/{len(coord_converged)} trials")
    if not all(coord_converged):
        print("  WARNING: some trials stopped on the round budget or on a cycle -- the "
              "``sparse`` arm is not a fixed point there; re-run with a larger --rounds "
              "or report the non-convergence.")

    print("\n" + "=" * 100)
    print("cross-deployment summary (the paper-relevant statistic is the WORST seed)")
    print("=" * 100)
    print(f"  {'arm':>15}{'worst-P_D mean':>16}{'worst-P_D MIN':>15}{'std':>9}"
          f"{'#TX':>6}{'#links':>8}")
    stats = {}
    for arm in ARMS:
        w = np.asarray(per_seed[arm]["worst"])
        stats[arm] = w
        print(f"  {arm:>15}{w.mean():>16.4f}{w.min():>15.4f}{w.std():>9.4f}"
              f"{np.mean(per_seed[arm]['n_tx']):>6.1f}"
              f"{np.mean(per_seed[arm]['n_links']):>8.1f}")

    base_w, sparse_w = stats["uncoordinated"], stats["sparse"]
    print()
    print(f"  uncoordinated worst-seed {base_w.min():.4f} -> sparse {sparse_w.min():.4f} "
          f"({sparse_w.min() - base_w.min():+.4f})")
    print(f"  mean over seeds          {base_w.mean():.4f} -> {sparse_w.mean():.4f} "
          f"({sparse_w.mean() - base_w.mean():+.4f})")
    print(f"  cross-seed std           {base_w.std():.4f} -> {sparse_w.std():.4f}")
    print(f"  every seed improved?     {bool(np.all(sparse_w > base_w))} "
          f"(per-seed deltas: {np.round(sparse_w - base_w, 4).tolist()})")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"coordination_experiment_L{args.looks}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
