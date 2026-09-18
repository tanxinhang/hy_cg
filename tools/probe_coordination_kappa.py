"""Is inter-UAV coordination a SUBSTITUTE for direct-path cancellation depth?

Claim under test — ``COORDINATION_INTERFERENCE_ROUTE.md`` §2.C, status row
"「协调与 κ_dc 是替代品，不可叠加计功」 = ⏳ 需按 11.1 口径重测（未跑）":

    Coordination and the direct-path cancellation depth ``kappa_dc`` attack the
    *same* term, so their gains must NOT be added.  Once the spectrum is quiet
    enough, paying for deeper cancellation buys nothing.

Why the claim has a structural form, not just an empirical one
--------------------------------------------------------------
``model.py:651-654`` builds the bistatic sensing denominator as::

    residual_direct = kappa_dc * I_sense_field[j]        # kappa_dc is LINEAR
    I_sense_field   = P_rad_sense @ direct_gain ,  P_rad_sense = P_sense * mask
    kappa_dc        = 10 ** (-direct_cancellation_db / 10)

so the two levers enter the denominator as a **product** -- a coefficient (the
linear factor ``kappa_dc``) times a field (the mask).  With ``gamma = signal /
(n0 + kappa_dc * field(mask) + eps) / (1 + INR)``, coordination shrinks ``field``
and a deeper cancellation shrinks the coefficient; as soon as
``kappa_dc * field << n0`` the second lever has nothing left to remove.  That
predicts **saturation / a negative interaction**, not a fixed additive offset --
which is exactly what "substitutes" means.

Notation, because the code and the sweep units differ: everywhere *below*
``kappa`` is the knob in **dB** (``--kappa 40`` = ``direct_cancellation_db=40``),
never the linear ``kappa_dc`` of ``model.py``.

A side effect worth not mistaking for a bug: ``#links`` responds strongly to
``kappa`` (the selector scores on a sensing SINR that contains the residual, so
better links let it satisfy the same requirement with fewer of them) -- 23.8 links
at 40 dB versus 11.6 at 60 dB for the same arm.  ``#links`` is therefore an
*outcome* of the sweep, not a fixed control; only ``P_FA`` is a control here.

Why every number in the old §2.C table is void
----------------------------------------------
It came from ``tools/probe_coordination_gain.py`` block C, which:

  * runs the **analytic surrogate** ``predicted_pd_for_links`` on a single
    FROZEN view -- documented as ordering-only (~2x off, >24 % low in low SNR);
  * defines "coordinated" as ``mask={i,j}`` for the worst target's own view,
    i.e. an isolated single-target scenario, **not** the release multi-target
    schedule (where the illuminator set is the union over all targets);
  * predates the A7 echo gate and the ``{i}``-mask convention.

Only the *direction* survived that table.  This probe re-measures it.

What this probe does
--------------------
It reuses the FORMAL experiment's paired machinery **verbatim**
(``tools/run_coordination_experiment.py``: ``make_cfg`` / ``run_trial`` /
``ARMS``), so a ``kappa=40`` row here is directly comparable cell-by-cell with
the 800-trial frontier
(``results/coordination_experiment_precision/coordination_experiment_L16.csv``
-- ``direct_cancellation_db`` defaults to 40.0 and no preset overrides it).
Exactly one thing is changed: ``interference.direct_cancellation_db``.

Therefore the default flags are the frontier's flags
(``--mc 200 --seeds 4 --penalty 0.1 --rounds 12``) and the default working
point is the frontier's working point (500 m / RCS 0.2 / L=16).  Anything else
is not comparable.

Pre-registered decision rule (fixed BEFORE the sweep, so it cannot be
reverse-engineered from the numbers)
--------------------------------------------------------------------
Let ``g(k)`` = (worst P_D of arm ``sparse``) - (worst P_D of arm
``uncoordinated``) at cancellation depth ``k``, and let ``r(k)`` = median sensing
``rinr`` of the *uncoordinated* arm in dB.

**Calibre.**  "Worst P_D" here means the **min over the 4 deployment seeds**, which
is the statistic the published frontier tables quote and the one
``run_coordination_experiment.py:237-238`` prints as its headline gain.  That
runner's *table column* labelled "worst-seed" is a different number: the **mean
over seeds**.  The two are not interchangeable (L=16 / uncoordinated: 0.7800 vs
0.7988).  This probe reports the min as primary -- so its kappa=40 row equals the
frontier's L=16 row -- and carries the mean alongside only as a sensitivity read,
because the collapse below is a difference of differences and must not depend on
the choice.

  * **SUBSTITUTES**  <=>  ``g(min k) - g(max k) >= 0.05``
                          **and** ``r(k) < 0 dB`` for some k above the lowest
                          (the uncoordinated arm leaves the interference-limited
                          regime, so coordination has nothing left to remove).
  * **NOT substitutes / additive**  <=>  that collapse rule does not fire while
                          ``r(k) > 0 dB`` throughout -- the two levers would then
                          be attacking different terms.
  * **CEILING guard**: a cell whose worst-seed is already ``>= 0.98`` is flagged
    ``[[CEILING]]``, because a saturated cell cannot show a gain and would fake
    sub-additivity.  The verdict spans every swept kappa and is always printed
    next to the flags, so a verdict that hinges on a ceiling cell is visible as
    such.

Four ``kappa`` values are swept because the old table's *shape* (a knee between
40 and 60 dB) is itself the thing being re-checked, not a single number.

Statistics reported: worst-target P_D per deployment seed (the frontier's
statistic, collapsed as min and as mean), plus its spread across seeds; median
``rinr`` as the ceiling-free mechanism readout; ``#TX`` / ``#links`` / ``P_FA`` as
controls.  Cross-``kappa`` comparison needs the same calibre as cross-L
comparison, since either collapsed statistic is taken over noisy estimates and the
downward bias of the min varies by cell.

Artifacts, one kappa at a time
------------------------------
``<out>/coordination_experiment_kappa<K>.csv`` -- provenance: per seed, per arm.
``<out>/meta_kappa<K>.json`` -- flags, convergence count, round statistics.

They are per-kappa **precisely so that several kappas can run in parallel**
without overwriting each other (an earlier design wrote one shared ``SUMMARY.md``
and lost three of four reports to the last writer).  The combined
``<out>/SUMMARY.md`` and the verdict are produced by::

    python tools/probe_coordination_kappa.py --report-only

which reads the CSVs back, so it can be re-run any time without re-doing MC.

Usage::

    python tools/probe_coordination_kappa.py --mc 5 --seeds 2      # smoke
    python tools/probe_coordination_kappa.py --mc 200 --seeds 4    # formal (all kappa)
    python tools/probe_coordination_kappa.py --kappa 50 --mc 200 --seeds 4 --no-summary
    python tools/probe_coordination_kappa.py --report-only         # combine artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from isac_sim.config import apply_overrides, validate_config  # noqa: E402
from isac_sim.coordination import illuminator_mask, select_with_coordination  # noqa: E402
from isac_sim.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from isac_sim.reporting import assign_fusion_nodes  # noqa: E402
from isac_sim.selection import select_c2f_adaptive  # noqa: E402

# The formal experiment's own machinery -- reused, not re-implemented.  Only the
# bookkeeping below is local; the physics, the arms and the pairing are theirs.
import run_coordination_experiment as rce  # noqa: E402

DEFAULT_KAPPAS = (40.0, 50.0, 60.0, 80.0)
CEILING = 0.98
SUBSTITUTION_MARGIN = 0.05
# The published detection threshold used throughout the coordination docs
# (RUN_LOG 5.4 / ROUTE 13.4).  Only used for the human-readable pass column --
# the verdict itself never touches it, so a wrong choice here cannot flip it.
PASS_TARGET = 0.95


def make_cfg(penalty: float, looks: int, kappa_db: float):
    """The formal runner's config, plus one override.

    ``rce.make_cfg`` pins the working point (500 m / RCS 0.2 / gate on / looks /
    penalty); injecting kappa on top keeps this probe's ``kappa=40`` row
    bit-identical to the frontier's own run, which is what makes the two
    comparable.  ``apply_overrides`` deep-copies, so nothing is shared.
    """
    cfg = rce.make_cfg(penalty, looks)
    cfg = apply_overrides(cfg, {"interference.direct_cancellation_db": float(kappa_db)})
    validate_config(cfg)
    return cfg


def sweep_one(kappa_db: float, seeds, mc: int, rounds: int, looks: int,
              penalty: float, verbose: bool = True):
    """One kappa across all seeds/trials.  Mirrors the runner's bookkeeping."""
    cfg_ref = make_cfg(0.0, looks, kappa_db)
    cfg_sparse = make_cfg(penalty, looks, kappa_db)
    n_tgt = cfg_ref.scale.Q
    arms = rce.ARMS

    per_seed = {a: {"worst": [], "mean": [], "n_tx": [], "n_links": [], "pfa": []}
                for a in arms}
    conv: list[bool] = []
    rounds_used: list[int] = []

    for seed in seeds:
        agg = {a: {"det": np.zeros(n_tgt), "n": 0, "n_tx": [], "n_links": [],
                   "fa_a": 0, "tfa_a": 0} for a in arms}
        for trial in range(mc):
            res = rce.run_trial(cfg_ref, cfg_sparse, seed, trial, rounds)
            conv.append(res["sparse"]["converged"])
            rounds_used.append(res["sparse"]["rounds"])
            for arm in arms:
                r = res[arm]
                agg[arm]["det"] += r["per_target"]
                agg[arm]["n"] += 1
                agg[arm]["n_tx"].append(r["n_tx"])
                agg[arm]["n_links"].append(r["n_links"])
                agg[arm]["fa_a"] += r["fa_active"]
                agg[arm]["tfa_a"] += r["tot_fa_active"]
        for arm in arms:
            a = agg[arm]
            pd = a["det"] / max(a["n"], 1)
            per_seed[arm]["worst"].append(float(pd.min()))
            per_seed[arm]["mean"].append(float(pd.mean()))
            per_seed[arm]["n_tx"].append(float(np.mean(a["n_tx"])))
            per_seed[arm]["n_links"].append(float(np.mean(a["n_links"])))
            per_seed[arm]["pfa"].append(a["fa_a"] / max(a["tfa_a"], 1))

    meta = {
        "kappa_db": float(kappa_db),
        "area_m": float(rce.AREA),
        "rcs_m2": float(rce.RCS),
        "looks": int(looks),
        "penalty": float(penalty),
        "round_budget": int(rounds),
        "mc_per_seed": int(mc),
        "seeds": [int(s) for s in seeds],
        "arms": list(arms),
        "n_trials": int(len(conv)),
        "n_converged": int(sum(conv)),
        "rounds_mean": float(np.mean(rounds_used)),
        "rounds_max": int(max(rounds_used)),
    }
    if verbose:
        print(f"  fixed point: converged {meta['n_converged']}/{meta['n_trials']}"
              f" ({meta['rounds_mean']:.2f} rounds mean, {meta['rounds_max']} max)")
        if meta["n_converged"] < meta["n_trials"]:
            print("  WARNING: not all trials reached a fixed point -- the `sparse` arm is "
                  "not a fixed point there; report the non-convergence, do not hide it.")

    summary = {}
    for arm in arms:
        w = np.asarray(per_seed[arm]["worst"])
        summary[arm] = {
            # The published frontier tables and the formal runner's headline
            # gain line (run_coordination_experiment.py:237-238) both quote the
            # MIN over seeds, so that is the primary statistic here.  The
            # runner's own table column prints the MEAN; keep it too, labelled,
            # so the two calibres can never be silently mixed.
            "worst_min": float(w.min()),        # PRIMARY -- the frontier's number
            "worst_mean": float(w.mean()),      # the runner's table column
            "std": float(w.std()),
            "mean_pd": float(np.mean(per_seed[arm]["mean"])),
            "n_tx": float(np.mean(per_seed[arm]["n_tx"])),
            "n_links": float(np.mean(per_seed[arm]["n_links"])),
            "pfa_lo": float(np.min(per_seed[arm]["pfa"])),
            "pfa_hi": float(np.max(per_seed[arm]["pfa"])),
            # kept per seed, not collapsed: the CSV is evidence, and a per-seed
            # row carrying the global mean would be a quietly wrong record.
            "per_seed_worst": list(per_seed[arm]["worst"]),
            "per_seed_mean": list(per_seed[arm]["mean"]),
            "per_seed_n_tx": list(per_seed[arm]["n_tx"]),
            "per_seed_n_links": list(per_seed[arm]["n_links"]),
            "per_seed_pfa": list(per_seed[arm]["pfa"]),
        }
    return summary, meta


def rinr_mechanism(kappa_db: float, seeds, trial: int, rounds: int, looks: int,
                   penalty: float):
    """Median sensing ``rinr`` per arm, collapsed over the swept deployment seeds.

    Ceiling-free on purpose: unlike P_D, ``rinr`` has no upper bound, so it still
    separates the two arms after both have saturated at P_D ~ 1.  Values are
    filtered to ``> 0`` because ``model.py`` leaves the diagonal at zero.

    The pre-registered verdict's second condition reads this number, so it is
    taken over the same seeds the sweep uses rather than off one deployment: a
    single deployment was first checked against the full seed set and agreed to
    ~0.35 dB, but a readout a verdict depends on should not rest on one sample.
    Returns ``(median over seeds, per-seed detail)``.
    """
    detail = {arm: [] for arm in rce.ARMS}
    for seed in seeds:
        cfg_ref = make_cfg(0.0, looks, kappa_db)
        cfg_sparse = make_cfg(penalty, looks, kappa_db)
        geom = generate_geometry(cfg_ref, np.random.default_rng([seed, trial]))
        base = build_base_gains(cfg_ref, geom, np.random.default_rng([seed, trial, 1]))
        t0 = compute_link_tables(cfg_ref, base, active_tx_mask=None)
        plan0 = assign_fusion_nodes(cfg_ref, base, t0, geom=geom)
        sel_ref, _, _ = select_c2f_adaptive(cfg_ref, base, t0, plan=plan0)
        coord = select_with_coordination(cfg_sparse, geom, rounds=rounds, seed=seed,
                                         base=base)
        for arm in rce.ARMS:
            selected = sel_ref if arm in ("uncoordinated", "mask_only") else coord.selected
            mask = None if arm == "uncoordinated" else illuminator_mask(
                selected, cfg_ref.scale.M)
            tabs = compute_link_tables(cfg_ref, base, dd_gain=base.eta_fine,
                                       active_tx_mask=mask)
            r = np.asarray(tabs.rinr, dtype=float)
            r = r[np.isfinite(r) & (r > 0.0)]
            detail[arm].append(10.0 * np.log10(float(np.median(r))) if r.size
                               else float("nan"))
    med = {arm: float(np.median(detail[arm])) for arm in rce.ARMS}
    return med, detail


def cell_paths(out: Path, kappa: float) -> tuple[Path, Path]:
    tag = f"{int(round(kappa))}"
    return (out / f"coordination_experiment_kappa{tag}.csv",
            out / f"meta_kappa{tag}.json")


def write_cell(out: Path, kappa: float, summary, meta, seeds) -> Path:
    path, meta_path = cell_paths(out, kappa)
    out.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kappa_db", "arm", "seed", "worst_pd", "mean_pd",
                    "n_tx", "n_links", "pfa_active"])
        for a in rce.ARMS:
            s = summary[a]
            for seed, wr, mn, ntx, nlk, pfa in zip(
                    seeds, s["per_seed_worst"], s["per_seed_mean"],
                    s["per_seed_n_tx"], s["per_seed_n_links"], s["per_seed_pfa"]):
                w.writerow([f"{kappa:.1f}", a, seed, f"{wr:.6f}", f"{mn:.6f}",
                            f"{ntx:.4f}", f"{nlk:.4f}", f"{pfa:.6f}"])
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def load_cell(out: Path, kappa: float):
    """Rebuild the summary dict from the CSV -- so reporting needs no MC."""
    path, _ = cell_paths(out, kappa)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"empty artifact: {path}")
    summary = {}
    for arm in rce.ARMS:
        rr = [r for r in rows if r["arm"] == arm]      # file order == sweep order
        if not rr:
            raise SystemExit(f"{path} has no rows for arm {arm!r}")
        worst = [float(r["worst_pd"]) for r in rr]
        pfa = [float(r["pfa_active"]) for r in rr]
        w = np.asarray(worst)
        summary[arm] = {
            "worst_min": float(w.min()),        # PRIMARY -- see run_kappas()
            "worst_mean": float(w.mean()),      # the runner's table column
            "std": float(w.std()),
            "mean_pd": float(np.mean([float(r["mean_pd"]) for r in rr])),
            "n_tx": float(np.mean([float(r["n_tx"]) for r in rr])),
            "n_links": float(np.mean([float(r["n_links"]) for r in rr])),
            "pfa_lo": float(np.min(pfa)),
            "pfa_hi": float(np.max(pfa)),
            "per_seed_worst": worst,
            "per_seed_mean": [float(r["mean_pd"]) for r in rr],
            "per_seed_n_tx": [float(r["n_tx"]) for r in rr],
            "per_seed_n_links": [float(r["n_links"]) for r in rr],
            "per_seed_pfa": pfa,
        }
    return summary


def print_grid(kappas, results) -> None:
    for kappa in kappas:
        print(f"\nkappa_dc = {kappa:.1f} dB")
        print(f"  {'arm':>15}{'worst-MIN':>12}{'mean-worst':>12}{'std':>9}"
              f"{'mean P_D':>10}{'#TX':>6}{'#links':>8}{'P_FA':>14}")
        for arm in rce.ARMS:
            s = results[kappa][arm]
            flag = "  [[CEILING]]" if s["worst_min"] >= CEILING else ""
            print(f"  {arm:>15}{s['worst_min']:>12.4f}{s['worst_mean']:>12.4f}"
                  f"{s['std']:>9.4f}{s['mean_pd']:>10.4f}{s['n_tx']:>6.1f}"
                  f"{s['n_links']:>8.1f}"
                  f"{s['pfa_lo']:>7.4f}-{s['pfa_hi']:<7.4f}{flag}")


def analyse(kappas, results, rinr):
    """Everything the pre-registered rule needs, as plain numbers.

    The rule is evaluated on ``worst_min`` -- the same statistic the published
    frontier quotes (``run_coordination_experiment.py:237-238``) and the same one
    the L=16 row here is cross-checked against.  ``g_mean`` reproduces the
    runner's *table column* calibre as a sensitivity read: the collapse is a
    difference of differences, so it must not depend on which of the two is used.
    """
    lo, hi = min(kappas), max(kappas)
    g = {k: results[k]["sparse"]["worst_min"] - results[k]["uncoordinated"]["worst_min"]
         for k in kappas}
    g_mean = {k: (results[k]["sparse"]["worst_mean"]
                  - results[k]["uncoordinated"]["worst_mean"]) for k in kappas}
    d_un = (results[hi]["uncoordinated"]["worst_min"]
            - results[lo]["uncoordinated"]["worst_min"])
    d_co = results[hi]["sparse"]["worst_min"] - results[lo]["sparse"]["worst_min"]
    dead = [k for k in kappas
            if k > lo and rinr.get(k, {}).get("uncoordinated", 1.0) < 0.0]
    ceiling = [(k, a) for k in kappas for a in rce.ARMS
               if results[k][a]["worst_min"] >= CEILING]
    collapse = g[lo] - g[hi]
    collapse_mean = g_mean[lo] - g_mean[hi]
    verdict = (collapse >= SUBSTITUTION_MARGIN) and bool(dead)
    return dict(lo=lo, hi=hi, g=g, g_mean=g_mean, d_un=d_un, d_co=d_co,
                interaction=d_un - d_co, dead=dead, ceiling=ceiling,
                collapse=collapse, collapse_mean=collapse_mean, verdict=verdict)


def print_analysis(a, kappas, results, meta) -> None:
    print("\n" + "=" * 100)
    print("interaction / substitution test (pre-registered rule, see module docstring)")
    print("=" * 100)
    print("  coordination gain vs kappa   (sparse - uncoordinated, worst-P_D MIN):")
    for k in kappas:
        cap = ("   [[CEILING cell -- gain capped by the 1.0 wall]]"
               if results[k]["uncoordinated"]["worst_min"] >= CEILING else "")
        print(f"    kappa {k:>5.1f} dB :  {a['g'][k]:+.4f}"
              f"   (mean-calibre {a['g_mean'][k]:+.4f}){cap}")
    print(f"  kappa {a['lo']:.0f}->{a['hi']:.0f} dB buys, uncoordinated : {a['d_un']:+.4f}")
    print(f"  kappa {a['lo']:.0f}->{a['hi']:.0f} dB buys, sparse       : {a['d_co']:+.4f}")
    print(f"  interaction (uncoordinated gain - sparse gain) : {a['interaction']:+.4f}"
          f"   [>0 => sub-additive = substitutes]")
    print(f"  coordination gain collapse  g({a['lo']:.0f}) - g({a['hi']:.0f}) : "
          f"{a['collapse']:+.4f}   (threshold {SUBSTITUTION_MARGIN:.2f}; "
          f"mean-calibre {a['collapse_mean']:+.4f})")
    print(f"  uncoordinated driven to noise-limited at kappa : "
          f"{a['dead'] if a['dead'] else 'nowhere in the sweep'}")
    if a["ceiling"]:
        print("  [[CEILING]] cells (worst-seed >= "
              f"{CEILING}): " + ", ".join(f"kappa={k:.0f}/{m}" for k, m in a["ceiling"]))
    print()
    if a["verdict"]:
        print("  VERDICT: **SUBSTITUTES** -- coordination's gain collapses as the")
        print("           cancellation depth rises, and deeper cancellation alone already")
        print("           leaves the interference-limited regime. Do NOT add their gains.")
        if a["ceiling"]:
            print("           CAVEAT: at least one cell is against the 1.0 wall; the")
            print("           collapse partly reflects missing headroom, not only physics.")
    else:
        print("  VERDICT: **NOT substitutes** as pre-registered -- the collapse rule and")
        print("           the noise-limited crossing did not both fire. Read the tables")
        print("           above; do not quote the old §2.C claim without re-deriving it.")
    if meta:
        bad = [k for k, m in meta.items() if m["n_converged"] < m["n_trials"]]
        if bad:
            print(f"  NOTE: kappa {bad} did not fully converge -- treat their `sparse`")
            print("        value as a worst case only.")


def write_summary(out: Path, kappas, results, rinr, a, meta) -> Path:
    path = out / "SUMMARY.md"
    lines = ["# coordination x direct-path cancellation depth (kappa_dc)", ""]
    if meta and kappas:
        m0 = meta[kappas[0]]
        lines.append(
            f"`tools/probe_coordination_kappa.py --mc {m0['mc_per_seed']} --seeds "
            f"{len(m0['seeds'])} --penalty {m0['penalty']} --looks {m0['looks']} "
            f"--rounds {m0['round_budget']}`")
        lines.append("")
        lines.append(f"working point: {m0['area_m']:.0f} m / RCS {m0['rcs_m2']} m^2 / "
                     f"L={m0['looks']} | {m0['n_trials']} trials per arm per cell")
    lines += ["", "| kappa_dc (dB) | arm | worst-P_D MIN | mean-of-worst | std | mean P_D "
                  "| #TX | #links | P_FA |", "|---|---|---|---|---|---|---|---|---|"]
    for kappa in kappas:
        for arm in rce.ARMS:
            s = results[kappa][arm]
            cel = " **CEILING**" if s["worst_min"] >= CEILING else ""
            lines.append(
                f"| {kappa:.0f} | {arm} | {s['worst_min']:.4f}{cel} "
                f"| {s['worst_mean']:.4f} "
                f"| {s['std']:.4f} | {s['mean_pd']:.4f} | {s['n_tx']:.1f} | "
                f"{s['n_links']:.1f} | {s['pfa_lo']:.4f}-{s['pfa_hi']:.4f} |")
    if rinr:
        lines += ["", "## median sensing rinr, uncoordinated arm (dB; >0 = interference "
                      "limited)", "", "| kappa_dc | uncoordinated | mask_only | sparse "
                      "|", "|---|---|---|---|"]
        for kappa in kappas:
            m = rinr[kappa]
            lines.append(f"| {kappa:.0f} | {m['uncoordinated']:+.2f} | "
                         f"{m['mask_only']:+.2f} | {m['sparse']:+.2f} |")
    lines += ["", "## coordination gain vs kappa (sparse - uncoordinated)", "",
              "Both calibres are listed because the runner prints them in different places "
              "(see ROUTE 13.6 / ROOT_CAUSE 3.6); the verdict uses the MIN calibre only.",
              "",
              "| kappa_dc | gain (worst-P_D **MIN**) | gain (mean-of-worst) "
              "| uncoordinated pass |", "|---|---|---|---|"]
    for kappa in kappas:
        per_seed = results[kappa]["uncoordinated"]["per_seed_worst"]
        n_pass = sum(1 for x in per_seed if x >= PASS_TARGET)
        lines.append(f"| {kappa:.0f} | **{a['g'][kappa]:+.4f}** "
                     f"| {a['g_mean'][kappa]:+.4f} | {n_pass}/{len(per_seed)} |")
    lines += ["", "## pre-registered verdict", "",
              f"- coordination gain (sparse - uncoordinated) at kappa={a['lo']:.0f}: "
              f"**{a['g'][a['lo']]:+.4f}**; at kappa={a['hi']:.0f}: "
              f"**{a['g'][a['hi']]:+.4f}**",
              f"- kappa {a['lo']:.0f}->{a['hi']:.0f} buys: uncoordinated "
              f"{a['d_un']:+.4f}, sparse {a['d_co']:+.4f}",
              f"- interaction (sub-additivity) = **{a['interaction']:+.4f}**",
              f"- uncoordinated noise-limited from kappa: "
              f"{a['dead'] if a['dead'] else 'nowhere in sweep'}",
              f"- **VERDICT: {'SUBSTITUTES' if a['verdict'] else 'NOT substitutes (per rule)'}**"]
    if a["ceiling"]:
        lines.append("- CEILING cells: "
                     + ", ".join(f"kappa={k:.0f}/{m}" for k, m in a["ceiling"]))
    if meta:
        lines.append("- convergence: " + ", ".join(
            f"kappa={k:.0f}:{m['n_converged']}/{m['n_trials']}" for k, m in meta.items()))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, nargs="+", default=list(DEFAULT_KAPPAS),
                    help="direct_cancellation_db values to sweep")
    ap.add_argument("--mc", type=int, default=200, help="trials per seed")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--penalty", type=float, default=0.1)
    ap.add_argument("--looks", type=int, default=16, help="CPI frames per observation")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "coordination_experiment_kappa")
    ap.add_argument("--no-mechanism", action="store_true",
                    help="skip the (extra) rinr mechanism block")
    ap.add_argument("--no-summary", action="store_true",
                    help="write only this kappa's CSV/meta (safe for parallel runs)")
    ap.add_argument("--report-only", action="store_true",
                    help="skip the sweep; rebuild SUMMARY.md + verdict from the artifacts")
    args = ap.parse_args()

    kappas = sorted({float(k) for k in args.kappa})
    seeds = [20260917, 101, 202, 303][: args.seeds]

    if args.report_only:
        results = {k: load_cell(args.out, k) for k in kappas}
        meta = {}
        for k in kappas:
            _, mp = cell_paths(args.out, k)
            meta[k] = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
        meta = {k: v for k, v in meta.items() if v}
    else:
        print("=" * 100)
        print(f"coordination x direct-cancellation sweep (REAL MC) | {rce.AREA:.0f} m | "
              f"RCS {rce.RCS} m^2 | L={args.looks} | penalty {args.penalty}")
        print(f"kappa_dc in {kappas} dB | {args.mc} trials x {len(seeds)} seeds "
              f"= {args.mc * len(seeds)} trials/arm/cell | arms {rce.ARMS}")
        print("claimed: coordination and kappa_dc are SUBSTITUTES (gain not additive)")
        print("=" * 100)
        results, meta = {}, {}
        for kappa in kappas:
            print(f"\nkappa_dc = {kappa:.1f} dB")
            summary, m = sweep_one(kappa, seeds, args.mc, args.rounds, args.looks,
                                   args.penalty)
            results[kappa], meta[kappa] = summary, m
            print(f"  {'arm':>15}{'worst-MIN':>12}{'mean-worst':>12}{'std':>9}"
                  f"{'mean P_D':>10}{'#TX':>6}{'#links':>8}{'P_FA':>14}")
            for arm in rce.ARMS:
                s = summary[arm]
                flag = "  [[CEILING]]" if s["worst_min"] >= CEILING else ""
                print(f"  {arm:>15}{s['worst_min']:>12.4f}{s['worst_mean']:>12.4f}"
                      f"{s['std']:>9.4f}{s['mean_pd']:>10.4f}{s['n_tx']:>6.1f}"
                      f"{s['n_links']:>8.1f}"
                      f"{s['pfa_lo']:>7.4f}-{s['pfa_hi']:<7.4f}{flag}")
            print(f"wrote {write_cell(args.out, kappa, summary, m, seeds)}")

    # ---- mechanism readout (ceiling-free) ---------------------------------
    # Recomputed here rather than read from the CSVs: it costs no MC, and the
    # report path then always agrees with the current implementation -- the
    # earlier single-deployment version would otherwise be silently inherited
    # by every SUMMARY written from old artifacts.
    rinr = {}
    if not args.no_mechanism:
        looks = meta[kappas[0]]["looks"] if meta else args.looks
        penalty = meta[kappas[0]]["penalty"] if meta else args.penalty
        rounds = meta[kappas[0]]["round_budget"] if meta else args.rounds
        mech_seeds = (meta[kappas[0]]["seeds"] if meta else seeds)
        print("\n" + "=" * 100)
        print("mechanism: median sensing rinr (dB). >0 dB = interference limited")
        print(f"(median over seeds {mech_seeds}; a regime readout, not a P_D statistic)")
        print("=" * 100)
        print(f"  {'kappa':>8}{'uncoordinated':>16}{'mask_only':>12}{'sparse':>10}"
              f"   per-seed uncoordinated")
        for kappa in kappas:
            m, det = rinr_mechanism(kappa, mech_seeds, 0, rounds, looks, penalty)
            rinr[kappa] = m
            regime = "INTERF" if m["uncoordinated"] > 0 else "NOISE "
            per_seed = " ".join(f"{x:+.1f}" for x in det["uncoordinated"])
            print(f"  {kappa:>8.1f}{m['uncoordinated']:>16.2f}{m['mask_only']:>12.2f}"
                  f"{m['sparse']:>10.2f}   [{regime} limited]  {per_seed}")

    if len(kappas) < 2:
        print("\n(one kappa only -- the substitution test needs at least two; "
              "run the rest, then `--report-only`)")
        return 0

    a = analyse(kappas, results, rinr)
    print_analysis(a, kappas, results, meta)
    if not args.no_summary:
        print(f"\nwrote {write_summary(args.out, kappas, results, rinr, a, meta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
