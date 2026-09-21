"""TP-UIC V1.1 experiment: a target-conditioned whitened GLRT on the residual.

V1 asked "is there energy in this protected patch", which on a scenario where ten
scatterers share a handful of delay bins cannot separate the arms -- dropping the
tested target leaves nine other echoes in the same subspace.  V1.1 changes the
detector, not the canceller::

    T_q = r~^H P_{B_q} r~,   B_q = P_-q C_res^{-1/2} T_e A_q

Three experiments, one for each question the revision note poses:

``arms``     Is the retention that TP-UIC buys visible as detection at the
             released work point (600 m, RCS 0.1, ten targets)?  Every arm is
             scored by ``T_q`` on a matched H1/H0 pair, with the CFAR level taken
             from the receiver's *own* stated ``C_res`` -- so the achieved
             ``P_FA`` is simultaneously the residual-covariance calibration error.
``masking``  Is what remains a cancellation problem or an identifiability
             problem?  ``rho`` (energy-weighted escape fraction), ``xi_rel``, the
             masking curve versus the number of co-located targets, and the
             continuous-DD audit that moves the tested template off the grid.
``single``   With the multi-target question removed, does cancellation transfer
             to detection at all?  One echo, one protection basis, no nuisance.
``oracle``   Is the analytic level a CFAR level once nothing else can be blamed?
             ``perfect_channel`` + ``truth`` + no belief error: the H0 residual
             is the other targets' echoes and noise, and nothing else.  This is
             the experiment that separates a mis-stated ``C_res`` from a
             mis-stated signal model, and the one the V1.1 headline lacked --
             it measured ``P_FA = 0.40 .. 0.94`` and could not attribute the gap.

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/run_tp_uic_v11.py --trials 40 --audit-trials 12 --single-trials 40 \
        --receiver-rule median --out results_tp_uic_v11

Nothing here touches the released path: ``cancellation.enable`` gates every
component and ``model.py`` is not imported for anything but geometry and gains.

Two fixes landed on 2026-09-19 and both change the numbers, so results from
before that date are not comparable to results from after it:

* the generated echo now carries its true coefficients on the **centre column
  only** (``isac_sim.receiver.cancellation.build_observation``); before the fix every
  fractional-DD tangent column also carried a unit-modulus scatterer, i.e. the
  generator wrote three echoes per target while the detector modelled one;
* the stage-2 joint support is now the *declared* protection set instead of the
  output of an energy threshold (``--candidate-policy``).  The old gate compared
  a threshold against ``||U_q^H r1||^2 / rank(U_q)``, whose level is set by the
  direct field the protection *deliberately retains*, so it selected every
  protected target on every trial and the threshold it printed was not a false
  alarm rate.  The old behaviour is kept as ``--candidate-policy statistic`` for
  the ablation, and every row records both selections (``gate_targets`` /
  ``supported_targets``).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim.receiver import cancellation as cx  # noqa: E402
from isac_sim.receiver import cancellation_glrt as gl  # noqa: E402
from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.sensing.model import build_base_gains, generate_geometry  # noqa: E402
from isac_sim.scenario.prior import perturbed_geometry  # noqa: E402
from tools.run_tp_uic_v1 import pick_receiver  # noqa: E402

AUDIT_ARMS = ("no_ic", "plain_ls", "tp_uic_full", "perfect_channel")


# --------------------------------------------------------------------------
# Configuration and one trial
# --------------------------------------------------------------------------
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
            "cancellation.tangent_order": int(args.tangent_order),
            "cancellation.interference_tangent_order": int(args.interference_tangent_order),
            "cancellation.max_protected_targets": int(args.max_protected_targets),
        },
    )


def make_trial(cfg: Config, args, index: int, *, truth_belief: bool = False):
    """One geometry, one matched ``(H1, H0)`` pair, one tested (receiver, target).

    Returns ``(obs1, obs0, receiver, weak)``.  ``receiver`` is the node the
    observation was built for and ``weak`` is the *target* index under test --
    two different integers that an earlier version of this driver conflated in
    its output rows, which silently destroyed the provenance of every recorded
    arm number.  They are returned separately so a row can carry both.

    ``truth_belief`` forces the receiver's tracker to be perfect (belief ==
    truth) regardless of ``--no-belief-error``; the oracle experiment needs that
    by definition rather than by flag.
    """
    rng = np.random.default_rng([cfg.run.seed, int(index)])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    belief = (
        geom
        if (truth_belief or args.no_belief_error)
        else perturbed_geometry(
            cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
        )
    )
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    receiver, weak = pick_receiver(cfg, base, args.receiver_rule)
    obs1, obs0 = cx.build_observation_pair(
        cfg, geom, belief, base, receiver, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0, exclude_target=weak, weak_index=weak,
    )
    return obs1, obs0, receiver, weak


def arms_with_gate(cfg: Config, obs, weak: int, policy: str = "protected_only"):
    """Two-pass support gate, identical to ``run_tp_uic_v1.passes``.

    Pass 1 runs with the gate closed so the receiver's own predicted residual
    ``i_res_pred`` is available; the level is ``n0 + i_res_pred / K`` and the
    gate is ``-ln(Pfa_target) * level``.  Pass 2 re-runs with that gate so the
    stage-2 candidate set comes from the receiver's own statistic rather than an
    oracle.  Sharing this helper with V1 is what keeps ``tp_uic_full`` from
    degenerating: with ``threshold = 0`` every target column clears the gate and
    the joint stage collapses onto plain LS.

    ``policy`` is forwarded to :func:`isac_sim.receiver.cancellation.cancellation_arms`
    in both passes.  It must be the same in both: the level is read out of pass
    1's stage-1 residual, and a pass 2 with a different support would be
    calibrating one arm and scoring another.
    """
    first = cx.cancellation_arms(cfg, obs, weak_target=weak, threshold=0.0,
                                 candidate_policy=policy)
    n0 = float(cx._noise_power(cfg))
    level = n0 + first["tp_uic_stage1"].i_res_pred / max(int(obs.y.size), 1)
    gate = -math.log(float(cfg.detect.Pfa_target)) * level
    return cx.cancellation_arms(cfg, obs, weak_target=weak, threshold=gate,
                                candidate_policy=policy), gate


def model_of(cfg, obs, name, arms, cache, label):
    """``residual_model`` memoised within a trial -- the linear probe is the cost.

    ``label`` rather than ``id(obs)`` keys the cache: object ids are recycled
    once an observation is garbage collected, so an id-keyed cache can hand a
    later trial the previous trial's model and quietly produce a plausible wrong
    number.  The cache itself is per trial for the same reason.
    """
    key = (label, name)
    if key not in cache:
        cache[key] = gl.residual_model(
            cfg, obs, name, arms, plans=gl.arm_plans(cfg, obs, arms)
        )
    return cache[key]


# --------------------------------------------------------------------------
# A. The arms, scored by T_q
# --------------------------------------------------------------------------
def experiment_arms(cfg: Config, args, out_dir: str) -> List[dict]:
    rows: List[dict] = []
    for t in range(int(args.trials)):
        obs1, obs0, receiver, weak = make_trial(cfg, args, t)
        arms1, gate1 = arms_with_gate(cfg, obs1, weak, args.candidate_policy)
        arms0, _ = arms_with_gate(cfg, obs0, weak, args.candidate_policy)
        cache: Dict = {}
        for name in gl.ARM_ORDER:
            row = {"trial": t, "arm": name, "receiver": int(receiver),
                   "target": int(weak), "gate": gate1}
            for label, obs, arms in (("h1", obs1, arms1), ("h0", obs0, arms0)):
                model = model_of(cfg, obs, name, arms, cache, label)
                got = gl.target_conditioned_glrt(
                    cfg, obs, arms[name], model, target=weak, p_fa=float(args.p_fa),
                    nuisance_manifold=int(args.manifold), dictionary=args.dictionary,
                )
                row.update({
                    f"t_{label}": got.statistic,
                    f"thr_{label}": got.threshold,
                    f"det_{label}": int(got.detected),
                    f"dof_{label}": got.dof_real,
                })
                if label == "h1":
                    row.update({
                        "kappa_db": arms[name].kappa_db,
                        "eta_survive": arms[name].eta_survive,
                        "eta_survive_q": arms[name].eta_survive_q,
                        "s_q_energy": arms[name].s_q_energy,
                        "rho_weighted": got.rho_weighted,
                        "rho_min": got.rho_min,
                        "xi_rel": got.xi_rel_q,
                        "ncp_unit": got.ncp_unit,
                        "ncp_best": got.ncp_best,
                        "calib_min_ratio": model.cov.min_ratio,
                        # Stage-2 provenance: how many targets the energy gate
                        # selected, and how many the joint fit actually modelled.
                        # Comparing the two is the direct measurement of what the
                        # declared support rule replaced.
                        "gate_targets": len(arms[name].gate_targets),
                        "supported_targets": len(arms[name].supported_targets),
                        "n_candidates": len(arms[name].candidates),
                    })
            rows.append(row)
        if (t + 1) % max(int(args.trials) // 5, 1) == 0 and not args.quiet:
            print("  arms %d/%d" % (t + 1, args.trials), flush=True)
    _write(os.path.join(out_dir, "arms.csv"), rows)
    return rows


# --------------------------------------------------------------------------
# B. Masking: curve, manifold, off-grid
# --------------------------------------------------------------------------
def experiment_masking(cfg: Config, args, out_dir: str) -> Dict[str, List[dict]]:
    curves: List[dict] = []
    audits: List[dict] = []
    for t in range(int(args.audit_trials)):
        obs1, _, _, weak = make_trial(cfg, args, t)
        arms1, _ = arms_with_gate(cfg, obs1, weak, args.candidate_policy)
        cache: Dict = {}
        for name in AUDIT_ARMS:
            model = model_of(cfg, obs1, name, arms1, cache, "h1")
            curve = gl.masking_curve(
                cfg, obs1, model, arms1[name], target=weak,
                p_fa=float(args.p_fa), dictionary=args.dictionary,
            )
            for n, rho_w, rho_min, ncp, dist in zip(
                curve.counts, curve.rho_weighted, curve.rho_min,
                curve.ncp_unit, curve.distance_bins,
            ):
                curves.append({
                    "trial": t, "arm": name, "n_nuisance_targets": n,
                    "rho_weighted": rho_w, "rho_min": rho_min,
                    "ncp_unit": ncp, "distance_bins": dist,
                })
            audit = gl.identifiability_audit(
                cfg, obs1, model, arms1[name], target=weak,
                p_fa=float(args.p_fa), dictionary=args.dictionary,
            )
            row = {
                "trial": t, "arm": name,
                "rho_grid_median": float(np.median(audit.rho_on_grid)),
                "rho_grid_min": audit.rho_min_on_grid,
                "rho_manifold_median": float(np.median(audit.rho_manifold)),
                "rho_manifold_min": audit.rho_min_manifold,
                "rho_manifold2_min": audit.rho_min_manifold2,
                "xi_rel_grid": audit.xi_rel_on_grid,
                "xi_rel_manifold": audit.xi_rel_manifold,
                "n_templates": audit.n_templates,
                "n_nuisance_grid": audit.n_nuisance_on_grid,
                "n_nuisance_manifold": audit.n_nuisance_manifold,
                "ncp_grid": audit.ncp_unit_on_grid,
                "off_grid_min": audit.off_grid_min,
                "off_grid_median": audit.off_grid_median,
                "off_grid_max": audit.off_grid_max,
            }
            for key, rho in audit.off_grid.items():
                row["off_%+.2f_%+.2f" % key] = float(np.median(rho))
            audits.append(row)
        if (t + 1) % max(int(args.audit_trials) // 3, 1) == 0 and not args.quiet:
            print("  masking %d/%d" % (t + 1, args.audit_trials), flush=True)
    _write(os.path.join(out_dir, "masking.csv"), curves)
    _write(os.path.join(out_dir, "identifiability.csv"), audits)
    return {"masking": curves, "identifiability": audits}


# --------------------------------------------------------------------------
# C. Single-target mechanism validation
# --------------------------------------------------------------------------
def experiment_single(cfg: Config, args, out_dir: str) -> List[dict]:
    rows: List[dict] = []
    for t in range(int(args.single_trials)):
        obs1, _, _, weak = make_trial(cfg, args, t)
        single = gl.restrict_to_target(cfg, obs1, weak)
        arms, _ = arms_with_gate(cfg, single, weak, args.candidate_policy)
        cache: Dict = {}
        for name in gl.ARM_ORDER:
            model = model_of(cfg, single, name, arms, cache, "single")
            got = gl.target_conditioned_glrt(
                cfg, single, arms[name], model, target=weak, p_fa=float(args.p_fa),
            )
            rows.append({
                "trial": t, "arm": name,
                "t_h1": got.statistic, "threshold": got.threshold,
                "detected": int(got.detected), "dof": got.dof_real,
                "rho_weighted": got.rho_weighted, "ncp_unit": got.ncp_unit,
                "ncp_best": got.ncp_best,
                "kappa_db": arms[name].kappa_db, "eta_survive": arms[name].eta_survive,
            })
        if (t + 1) % max(int(args.single_trials) // 5, 1) == 0 and not args.quiet:
            print("  single %d/%d" % (t + 1, args.single_trials), flush=True)
    _write(os.path.join(out_dir, "single_target.csv"), rows)
    return rows


# --------------------------------------------------------------------------
# D. Oracle CFAR closure: perfect_channel + truth, H1/H0 pair
# --------------------------------------------------------------------------
ORACLE_ARMS = ("perfect_channel", "no_ic", "tp_uic_stage1")


def experiment_oracle(cfg: Config, args, out_dir: str) -> List[dict]:
    """Is the analytic GLRT level a CFAR level once nothing else can be blamed?

    ``perfect_channel`` removes the true direct field exactly and the ``truth``
    dictionary states every other target's echo exactly, so on the H0
    observation the residual is *exactly* "the other nine echoes plus noise" and
    the nuisance projection removes those nine exactly.  Whatever ``P_FA`` the
    analytic threshold then achieves is a property of the detector and of the
    echo the generator actually wrote -- no estimator error, no belief error, no
    cancellation depth in the way.

    That is the experiment V1.1 could not run.  Its headline measured
    ``P_FA = 0.40 .. 0.94`` at a threshold that should give 0.05 and could not
    separate three explanations: a mis-stated ``C_res``, a belief error, or a
    mis-stated *signal model*.  Isolating the three is what makes the remaining
    gap attributable, and it is cheap: one matched pair per trial and three arms.
    """
    rows: List[dict] = []
    for t in range(int(args.oracle_trials)):
        obs1, obs0, receiver, weak = make_trial(cfg, args, t, truth_belief=True)
        arms1, gate1 = arms_with_gate(cfg, obs1, weak, args.candidate_policy)
        arms0, _ = arms_with_gate(cfg, obs0, weak, args.candidate_policy)
        cache: Dict = {}
        for name in ORACLE_ARMS:
            row = {"trial": t, "arm": name, "receiver": int(receiver),
                   "target": int(weak), "gate": gate1}
            for label, obs, arms in (("h1", obs1, arms1), ("h0", obs0, arms0)):
                model = model_of(cfg, obs, name, arms, cache, label)
                got = gl.target_conditioned_glrt(
                    cfg, obs, arms[name], model, target=weak, p_fa=float(args.p_fa),
                    nuisance_manifold=int(args.manifold), dictionary="truth",
                )
                row.update({
                    f"t_{label}": got.statistic,
                    f"thr_{label}": got.threshold,
                    f"det_{label}": int(got.detected),
                    f"dof_{label}": got.dof_real,
                    f"cov_rank_{label}": model.cov.rank,
                    f"min_ratio_{label}": model.cov.min_ratio,
                })
            rows.append(row)
        if (t + 1) % max(int(args.oracle_trials) // 5, 1) == 0 and not args.quiet:
            print("  oracle %d/%d" % (t + 1, args.oracle_trials), flush=True)
    _write(os.path.join(out_dir, "oracle_cfar.csv"), rows)
    return rows


def report_oracle(rows: List[dict], args) -> None:
    print("\n=== D. oracle CFAR closure: perfect_channel + truth, p_fa = %.3f ===" % args.p_fa)
    print("  no belief error, truth dictionary, direct field removed exactly:")
    print("  the H0 residual is other targets' echoes + noise, nothing else.")
    print()
    print("  %-16s %9s %9s %8s %8s %9s %9s" % (
        "arm", "thr", "T_H0 med", "P_FA", "P_D", "ratio med", "C_res rank"))
    for name in ORACLE_ARMS:
        sub = [r for r in rows if r["arm"] == name]
        if not sub:
            continue
        ratio = []
        for r in sub:
            if float(r.get("t_h0", 0.0)) > 0.0:
                ratio.append(float(r["t_h1"]) / float(r["t_h0"]))
        ratio_med = st.median(ratio) if ratio else float("nan")
        print("  %-16s %9.2f %9.2f %8.3f %8.3f %9.2f %9.0f" % (
            name, _med(sub, "thr_h1"), _med(sub, "t_h0"), _rate(sub, "det_h0"),
            _rate(sub, "det_h1"), ratio_med, _med(sub, "cov_rank_h1")))
    print("\n  A calibrated detector lands on P_FA = %.3f here.  Anything above it," % args.p_fa)
    print("  with belief error and estimation error both absent by construction,")
    print("  is the *generated echo* disagreeing with the nuisance model.")



def _write(path: str, rows: List[dict]) -> None:
    if not rows:
        return
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


def _med(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return st.median(vals) if vals else float("nan")


def _rate(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return (sum(vals) / len(vals)) if vals else float("nan")


def report_arms(rows: List[dict], args) -> None:
    print("\n=== A. arm comparison, T_q on a matched H1/H0 pair (p_fa = %.3f) ===" % args.p_fa)
    print("  receiver rule = %s ; nuisance manifold order = %d ; dictionary = %s"
          % (args.receiver_rule, args.manifold, args.dictionary))
    print("  stage-2 joint support = %s" % args.candidate_policy)
    print()
    print("  %-16s %8s %8s %8s %9s %6s %6s %8s %8s %8s %6s %6s" % (
        "arm", "C_IC", "eta_surv", "eta_q", "T1 med", "P_D", "P_FA", "rho", "xi_rel",
        "ncp", "gate", "supp"))
    for name in gl.ARM_ORDER:
        sub = [r for r in rows if r["arm"] == name]
        if not sub:
            continue
        print("  %-16s %8.2f %8.4f %8.4f %9.2f %6.3f %6.3f %8.4f %8.4f %8.3f %6.1f %6.1f" % (
            name, _med(sub, "kappa_db"), _med(sub, "eta_survive"),
            _med(sub, "eta_survive_q"), _med(sub, "t_h1"),
            _rate(sub, "det_h1"), _rate(sub, "det_h0"), _med(sub, "rho_weighted"),
            _med(sub, "xi_rel"), _med(sub, "ncp_unit"),
            _med(sub, "gate_targets"), _med(sub, "supported_targets")))
    print("\n  eta_surv is the whole echo field; eta_q is the *tested target* alone.")
    print("  Q1 ('does TP-UIC damage the weak target less') is a question about")
    print("  eta_q -- eta_surv also moves when the stage-2 support changes size.")
    print("\n  P_FA is measured at the *analytic* threshold, so it is also the")
    print("  residual-covariance calibration check: a well-stated C_res lands on %.3f." % args.p_fa)


def report_masking(curves: List[dict], audits: List[dict], args) -> None:
    print("\n=== B. masking: rho against the number of co-located targets ===")
    counts = sorted({int(r["n_nuisance_targets"]) for r in curves})
    print("  %-16s %s" % ("arm", "".join("%10s" % ("n=%d" % n) for n in counts)))
    for name in AUDIT_ARMS:
        row = []
        for n in counts:
            sub = [r for r in curves if r["arm"] == name and int(r["n_nuisance_targets"]) == n]
            row.append("%10.2e" % _med(sub, "rho_weighted") if sub else "%10s" % "-")
        print("  %-16s %s" % (name, "".join(row)))
    print("\n=== B2. continuous-DD audit (rho of the tested template) ===")
    print("  %-16s %10s %10s %10s | %s" % (
        "arm", "on-grid", "manifold", "manifold2", "off-grid medians (+dk,+dl)"))
    keys = [k for k in audits[0] if k.startswith("off_")] if audits else []
    for name in AUDIT_ARMS:
        sub = [r for r in audits if r["arm"] == name]
        if not sub:
            continue
        offs = " ".join("%s:%.2e" % (k[4:], _med(sub, k)) for k in keys)
        print("  %-16s %10.2e %10.2e %10.2e | %s" % (
            name, _med(sub, "rho_grid_median"), _med(sub, "rho_manifold_median"),
            _med(sub, "rho_manifold2_min"), offs))
    print("\n  rho = fraction of a template's whitened matched-filter energy that")
    print("  survives projecting out the other targets.  xi_rel is the worst-direction")
    print("  version over the whole block.")


def report_single(rows: List[dict], args) -> None:
    print("\n=== C. single-target mechanism validation (one echo, no nuisance) ===")
    print("  %-16s %8s %8s %9s %6s %9s %9s" % (
        "arm", "C_IC", "eta_surv", "T med", "P_D", "rho", "ncp"))
    for name in gl.ARM_ORDER:
        sub = [r for r in rows if r["arm"] == name]
        if not sub:
            continue
        print("  %-16s %8.2f %8.4f %9.2f %6.3f %9.4f %9.3f" % (
            name, _med(sub, "kappa_db"), _med(sub, "eta_survive"), _med(sub, "t_h1"),
            _rate(sub, "detected"), _med(sub, "rho_weighted"), _med(sub, "ncp_unit")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-cpi", type=int, default=1)
    ap.add_argument("--tangent-order", type=int, default=1)
    ap.add_argument("--interference-tangent-order", type=int, default=0)
    ap.add_argument("--max-protected-targets", type=int, default=3)
    ap.add_argument("--receiver-rule", choices=("worst", "median"), default="median")
    ap.add_argument("--p-fa", type=float, default=0.05)
    ap.add_argument("--manifold", type=int, default=0,
                    help="nuisance tangent order used by the headline experiment "
                         "(0 = on-grid nuisance only)")
    ap.add_argument("--dictionary", choices=("belief", "truth"), default="belief")
    ap.add_argument("--no-belief-error", action="store_true",
                    help="give the receiver a perfect tracker (diagnostic)")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--audit-trials", type=int, default=12)
    ap.add_argument("--single-trials", type=int, default=40)
    ap.add_argument("--oracle-trials", type=int, default=60,
                    help="trials for the D. oracle CFAR closure experiment")
    ap.add_argument("--candidate-policy",
                    choices=("protected_only", "statistic"),
                    default="protected_only",
                    help="stage-2 joint support: 'protected_only' (default) models "
                         "exactly the protection set, 'statistic' keeps the "
                         "V1/V1.1 energy-threshold gate for the ablation")
    ap.add_argument("--skip", default="", help="comma list of arms,masking,single,oracle to skip")
    ap.add_argument("--out", default="results_tp_uic_v11")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    cfg = build_config(args)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    print("TP-UIC V1.1 : %s, area=%.0f m, RCS=%.2f m^2, seed=%d" % (
        args.preset, args.area, args.rcs, args.seed))
    print("  K = %d bins, %d UAVs, %d targets" % (
        cfg.waveform.N * cfg.waveform.L, cfg.scale.M, cfg.scale.Q))
    print("  stage-2 joint support: %s" % args.candidate_policy)
    print("  detector: target-conditioned whitened GLRT, p_fa = %.3f" % args.p_fa)

    if "arms" not in skip:
        rows = experiment_arms(cfg, args, args.out)
        report_arms(rows, args)
        print("  [%.0f s]" % (time.time() - t0))
    if "masking" not in skip:
        got = experiment_masking(cfg, args, args.out)
        report_masking(got["masking"], got["identifiability"], args)
        print("  [%.0f s]" % (time.time() - t0))
    if "single" not in skip:
        rows = experiment_single(cfg, args, args.out)
        report_single(rows, args)
    if "oracle" not in skip:
        rows = experiment_oracle(cfg, args, args.out)
        report_oracle(rows, args)
    print("\nwrote %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
