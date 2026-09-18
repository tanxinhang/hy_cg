"""TP-UIC V1: does an executable canceller reproduce the constant ``kappa``?

Motivation
----------
``interference.direct_cancellation_db`` is a constant that multiplies the
aggregated direct field at every sensing receiver.  ``KAPPA_DERIVATION.md``
shows *what value the scenario requires*; this tool asks the complementary
question: *what value does a receiver actually deliver*, and what does it cost.

It runs :mod:`isac_sim.cancellation` -- a sample-level (4096-bin DD-domain)
target-preserving interference canceller -- on the release geometry and reports
four numbers per arm, none of which is "cancellation dB" alone::

    C_IC        interference suppression (the effective kappa)
    eta_survive target retention, ||s - f(s)||^2 / ||s||^2
    P_D_weak    detection probability of the weakest target at P_FA = 0.05
    cal_err     |predicted - measured| residual, the covariance calibration

The joint point of the two quantities is the whole argument: an arm can buy
suppression by eating the target, and only reporting both makes that visible.

Design of the measurement
-------------------------
* One receiver per trial: by default the one with the worst echo-to-direct
  margin.  That is the receiver the method is for; averaging over all 15 hides
  it.  ``--receiver-rule median`` reports the middle margin alongside it.
* **H1 and H0 are built as a matched pair** (``build_observation_pair``): both
  share ``x_direct``, the direct dictionary, the target dictionary and every
  other target's echo; only the tested target's echo is dropped for H0 and only
  the noise is redrawn.  Building them with two independent calls gives the two
  hypotheses different illumination, and every detection number collapses.
* All arms are linear maps on the observation, so every reported residual is a
  component transfer, not a difference of totals (see
  :class:`isac_sim.cancellation.CancellationResult`).  A plain ``||r||^2``
  would credit an estimator for the interference it removed while silently
  charging it for the target it ate.
* The candidate gate of the joint stage is set from the receiver's *own*
  predicted residual (``i_res_pred``), never from the truth.  Two passes per
  observation: the first with the gate closed yields the prediction, the second
  uses it.
* Two thresholds are in play and they are **not** the same thing, so they are
  reported separately:
  - ``gate`` (a column of ``arms_mc.csv``) is the *structural* level
    ``-ln(P_FA) * (n0 + I_res_pred / K)`` derived from the prediction.  It is
    what the joint stage uses to accept a candidate; it is not a calibrated
    CFAR threshold.
  - the reported ``P_D``/``P_FA`` come from the **empirical per-arm H0 pool**
    (``quantile(H0, 1 - P_FA)``), which is what makes the arms comparable at
    the same false-alarm rate.  The achieved ``P_FA`` is printed next to it.

Usage::

    python tools/run_tp_uic_v1.py --trials 80
    python tools/run_tp_uic_v1.py --area 800 --rcs 0.05 --trials 40 --sweeps
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isac_sim import cancellation as cx  # noqa: E402
from isac_sim.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.model import build_base_gains, generate_geometry  # noqa: E402
from isac_sim.prior import perturbed_geometry  # noqa: E402

SEED = 2026

# The eight arms of the method note, in the order they should be read.
ARMS = (
    "no_ic",
    "fixed_kappa",
    "plain_ls",
    "ridge_ls",
    "protected_ls",
    "tp_uic_stage1",
    "tp_uic_full",
    "perfect_channel",
)


# --------------------------------------------------------------------------
# Configuration and scenario selection
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
            "interference.direct_cancellation_db": float(args.kappa_db),
        },
    )


def pick_receiver(cfg: Config, base, rule: str = "worst") -> tuple[int, int]:
    """Pick the (receiver, target) pair to evaluate.

    Margins are read straight off the link tables (no sample-level build), so
    the choice is cheap and identical for every arm.

    ``"worst"`` takes the lowest echo-to-direct margin -- the receiver the
    method is *for*.  ``"median"`` takes the middle one and is reported
    alongside it so the result is not hostage to a single harshest geometry.

    Note on history: an earlier revision of this docstring blamed the "worst"
    rule for ``P_D`` sitting at the CFAR floor.  That was wrong.  The real cause
    was that H0 was built by a second, independent ``build_observation`` call,
    so the two hypotheses had *different direct fields*; see
    :func:`isac_sim.cancellation.build_observation_pair`.  With the pair fixed,
    both rules separate the arms (measured H1/H0: 2.1 dB for the untouched
    arms, 4.6--4.9 dB for the cancelling ones).
    """
    m, q_count = cfg.scale.M, cfg.scale.Q
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    direct_field = sense @ base.direct_gain  # (M,)
    candidates = []
    for j in range(m):
        for q in range(q_count):
            echo = float(np.sum(sense * base.target_gain[:, j, q]))
            if echo <= 0.0:
                continue
            candidates.append((echo / max(float(direct_field[j]), 1e-30), j, q))
    if not candidates:
        return 0, 0
    candidates.sort()
    if rule == "worst":
        _, j, q = candidates[0]
    elif rule == "median":
        _, j, q = candidates[len(candidates) // 2]
    else:
        raise ValueError(f"unknown receiver rule {rule!r}; expected 'worst' or 'median'")
    return int(j), int(q)


# --------------------------------------------------------------------------
# One trial
# --------------------------------------------------------------------------
def evaluate(cfg: Config, geom_true, geom_belief, base, receiver: int, weak: int,
             rng: np.random.Generator) -> dict:
    """Two-pass, H1/H0 evaluation of every arm for one trial."""
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    g_proc = cfg.waveform.N * cfg.waveform.L
    n0 = float(cx._noise_power(cfg))

    common = dict(
        rng=rng, sense_power=sense, radiated_power=sense,
        processing_gain=g_proc, hw_gain=1.0, weak_index=weak,
    )
    obs1, obs0 = cx.build_observation_pair(
        cfg, geom_true, geom_belief, base, receiver,
        exclude_target=weak, **common)

    def passes(obs):
        """Gate closed -> read the predicted residual -> gate open."""
        first = cx.cancellation_arms(cfg, obs, weak_target=weak, threshold=0.0)
        level = n0 + first["tp_uic_stage1"].i_res_pred / max(obs.y.size, 1)
        gate = -math.log(cfg.detect.Pfa_target) * level
        return cx.cancellation_arms(cfg, obs, weak_target=weak, threshold=gate), gate

    arms1, gate = passes(obs1)
    arms0, _ = passes(obs0)
    echo_power = float(np.vdot(obs1.s_target, obs1.s_target).real)
    return {
        "arms1": arms1,
        "arms0": arms0,
        "gate": gate,
        "echo_power": echo_power,
        "direct_field": arms1["no_ic"].i_in,
        "receiver": receiver,
        "weak": weak,
    }


def run(cfg: Config, args, label: str, trials: int) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    t0 = time.time()
    for trial in range(trials):
        rng = np.random.default_rng([cfg.run.seed, trial])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        belief = perturbed_geometry(
            cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
        )
        receiver, weak = pick_receiver(cfg, base, args.receiver_rule)
        res = evaluate(cfg, geom, belief, base, receiver, weak, rng)
        for arm in ARMS:
            a1, a0 = res["arms1"][arm], res["arms0"][arm]
            rows.append({
                "trial": trial, "arm": arm, "receiver": receiver, "weak_target": weak,
                "kappa_db": a1.kappa_db, "kappa_pred_db": a1.kappa_pred_db,
                "cal_err_db": a1.calibration_error_db,
                "eta_protect": a1.eta_protect, "eta_survive": a1.eta_survive,
                "noise_enh_db": a1.noise_enhance_db,
                "protect_dim": a1.protect_dim, "candidates": len(a1.candidates),
                "i_in": a1.i_in, "i_res": a1.i_res,
                "i_res_structural": a1.i_res_structural,
                "i_res_estimate": a1.i_res_estimate,
                "i_res_pred": a1.i_res_pred,
                "i_res_retained": a1.i_res_retained,
                "i_res_pred_estimate": a1.i_res_pred_estimate,
                "stat_h1": a1.matched_stat, "stat_h0": a0.matched_stat,
                "echo_to_direct_db": 10.0 * math.log10(
                    max(res["echo_power"], 1e-30) / max(res["direct_field"], 1e-30)),
                "gate": res["gate"],
            })
        if args.verbose and (trial + 1) % 20 == 0:
            print("   [%s] %d/%d  (%.1fs)" % (label, trial + 1, trials, time.time() - t0),
                  flush=True)
    return rows, summarise(rows, cfg.detect.Pfa_target)


def summarise(rows: list[dict], pfa: float) -> dict[str, dict]:
    """Per-arm medians plus the empirical CFAR/PD pair."""
    out: dict[str, dict] = {}
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        h0 = np.array([r["stat_h0"] for r in sub], dtype=float)
        h1 = np.array([r["stat_h1"] for r in sub], dtype=float)
        thr = float(np.quantile(h0, 1.0 - pfa))
        out[arm] = {
            "arm": arm,
            "n": len(sub),
            "kappa_db": float(np.median([r["kappa_db"] for r in sub])),
            "kappa_pred_db": float(np.median([r["kappa_pred_db"] for r in sub])),
            "cal_err_db": float(np.median([r["cal_err_db"] for r in sub])),
            "eta_protect": float(np.median([r["eta_protect"] for r in sub])),
            "eta_survive": float(np.median([r["eta_survive"] for r in sub])),
            "noise_enh_db": float(np.median([r["noise_enh_db"] for r in sub])),
            "protect_dim": float(np.median([r["protect_dim"] for r in sub])),
            "candidates": float(np.median([r["candidates"] for r in sub])),
            "echo_to_direct_db": float(np.median([r["echo_to_direct_db"] for r in sub])),
            "p_fa": float(np.mean(h0 > thr)),
            "p_d_weak": float(np.mean(h1 > thr)),
        }
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
HEAD = ("%-16s %9s %9s %8s %9s %9s %9s %7s %7s" %
        ("arm", "C_IC[dB]", "pred[dB]", "calerr", "etaProt", "etaSurv", "noiseEnh",
         "P_D", "P_FA"))


def print_report(cfg, args, table: dict[str, dict], extra: str = "") -> None:
    print()
    print("=" * 100)
    print("TP-UIC V1  |  preset=%s  area=%.0f m  RCS=%.3g m^2  kappa(assumed)=%.0f dB"
          % (args.preset, args.area, args.rcs, args.kappa_db))
    print("           |  n_cpi=%d  tangent=%d  interf_tangent=%d  protect<=%d  %s"
          % (args.n_cpi, args.tangent_order, args.interference_tangent_order,
             args.max_protected_targets, extra))
    print("=" * 100)
    print(HEAD)
    print("-" * len(HEAD))
    for arm in ARMS:
        v = table.get(arm)
        if not v:
            continue
        print("%-16s %9.2f %9.2f %8.2f %9.4f %9.4f %9.1f %7.3f %7.3f" % (
            arm, v["kappa_db"], v["kappa_pred_db"], v["cal_err_db"],
            v["eta_protect"], v["eta_survive"], v["noise_enh_db"],
            v["p_d_weak"], v["p_fa"]))
    print("-" * len(HEAD))
    med = table.get("no_ic", {})
    print("echo-to-direct margin at the selected receiver: %.1f dB (median)"
          % med.get("echo_to_direct_db", float("nan")))
    print("C_IC      = 10log10(||x_direct||^2 / (||x_direct - f(x_direct)||^2 + ||f(n)||^2));")
    print("            the two terms are eq. (3)'s retention and estimation parts, orthogonal,")
    print("            so the sum is exact.  f is the arm's own map, never applied to the total.")
    print("etaSurv   = ||s - f(s)||^2 / ||s||^2; below 1 means the canceller ate the echo.")
    print("            Above 1 is legitimate: the joint stage can add coherent echo energy.")
    print("pred/cal  = eq. (3) and |pred - measured|.  eq. (3) describes the CONSERVATIVE")
    print("            stage-1 estimate, so for tp_uic_full the calerr is the depth the joint")
    print("            stage buys beyond what the receiver could have promised itself.")
    print("P_D/P_FA  = CFAR calibrated per arm on its own H0 pool at P_FA=%.2f."
          % cfg.detect.Pfa_target)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
SWEEP_AXES = (
    ("n_cpi", (1, 2, 4, 8, 16, 32, 64)),
    ("tangent_order", (0, 1)),
    ("interference_tangent_order", (0, 1)),
    ("max_protected_targets", (0, 1, 2, 3, 5, 10)),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="small-uav-compact-800m")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--kappa-db", type=float, default=40.0)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--sweep-trials", type=int, default=24)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-cpi", type=int, default=1)
    ap.add_argument("--tangent-order", type=int, default=1)
    ap.add_argument("--interference-tangent-order", type=int, default=0)
    ap.add_argument("--max-protected-targets", type=int, default=3)
    ap.add_argument("--receiver-rule", choices=("worst", "median"), default="worst",
                    help="'worst' = harshest echo-to-direct margin (the method's "
                         "target case); 'median' = the middle margin, reported "
                         "alongside so the result is not hostage to one geometry")
    ap.add_argument("--sweeps", action="store_true",
                    help="also run the reference-budget and protection-budget sweeps")
    ap.add_argument("--out", default="results_tp_uic_v1")
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    cfg = build_config(args)

    print("TP-UIC V1 experiment")
    print("  %.0f m, RCS %.3g m^2, %d UAV / %d targets, gate=%s"
          % (args.area, args.rcs, cfg.scale.M, cfg.scale.Q,
             "on" if cfg.cancellation.enable else "off"))
    print("  receiver rule = %s ; weak target = the lowest-margin echo there"
          % args.receiver_rule)

    t0 = time.time()
    rows, table = run(cfg, args, "main", int(args.trials))
    write_csv(out_dir / "arms_mc.csv", rows)
    print_report(cfg, args, table, extra="main")
    print("\nwrote %s (%d rows, %.0fs)"
          % ((out_dir / "arms_mc.csv").relative_to(ROOT), len(rows), time.time() - t0))

    if args.sweeps:
        sweep_rows: list[dict] = []
        for axis, values in SWEEP_AXES:
            for value in values:
                sub = argparse.Namespace(**vars(args))
                setattr(sub, axis, value)
                sub.receiver_rule = args.receiver_rule
                scfg = build_config(sub)
                _r, st = run(scfg, sub, "%s=%s" % (axis, value), int(args.sweep_trials))
                for arm in ARMS:
                    if arm in st:
                        sweep_rows.append({"axis": axis, "value": value, **st[arm]})
                if args.verbose:
                    v = st.get("tp_uic_stage1", {})
                    print("   %-26s C_IC=%7.2f  etaSurv=%.4f  P_D=%.3f  prot=%3.0f"
                          % ("%s=%s" % (axis, value), v.get("kappa_db", float("nan")),
                             v.get("eta_survive", float("nan")),
                             v.get("p_d_weak", float("nan")),
                             v.get("protect_dim", float("nan"))), flush=True)
        write_csv(out_dir / "sweeps.csv", sweep_rows)
        print("wrote %s (%d rows)"
              % ((out_dir / "sweeps.csv").relative_to(ROOT), len(sweep_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
