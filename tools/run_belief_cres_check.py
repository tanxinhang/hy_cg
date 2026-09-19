"""Does charging the belief error to ``C_res`` calibrate the detector?

The TP-UIC detector builds every echo template at the *believed* target state
while the echo arrives from the true one, so the templates are misplaced and the
echoes leak into the residual.  The measured consequence is a false-alarm rate
well above the level the analytic threshold is derived for (0.125--0.438 in
belief mode against a nominal 0.05), which is why every ``P_D`` in the V1.2
tables carries an "uncalibrated" footnote.

``cancellation.belief_error_in_cres`` adds the first-order covariance of that
misalignment to ``C_res`` (tangent columns scaled by the belief error expressed
in DD bins).  This script measures what that buys and what it costs, on the same
trials, with everything else held fixed:

    P_FA   should fall toward 0.05   (the point of the term)
    P_D    may fall slightly         (C_res is larger, so the test is weaker)

Both are reported rather than one being optimised: a term that buys calibration
by destroying detection is not a calibration, it is a different detector.

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/run_belief_cres_check.py --trials 60 --out results_belief_cres
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from isac_sim import cancellation as cx
from isac_sim import cancellation_glrt as gl
from isac_sim.config import Config, apply_preset
from isac_sim.model import build_base_gains, generate_geometry
from isac_sim.prior import perturbed_geometry

ARMS = ("perfect_channel", "tp_uic_stage1", "tp_uic_full")


def make_trial(cfg: Config, index: int, receiver_rule: str = "median"):
    rng = np.random.default_rng([cfg.run.seed, int(index)])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    belief = perturbed_geometry(
        cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
    )
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    weak = pick_receiver(base)
    obs1, obs0 = cx.build_observation_pair(
        cfg, geom, belief, base, weak[0], rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0, exclude_target=weak[1], weak_index=weak[1],
    )
    return obs1, obs0, weak[1]


def pick_receiver(base):
    """Receiver with the median direct field, weakest target at that receiver.

    Mirrors the ``median`` rule the other drivers use: the median total direct
    power over receivers, then the target whose echo there is weakest -- the one
    the method exists for.
    """
    m = base.direct_gain.shape[0]
    order = sorted(range(m), key=lambda j: float(np.sum(base.direct_gain[:, j])))
    j = order[m // 2]
    echo = np.asarray([float(np.sum(base.target_gain[:, j, q]))
                       for q in range(base.target_gain.shape[2])])
    return j, int(np.argmin(echo))


def arms_with_gate(cfg: Config, obs, weak: int):
    first = cx.cancellation_arms(cfg, obs, weak_target=weak, threshold=0.0,
                                 candidate_policy="protected_only")
    n0 = float(cx._noise_power(cfg))
    level = n0 + first["tp_uic_stage1"].i_res_pred / max(int(obs.y.size), 1)
    gate = -math.log(float(cfg.detect.Pfa_target)) * level
    return cx.cancellation_arms(cfg, obs, weak_target=weak, threshold=gate,
                                candidate_policy="protected_only")


def evaluate(cfg: Config, obs, arms, weak: int, belief_term: bool) -> Dict[str, float]:
    cfg.cancellation.belief_error_in_cres = bool(belief_term)
    cache: Dict = {}
    out: Dict[str, float] = {}
    for name in ARMS:
        key = name
        if key not in cache:
            cache[key] = gl.residual_model(
                cfg, obs, name, arms, plans=gl.arm_plans(cfg, obs, arms),
                dictionary="belief",
            )
        got = gl.target_conditioned_glrt(
            cfg, obs, arms[name], cache[key], target=weak,
            p_fa=float(cfg.detect.Pfa_target), dictionary="belief",
        )
        out[name] = got.statistic
        out[name + "_thr"] = got.threshold
        out[name + "_det"] = float(got.detected)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--out", default="results_belief_cres")
    args = ap.parse_args(argv)

    cfg = apply_preset(Config(), "paper-canonical")
    cfg.geometry.area_xy = args.area
    cfg.detect.target_rcs = args.rcs
    cfg.cancellation.enable = True

    sig_l, sig_k = gl._belief_bin_sigmas(cfg)
    print("belief error in C_res : area %.0f m, RCS %.2f, %d trials"
          % (args.area, args.rcs, args.trials))
    print("  declared belief: sigma_p = %.1f m, sigma_v = %.1f m/s"
          % (cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps))
    print("  => %.3f delay bins, %.3f Doppler bins" % (sig_l, sig_k))
    print()

    rows: List[dict] = []
    for t in range(int(args.trials)):
        obs1, obs0, weak = make_trial(cfg, t)
        arms1 = arms_with_gate(cfg, obs1, weak)
        arms0 = arms_with_gate(cfg, obs0, weak)
        for term in (False, True):
            h1 = evaluate(cfg, obs1, arms1, weak, term)
            h0 = evaluate(cfg, obs0, arms0, weak, term)
            for name in ARMS:
                rows.append({
                    "trial": t, "arm": name, "belief_term": int(term),
                    "t_h1": h1[name], "thr_h1": h1[name + "_thr"],
                    "det_h1": h1[name + "_det"],
                    "t_h0": h0[name], "thr_h0": h0[name + "_thr"],
                    "det_h0": h0[name + "_det"],
                })
        if (t + 1) % 10 == 0:
            print("  trial %d/%d" % (t + 1, args.trials), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "belief_cres.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("  %-16s %10s %10s %10s %10s" % (
        "arm", "P_FA off", "P_FA on", "P_D off", "P_D on"))
    for name in ARMS:
        def rate(term, key):
            sub = [r for r in rows if r["arm"] == name and r["belief_term"] == term]
            return sum(r[key] for r in sub) / max(len(sub), 1)
        print("  %-16s %10.3f %10.3f %10.3f %10.3f" % (
            name, rate(0, "det_h0"), rate(1, "det_h0"),
            rate(0, "det_h1"), rate(1, "det_h1")))
    print()
    print("  nominal P_FA = %.3f" % cfg.detect.Pfa_target)
    print("wrote %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
