"""P1-4: does searching over the belief uncertainty rescue the detector?

Measured so far, at 600 m / RCS 0.1:

* the declared belief error is ``0.96`` delay bins and ``1.26`` Doppler bins
  -- i.e. the tracker's uncertainty is about **one whole resolution cell**;
* with an array the escape fraction goes 0.18 -> 0.89, yet AUC stays ~0.54
  (chance), so *separability* is not what is missing;
* the analytic level is not a CFAR level here (``dof = 28`` against a threshold
  of 20.67), so ``P_D``/``P_FA`` read at that threshold are not interpretable.

The remaining hypothesis is therefore a **placement** one: the detector builds
the tested template at the believed cell while the echo sits one cell away, so
the statistic never sees the echo no matter how separable the cell is.

This script searches a grid of off-grid offsets around the belief for each of the
tested target's echoes and takes the maximum statistic, then compares against the
no-search baseline.  It uses the existing ``template_override`` hook, so nothing
in the production detector changes.

Pre-registered criterion (fixed before the run, and checked for the defect that
bit the earlier probes -- a rule whose PASS trigger equals its FAIL threshold)::

    PASS   AUC improves by >= 0.05 at m_rx = 8
    FAIL   |AUC change| < 0.02
    grey   in between -- reported as inconclusive, not rounded into a verdict

AUC is the primary metric on purpose: taking a maximum over a search grid
inflates the null statistic, so a fixed threshold would make the search look
better for the wrong reason.  ``P_D`` at an *empirically matched* ``P_FA`` is
reported alongside, with its small-sample caveat.

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/run_offgrid_search.py --trials 20 --m-list 1,8 --out results_offgrid
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from isac_sim import cancellation as cx
from isac_sim import cancellation_glrt as gl
from isac_sim.config import Config, apply_overrides, apply_preset
from isac_sim.model import build_base_gains, generate_geometry
from isac_sim.prior import perturbed_geometry

PASS_GAIN = 0.05
FAIL_GAIN = 0.02
ARM = "perfect_channel"


def build_cfg(area: float, rcs: float, seed: int, m_rx: int) -> Config:
    cfg = apply_preset(Config(), "paper-canonical")
    cfg = apply_overrides(cfg, {
        "geometry.area_xy": float(area),
        "detect.target_rcs": float(rcs),
        "run.seed": int(seed),
        "run.verbose": False,
        "cancellation.enable": True,
    })
    if int(m_rx) > 1:
        cfg.aperture.enable = True
        cfg.aperture.m_rx = int(m_rx)
    return cfg


def offset_template(cfg: Config, obs, target: int, dk: float, dl: float,
                    place: str = "belief") -> np.ndarray:
    """The tested target's echo block, rebuilt ``(dk, dl)`` bins off the belief.

    One column per illuminating path, amplitude ``sqrt(power)`` and the same
    array lift the dictionary uses -- so this is exactly the dictionary block,
    merely placed somewhere else.

    ``place="truth"`` uses the *true* bins instead, i.e. an oracle placement.
    It cannot be used by a receiver, and that is the point: it is the upper bound
    that decides whether the deficit is placement at all.  If oracle placement
    does not move the AUC, no search can, and the whole line is closed.
    """
    m_rx = int(cfg.aperture.m_rx) if cfg.aperture.enable else 1
    if place == "truth":
        sources = obs.targets
    else:
        sources = obs.targets_belief if obs.targets_belief is not None else obs.targets
    cols: List[np.ndarray] = []
    for s in sources:
        if int(s.target) != int(target):
            continue
        k = cx.kernel_vector(cfg, float(s.doppler_bin) + dk, float(s.delay_bin) + dl)
        if m_rx > 1:
            k = np.kron(k, cx.steering_vector(m_rx, float(s.u)))
        cols.append(math.sqrt(max(float(s.power), 0.0)) * k)
    if not cols:
        raise ValueError("target %d has no believed source" % target)
    return np.stack(cols, axis=1)


def grid(half: float, step: float) -> List[Tuple[float, float]]:
    n = int(round(half / step))
    vals = [i * step for i in range(-n, n + 1)]
    return [(dk, dl) for dk in vals for dl in vals]


def make_trial(cfg: Config, index: int):
    rng = np.random.default_rng([cfg.run.seed, int(index)])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    belief = perturbed_geometry(
        cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
    )
    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    j = 7
    echo = [float(np.sum(base.target_gain[:, j, q])) for q in range(cfg.scale.Q)]
    weak = int(np.argmin(echo))
    obs1, obs0 = cx.build_observation_pair(
        cfg, geom, belief, base, j, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0, exclude_target=weak, weak_index=weak,
    )
    return obs1, obs0, weak


def trial_stats(cfg: Config, index: int, offsets: List[Tuple[float, float]]):
    obs1, obs0, weak = make_trial(cfg, index)
    arms1 = cx.cancellation_arms(cfg, obs1, weak_target=weak,
                                 candidate_policy="protected_only")
    arms0 = cx.cancellation_arms(cfg, obs0, weak_target=weak,
                                 candidate_policy="protected_only")
    model1 = gl.residual_model(cfg, obs1, ARM, arms1,
                               plans=gl.arm_plans(cfg, obs1, arms1))
    model0 = gl.residual_model(cfg, obs0, ARM, arms0,
                               plans=gl.arm_plans(cfg, obs0, arms0))

    def stat(obs, arms, model, override):
        got = gl.target_conditioned_glrt(
            cfg, obs, arms[ARM], model, target=weak,
            p_fa=float(cfg.detect.Pfa_target), dictionary="belief",
            template_override=override,
        )
        return float(got.statistic)

    base1 = stat(obs1, arms1, model1, None)
    base0 = stat(obs0, arms0, model0, None)
    # oracle placement: the same block, built on the true bins.  Upper bound for
    # every placement strategy, including the search below.
    orc1 = stat(obs1, arms1, model1, offset_template(cfg, obs1, weak, 0.0, 0.0, "truth"))
    orc0 = stat(obs0, arms0, model0, offset_template(cfg, obs0, weak, 0.0, 0.0, "truth"))
    best1, best0 = base1, base0
    for dk, dl in offsets:
        if dk == 0.0 and dl == 0.0:
            continue
        t1 = offset_template(cfg, obs1, weak, dk, dl)
        t0 = offset_template(cfg, obs0, weak, dk, dl)
        best1 = max(best1, stat(obs1, arms1, model1, t1))
        best0 = max(best0, stat(obs0, arms0, model0, t0))
    return base1, base0, best1, best0, orc1, orc0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--m-list", default="1,8")
    ap.add_argument("--half", type=float, default=1.5)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results_offgrid")
    args = ap.parse_args(argv)

    offsets = grid(args.half, args.step)
    print("off-grid search: %d offsets (%+.2f bins, step %.2f)"
          % (len(offsets), args.half, args.step))
    print("pre-registered: PASS if AUC(8) gains >= %.2f ; FAIL if |gain| < %.2f"
          % (PASS_GAIN, FAIL_GAIN))
    print()

    rows: List[Dict] = []
    t0 = time.time()
    for m in [int(x) for x in args.m_list.split(",") if x.strip()]:
        cfg = build_cfg(args.area, args.rcs, args.seed, m)
        for t in range(int(args.trials)):
            b1, b0, s1, s0, o1, o0 = trial_stats(cfg, t, offsets)
            rows.append({
                "trial": t, "m_rx": m,
                "base_h1": b1, "base_h0": b0,
                "search_h1": s1, "search_h0": s0, "oracle_h1": o1, "oracle_h0": o0,
            })
        print("  m_rx=%-3d done  [%.0f s]" % (m, time.time() - t0), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "offgrid_search.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("  %-6s %-10s %8s %10s" % ("m_rx", "variant", "AUC", "P_D@P_FA=0.05"))
    aucs: Dict[int, float] = {}
    for m in sorted({int(r["m_rx"]) for r in rows}):
        sub = [r for r in rows if int(r["m_rx"]) == m]
        for label, k1, k0 in (("no search", "base_h1", "base_h0"),
                              ("search", "search_h1", "search_h0"),
                              ("oracle", "oracle_h1", "oracle_h0")):
            t1 = [r[k1] for r in sub]
            t0s = [r[k0] for r in sub]
            wins = sum(1 for a in t1 for b in t0s if a > b) + 0.5 * sum(
                1 for a in t1 for b in t0s if a == b)
            auc = wins / max(len(t1) * len(t0s), 1)
            # empirically matched level: the pooled H0 quantile
            thr = float(np.quantile(t0s, 1.0 - float(cfg.detect.Pfa_target)))
            p_d = float(np.mean([a > thr for a in t1]))
            print("  %-6d %-10s %8.3f %10.3f" % (m, label, auc, p_d))
            aucs[(m, label)] = auc

    print()
    gain = aucs.get((8, "search"), float("nan")) - aucs.get((8, "no search"), float("nan"))
    verdict = ("PASS" if gain >= PASS_GAIN
               else ("FAIL" if abs(gain) < FAIL_GAIN else "GREY"))
    print("  AUC gain at m_rx = 8 : %+.3f  ->  %s" % (gain, verdict))
    print("  (P_D@P_FA uses the pooled H0 quantile over %d trials: indicative only.)"
          % int(args.trials))
    print("wrote %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
