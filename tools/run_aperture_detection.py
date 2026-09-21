"""P1-2, second half: does the lifted (spatial-DD) manifold actually detect better?

The first half put the array into the production collision penalty and found it
buys +0.22 dB and no ``P_D``: the *masking* channel is not the bottleneck.  This
measures the other channel -- the one the angular probe was actually about.  With
``aperture.enable`` the observation space becomes ``DD (x) array`` and the escape
fraction ``rho`` of a co-bin pair goes from ~0 to 0.47--0.87.  The question here
is whether that reaches the *decision*.

Reported per aperture, on paired (H1, H0) observations built from one geometry:

* ``P_D`` / ``P_FA`` at the analytic level for the requested ``p_fa``;
* AUC from the paired statistics -- threshold-free, so a level that is simply
  wrong in the lifted space cannot masquerade as a result;
* ``rho`` and the raw statistics, so a change in level and a change in
  separation can be told apart.

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/run_aperture_detection.py --trials 20 --out results_aperture_det
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.sensing.model import build_base_gains, generate_geometry
from isac_sim.scenario.prior import perturbed_geometry

ARMS = ("perfect_channel", "tp_uic_full")


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


def make_trial(cfg: Config, index: int, receiver_rule: str = "fixed"):
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


def score(cfg: Config, obs, weak: int, name: str):
    arms = cx.cancellation_arms(cfg, obs, weak_target=weak,
                                candidate_policy="protected_only")
    model = gl.residual_model(cfg, obs, name, arms,
                              plans=gl.arm_plans(cfg, obs, arms))
    got = gl.target_conditioned_glrt(cfg, obs, arms[name], model, target=weak,
                                     p_fa=float(cfg.detect.Pfa_target),
                                     dictionary="belief")
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--m-list", default="1,4,8")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results_aperture_det")
    args = ap.parse_args(argv)

    m_list = [int(x) for x in args.m_list.split(",") if x.strip()]
    rows: List[Dict] = []
    t0 = time.time()
    for m in m_list:
        cfg = build_cfg(args.area, args.rcs, args.seed, m)
        for t in range(int(args.trials)):
            obs1, obs0, weak = make_trial(cfg, t)
            for name in ARMS:
                g1 = score(cfg, obs1, weak, name)
                g0 = score(cfg, obs0, weak, name)
                rows.append({
                    "trial": t, "m_rx": m, "arm": name,
                    "t_h1": g1.statistic, "t_h0": g0.statistic,
                    "thr": g1.threshold, "dof": g1.dof_real,
                    "det_h1": float(g1.detected), "det_h0": float(g0.detected),
                    "rho_weighted": g1.rho_weighted, "rho_min": g1.rho_min,
                })
        print("  m_rx=%-3d done  [%.0f s]" % (m, time.time() - t0), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "aperture_detection.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("  %-6s %-16s %8s %8s %8s %8s" % ("m_rx", "arm", "P_D", "P_FA", "AUC", "rho"))
    for m in m_list:
        for name in ARMS:
            sub = [r for r in rows if r["m_rx"] == m and r["arm"] == name]
            t1 = [r["t_h1"] for r in sub]
            t0s = [r["t_h0"] for r in sub]
            wins = sum(1 for a in t1 for b in t0s if a > b) + 0.5 * sum(
                1 for a in t1 for b in t0s if a == b)
            auc = wins / max(len(t1) * len(t0s), 1)
            print("  %-6d %-16s %8.3f %8.3f %8.3f %8.3f" % (
                m, name,
                float(np.mean([r["det_h1"] for r in sub])),
                float(np.mean([r["det_h0"] for r in sub])),
                auc, float(np.mean([r["rho_weighted"] for r in sub])),
            ))
    print()
    print("  stats are paired within a trial; AUC is over all (H1, H0) pairs.")
    print("wrote %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
