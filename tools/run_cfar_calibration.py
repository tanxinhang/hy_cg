"""CFAR calibration: is the declared P_FA the P_FA the detector delivers?

The current evidence for constant false-alarm behaviour is ``E[T] = dof/2`` --
the *mean* of the null statistic matching its nominal mean.  That is a necessary
condition and close to a vacuous one: a null distribution can have the right
mean and the wrong tail, and the tail is the only part a detector uses.  A
threshold set at the nominal 0.95 quantile of a mis-modelled tail delivers
whatever false-alarm rate that tail implies.

This driver measures the tail instead, at three complexity levels, so a failure
can be attributed to the thing that caused it:

* ``oracle``  -- perfect channel, truth dictionary, no belief error.  If the
  rate is wrong here, the detector's own null model is wrong and nothing
  downstream can be trusted.
* ``truth``   -- belief error in the *geometry*, but the target dictionary is
  still built on the truth.  Isolates the residual / protection side.
* ``belief``  -- the production configuration: the receiver matches against the
  dictionary it believes.  This is the number the paper is entitled to quote.

Reported per level: the empirical false-alarm rate with a Wilson interval, the
empirical-to-nominal threshold ratio (``q95_empirical / threshold``), and the
ranking of every null statistic so a QQ check is possible offline.

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/run_cfar_calibration.py --trials 5 --nulls 100 --pfa 0.05
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim.receiver import cancellation as cx  # noqa: E402
from isac_sim.receiver import cancellation_glrt as gl  # noqa: E402
from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.sensing.model import build_base_gains, generate_geometry  # noqa: E402
from isac_sim.scenario.prior import perturbed_geometry  # noqa: E402

ARM = "tp_uic_full"
LEVELS = ("oracle", "truth", "belief")


def build_cfg(area: float, rcs: float, seed: int, m_rx: int,
              belief_sigma_pos: float | None = None,
              belief_error_in_cres: bool = False,
              m: int = 15, q: int = 10,
              covariance_protection: bool = False,
              adaptive_soft: bool = False,
              adaptive_risk_slack: float = 0.0) -> Config:
    cfg = apply_preset(Config(), "small-uav-compact-800m")
    over = {
        "geometry.area_xy": area,
        "scale.M": int(m),
        "scale.Q": int(q),
        "detect.target_rcs": rcs,
        "run.seed": seed,
        "run.verbose": False,
        "cancellation.enable": True,
        "cancellation.n_cpi": 1,
        "cancellation.covariance_protection": bool(covariance_protection),
        "cancellation.belief_error_in_cres": bool(belief_error_in_cres),
        "cancellation.adaptive_soft_enable": bool(adaptive_soft),
        "cancellation.adaptive_soft_risk_slack": float(adaptive_risk_slack),
        "aperture.enable": bool(m_rx > 1),
        "aperture.m_rx": int(m_rx),
    }
    if belief_sigma_pos is not None:
        # Attribution control, not a tuning knob: setting this to 0 makes the
        # belief geometry identical to the truth while leaving every *code path*
        # on the belief route.  If the false-alarm rate collapses back to the
        # nominal value, the excess is caused by the tracker error and is a
        # property of the scenario; if it does not, the belief route itself is
        # mis-modelled and no amount of scenario talk explains it.
        over["prior.belief_sigma_pos_m"] = float(belief_sigma_pos)
        over["prior.belief_sigma_vel_mps"] = 0.0
    return apply_overrides(cfg, over)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def null_observation(cfg: Config, geom, geom_belief, base, receiver: int,
                     weak: int, rng: np.random.Generator):
    """One H0 draw; dictionaries stay fixed while coefficients/noise vary."""
    m = int(cfg.scale.M)
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    _obs1, obs0 = cx.build_observation_pair(
        cfg, geom, geom_belief, base, receiver, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0, exclude_target=weak, weak_index=weak,
    )
    return obs0


def null_statistic(
    cfg: Config, obs0, level: str, p_fa: float, *, model0=None,
    nuisance_manifold: int = 0,
    belief_error_threshold_only: bool = False,
    threshold_override: float | None = None,
):
    """Score one H0 draw, caching the geometry-fixed affine receiver model."""
    null_cov_model = None
    if model0 is None:
        arms0 = cx.cancellation_arms(
            cfg, obs0, weak_target=int(obs0.weak_index),
            candidate_policy="protected_only",
        )
        model0 = gl.residual_model(
            cfg, obs0, ARM, arms0, plans=gl.arm_plans(cfg, obs0, arms0),
            dictionary=("truth" if level in ("oracle", "truth") else "belief"),
        )
        if belief_error_threshold_only and level == "belief":
            null_cfg = apply_overrides(
                cfg, {"cancellation.belief_error_in_cres": True}
            )
            null_cov_model = gl.residual_model(
                null_cfg, obs0, ARM, arms0,
                plans=gl.arm_plans(null_cfg, obs0, arms0),
                dictionary="belief",
            )
    got = gl.target_conditioned_glrt(
        cfg, obs0, None, model0, target=int(obs0.weak_index), p_fa=float(p_fa),
        dictionary=("truth" if level in ("oracle", "truth") else "belief"),
        nuisance_manifold=int(nuisance_manifold),
        null_cov_model=null_cov_model,
        threshold_override=threshold_override,
    )
    return float(got.statistic), float(got.threshold), int(got.dof_real), model0


def run(cfg: Config, args, m_rx: int) -> List[dict]:
    rows: List[dict] = []
    for trial in range(int(args.trials)):
        geom_rng = np.random.default_rng([cfg.run.seed, int(trial)])
        geom = generate_geometry(cfg, geom_rng)
        base = build_base_gains(cfg, geom, geom_rng)
        belief = perturbed_geometry(
            cfg, geom, cfg.prior.belief_sigma_pos_m,
            cfg.prior.belief_sigma_vel_mps, geom_rng,
        )
        # ``oracle`` uses the truth as its own belief: no tracker error at all.
        belief_for = {"oracle": geom, "truth": belief, "belief": belief}
        for receiver in ([int(x) for x in args.receivers.split(",")]
                         if args.receivers else [7]):
            echo = [float(np.sum(base.target_gain[:, receiver, q]))
                    for q in range(cfg.scale.Q)]
            weak = int(np.argmin(echo))
            for level in LEVELS:
                stats: List[float] = []
                thr = float("nan")
                dof = 0
                model0 = None
                calibrated_threshold = None
                for r in range(int(args.nulls)):
                    rng = np.random.default_rng(
                        [cfg.run.seed, int(trial), int(receiver), 104729 + int(r)]
                    )
                    obs0 = null_observation(
                        cfg, geom, belief_for[level], base, receiver, weak, rng
                    )
                    s, thr, dof, model0 = null_statistic(
                        cfg, obs0, level, args.pfa, model0=model0,
                        nuisance_manifold=args.nuisance_manifold,
                        belief_error_threshold_only=args.belief_error_threshold_only,
                        threshold_override=calibrated_threshold,
                    )
                    if calibrated_threshold is None:
                        calibrated_threshold = float(thr)
                    stats.append(s)
                exc = sum(1 for s in stats if s > thr)
                lo, hi = wilson(exc, len(stats))
                q95 = float(np.quantile(stats, 0.95)) if stats else float("nan")
                rows.append({
                    "m_rx": int(m_rx),
                    "trial": int(trial),
                    "receiver": int(receiver),
                    "weak_target": int(weak),
                    "level": level,
                    "p_fa_nominal": float(args.pfa),
                    "p_fa_empirical": exc / max(len(stats), 1),
                    "p_fa_lo": lo,
                    "p_fa_hi": hi,
                    "n_nulls": len(stats),
                    "threshold": thr,
                    "empirical_q95": q95,
                    "threshold_ratio": q95 / thr if thr > 0 else float("nan"),
                    "mean_over_dof_half": (float(np.mean(stats)) / (dof / 2.0))
                                          if dof else float("nan"),
                    "dof_real": int(dof),
                    "covariance_protection": bool(
                        cfg.cancellation.covariance_protection
                    ),
                    "receiver_arm": str(ARM),
                })
    return rows


def report(rows: Sequence[dict], args) -> None:
    print("\n=== CFAR calibration (nominal P_FA = %.3f, %d nulls per unit) ==="
          % (args.pfa, args.nulls))
    print("  oracle = perfect channel + truth dictionary; truth = belief error in")
    print("  the geometry only; belief = the production configuration.")
    print()
    print("  %-5s %-8s %10s %16s %12s %12s" % (
        "m_rx", "level", "emp. P_FA", "95% Wilson", "E[T]/(dof/2)", "q95/thr"))
    for m_rx in sorted({r["m_rx"] for r in rows}):
        for level in LEVELS:
            sub = [r for r in rows if r["m_rx"] == m_rx and r["level"] == level]
            if not sub:
                continue
            exc = sum(int(round(r["p_fa_empirical"] * r["n_nulls"])) for r in sub)
            n = sum(int(r["n_nulls"]) for r in sub)
            lo, hi = wilson(exc, n)
            mean_ratio = sum(r["mean_over_dof_half"] for r in sub) / len(sub)
            thr_ratio = sum(r["threshold_ratio"] for r in sub) / len(sub)
            print("  %-5d %-8s %10.4f  [%.4f, %.4f] %12.3f %12.3f"
                  % (m_rx, level, exc / max(n, 1), lo, hi, mean_ratio, thr_ratio))

    print("\n  How to read this:")
    print("    * E[T]/(dof/2) ~ 1 with a wrong P_FA is the exact failure the old")
    print("      check could not see: right mean, wrong tail.")
    print("    * q95/thr > 1 means the null tail is heavier than the model, so")
    print("      the detector alarms more often than it claims.")
    print("    * a Wilson interval excluding the nominal value is a calibration")
    print("      failure, not a sampling artefact -- report it, do not retune.")


def main(argv=None) -> int:
    global ARM, LEVELS
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--nulls", type=int, default=100)
    ap.add_argument("--pfa", type=float, default=0.05)
    ap.add_argument("--m-list", default="1")
    ap.add_argument("--m", type=int, default=15)
    ap.add_argument("--q", type=int, default=10)
    ap.add_argument("--covariance-protection", action="store_true")
    ap.add_argument(
        "--receiver-arm", choices=("tp_uic_full", "adaptive_soft_tpuic"),
        default="tp_uic_full",
    )
    ap.add_argument("--adaptive-risk-slack", type=float, default=0.001)
    ap.add_argument(
        "--levels", default="oracle,truth,belief",
        help="comma-separated subset of oracle,truth,belief",
    )
    ap.add_argument("--receivers", default="")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--belief-sigma-pos", type=float, default=None,
                    help="override prior.belief_sigma_pos_m (position std, m) "
                         "to attribute the excess false-alarm rate; 0 makes the "
                         "belief equal to the truth on the belief code path")
    ap.add_argument(
        "--belief-error-in-cres", action="store_true",
        help="include first-order belief-template mismatch in residual covariance",
    )
    ap.add_argument(
        "--belief-error-threshold-only", action="store_true",
        help=("keep the baseline whitening/statistic and use belief-error "
              "covariance only for its weighted-exponential null threshold"),
    )
    ap.add_argument(
        "--nuisance-manifold", type=int, default=0,
        help="tangent order used to project nuisance-target belief mismatch",
    )
    ap.add_argument("--out", default="results_cfar_calibration")
    args = ap.parse_args(argv)
    if args.belief_error_in_cres and args.belief_error_threshold_only:
        ap.error("choose either full belief covariance or threshold-only calibration")
    ARM = str(args.receiver_arm)
    requested_levels = tuple(v.strip() for v in args.levels.split(",") if v.strip())
    if not requested_levels or any(v not in LEVELS for v in requested_levels):
        ap.error("--levels must be a non-empty subset of oracle,truth,belief")
    LEVELS = requested_levels

    os.makedirs(args.out, exist_ok=True)
    all_rows: List[dict] = []
    for m_rx in [int(x) for x in args.m_list.split(",")]:
        cfg = build_cfg(
            args.area, args.rcs, args.seed, m_rx, args.belief_sigma_pos,
            args.belief_error_in_cres, args.m, args.q,
            args.covariance_protection,
            args.receiver_arm == "adaptive_soft_tpuic",
            args.adaptive_risk_slack,
        )
        print("m_rx = %d : %d trials x %d nulls x %d levels"
              % (m_rx, args.trials, args.nulls, len(LEVELS)), flush=True)
        all_rows.extend(run(cfg, args, m_rx))

    path = os.path.join(args.out, "cfar_calibration.csv")
    keys: List[str] = []
    for row in all_rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    report(all_rows, args)
    print("\nwrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
