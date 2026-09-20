"""Detector evidence protocol: conditional, paired and marginal AUC.

Why the old protocol was not enough
-----------------------------------
Every off-grid / aperture probe so far reported one number::

    for x in H1_all_trials:
        for y in H0_all_trials:
            compare(x, y)

That is a **marginal** AUC -- draw one H1 and one H0 from the whole deployment
distribution and ask which is larger.  It is a well-defined quantity, but every
conclusion actually being drawn from it is a *conditional* one: "given this
scene, does this statistic carry detection information about this target?"  The
marginal version mixes geometry, target strength, belief error and residual
scale across units, so a hard scene's H1 is compared against an easy scene's H0
and the mixture can sit at 0.5 while every individual scene separates.

Measured on this repository (``perfect_channel``, ``m_rx=8``): the pooled
cross-AUC reads 0.540 and was reported as "the detector is random", while the
matched-trial probability ``P(T_H1 > T_H0)`` on the same data reads 0.65.  Two
different questions, two different answers, and the reported one was not the
one being asked.

This driver reports all three, and never one alone:

* **conditional** -- AUC inside one unit (fixed geometry, fixed receiver, fixed
  target), over independent noise realisations.  Answers "does the statistic
  separate H1 from H0 *here*".  Averaged over units with a spread.
* **paired**     -- ``P(T_H1^{(r)} > T_H0^{(r)})`` over matched realisations
  ``r``.  Same question, using only the matched pair, so the nuisance
  (geometry, direct field, target strength) cancels exactly.
* **marginal**   -- the old pooled cross-unit AUC, kept so a reader can see how
  much of the earlier literature's 0.5 was mixture rather than randomness.

Wilson intervals are reported for the paired probability; the conditional AUC is
report**ed** per unit with its spread, because averaging AUCs over units is not
itself an AUC.

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/run_detector_evidence.py --trials 20 --realisations 8 --m-list 1,8
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim import cancellation as cx  # noqa: E402
from isac_sim import cancellation_glrt as gl  # noqa: E402
from isac_sim.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.model import build_base_gains, generate_geometry  # noqa: E402
from isac_sim.prior import perturbed_geometry  # noqa: E402

ARM = "tp_uic_full"


def build_cfg(
    area: float, rcs: float, seed: int, m_rx: int,
    m: int = 15, q: int = 10, covariance_protection: bool = False,
    adaptive_soft: bool = False, adaptive_risk_slack: float = 0.0,
    belief_error_in_cres: bool = False,
) -> Config:
    cfg = apply_preset(Config(), "small-uav-compact-800m")
    return apply_overrides(
        cfg,
        {
            "geometry.area_xy": area,
            "scale.M": int(m),
            "scale.Q": int(q),
            "detect.target_rcs": rcs,
            "run.seed": seed,
            "run.verbose": False,
            "cancellation.enable": True,
            "cancellation.n_cpi": 1,
            "cancellation.covariance_protection": bool(covariance_protection),
            "cancellation.adaptive_soft_enable": bool(adaptive_soft),
            "cancellation.adaptive_soft_risk_slack": float(adaptive_risk_slack),
            "cancellation.belief_error_in_cres": bool(belief_error_in_cres),
            "aperture.enable": bool(m_rx > 1),
            "aperture.m_rx": int(m_rx),
        },
    )


def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval -- the right one for a proportion at small ``n``.

    The normal approximation gives intervals outside [0, 1] and is badly
    over-confident near the boundary, which is exactly where these numbers sit.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def auc(h1: Sequence[float], h0: Sequence[float]) -> float:
    """``Pr(T_H1 > T_H0)`` with ties counted as one half."""
    if not h1 or not h0:
        return float("nan")
    a = np.asarray(h1, dtype=float)[:, None]
    b = np.asarray(h0, dtype=float)[None, :]
    return float(np.mean((a > b) + 0.5 * (a == b)))


def offset_template(cfg: Config, obs, target: int, dk: float, dl: float,
                    place: str = "belief") -> np.ndarray:
    """The tested target's echo block, ``(dk, dl)`` bins off the belief.

    ``place="truth"`` places it on the true bins: an oracle a receiver cannot
    have.  It is the placement upper bound -- if it does not move the evidence,
    no search can.
    """
    m_rx = int(cfg.aperture.m_rx) if cfg.aperture.enable else 1
    sources = obs.targets if place == "truth" else (
        obs.targets_belief if obs.targets_belief is not None else obs.targets
    )
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


def state_sigma_templates(cfg: Config, obs, target: int) -> List[np.ndarray]:
    """Eight correlated horizontal-state ±1σ templates for one target."""
    sources = obs.targets_belief if obs.targets_belief is not None else obs.targets
    sources = [s for s in sources if int(s.target) == int(target)]
    scales = np.asarray([
        float(cfg.prior.belief_sigma_pos_m),
        float(cfg.prior.belief_sigma_pos_m),
        0.0,
        float(cfg.prior.belief_sigma_vel_mps),
        float(cfg.prior.belief_sigma_vel_mps),
        0.0,
    ])
    m_rx = int(cfg.aperture.m_rx) if cfg.aperture.enable else 1
    templates: List[np.ndarray] = []
    for dim in (0, 1, 3, 4):
        if scales[dim] <= 0.0:
            continue
        for sign in (1.0, -1.0):
            delta = sign * scales[dim]
            cols = []
            for src in sources:
                gk = src.jacobian_doppler_state
                gl = src.jacobian_delay_state
                gu = src.jacobian_bearing_state
                dk = 0.0 if gk is None else float(gk[dim]) * delta
                dl = 0.0 if gl is None else float(gl[dim]) * delta
                du = 0.0 if gu is None else float(gu[dim]) * delta
                kernel = cx.kernel_vector(
                    cfg, float(src.doppler_bin) + dk,
                    float(src.delay_bin) + dl,
                )
                if m_rx > 1:
                    u = float(np.clip(float(src.u) + du, -1.0, 1.0))
                    kernel = np.kron(kernel, cx.steering_vector(m_rx, u))
                cols.append(math.sqrt(max(float(src.power), 0.0)) * kernel)
            if cols:
                templates.append(np.stack(cols, axis=1))
    return templates


def one_realisation(cfg: Config, geom, belief, base, receiver: int, weak: int,
                    rng: np.random.Generator, offsets, with_search: bool,
                    model_cache: dict | None = None):
    """``{arm: (T_H1, T_H0)}`` for one noise draw of one fixed unit."""
    m = int(cfg.scale.M)
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    obs1, obs0 = cx.build_observation_pair(
        cfg, geom, belief, base, receiver, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0, exclude_target=weak, weak_index=weak,
    )
    cached = None if model_cache is None else model_cache.get("models")
    if cached is None:
        arms1 = cx.cancellation_arms(cfg, obs1, weak_target=weak,
                                     candidate_policy="protected_only")
        arms0 = cx.cancellation_arms(cfg, obs0, weak_target=weak,
                                     candidate_policy="protected_only")
        model1 = gl.residual_model(cfg, obs1, ARM, arms1,
                                   plans=gl.arm_plans(cfg, obs1, arms1))
        model0 = gl.residual_model(cfg, obs0, ARM, arms0,
                                   plans=gl.arm_plans(cfg, obs0, arms0))
        if model_cache is not None:
            model_cache["models"] = (model1, model0)
    else:
        model1, model0 = cached
        # ``target_conditioned_glrt`` consumes the affine model and observation;
        # its legacy result argument is retained for API compatibility only.
        arms1 = {ARM: None}
        arms0 = {ARM: None}

    def stat(
        obs, arms, model, override, nuisance_manifold=0,
        centre_only: bool = True,
    ):
        got = gl.target_conditioned_glrt(
            cfg, obs, arms[ARM], model, target=weak,
            p_fa=float(cfg.detect.Pfa_target), dictionary="belief",
            template_override=override,
            nuisance_manifold=int(nuisance_manifold),
            centre_only=bool(centre_only),
        )
        return float(got.statistic)

    out: Dict[str, Tuple[float, float]] = {}
    out["baseline"] = (stat(obs1, arms1, model1, None),
                       stat(obs0, arms0, model0, None))
    out["manifold"] = (
        stat(obs1, arms1, model1, None, nuisance_manifold=1),
        stat(obs0, arms0, model0, None, nuisance_manifold=1),
    )
    out["composite"] = (
        stat(obs1, arms1, model1, None, centre_only=False),
        stat(obs0, arms0, model0, None, centre_only=False),
    )
    sigma_templates1 = state_sigma_templates(cfg, obs1, weak)
    sigma_templates0 = state_sigma_templates(cfg, obs0, weak)
    sigma_stats1 = [stat(obs1, arms1, model1, None)]
    sigma_stats0 = [stat(obs0, arms0, model0, None)]
    for template1, template0 in zip(sigma_templates1, sigma_templates0):
        sigma_stats1.append(stat(obs1, arms1, model1, template1))
        sigma_stats0.append(stat(obs0, arms0, model0, template0))
    out["sigma_search"] = (max(sigma_stats1), max(sigma_stats0))
    out["sigma_mean"] = (
        float(np.mean(sigma_stats1)), float(np.mean(sigma_stats0))
    )
    out["oracle"] = (
        stat(obs1, arms1, model1,
             offset_template(cfg, obs1, weak, 0.0, 0.0, "truth")),
        stat(obs0, arms0, model0,
             offset_template(cfg, obs0, weak, 0.0, 0.0, "truth")),
    )
    if with_search:
        # A maximum over the grid is taken *within* each hypothesis.  The null
        # is inflated by the same search, which is why AUC and not a fixed
        # threshold is the metric here -- see the module docstring.
        best1 = stat(obs1, arms1, model1, None)
        best0 = stat(obs0, arms0, model0, None)
        for dk, dl in offsets:
            if dk == 0.0 and dl == 0.0:
                continue
            best1 = max(best1, stat(obs1, arms1, model1,
                                    offset_template(cfg, obs1, weak, dk, dl)))
            best0 = max(best0, stat(obs0, arms0, model0,
                                    offset_template(cfg, obs0, weak, dk, dl)))
        out["search"] = (best1, best0)
    return out


def one_receiver_comparison(
    cfg: Config, geom, belief, base, receiver: int, weak: int,
    rng: np.random.Generator, model_cache: dict | None = None,
):
    """Paired centre-template evidence for hard and adaptive receivers."""
    m = int(cfg.scale.M)
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    obs1, obs0 = cx.build_observation_pair(
        cfg, geom, belief, base, receiver, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0, exclude_target=weak, weak_index=weak,
    )
    models = None if model_cache is None else model_cache.get("receiver_models")
    if models is None:
        arms1 = cx.cancellation_arms(
            cfg, obs1, weak_target=weak, candidate_policy="protected_only"
        )
        arms0 = cx.cancellation_arms(
            cfg, obs0, weak_target=weak, candidate_policy="protected_only"
        )
        models = {}
        for label, name in (
            ("hard", "tp_uic_full"),
            ("adaptive", "adaptive_soft_tpuic"),
        ):
            models[label] = (
                gl.residual_model(
                    cfg, obs1, name, arms1,
                    plans=gl.arm_plans(cfg, obs1, arms1),
                ),
                gl.residual_model(
                    cfg, obs0, name, arms0,
                    plans=gl.arm_plans(cfg, obs0, arms0),
                ),
            )
        if model_cache is not None:
            model_cache["receiver_models"] = models

    out = {}
    for label, (model1, model0) in models.items():
        stat1 = gl.target_conditioned_glrt(
            cfg, obs1, None, model1, target=weak,
            p_fa=float(cfg.detect.Pfa_target), dictionary="belief",
        ).statistic
        stat0 = gl.target_conditioned_glrt(
            cfg, obs0, None, model0, target=weak,
            p_fa=float(cfg.detect.Pfa_target), dictionary="belief",
        ).statistic
        out[label] = (float(stat1), float(stat0))
    return out


def run(cfg: Config, args, m_rx: int) -> List[dict]:
    offsets = grid(args.half, args.step)
    rows: List[dict] = []
    for trial in range(
        int(args.trial_start), int(args.trial_start) + int(args.trials)
    ):
        geom_rng = np.random.default_rng([cfg.run.seed, int(trial)])
        geom = generate_geometry(cfg, geom_rng)
        base = build_base_gains(cfg, geom, geom_rng)
        belief = perturbed_geometry(
            cfg, geom, cfg.prior.belief_sigma_pos_m,
            cfg.prior.belief_sigma_vel_mps, geom_rng,
        )
        for receiver in ([int(x) for x in args.receivers.split(",")]
                         if args.receivers else [7]):
            echo = [float(np.sum(base.target_gain[:, receiver, q]))
                    for q in range(cfg.scale.Q)]
            weak = int(np.argmin(echo))
            h1: Dict[str, List[float]] = {}
            h0: Dict[str, List[float]] = {}
            model_cache: dict = {}
            for r in range(int(args.realisations)):
                rng = np.random.default_rng(
                    [cfg.run.seed, int(trial), int(receiver), 7919 + int(r)]
                )
                got = (
                    one_receiver_comparison(
                        cfg, geom, belief, base, receiver, weak, rng, model_cache
                    )
                    if args.receiver_compare else
                    one_realisation(
                        cfg, geom, belief, base, receiver, weak,
                        rng, offsets, bool(args.search), model_cache
                    )
                )
                for arm, (t1, t0) in got.items():
                    h1.setdefault(arm, []).append(t1)
                    h0.setdefault(arm, []).append(t0)
            # One row per *realisation*, not per unit: the conditional and
            # paired figures are aggregates of these, and the marginal one needs
            # every draw -- collapsing to a unit-level row first would make the
            # pooled column a sample of a sample.
            for arm in h1:
                for r, (t1, t0) in enumerate(zip(h1[arm], h0[arm])):
                    rows.append({
                        "m_rx": int(m_rx),
                        "trial": int(trial),
                        "receiver": int(receiver),
                        "weak_target": int(weak),
                        "arm": arm,
                        "realisation": int(r),
                        "t_h1": float(t1),
                        "t_h0": float(t0),
                        "dT": float(t1 - t0),
                        "covariance_protection": bool(
                            cfg.cancellation.covariance_protection
                        ),
                    })
    return rows


def _units(rows: Sequence[dict], m_rx: int, arm: str) -> List[List[dict]]:
    """Group rows into units: one fixed scene, all its noise realisations."""
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        if r["m_rx"] != m_rx or r["arm"] != arm:
            continue
        groups.setdefault((r["trial"], r["receiver"]), []).append(r)
    return list(groups.values())


def report(rows: List[dict], args) -> None:
    print("\n=== Detector evidence protocol (trials = %d, realisations = %d) ==="
          % (args.trials, args.realisations))
    print("  conditional = AUC within one fixed scene, over noise draws")
    print("  paired      = P(T_H1 > T_H0) on matched realisations")
    print("  marginal    = pooled cross-scene AUC (the old, mixed question)")
    print()
    header = "  %-5s %-9s %12s %12s %12s %16s" % (
        "m_rx", "arm", "cond. AUC", "paired P", "paired 95%", "marginal AUC")
    print(header)
    for m_rx in sorted({r["m_rx"] for r in rows}):
        for arm in (
            "hard", "adaptive",
            "baseline", "sigma_search", "sigma_mean", "composite",
            "manifold", "search", "oracle"
        ):
            units = _units(rows, m_rx, arm)
            if not units:
                continue
            conds = [auc([u["t_h1"] for u in g], [u["t_h0"] for u in g]) for g in units]
            wins = sum(1 for g in units for u in g if u["dT"] > 0.0)
            n = sum(len(g) for g in units)
            lo, hi = wilson(wins, n)
            # Marginal / pooled: every H1 against every H0 across all scenes --
            # i.e. exactly the quantity the earlier probes reported.
            sub = [r for r in rows if r["m_rx"] == m_rx and r["arm"] == arm]
            marg = auc([r["t_h1"] for r in sub], [r["t_h0"] for r in sub])
            print("  %-5d %-9s %12.3f %12.3f  [%.2f, %.2f] %12.3f"
                  % (m_rx, arm, st.mean(conds), wins / max(n, 1), lo, hi, marg))
            if len(conds) > 1:
                print("         %-9s (conditional AUC spread across %d scenes: %.3f)"
                      % ("", len(conds), st.stdev(conds)))

    print("\n  How to read this:")
    print("    * marginal ~ 0.5 with conditional > 0.5  =>  the statistic DOES")
    print("      separate, and the old pooled number was measuring scene mixture.")
    print("    * both ~ 0.5  =>  the statistic really carries no evidence here.")
    print("    * a Wilson interval spanning 0.5 means the sample cannot decide,")
    print("      which must be said as 'not established', never as 'closed'.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--trial-start", type=int, default=0)
    ap.add_argument("--realisations", type=int, default=8)
    ap.add_argument("--m-list", default="1,8")
    ap.add_argument("--m", type=int, default=15)
    ap.add_argument("--q", type=int, default=10)
    ap.add_argument("--covariance-protection", action="store_true")
    ap.add_argument(
        "--receiver-compare", action="store_true",
        help="compare hard TP-UIC with belief-adaptive soft TP-UIC",
    )
    ap.add_argument("--adaptive-risk-slack", type=float, default=0.001)
    ap.add_argument("--belief-error-in-cres", action="store_true")
    ap.add_argument("--half", type=float, default=1.5)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--search", action="store_true",
                    help="add the off-grid max-over-grid arm (much slower)")
    ap.add_argument("--receivers", default="",
                    help="comma-separated receiver indices (default: 7)")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results_detector_evidence")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    all_rows: List[dict] = []
    for m_rx in [int(x) for x in args.m_list.split(",")]:
        cfg = build_cfg(
            args.area, args.rcs, args.seed, m_rx,
            args.m, args.q, args.covariance_protection,
            args.receiver_compare, args.adaptive_risk_slack,
            args.belief_error_in_cres,
        )
        print("m_rx = %d : %d trials x %d realisations"
              % (m_rx, args.trials, args.realisations), flush=True)
        all_rows.extend(run(cfg, args, m_rx))

    path = os.path.join(args.out, "detector_evidence.csv")
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
