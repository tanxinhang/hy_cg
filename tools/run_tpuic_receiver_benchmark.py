#!/usr/bin/env python3
"""
Receiver-only TP-UIC benchmark for hy_cg.

Purpose
-------
This experiment isolates Innovation I (receiver-side interference cancellation).
It deliberately DOES NOT run:
  * power allocation
  * UAV formation / trajectory optimization
  * reporting-link selection
  * fusion-node selection
  * cooperative LLR fusion

For every generated scene, receiver, target, and Monte-Carlo realization it:
  1) builds a paired H1/H0 observation with the SAME geometry/direct field;
  2) runs the same observation through multiple cancellation baselines;
  3) builds each arm's residual covariance model;
  4) evaluates the target-conditioned whitened GLRT statistic;
  5) calibrates the threshold using calibration-scene H0 samples only;
  6) evaluates held-out test-scene P_FA / P_D;
  7) reports cancellation depth, target survival, direct INR, and an
     interference-target subspace conflict index.

This is intended as the fixed experiment harness for:
  B0 no_ic
  B1 plain_ls
  B2 ridge_ls
  B3 protected_ls
  B4 tp_uic_stage1
  B5 tp_uic_full        (current TP-UIC V1)
  B6 perfect_channel    (oracle)

When TP-UIC V2 is implemented as a normal cancellation arm, add its arm name
to --arms; do NOT change the experiment protocol.

Example smoke run
-----------------
python tools/run_tpuic_receiver_benchmark.py \
    --out studies/direction3/data/tpuic_receiver_smoke \
    --cal-scenes 2 --test-scenes 2 --realisations 8 \
    --receivers 0 --targets 0 \
    --direct-gain-boost-db 0,20 \
    --direct-dd-sigma 0.10

Example formal small-system run
-------------------------------
python tools/run_tpuic_receiver_benchmark.py \
    --out studies/direction3/data/tpuic_receiver_v1 \
    --cal-scenes 50 --test-scenes 50 --realisations 64 \
    --receivers all --targets all \
    --target-rcs 0.05 --area-xy 800 \
    --direct-gain-boost-db 0,10,20,30,40 \
    --direct-dd-sigma 0.10 \
    --belief-pos-sigma 20 --belief-vel-sigma 3

Notes
-----
* `direct_gain_boost_db` scales ONLY the UAV-UAV direct gain in BaseGains.
  It does not scale target_gain, so it is an explicit direct-INR stress axis.
* The default target RCS is 0.05 m^2.
* The empirical detector threshold is NEVER calibrated on test scenes.
* Scene snapshots are saved to <out>/scenes/*.npz.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata

from isac_sim.cooperation.scientific_validation import save_frozen_scenario
from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.scenario.belief import BeliefState
from isac_sim.sensing.model import (
    build_base_gains,
    generate_geometry,
    radar_hardware_gain,
)


# Default arm set: five arms with five distinct behaviours.
#
# Measured on the smoke run (paired relative difference of the detection
# statistic within the same scene/realisation/boost):
#   ridge_ls        == plain_ls        (median rel. diff 9.8e-07)
#   tp_uic_stage1   == protected_ls    (median rel. diff 4.5e-07)
#   no_ic           ~= plain_ls        (median rel. diff 7.2e-05)
# so the seven-arm list carries three redundant rows per table and three
# redundant GLRT evaluations per observation for no information gain.
#
# The two dropped arms are NOT deleted from the protocol: pass them explicitly
# via --arms to restore the full ablation.  Their redundancy is a result, not a
# definition -- `ridge_ls == plain_ls` means the interference prior is inert on
# this configuration, which is exactly why `tp_uic_stage1 == protected_ls`.
DEFAULT_ARMS = (
    "no_ic",
    "plain_ls",
    "protected_ls",
    "tp_uic_full",
    "perfect_channel",
)

DETECTION_AWARE_ARM = "detection_aware_tpuic"
DETECTION_AWARE_CANDIDATES = (
    "tp_uic_full",
    "protected_ls",
    "plain_ls",
)


def _parse_int_selection(value: str, count: int) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(range(count))
    out = tuple(sorted({int(v.strip()) for v in value.split(",") if v.strip()}))
    if not out:
        raise ValueError("empty index selection")
    if any(v < 0 or v >= count for v in out):
        raise ValueError(f"indices must lie in [0, {count - 1}]")
    return out


def _parse_float_grid(value: str) -> tuple[float, ...]:
    out = tuple(float(v.strip()) for v in value.split(",") if v.strip())
    if not out:
        raise ValueError("empty float grid")
    if not all(np.isfinite(v) for v in out):
        raise ValueError("grid values must be finite")
    return out


def _orth(A: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[1] == 0:
        return np.zeros((A.shape[0], 0), dtype=complex)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if not s.size:
        return np.zeros((A.shape[0], 0), dtype=complex)
    keep = s > rtol * max(float(s[0]), 1e-30)
    return U[:, keep]


def interference_target_conflict(obs, target: int, interference_block: int = 1) -> float:
    """
    Largest squared canonical correlation between the assumed direct-interference
    subspace span(X) and target-q belief subspace span(A_q).

        xi = sigma_max^2(U_I^H U_T) in [0, 1].

    xi ~ 0: easy separation.
    xi ~ 1: direct interference and target evidence are nearly aligned.
    """
    ids = np.asarray(obs.A_target_ids)
    sel = ids == int(target)
    if not np.any(sel) or obs.X.shape[1] == 0:
        return 0.0
    # Conflict is a property of the nominal physical direct manifold, not of
    # how many nuisance Jacobians a particular estimator chooses to append.
    # Otherwise the stress axis itself changes across the four-arm ablation.
    X_nominal = obs.X[:, ::max(int(interference_block), 1)]
    Ui = _orth(X_nominal)
    Ut = _orth(obs.A[:, sel])
    if Ui.shape[1] == 0 or Ut.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(Ui.conj().T @ Ut, compute_uv=False)
    return float(np.clip((s[0] ** 2) if s.size else 0.0, 0.0, 1.0))


def direct_inr_db(obs) -> float:
    noise_energy = float(obs.y.size) * float(obs.sigma2)
    direct_energy = float(np.vdot(obs.x_direct, obs.x_direct).real)
    if direct_energy <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(direct_energy / max(noise_energy, 1e-300))


def _depth_db(i_in: float, i_res: float) -> float:
    if i_in <= 0.0:
        return float("nan")
    if i_res <= 0.0:
        return float("inf")
    return 10.0 * math.log10(i_in / i_res)


def _wilson(success_rate: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = float(success_rate)
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(
        p * (1.0 - p) / n + z * z / (4.0 * n * n)
    ) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _auc(h0: Iterable[float], h1: Iterable[float]) -> float:
    x0 = np.asarray(tuple(h0), dtype=float)
    x1 = np.asarray(tuple(h1), dtype=float)
    if x0.size == 0 or x1.size == 0:
        return float("nan")
    values = np.concatenate([x0, x1])
    ranks = rankdata(values, method="average")
    r1 = float(np.sum(ranks[x0.size:]))
    return float((r1 - x1.size * (x1.size + 1) / 2.0) / (x0.size * x1.size))


def _json_float(v: float):
    v = float(v)
    if np.isfinite(v):
        return v
    if np.isnan(v):
        return None
    return "inf" if v > 0 else "-inf"


def _all_arms(cfg, obs) -> dict:
    """Assemble every cancellation arm ONCE per observation.

    Do NOT pass ``only=<arm>`` here: the assembly pruner in
    ``arms_basic.build_basic_arms`` does not build ``tp_uic_stage1`` under the
    default ``candidate_policy="protected_only"`` (it is only built when
    ``only is None`` or the legacy ``"statistic"`` policy is used), so a
    per-arm call raises KeyError.  ``only=None`` is also the numerically
    reference path, and it is what production is validated against.
    """
    return cx.cancellation_arms(cfg, obs, weak_target=int(obs.weak_index))


def _run_arm(cfg, obs, arm: str, results: dict, *, dictionary: str = "belief"):
    if arm not in results:
        raise KeyError(f"arm {arm!r} was not produced; available={sorted(results)}")
    result = results[arm]
    model = gl.residual_model(cfg, obs, arm, results, dictionary=dictionary)
    if str(cfg.cancellation.target_glrt_mode) == "neighbourhood_max":
        out = gl.target_neighbourhood_glrt(
            cfg, obs, result, model, target=int(obs.weak_index),
            p_fa=float(cfg.detect.Pfa_target), dictionary=dictionary,
        )
    else:
        out = gl.target_conditioned_glrt(
            cfg, obs, result, model,
            target=int(obs.weak_index),
            p_fa=float(cfg.detect.Pfa_target),
            dictionary=dictionary,
            centre_only=True,
        )
    return result, model, out


def _run_detection_aware(cfg, obs, results: dict):
    """Select the existing cancellation operator with maximum GLRT information.

    The score is ``ncp_unit = mu^H R_res^-1 mu`` after nuisance rejection.  It
    depends on the belief dictionary, cancellation transfer and residual
    covariance, but not on the realised H1/H0 statistic or truth echo.  Ties
    prefer TP-UIC full, then protected LS, then plain LS.
    """
    evaluated = []
    for candidate in DETECTION_AWARE_CANDIDATES:
        result, model, out = _run_arm(cfg, obs, candidate, results)
        evaluated.append((float(out.ncp_unit), candidate, result, model, out))
    best = max(evaluated, key=lambda item: item[0])
    return best[1], best[2], best[3], best[4]


def _make_cfg(args) -> Config:
    cfg = apply_preset(Config(), "paper-canonical")
    cfg = apply_overrides(cfg, {
        "geometry.area_xy": float(args.area_xy),
        "scale.M": int(args.uavs),
        "scale.Q": int(args.targets_count),
        "detect.target_rcs": float(args.target_rcs),
        "radio.P_default": float(args.total_power_w),
        "radio.rho": float(args.sensing_fraction),
        "run.seed": int(args.master_seed),
        "run.verbose": False,

        # Receiver-only experiment.
        "cancellation.enable": True,
        "cancellation.protect_targets": True,
        "cancellation.max_protected_targets": int(args.max_protected_targets),
        "cancellation.tangent_order": 1,
        "cancellation.belief_error_in_cres": True,
        "cancellation.interference_tangent_order": int(args.interference_tangent_order),
        "cancellation.interference_uncertainty_weighted": bool(
            args.interference_uncertainty_weighted
        ),
        "cancellation.direct_mismatch_in_cres": bool(args.direct_mismatch_in_cres),
        "cancellation.direct_mismatch_covariance_model": str(
            args.direct_mismatch_covariance_model
        ),
        "cancellation.direct_mismatch_covariance_scale": float(
            args.direct_mismatch_covariance_scale
        ),
        "cancellation.oracle_direct_residual_in_cres": bool(
            args.oracle_direct_residual_in_cres
        ),
        "cancellation.target_glrt_mode": str(args.target_glrt_mode),
        "cancellation.target_glrt_radius_bins": float(args.target_glrt_radius_bins),
        "cancellation.target_glrt_grid_points": int(args.target_glrt_grid_points),
        "cancellation.target_statistic_normalization": str(
            args.target_statistic_normalization
        ),
        "cancellation.direct_estimation_sigma_delay_bins": float(args.direct_dd_sigma),
        "cancellation.direct_estimation_sigma_doppler_bins": float(args.direct_dd_sigma),

        # Keep the current V1 hard receiver unchanged unless explicitly changed.
        "cancellation.covariance_protection": False,
        "cancellation.adaptive_soft_enable": False,

        # Physical target belief used by the receiver dictionary/protection.
        "prior.belief_mode": True,
        "prior.belief_sigma_pos_m": float(args.belief_pos_sigma),
        "prior.belief_sigma_vel_mps": float(args.belief_vel_sigma),

        # Keep the array dimension explicit.
        "aperture.enable": True,
        "aperture.m_rx": int(args.m_rx),
    })
    return cfg


def _scene(
    cfg: Config,
    scene_seed: int,
    snapshot_path: Path,
):
    rng = np.random.default_rng([int(cfg.run.seed), int(scene_seed), 101])
    truth = generate_geometry(cfg, rng)

    # Use a dedicated substream so geometry/channel draws do not move if the
    # belief model is changed later.
    belief_rng = np.random.default_rng([int(cfg.run.seed), int(scene_seed), 202])
    belief = BeliefState.from_truth(cfg, truth, belief_rng)

    base_rng = np.random.default_rng([int(cfg.run.seed), int(scene_seed), 303])
    base = build_base_gains(cfg, truth, base_rng)

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    save_frozen_scenario(snapshot_path, truth, belief, base)
    return truth, belief, base


def _boost_direct(base, boost_db: float):
    factor = 10.0 ** (float(boost_db) / 10.0)
    return replace(
        base,
        direct_gain=np.asarray(base.direct_gain, dtype=float) * factor,
    )


def _record_one(
    cfg,
    truth,
    belief,
    base,
    split: str,
    scene_id: int,
    receiver: int,
    target: int,
    realisation: int,
    boost_db: float,
    arms: tuple[str, ...],
    direct_error_scope: str,
    target_dictionary: str,
):
    geom_belief = belief.as_geometry(truth)

    m = int(cfg.scale.M)
    sense_power = np.full(
        m, float(cfg.radio.P_default) * float(cfg.radio.rho), dtype=float
    )
    active_mask = np.ones(m, dtype=bool)
    radiated_power = sense_power.copy()
    processing_gain = float(cfg.waveform.N * cfg.waveform.L)
    hw_gain = float(radar_hardware_gain(cfg))

    # IMPORTANT: seed excludes `arm`, therefore every arm gets the exact same
    # physical H1/H0 observation. It also excludes `boost_db`, so direct-gain
    # sweeps are paired in phases/noise/DD estimation error.
    obs_rng = np.random.default_rng([
        int(cfg.run.seed),
        int(scene_id),
        int(receiver),
        int(target),
        int(realisation),
        404,
    ])
    if direct_error_scope == "scene":
        direct_error_rng = np.random.default_rng([
            int(cfg.run.seed), int(scene_id), int(receiver), int(target), 405,
        ])
    elif direct_error_scope == "realisation":
        direct_error_rng = None
    else:
        raise ValueError(f"unknown direct-error scope {direct_error_scope!r}")
    obs1, obs0 = cx.build_observation_pair(
        cfg, truth, geom_belief, base, int(receiver),
        rng=obs_rng, direct_error_rng=direct_error_rng,
        sense_power=sense_power,
        radiated_power=radiated_power,
        processing_gain=processing_gain,
        hw_gain=hw_gain,
        exclude_target=int(target),
        active_mask=active_mask,
        weak_index=int(target),
        share_noise=False,
    )

    xi = interference_target_conflict(
        obs1, target,
        1 + 2 * int(cfg.cancellation.interference_tangent_order),
    )
    inr_db = direct_inr_db(obs1)

    # The protection budget selects targets by WEAKEST echo power
    # (``protection.protected_target_ids``), independently of which target the
    # detector is being scored on.  Recording whether the tested target is in
    # the protected set is the only way to interpret the protected arms: with
    # ``max_protected_targets=1`` the tested target is protected only when it
    # happens to be the weakest one.  This column is bookkeeping only -- it does
    # not feed any arm.
    protected_ids = frozenset(
        int(q) for q in cx.protected_target_ids(cfg, obs1.targets)
    )
    tested_protected = int(int(target) in protected_ids)

    rows = []
    results1 = _all_arms(cfg, obs1)
    results0 = _all_arms(cfg, obs0)
    for arm in arms:
        if arm == DETECTION_AWARE_ARM:
            selected1, r1, _m1, g1 = _run_detection_aware(cfg, obs1, results1)
            selected0, _r0, _m0, g0 = _run_detection_aware(cfg, obs0, results0)
            if selected1 != selected0:
                raise RuntimeError(
                    "detection-aware selection changed between paired H1/H0 "
                    f"observations ({selected1!r} != {selected0!r}); selection "
                    "must not depend on target presence"
                )
            selected_base_arm = selected1
        else:
            r1, _m1, g1 = _run_arm(
                cfg, obs1, arm, results1, dictionary=target_dictionary
            )
            _r0, _m0, g0 = _run_arm(
                cfg, obs0, arm, results0, dictionary=target_dictionary
            )
            selected_base_arm = arm

        rows.append({
            "split": split,
            "scene_id": int(scene_id),
            "receiver": int(receiver),
            "target": int(target),
            "realisation": int(realisation),
            "direct_gain_boost_db": float(boost_db),
            "arm": arm,
            "selected_base_arm": selected_base_arm,

            "conflict_index": xi,
            "direct_inr_db": inr_db,
            "tested_protected": tested_protected,
            "n_protected_targets": len(protected_ids),

            "stat_h0": float(g0.statistic),
            "stat_h1": float(g1.statistic),
            "raw_stat_h0": float(g0.raw_statistic),
            "raw_stat_h1": float(g1.raw_statistic),
            "target_direction_inflation_h0": float(
                g0.raw_statistic / max(g0.dof_real / 2.0, 1.0)
            ),
            "whitened_residual_power_h0": float(g0.whitened_residual_power),
            "whitened_residual_ratio_h0": float(
                g0.whitened_residual_power / max(obs0.y.size, 1)
            ),
            "dof_real": int(g1.dof_real),
            "identifiable_fraction": float(g1.rho_weighted),
            "ncp_unit": float(g1.ncp_unit),
            "glrt_neighbourhood_size": int(g1.neighbourhood_size),
            "glrt_selected_offset_delay": float(g1.selected_offset_delay),
            "glrt_selected_offset_doppler": float(g1.selected_offset_doppler),

            "kappa_accounted_db": float(r1.kappa_db),
            "kappa_structural_db": _depth_db(
                float(r1.i_in), float(r1.i_res_structural)
            ),
            "i_in": float(r1.i_in),
            "i_res": float(r1.i_res),
            "i_res_structural": float(r1.i_res_structural),
            "i_res_estimate": float(r1.i_res_estimate),

            "eta_survive_q": float(r1.eta_survive_q),
            "eta_survive_field": float(r1.eta_survive),
            "eta_survive_pred_q": float(r1.eta_survive_pred_q),
            "eta_survive_risk_q": float(r1.eta_survive_risk_q),
            "eta_protect": float(r1.eta_protect),
            "s_q_energy": float(r1.s_q_energy),
            "post_ic_sinr": float(
                r1.eta_survive_q * r1.s_q_energy
                / max(r1.i_res + float(obs1.y.size) * float(obs1.sigma2), 1e-300)
            ),

            "protect_dim": int(r1.protect_dim),
            "n_coefficients": int(r1.n_coefficients),
            "calibration_error_db": float(r1.calibration_error_db),
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _calibrated_threshold(values: np.ndarray, p_fa: float, mode: str) -> float:
    if mode == "empirical_quantile":
        return float(np.quantile(values, 1.0 - float(p_fa), method="higher"))
    if mode != "split_conformal":
        raise ValueError(f"unknown threshold calibration mode {mode!r}")
    n = int(values.size)
    rank = int(math.ceil((n + 1) * (1.0 - float(p_fa))))
    if rank > n:
        return float("inf")
    return float(np.sort(values)[rank - 1])


def _summarise(rows: list[dict], p_fa: float, threshold_calibration: str):
    calibration = defaultdict(list)
    test = defaultdict(list)

    key_fields = ("direct_gain_boost_db", "receiver", "target", "arm")
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        (calibration if row["split"] == "calibration" else test)[key].append(row)

    summaries = []
    scene_rows = []

    for key, test_rows in sorted(test.items()):
        cal_rows = calibration.get(key, [])
        if not cal_rows:
            continue

        cal_h0 = np.asarray([r["stat_h0"] for r in cal_rows], dtype=float)
        threshold = _calibrated_threshold(cal_h0, p_fa, threshold_calibration)

        h0 = np.asarray([r["stat_h0"] for r in test_rows], dtype=float)
        h1 = np.asarray([r["stat_h1"] for r in test_rows], dtype=float)
        pfa = float(np.mean(h0 > threshold))
        pd = float(np.mean(h1 > threshold))
        pd_lo, pd_hi = _wilson(pd, h1.size)
        pfa_lo, pfa_hi = _wilson(pfa, h0.size)

        boost, receiver, target, arm = key

        def med(name):
            x = np.asarray([r[name] for r in test_rows], dtype=float)
            finite = x[np.isfinite(x)]
            return float(np.median(finite)) if finite.size else float("nan")

        summaries.append({
            "direct_gain_boost_db": float(boost),
            "receiver": int(receiver),
            "target": int(target),
            "arm": str(arm),
            "threshold": threshold,
            "empirical_pfa": pfa,
            "pfa_ci95_low": pfa_lo,
            "pfa_ci95_high": pfa_hi,
            "empirical_pd": pd,
            "pd_ci95_low": pd_lo,
            "pd_ci95_high": pd_hi,
            "auc": _auc(h0, h1),
            "n_cal_h0": int(cal_h0.size),
            "n_test_h0": int(h0.size),
            "n_test_h1": int(h1.size),
            "median_conflict_index": med("conflict_index"),
            "median_direct_inr_db": med("direct_inr_db"),
            "median_kappa_accounted_db": med("kappa_accounted_db"),
            "median_kappa_structural_db": med("kappa_structural_db"),
            "median_eta_survive_q": med("eta_survive_q"),
            "median_eta_survive_risk_q": med("eta_survive_risk_q"),
            "median_identifiable_fraction": med("identifiable_fraction"),
            "median_protect_dim": med("protect_dim"),
        })

        by_scene = defaultdict(list)
        for row in test_rows:
            by_scene[int(row["scene_id"])].append(row)
        for scene_id, items in sorted(by_scene.items()):
            s0 = np.asarray([r["stat_h0"] for r in items], dtype=float)
            s1 = np.asarray([r["stat_h1"] for r in items], dtype=float)
            scene_rows.append({
                "direct_gain_boost_db": float(boost),
                "receiver": int(receiver),
                "target": int(target),
                "arm": str(arm),
                "scene_id": int(scene_id),
                "threshold": threshold,
                "scene_pfa": float(np.mean(s0 > threshold)),
                "scene_pd": float(np.mean(s1 > threshold)),
                "scene_conflict_index": float(np.median(
                    [r["conflict_index"] for r in items]
                )),
                "scene_direct_inr_db": float(np.median(
                    [r["direct_inr_db"] for r in items]
                )),
            })

    return summaries, scene_rows


def run(args) -> dict:
    cfg = _make_cfg(args)
    receivers = _parse_int_selection(args.receivers, int(cfg.scale.M))
    targets = _parse_int_selection(args.targets, int(cfg.scale.Q))
    boosts = _parse_float_grid(args.direct_gain_boost_db)

    arms = tuple(v.strip() for v in args.arms.split(",") if v.strip())
    if not arms:
        raise ValueError("at least one arm is required")
    if args.target_dictionary == "truth" and DETECTION_AWARE_ARM in arms:
        raise ValueError("truth-template oracle cannot use detection-aware arm selection")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total_scenes = int(args.cal_scenes) + int(args.test_scenes)

    for scene_id in range(total_scenes):
        split = "calibration" if scene_id < int(args.cal_scenes) else "test"
        truth, belief, base0 = _scene(
            cfg, scene_id, out_dir / "scenes" / f"scene_{scene_id:05d}.npz"
        )

        for boost_db in boosts:
            base = _boost_direct(base0, boost_db)
            for receiver in receivers:
                for target in targets:
                    for realisation in range(int(args.realisations)):
                        rows.extend(_record_one(
                            cfg, truth, belief, base,
                            split=split,
                            scene_id=scene_id,
                            receiver=receiver,
                            target=target,
                            realisation=realisation,
                            boost_db=boost_db,
                            arms=arms,
                            direct_error_scope=str(args.direct_error_scope),
                            target_dictionary=str(args.target_dictionary),
                        ))

    summaries, scene_rows = _summarise(
        rows, float(args.p_fa), str(args.threshold_calibration)
    )
    _write_csv(out_dir / "records.csv", rows)
    _write_csv(out_dir / "summary.csv", summaries)
    _write_csv(out_dir / "scene_summary.csv", scene_rows)

    manifest = {
        "protocol": "tpuic_receiver_only_empirical_roc_v1",
        "purpose": "Innovation-I receiver isolation; no network optimization",
        "observation_pairing_scope": "one receiver-target pair",
        "global_no_target_h0_available": False,
        "multi_target_joint_alarm_compatible": False,
        "multi_target_warning": (
            "Each target q uses a separately seeded observation and an H0 "
            "that excludes only q while retaining other targets. Rows with "
            "different target values must not be combined as one physical "
            "multi-target global-null observation."
        ),
        "master_seed": int(args.master_seed),
        "uavs": int(args.uavs),
        "targets_count": int(args.targets_count),
        "receivers": list(receivers),
        "targets": list(targets),
        "arms": list(arms),
        "calibration_scenes": int(args.cal_scenes),
        "test_scenes": int(args.test_scenes),
        "realisations_per_scene_target_receiver": int(args.realisations),
        "p_fa": float(args.p_fa),
        "threshold_calibration": str(args.threshold_calibration),
        "direct_error_scope": str(args.direct_error_scope),
        "target_rcs_m2": float(args.target_rcs),
        "area_xy_m": float(args.area_xy),
        "m_rx": int(args.m_rx),
        "total_power_w_per_uav": float(args.total_power_w),
        "sensing_fraction": float(args.sensing_fraction),
        "sensing_power_w_per_uav": float(
            args.total_power_w * args.sensing_fraction
        ),
        "direct_gain_boost_db_grid": list(boosts),
        "direct_estimation_sigma_delay_bins": float(args.direct_dd_sigma),
        "direct_estimation_sigma_doppler_bins": float(args.direct_dd_sigma),
        "interference_tangent_order": int(args.interference_tangent_order),
        "interference_uncertainty_weighted": bool(args.interference_uncertainty_weighted),
        "direct_mismatch_in_cres": bool(args.direct_mismatch_in_cres),
        "direct_mismatch_covariance_model": str(
            args.direct_mismatch_covariance_model
        ),
        "direct_mismatch_covariance_scale": float(
            args.direct_mismatch_covariance_scale
        ),
        "oracle_direct_residual_in_cres": bool(args.oracle_direct_residual_in_cres),
        "target_glrt_mode": str(args.target_glrt_mode),
        "target_dictionary": str(args.target_dictionary),
        "target_glrt_radius_bins": float(args.target_glrt_radius_bins),
        "target_glrt_grid_points": int(args.target_glrt_grid_points),
        "target_statistic_normalization": str(args.target_statistic_normalization),
        "belief_sigma_pos_m": float(args.belief_pos_sigma),
        "belief_sigma_vel_mps": float(args.belief_vel_sigma),
        "max_protected_targets": int(args.max_protected_targets),
        "fixed_network_variables": {
            "power_optimization": False,
            "formation_optimization": False,
            "reporting_optimization": False,
            "fusion_optimization": False,
            "active_mask": "all UAVs active",
        },
        "primary_endpoints": [
            "empirical_pd_at_empirically_calibrated_pfa",
            "eta_survive_q",
            "kappa_structural_db",
        ],
        "secondary_endpoints": [
            "auc",
            "identifiable_fraction",
            "kappa_accounted_db",
            "residual_covariance_calibration_error_db",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Console table: deliberately compact.
    print("\nTP-UIC receiver-only benchmark")
    print("=" * 108)
    print(
        f"{'boost':>7} {'rx':>3} {'q':>3} {'arm':<20} "
        f"{'PFA':>7} {'PD':>7} {'AUC':>7} {'kappa_s':>9} {'eta_q':>8} {'xi':>7}"
    )
    for s in summaries:
        print(
            f"{s['direct_gain_boost_db']:7.1f} "
            f"{s['receiver']:3d} {s['target']:3d} "
            f"{s['arm']:<20} "
            f"{s['empirical_pfa']:7.3f} "
            f"{s['empirical_pd']:7.3f} "
            f"{s['auc']:7.3f} "
            f"{s['median_kappa_structural_db']:9.2f} "
            f"{s['median_eta_survive_q']:8.3f} "
            f"{s['median_conflict_index']:7.3f}"
        )
    print("=" * 108)
    print(f"records : {out_dir / 'records.csv'}")
    print(f"summary : {out_dir / 'summary.csv'}")
    print(f"scenes  : {out_dir / 'scene_summary.csv'}")
    print(f"manifest: {out_dir / 'manifest.json'}")

    # JSON-safe compact return.
    return {
        "manifest": manifest,
        "n_records": len(rows),
        "n_summary_rows": len(summaries),
        "summary": [
            {k: (_json_float(v) if isinstance(v, float) else v)
             for k, v in row.items()}
            for row in summaries
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--out", required=True)

    # Core scenario.
    p.add_argument("--master-seed", type=int, default=20260922)
    p.add_argument("--uavs", type=int, default=6)
    p.add_argument("--targets-count", type=int, default=3)
    p.add_argument("--area-xy", type=float, default=800.0)
    p.add_argument("--target-rcs", type=float, default=0.05)
    p.add_argument("--m-rx", type=int, default=4)

    # Fixed sensing power: this experiment is NOT a power-allocation experiment.
    p.add_argument("--total-power-w", type=float, default=1.0)
    p.add_argument("--sensing-fraction", type=float, default=0.8)

    # Receiver/target selection.
    p.add_argument("--receivers", default="0",
                   help="'all' or comma-separated zero-based receiver indices")
    p.add_argument("--targets", default="0",
                   help="'all' or comma-separated zero-based target indices")

    # Baselines.
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))

    # Calibration/test protocol.
    p.add_argument("--cal-scenes", type=int, default=4)
    p.add_argument("--test-scenes", type=int, default=4)
    p.add_argument("--realisations", type=int, default=16)
    p.add_argument("--p-fa", type=float, default=0.05)
    p.add_argument("--threshold-calibration",
                   choices=("empirical_quantile", "split_conformal"),
                   default="empirical_quantile")

    # Stress axes.
    p.add_argument(
        "--direct-gain-boost-db", default="0,10,20,30",
        help="comma-separated dB boost applied only to direct UAV-UAV gain",
    )
    p.add_argument(
        "--direct-dd-sigma", type=float, default=0.10,
        help="std. dev. of direct-path delay and Doppler estimation error [bins]",
    )
    p.add_argument("--direct-error-scope", choices=("realisation", "scene"),
                   default="realisation",
                   help="hold DD error fixed per scene or redraw per realisation")
    p.add_argument("--interference-tangent-order", type=int, choices=(0, 1), default=0)
    p.add_argument("--interference-uncertainty-weighted", action="store_true")
    p.add_argument("--no-direct-mismatch-in-cres", dest="direct_mismatch_in_cres",
                   action="store_false")
    p.set_defaults(direct_mismatch_in_cres=True)
    p.add_argument("--direct-mismatch-covariance-model",
                   choices=("first_order", "sigma_point"), default="first_order")
    p.add_argument("--direct-mismatch-covariance-scale", type=float, default=1.0)
    p.add_argument("--oracle-direct-residual-in-cres", action="store_true",
                   help="diagnostic truth-leaking rank-one C_res correction")
    p.add_argument("--target-glrt-mode", choices=("centre", "neighbourhood_max"),
                   default="centre")
    p.add_argument("--target-dictionary", choices=("belief", "truth"),
                   default="belief", help="truth is an oracle diagnostic only")
    p.add_argument("--target-glrt-radius-bins", type=float, default=0.0)
    p.add_argument("--target-glrt-grid-points", type=int, default=3)
    p.add_argument("--target-statistic-normalization",
                   choices=("none", "whitened_energy"), default="none")
    p.add_argument("--belief-pos-sigma", type=float, default=20.0)
    p.add_argument("--belief-vel-sigma", type=float, default=3.0)
    # Protection budget.  1 = protect ONE target (the weakest by echo power).
    #
    # Do not raise this to ``Q`` without reading the measurement below: the
    # budget is spent on targets, and Q=3 with budget 3 protects the entire
    # 45-column target dictionary, which costs cancellation depth for no
    # detection gain.  Measured at boost +40 dB, 18 (receiver,target) keys,
    # paired difference within the same key:
    #     budget 1:  Dkappa(protected_ls - plain_ls) = -0.03 dB  [-0.21, -0.00]
    #     budget 3:  Dkappa(protected_ls - plain_ls) = -1.67 dB  [-3.87, -0.54]
    # with DAUC = -0.003 in both cases.  Budget 0 means "protect everything"
    # and is numerically identical to budget 3 when Q=3.
    #
    # Caveat that no budget can fix: the budget picks targets by WEAKEST echo
    # power, not by which target is being scored.  Under budget 1 the tested
    # target is inside the protected set in only 33% of records -- that is why
    # this script records ``tested_protected``.
    p.add_argument("--max-protected-targets", type=int, default=1)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cal_scenes <= 0 or args.test_scenes <= 0:
        raise ValueError("cal-scenes and test-scenes must both be positive")
    if args.realisations <= 0:
        raise ValueError("realisations must be positive")
    if not 0.0 < args.p_fa < 1.0:
        raise ValueError("p-fa must lie in (0,1)")
    if args.target_rcs <= 0.0:
        raise ValueError("target-rcs must be positive")
    if not 0.0 < args.sensing_fraction <= 1.0:
        raise ValueError("sensing-fraction must lie in (0,1]")
    if args.direct_dd_sigma < 0.0:
        raise ValueError("direct-dd-sigma must be non-negative")

    result = run(args)
    print(json.dumps(
        {
            "protocol": result["manifest"]["protocol"],
            "n_records": result["n_records"],
            "n_summary_rows": result["n_summary_rows"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
