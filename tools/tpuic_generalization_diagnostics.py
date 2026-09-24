"""Layered diagnostics for the receiver-target generalization experiment."""
from __future__ import annotations

import time

import numpy as np

from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl


def _score(cfg, obs, target, arm="tp_uic_full"):
    results = cx.cancellation_arms(cfg, obs, weak_target=target)
    result = results[arm]
    model = gl.residual_model(cfg, obs, arm, results)
    detector = gl.target_neighbourhood_glrt(
        cfg, obs, result, model, target=target, aggregation="max")
    return result, detector


def _dd_rmse(estimated, truth):
    if not estimated:
        return 0.0, 0.0
    delay = [(float(a.delay_bin) - float(b.delay_bin)) ** 2
             for a, b in zip(estimated, truth)]
    doppler = [(float(a.doppler_bin) - float(b.doppler_bin)) ** 2
               for a, b in zip(estimated, truth)]
    return float(np.sqrt(np.mean(delay))), float(np.sqrt(np.mean(doppler)))


def _summary(results, detectors, elapsed):
    mean = lambda values: float(np.mean(np.asarray(values, dtype=float)))
    return {
        "statistic": float(sum(item.statistic for item in detectors)),
        "raw_statistic": float(sum(item.raw_statistic for item in detectors)),
        "whitened_residual_power": mean(
            [item.whitened_residual_power for item in detectors]),
        "glrt_residual_power": mean([item.residual_power for item in detectors]),
        "rho_weighted": mean([item.rho_weighted for item in detectors]),
        "xi_rel_q": mean([item.xi_rel_q for item in detectors]),
        "i_res": mean([item.i_res for item in results]),
        "i_res_structural": mean([item.i_res_structural for item in results]),
        "i_res_pred": mean([item.i_res_pred for item in results]),
        "eta_survive_q": mean([item.eta_survive_q for item in results]),
        "eta_survive_risk_q": mean([item.eta_survive_risk_q for item in results]),
        "elapsed_seconds": float(elapsed),
    }


def score_looks(cfg, observations, target, arm="tp_uic_full"):
    started = time.perf_counter()
    pairs = [_score(cfg, obs, target, arm) for obs in observations]
    return _summary([item[0] for item in pairs], [item[1] for item in pairs],
                    time.perf_counter() - started)


def crossfit_gn(cfg, observations, target, max_nfev):
    started = time.perf_counter()
    results, detectors, optimizer = [], [], []
    initial_delay, initial_doppler = [], []
    final_delay, final_doppler = [], []
    for held_index in range(2):
        reference = observations[1 - held_index]
        sources, diagnostic = cx.refine_direct_dd_joint_gn_diagnostics(
            cfg, [reference], max_nfev=max_nfev)
        held = cx.apply_direct_dd(cfg, observations[held_index], sources)
        result, detector = _score(cfg, held, target)
        results.append(result); detectors.append(detector); optimizer.append(diagnostic)
        before = _dd_rmse(reference.direct_est or [], reference.direct)
        after = _dd_rmse(sources, reference.direct)
        initial_delay.append(before[0]); initial_doppler.append(before[1])
        final_delay.append(after[0]); final_doppler.append(after[1])
    out = _summary(results, detectors, time.perf_counter() - started)
    mean = lambda values: float(np.mean(np.asarray(values, dtype=float)))
    out.update({
        "gn_initial_cost": mean([item.initial_cost for item in optimizer]),
        "gn_final_cost": mean([item.final_cost for item in optimizer]),
        "gn_nfev": mean([item.nfev for item in optimizer]),
        "gn_optimality": mean([item.optimality for item in optimizer]),
        "gn_active_bound_count": mean(
            [item.active_bound_count for item in optimizer]),
        "oracle_initial_delay_rmse": mean(initial_delay),
        "oracle_final_delay_rmse": mean(final_delay),
        "oracle_initial_doppler_rmse": mean(initial_doppler),
        "oracle_final_doppler_rmse": mean(final_doppler),
    })
    return out


def observation_diagnostics(cfg, observation, target):
    protected = cx.protected_target_ids(cfg, observation.targets_belief or [])
    basis = observation.basis_belief
    return {
        "tested_target_protected": int(target in protected),
        "protected_targets": ";".join(str(value) for value in sorted(protected)),
        "protection_rank": int(basis.shape[1] if basis is not None else 0),
        "direct_dictionary_condition": float(np.linalg.cond(observation.X)),
    }


def prefix(values, label):
    return {f"{label}_{key}": value for key, value in values.items()}


__all__ = ["crossfit_gn", "observation_diagnostics", "prefix", "score_looks"]
