"""End-to-end matrix-information / exact-z / fixed-window audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, ncx2

from isac_sim.cooperation.matrix_information_ao import (
    monotone_matrix_information_ao,
    protection_subsets,
)
from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.scenario.belief import BeliefState
from isac_sim.sensing.model import (
    build_base_gains,
    generate_geometry,
    radar_hardware_gain,
)


def _pd(information: np.ndarray, dof: np.ndarray, frames: int, p_fa: float) -> np.ndarray:
    df = np.asarray(dof, dtype=float) * int(frames)
    ncp = np.asarray(information, dtype=float) * int(frames)
    threshold = chi2.ppf(1.0 - float(p_fa), df)
    return np.asarray(ncx2.sf(threshold, df, ncp), dtype=float)


def run(trial: int = 32, p_fa: float = 0.05, area_xy: float = 600.0) -> dict:
    cfg = apply_preset(Config(), "small-uav-compact-800m")
    cfg = apply_overrides(cfg, {
        "geometry.area_xy": float(area_xy),
        "detect.target_rcs": 0.1,
        "scale.M": 6,
        "scale.Q": 3,
        "run.seed": 2026,
        "run.verbose": False,
        "cancellation.enable": True,
        "cancellation.belief_error_in_cres": True,
    })
    rng = np.random.default_rng([cfg.run.seed, int(trial)])
    truth = generate_geometry(cfg, rng)
    base_truth = build_base_gains(cfg, truth, rng)
    belief_state = BeliefState.from_truth(cfg, truth, rng)
    belief = belief_state.as_geometry(truth)
    base_belief = build_base_gains(
        cfg, belief, rng, channel=base_truth,
        rcs_view=cfg.prior.scheduler_rcs.lower(),
    )
    m, q = int(cfg.scale.M), int(cfg.scale.Q)
    active = tuple(True for _ in range(m))
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    gain = float(radar_hardware_gain(cfg))
    pg = float(cfg.waveform.N * cfg.waveform.L)
    targets = tuple(range(q))
    cache: dict[tuple[int, frozenset[int]], tuple[np.ndarray, np.ndarray, int]] = {}

    def receiver_case(receiver: int, protected: frozenset[int]):
        key = (int(receiver), frozenset(protected))
        if key in cache:
            return cache[key]
        obs = cx.build_observation(
            cfg, truth, belief, base_truth, int(receiver),
            rng=np.random.default_rng([cfg.run.seed, int(trial), 7000 + int(receiver)]),
            sense_power=sense, radiated_power=sense,
            processing_gain=pg, hw_gain=gain,
            active_mask=np.ones(m, dtype=bool), base_belief=base_belief,
            protected_targets=frozenset(protected),
        )
        arms = cx.cancellation_arms(cfg, obs, only="tp_uic_full")
        result = arms["tp_uic_full"]
        model = gl.residual_model(cfg, obs, "tp_uic_full", arms)
        infos, dofs = [], []
        for target in targets:
            item = gl.detection_information(cfg, obs, result, model, target)
            infos.append(item.value)
            dofs.append(item.dof_real)
        value = (np.asarray(infos), np.asarray(dofs, dtype=int), int(result.protect_dim))
        cache[key] = value
        return value

    # The old rule protects all Q targets here (max_protected_targets == Q).
    baseline_z = tuple(frozenset(targets) for _ in range(m))
    rank_budget = tuple(receiver_case(j, baseline_z[j])[2] for j in range(m))
    candidates = []
    for j in range(m):
        rank_lookup = {
            subset: receiver_case(j, subset)[2]
            for size in range(q + 1)
            for subset_tuple in __import__("itertools").combinations(targets, size)
            for subset in (frozenset(subset_tuple),)
        }
        candidates.append(protection_subsets(targets, rank_lookup.__getitem__, rank_budget[j]))

    def evaluate(_active, protection):
        return sum((receiver_case(j, protection[j])[0] for j in range(m)), np.zeros(q))

    baseline_info = evaluate(active, baseline_z)
    baseline_dof = sum((receiver_case(j, baseline_z[j])[1] for j in range(m)), np.zeros(q, dtype=int))
    result = monotone_matrix_information_ao(
        active, baseline_z, (active,), candidates, evaluate, max_rounds=20
    )
    optimized_dof = sum(
        (receiver_case(j, result.protection[j])[1] for j in range(m)),
        np.zeros(q, dtype=int),
    )
    frames = (1, 2, 4, 8, 16, 64, 256, 1024, 4096)
    optimized_pd = {
        str(k): _pd(result.information, optimized_dof, k, p_fa).tolist()
        for k in frames
    }
    first_k_08 = None
    for k in range(1, 10001):
        if float(np.min(_pd(result.information, optimized_dof, k, p_fa))) >= 0.8:
            first_k_08 = k
            break
    output = {
        "trial": int(trial),
        "area_xy_m": float(area_xy),
        "p_fa": float(p_fa),
        "rank_budget": list(rank_budget),
        "baseline_protection": [sorted(v) for v in baseline_z],
        "optimized_protection": [sorted(v) for v in result.protection],
        "baseline_information": baseline_info.tolist(),
        "optimized_information": result.information.tolist(),
        "baseline_dof": baseline_dof.tolist(),
        "optimized_dof": optimized_dof.tolist(),
        "information_gain_ratio": (
            result.information / np.maximum(baseline_info, 1e-30)
        ).tolist(),
        "ao_history": list(result.history),
        "baseline_pd": {str(k): _pd(baseline_info, baseline_dof, k, p_fa).tolist() for k in frames},
        "optimized_pd": optimized_pd,
        "optimized_fixed_total_energy_pd": {
            str(k): _pd(result.information / int(k), optimized_dof, k, p_fa).tolist()
            for k in frames
        },
        "minimum_frames_for_worst_pd_0.8_fixed_per_frame_energy": first_k_08,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=32)
    parser.add_argument("--p-fa", type=float, default=0.05)
    parser.add_argument("--area-xy", type=float, default=600.0)
    parser.add_argument(
        "--out",
        default="studies/direction3/data/matrix_information_chain_trial32/result.json",
    )
    args = parser.parse_args()
    output = run(args.trial, args.p_fa, args.area_xy)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
