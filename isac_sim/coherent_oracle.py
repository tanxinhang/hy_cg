"""Phase-aware coherent-illumination oracle for active sensing.

The released detector uses a zero-mean Swerling-II energy model and therefore
contains no stable target phase that an operational coherent combiner could
track.  This module deliberately does not alter that canonical model.  It adds
an optimistic, explicitly labelled counterfactual: transmitters observed at
the same receiver, with the same look count and refinement mode, are assumed
to be waveform-, clock-, delay-, Doppler-, and phase-aligned before the energy
detector.  It is useful as an upper-bound diagnostic, not as deployable evidence.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .active_information import (
    ActiveDetectionResult,
    ActiveObservation,
    active_observation_gammas,
    received_information,
)
from .config import Config, validate_config
from .model import BaseGains, LinkTables
from .reporting import ReportingPlan, report_chi


@dataclass(frozen=True)
class CoherentOracleInformation:
    """Scenario information before and after ideal coherent illumination."""

    scenario_direct_information: tuple[float, ...]
    scenario_coherent_information: tuple[float, ...]

    @property
    def robust_direct_information(self) -> float:
        return float(min(self.scenario_direct_information, default=0.0))

    @property
    def robust_coherent_information(self) -> float:
        return float(min(self.scenario_coherent_information, default=0.0))

    @property
    def robust_gain(self) -> float:
        return self.robust_coherent_information - self.robust_direct_information


def _coherent_groups(
    observations: Sequence[ActiveObservation],
) -> tuple[tuple[int, ...], ...]:
    groups: dict[tuple[int, int, bool], list[int]] = {}
    for index, observation in enumerate(observations):
        key = (
            int(observation.link[1]),
            int(observation.mode.looks),
            bool(observation.mode.refined),
        )
        groups.setdefault(key, []).append(index)
    return tuple(tuple(indices) for indices in groups.values())


def coherent_group_gamma(
    gammas: np.ndarray, phase_error_std_rad: float = 0.0
) -> np.ndarray:
    """Return effective SINR for ideal same-receiver coherent illumination.

    Independent zero-mean Gaussian phase errors with standard deviation
    ``sigma`` attenuate each expected pairwise field cross-term by
    ``exp(-sigma**2)``.  Infinite phase uncertainty therefore approaches the
    incoherent sum rather than inventing a coherent gain.
    """
    values = np.asarray(gammas, dtype=float)
    sigma = float(phase_error_std_rad)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("gammas must contain at least one coherent branch")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("phase_error_std_rad must be finite and non-negative")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("gammas must be finite and non-negative")
    incoherent = np.sum(values, axis=-1)
    field_sum = np.sum(np.sqrt(values), axis=-1)
    cross_terms = np.maximum(field_sum * field_sum - incoherent, 0.0)
    return incoherent + np.exp(-(sigma ** 2)) * cross_terms


def coherent_oracle_information(
    cfg: Config,
    base: BaseGains,
    coarse_tables: LinkTables,
    refined_tables: LinkTables,
    q: int,
    fusion: int,
    observations: Sequence[ActiveObservation],
    *,
    transmitter_reference_scales: np.ndarray | None = None,
    phase_error_std_rad: float = 0.0,
) -> CoherentOracleInformation:
    """Compare direct exact-LLR evidence with the coherent oracle evidence."""
    validate_config(cfg)
    chosen = tuple(observations)
    scenario_count = (
        len(cfg.active_sensing.aspect_angles_deg)
        if cfg.active_sensing.aspect_enable else 1
    )
    if not chosen:
        zeros = (0.0,) * scenario_count
        return CoherentOracleInformation(zeros, zeros)
    gammas = active_observation_gammas(
        cfg, base, coarse_tables, refined_tables, q, chosen,
        transmitter_reference_scales,
    )
    plan = ReportingPlan(
        mode="explicit", f_q=np.full(cfg.scale.Q, int(fusion), dtype=int)
    )
    direct = np.zeros(gammas.shape[0], dtype=float)
    coherent = np.zeros(gammas.shape[0], dtype=float)
    for group in _coherent_groups(chosen):
        representative = chosen[group[0]]
        chi = float(np.clip(
            report_chi(coarse_tables, plan, representative.link, q), 0.0, 1.0
        ))
        looks = int(representative.mode.looks)
        for index in group:
            direct += received_information(
                gammas[:, index], looks, chi,
                cfg.active_sensing.information_metric,
            )
        effective = coherent_group_gamma(
            gammas[:, group], phase_error_std_rad=phase_error_std_rad
        )
        coherent += received_information(
            effective, looks, chi, cfg.active_sensing.information_metric
        )
    return CoherentOracleInformation(
        tuple(float(value) for value in direct),
        tuple(float(value) for value in coherent),
    )


def evaluate_coherent_tx_detection_oracle(
    cfg: Config,
    base: BaseGains,
    coarse_tables: LinkTables,
    refined_tables: LinkTables,
    q: int,
    fusion: int,
    observations: Sequence[ActiveObservation],
    *,
    calibration_samples: int | None = None,
    evaluation_samples: int | None = None,
    seed: int = 0xC0E2E17,
    transmitter_reference_scales: np.ndarray | None = None,
    phase_error_std_rad: float = 0.0,
    transport_mode: str = "receiver_local_llr",
) -> ActiveDetectionResult:
    """Monte Carlo operating point of the same-receiver coherent oracle."""
    validate_config(cfg)
    if transport_mode not in {"direct_llr", "receiver_local_llr"}:
        raise ValueError(
            "transport_mode must be 'direct_llr' or 'receiver_local_llr'"
        )
    if cfg.detect.comm_error_model != "erasure":
        raise ValueError("coherent oracle requires detect.comm_error_model='erasure'")
    n_cal = int(calibration_samples or cfg.detect.fused_calibration_samples)
    n_eval = int(evaluation_samples or n_cal)
    if n_cal < 1 or n_eval < 1:
        raise ValueError("calibration and evaluation sample counts must be positive")
    chosen = tuple(observations)
    scenario_count = (
        len(cfg.active_sensing.aspect_angles_deg)
        if cfg.active_sensing.aspect_enable else 1
    )
    if not chosen:
        return ActiveDetectionResult(
            (0.0,) * scenario_count,
            (float(cfg.detect.Pfa_target),) * scenario_count,
            (0.0,) * scenario_count,
        )
    gammas = active_observation_gammas(
        cfg, base, coarse_tables, refined_tables, q, chosen,
        transmitter_reference_scales,
    )
    groups = _coherent_groups(chosen)
    plan = ReportingPlan(
        mode="explicit", f_q=np.full(cfg.scale.Q, int(fusion), dtype=int)
    )
    pd_values: list[float] = []
    pfa_values: list[float] = []
    thresholds: list[float] = []
    for scenario in range(gammas.shape[0]):
        h0_cal = np.zeros(n_cal, dtype=float)
        h0_eval = np.zeros(n_eval, dtype=float)
        h1_eval = np.zeros(n_eval, dtype=float)
        receiver_erasure_masks: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        for group_number, group in enumerate(groups):
            representative = chosen[group[0]]
            effective_gamma = float(coherent_group_gamma(
                gammas[scenario, group], phase_error_std_rad
            ))
            looks = int(representative.mode.looks)
            coefficient = effective_gamma / (1.0 + effective_gamma)
            offset = -looks * np.log1p(effective_gamma)
            chi = float(np.clip(
                report_chi(coarse_tables, plan, representative.link, q), 0.0, 1.0
            ))
            signature = "|".join(
                f"{chosen[index].link[0]}:{chosen[index].link[1]}:"
                f"{chosen[index].mode.power_scale:.12g}"
                for index in group
            )
            group_key = zlib.crc32(signature.encode("utf-8"))
            statistic_rng = np.random.default_rng([
                int(seed), int(q), int(scenario), int(group_number),
                int(group_key), 0xC0E2E17,
            ])
            receiver = int(representative.link[1])
            if transport_mode == "receiver_local_llr":
                if receiver not in receiver_erasure_masks:
                    erasure_rng = np.random.default_rng([
                        int(seed), int(q), int(scenario), receiver, 0x10CA11A6,
                    ])
                    receiver_erasure_masks[receiver] = (
                        erasure_rng.random(n_cal) < chi,
                        erasure_rng.random(n_eval) < chi,
                        erasure_rng.random(n_eval) < chi,
                    )
                cal_arrives, h0_arrives, h1_arrives = (
                    receiver_erasure_masks[receiver]
                )
            else:
                erasure_rng = np.random.default_rng([
                    int(seed), int(q), int(scenario), receiver,
                    int(group_key), 0xE2A5E2A5,
                ])
                cal_arrives = erasure_rng.random(n_cal) < chi
                h0_arrives = erasure_rng.random(n_eval) < chi
                h1_arrives = erasure_rng.random(n_eval) < chi
            cal = offset + coefficient * statistic_rng.gamma(looks, 1.0, n_cal)
            h0 = offset + coefficient * statistic_rng.gamma(looks, 1.0, n_eval)
            h1 = offset + coefficient * statistic_rng.gamma(
                looks, 1.0 + effective_gamma, n_eval
            )
            h0_cal += np.where(cal_arrives, cal, 0.0)
            h0_eval += np.where(h0_arrives, h0, 0.0)
            h1_eval += np.where(h1_arrives, h1, 0.0)
        threshold = float(np.quantile(
            h0_cal, 1.0 - cfg.detect.Pfa_target, method="higher"
        ))
        thresholds.append(threshold)
        pfa_values.append(float(np.mean(h0_eval > threshold)))
        pd_values.append(float(np.mean(h1_eval > threshold)))
    return ActiveDetectionResult(
        tuple(pd_values), tuple(pfa_values), tuple(thresholds)
    )
