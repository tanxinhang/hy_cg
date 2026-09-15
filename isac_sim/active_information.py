"""Detector-consistent information for active observation design.

This module turns a passive bistatic link ``(i,j,q)`` into an active observation
``(i,j,q,m)`` by attaching a discrete sensing mode ``m``.  Each mode controls
power, independent looks and DD refinement.  A finite set of target-aspect
scenarios changes the RCS of different bistatic views in different directions.

The pricing objective is received hypothesis-separation information.  When a
packet-erasure indicator is observed at the fusion node and is independent of
the hypothesis, the received KL divergence is exactly ``chi * KL_local``.
Independent observations are therefore additive.  The branch-and-bound oracle
uses this modular quantity for a valid optimistic bound and returns an exact
certificate unless its declared node limit is reached.  Final publication
claims must still be evaluated with the true-erasure Monte Carlo detector.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import Config, Link, validate_config
from .llr import llr_jeffreys, llr_kld
from .model import BaseGains, LinkTables
from .reporting import ReportingPlan, is_local_observation, report_chi
from .selection import feasible_links_for_target


@dataclass(frozen=True)
class SensingMode:
    """One discrete acquisition mode for a bistatic observation."""

    name: str
    power_scale: float
    looks: int
    refined: bool

    @property
    def energy(self) -> float:
        """Normalized look-power energy consumed by one observation."""
        return float(self.power_scale * self.looks)


@dataclass(frozen=True)
class ActiveObservation:
    """A physical bistatic link paired with one acquisition mode."""

    link: Link
    mode: SensingMode


@dataclass(frozen=True)
class InformationPricingResult:
    """Certified solution of one target--fusion active-information problem."""

    target: int
    fusion: int
    observations: tuple[ActiveObservation, ...]
    scenario_information: tuple[float, ...]
    robust_information: float
    objective: float
    energy: float
    remote_reports: int
    explored_nodes: int
    exact: bool
    upper_bound: float
    total_feasible_links: int = 0
    retained_candidate_links: int = 0
    certificate_scope: str = "shortlist"
    candidate_links: tuple[Link, ...] = ()

    @property
    def certificate_gap(self) -> float:
        return float(max(self.upper_bound - self.objective, 0.0))


@dataclass(frozen=True)
class ActiveDetectionResult:
    """Scenario-wise operating point of a mixed-mode exact-LLR bundle."""

    scenario_pd: tuple[float, ...]
    scenario_pfa: tuple[float, ...]
    thresholds: tuple[float, ...]

    @property
    def worst_pd(self) -> float:
        return float(min(self.scenario_pd, default=0.0))

    @property
    def worst_pfa(self) -> float:
        return float(max(self.scenario_pfa, default=0.0))


def configured_sensing_modes(cfg: Config) -> tuple[SensingMode, ...]:
    """Materialize and validate the modes stored in ``cfg.active_sensing``."""
    validate_config(cfg)
    a = cfg.active_sensing
    return tuple(
        SensingMode(str(name), float(power), int(looks), bool(refined))
        for name, power, looks, refined in zip(
            a.mode_names, a.power_scales, a.looks, a.refined
        )
    )


def aspect_scenario_factors(
    cfg: Config, base: BaseGains, q: int, links: Sequence[Link]
) -> np.ndarray:
    """Return path-specific RCS multipliers for each declared aspect scenario.

    The bounded ``floor + (1-floor) cos^2(.)`` law is a transparent research
    abstraction, not a calibrated target signature.  Unlike the scalar V1.3
    interval, it can rank two bistatic views differently in two scenarios and
    therefore creates genuine multi-view diversity.
    """
    if not links:
        return np.zeros((1, 0), dtype=float)
    active = cfg.active_sensing
    if not active.aspect_enable:
        return np.ones((1, len(links)), dtype=float)
    view = np.asarray(
        [base.aspect_azimuth[i, j, q] for i, j in links], dtype=float
    )
    angles = np.deg2rad(np.asarray(active.aspect_angles_deg, dtype=float))
    floor = float(active.aspect_floor)
    return floor + (1.0 - floor) * np.cos(angles[:, None] - view[None, :]) ** 2


def received_information(
    gamma: np.ndarray | float,
    looks: int,
    chi: float,
    metric: str = "forward_kl",
) -> np.ndarray | float:
    """Information retained after an observed, hypothesis-independent erasure."""
    if not 0.0 <= chi <= 1.0:
        raise ValueError("packet success probability chi must lie in [0, 1]")
    if metric == "forward_kl":
        local = llr_kld(gamma, looks)
    elif metric == "jeffreys":
        local = llr_jeffreys(gamma, looks)
    else:
        raise ValueError(f"unknown information metric {metric!r}")
    out = float(chi) * np.asarray(local, dtype=float)
    return float(out) if np.ndim(gamma) == 0 else out


def observation_information_matrix(
    cfg: Config,
    base: BaseGains,
    coarse_tables: LinkTables,
    refined_tables: LinkTables,
    q: int,
    fusion: int,
    links: Sequence[Link],
    modes: Sequence[SensingMode] | None = None,
) -> np.ndarray:
    """Return information with shape ``(scenario, link, mode)``."""
    chosen_modes = tuple(modes) if modes is not None else configured_sensing_modes(cfg)
    factors = aspect_scenario_factors(cfg, base, q, links)
    plan = ReportingPlan(
        mode="explicit", f_q=np.full(cfg.scale.Q, int(fusion), dtype=int)
    )
    values = np.zeros((factors.shape[0], len(links), len(chosen_modes)), dtype=float)
    for link_index, link in enumerate(links):
        i, j = link
        chi = float(np.clip(report_chi(coarse_tables, plan, link, q), 0.0, 1.0))
        for mode_index, mode in enumerate(chosen_modes):
            source = refined_tables if mode.refined else coarse_tables
            nominal_gamma = max(float(source.gamma_sense[i, j, q]), 0.0)
            gamma = nominal_gamma * mode.power_scale * factors[:, link_index]
            values[:, link_index, mode_index] = received_information(
                gamma, mode.looks, chi, cfg.active_sensing.information_metric
            )
    return values


def active_candidate_links(
    cfg: Config,
    links: Sequence[Link],
    values: np.ndarray,
    *,
    strategy: str | None = None,
) -> tuple[list[Link], np.ndarray, str]:
    """Retain robust, scenario-strong, and complementary candidate families."""
    chosen_strategy = strategy or cfg.active_sensing.candidate_strategy
    if values.ndim != 3 or values.shape[1] != len(links) or values.shape[2] == 0:
        raise ValueError("candidate value matrix does not match physical links")
    if chosen_strategy == "full":
        if len(links) > cfg.active_sensing.complete_pool_max_links:
            raise ValueError(
                "complete candidate pool exceeds active_sensing.complete_pool_max_links"
            )
        return list(links), values, "complete_pool"

    best_by_scenario = np.max(values, axis=2)
    # A link selects one mode before the aspect scenario is known.
    robust = np.max(np.min(values, axis=0), axis=1)
    limit = min(int(cfg.active_sensing.max_candidates_per_pair), len(links))
    if chosen_strategy == "robust_singleton":
        keep = list(np.argsort(-robust, kind="stable")[:limit])
    elif chosen_strategy == "scenario_union":
        priority: list[int] = []
        scenario_k = int(cfg.active_sensing.scenario_topk_per_scenario)
        for scenario in range(values.shape[0]):
            priority.extend(
                int(index) for index in
                np.argsort(-best_by_scenario[scenario], kind="stable")[:scenario_k]
            )
        pair_scores: list[tuple[float, int, int]] = []
        for left in range(len(links)):
            for right in range(left + 1, len(links)):
                combined = max(
                    float(np.min(values[:, left, left_mode]
                                 + values[:, right, right_mode]))
                    for left_mode in range(values.shape[2])
                    for right_mode in range(values.shape[2])
                )
                complement = combined - max(float(robust[left]), float(robust[right]))
                pair_scores.append((complement, left, right))
        pair_scores.sort(key=lambda item: (-item[0], item[1], item[2]))
        for _, left, right in pair_scores[:cfg.active_sensing.complementary_pair_topk]:
            priority.extend((left, right))
        priority.extend(int(index) for index in np.argsort(-robust, kind="stable"))
        keep = list(dict.fromkeys(priority))[:limit]
    else:
        raise ValueError(f"unknown active candidate strategy {chosen_strategy!r}")
    index = np.asarray(keep, dtype=int)
    return [links[item] for item in keep], values[:, index, :], "shortlist_union"


def active_observation_gammas(
    cfg: Config,
    base: BaseGains,
    coarse_tables: LinkTables,
    refined_tables: LinkTables,
    q: int,
    observations: Sequence[ActiveObservation],
    transmitter_reference_scales: np.ndarray | None = None,
) -> np.ndarray:
    """Return scenario-by-observation SINR with each mode applied once."""
    links = [observation.link for observation in observations]
    factors = aspect_scenario_factors(cfg, base, q, links)
    gammas = np.zeros_like(factors)
    reference = (
        np.ones(cfg.scale.M, dtype=float)
        if transmitter_reference_scales is None
        else np.asarray(transmitter_reference_scales, dtype=float)
    )
    if (
        reference.shape != (cfg.scale.M,)
        or not np.all(np.isfinite(reference))
        or np.any(reference <= 0.0)
    ):
        raise ValueError("transmitter reference scales must be positive with shape (M,)")
    for index, observation in enumerate(observations):
        i, j = observation.link
        source = refined_tables if observation.mode.refined else coarse_tables
        gammas[:, index] = (
            max(float(source.gamma_sense[i, j, q]), 0.0)
            * observation.mode.power_scale / reference[i]
            * factors[:, index]
        )
    return gammas


def evaluate_active_detection(
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
    seed: int = 0xA5715E,
    transmitter_reference_scales: np.ndarray | None = None,
    transport_mode: str = "direct_llr",
) -> ActiveDetectionResult:
    """Evaluate a mixed-mode exact-LLR sum under observed true erasures.

    Every observation uses its own SINR and Gamma shape ``mode.looks``.  H0
    calibration, H0 evaluation, and H1 evaluation use disjoint deterministic
    draws.  Physical-statistic and report-erasure streams are separated, and
    neither is keyed by the fusion destination.  Consequently, changing only
    the fusion UAV changes report success probability and locality without
    resampling the underlying echo.  Stream keys include the physical link and
    mode, so common random numbers remain stable when another method changes
    its selected bundle.
    """
    validate_config(cfg)
    if transport_mode not in {"direct_llr", "receiver_local_llr"}:
        raise ValueError(
            "transport_mode must be 'direct_llr' or 'receiver_local_llr'"
        )
    if cfg.detect.comm_error_model != "erasure":
        raise ValueError("mixed-mode exact LLR requires detect.comm_error_model='erasure'")
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
        for index, observation in enumerate(chosen):
            i, j = observation.link
            gamma = max(float(gammas[scenario, index]), 0.0)
            looks = int(observation.mode.looks)
            coefficient = gamma / (1.0 + gamma)
            offset = -looks * np.log1p(gamma)
            chi = float(np.clip(
                report_chi(coarse_tables, plan, observation.link, q), 0.0, 1.0
            ))
            # Key by physical mode parameters, not its human-readable label;
            # factorial ablations that contain an identical mode then share
            # exactly the same standard variates.
            physical_mode = (
                f"{observation.mode.power_scale:.12g}|{looks}|"
                f"{int(observation.mode.refined)}"
            )
            mode_key = zlib.crc32(physical_mode.encode("utf-8"))
            # Fusion is intentionally absent from both keys.  It is a routing
            # decision, not part of the physical echo experiment.  Shared
            # erasure uniforms also provide monotone common-random-number
            # coupling when two destinations have different success
            # probabilities; only ``chi`` changes between those evaluations.
            statistic_rng = np.random.default_rng([
                int(seed), int(q), int(scenario), int(i), int(j),
                int(mode_key), 0x571A7157,
            ])
            if transport_mode == "receiver_local_llr":
                if j not in receiver_erasure_masks:
                    erasure_rng = np.random.default_rng([
                        int(seed), int(q), int(scenario), int(j), 0x10CA11A6,
                    ])
                    receiver_erasure_masks[j] = (
                        erasure_rng.random(n_cal) < chi,
                        erasure_rng.random(n_eval) < chi,
                        erasure_rng.random(n_eval) < chi,
                    )
                cal_arrives, h0_arrives, h1_arrives = receiver_erasure_masks[j]
            else:
                erasure_rng = np.random.default_rng([
                    int(seed), int(q), int(scenario), int(i), int(j),
                    int(mode_key), 0xE2A5E2A5,
                ])
                cal_arrives = erasure_rng.random(n_cal) < chi
                h0_arrives = erasure_rng.random(n_eval) < chi
                h1_arrives = erasure_rng.random(n_eval) < chi
            cal = offset + coefficient * statistic_rng.gamma(looks, 1.0, n_cal)
            h0 = offset + coefficient * statistic_rng.gamma(looks, 1.0, n_eval)
            h1 = offset + coefficient * statistic_rng.gamma(
                looks, 1.0 + gamma, n_eval
            )
            h0_cal += np.where(cal_arrives, cal, 0.0)
            h0_eval += np.where(h0_arrives, h0, 0.0)
            h1_eval += np.where(h1_arrives, h1, 0.0)
        threshold = float(np.quantile(
            h0_cal, 1.0 - cfg.detect.Pfa_target, method="higher"
        ))
        thresholds.append(threshold)
        # Strict inequality matches the released detector.  This matters for
        # true erasure because the fused law has a discrete atom at zero.
        pfa_values.append(float(np.mean(h0_eval > threshold)))
        pd_values.append(float(np.mean(h1_eval > threshold)))
    return ActiveDetectionResult(
        tuple(pd_values), tuple(pfa_values), tuple(thresholds)
    )


def price_active_information_bundle(
    cfg: Config,
    base: BaseGains,
    coarse_tables: LinkTables,
    refined_tables: LinkTables,
    q: int,
    fusion: int,
    *,
    modes: Sequence[SensingMode] | None = None,
    max_observations: int | None = None,
    candidate_links: Sequence[Link] | None = None,
    candidate_strategy: str | None = None,
) -> InformationPricingResult:
    """Solve one robust active-observation pricing problem by branch-and-bound.

    At most one mode may be selected for each physical link.  The objective is
    the minimum additive information across aspect scenarios minus declared
    energy and report prices.  The upper bound independently takes the best
    remaining mode for each scenario and ignores future non-negative costs;
    hence pruning is valid even though the bound is intentionally optimistic.
    """
    validate_config(cfg)
    active = cfg.active_sensing
    chosen_modes = tuple(modes) if modes is not None else configured_sensing_modes(cfg)
    if not chosen_modes:
        raise ValueError("at least one sensing mode is required")
    plan = ReportingPlan(
        mode="explicit", f_q=np.full(cfg.scale.Q, int(fusion), dtype=int)
    )
    feasible = feasible_links_for_target(cfg, base, coarse_tables, q, plan)
    if candidate_links is None:
        all_links = feasible
    else:
        feasible_set = set(feasible)
        declared = list(dict.fromkeys(candidate_links))
        invalid = [link for link in declared if link not in feasible_set]
        if invalid:
            raise ValueError(f"declared active candidate links are infeasible: {invalid}")
        all_links = declared
    if not all_links:
        scenario_count = len(active.aspect_angles_deg) if active.aspect_enable else 1
        return InformationPricingResult(
            q, fusion, (), (0.0,) * scenario_count, 0.0, 0.0, 0.0, 0,
            1, True, 0.0,
            total_feasible_links=len(feasible),
            retained_candidate_links=0,
            certificate_scope=(
                "complete_pool" if candidate_strategy == "full" else "declared_pool"
                if candidate_links is not None else "shortlist_union"
            ),
        )

    all_values = observation_information_matrix(
        cfg, base, coarse_tables, refined_tables, q, fusion, all_links, chosen_modes
    )
    if candidate_links is None:
        links, values, certificate_scope = active_candidate_links(
            cfg, all_links, all_values, strategy=candidate_strategy
        )
    else:
        links, values, certificate_scope = list(all_links), all_values, "declared_pool"
    order = np.argsort(-np.max(np.min(values, axis=0), axis=1), kind="stable")
    links = [links[int(index)] for index in order]
    values = values[:, order, :]

    limit = (
        int(cfg.selector.max_links_per_target)
        if max_observations is None else int(max_observations)
    )
    limit = min(max(limit, 0), len(links))
    energy_budget = float(active.energy_budget_per_target)
    energy_price = float(active.energy_price)
    report_price = float(active.report_price)
    remote = np.asarray([
        0 if is_local_observation(plan, link, q) else 1 for link in links
    ], dtype=int)
    energies = np.asarray([mode.energy for mode in chosen_modes], dtype=float)

    best_score = 0.0
    best_info = np.zeros(values.shape[0], dtype=float)
    best_choice: tuple[tuple[int, int], ...] = ()
    best_energy = 0.0
    best_remote = 0
    explored = 0
    truncated = False

    def optimistic_bound(index: int, count: int, info: np.ndarray,
                         energy: float, reports: int) -> float:
        slots = limit - count
        if slots <= 0 or index >= len(links):
            return float(np.min(info) - energy_price * energy - report_price * reports)
        remaining = values[:, index:, :].max(axis=2)
        additions = np.zeros(values.shape[0], dtype=float)
        for scenario in range(values.shape[0]):
            row = np.sort(remaining[scenario])[::-1]
            additions[scenario] = float(np.sum(row[:slots]))
        return float(
            np.min(info + additions)
            - energy_price * energy
            - report_price * reports
        )

    root_bound = optimistic_bound(0, 0, np.zeros(values.shape[0]), 0.0, 0)

    def visit(index: int, choice: tuple[tuple[int, int], ...], info: np.ndarray,
              energy: float, reports: int) -> None:
        nonlocal best_score, best_info, best_choice, best_energy, best_remote
        nonlocal explored, truncated
        if explored >= int(active.branch_node_limit):
            truncated = True
            return
        explored += 1
        score = float(np.min(info) - energy_price * energy - report_price * reports)
        candidate_key = (score, float(np.min(info)), -energy, -reports)
        incumbent_key = (
            best_score, float(np.min(best_info)), -best_energy, -best_remote
        )
        if candidate_key > incumbent_key:
            best_score = score
            best_info = info.copy()
            best_choice = choice
            best_energy = energy
            best_remote = reports
        if index >= len(links) or len(choice) >= limit:
            return
        if optimistic_bound(index, len(choice), info, energy, reports) <= best_score + 1e-12:
            return

        feasible_modes = [
            mode_index for mode_index, mode_energy in enumerate(energies)
            if energy + mode_energy <= energy_budget + 1e-12
        ]
        feasible_modes.sort(
            key=lambda mode_index: float(np.min(values[:, index, mode_index])),
            reverse=True,
        )
        for mode_index in feasible_modes:
            visit(
                index + 1,
                choice + ((index, mode_index),),
                info + values[:, index, mode_index],
                energy + float(energies[mode_index]),
                reports + int(remote[index]),
            )
        visit(index + 1, choice, info, energy, reports)

    visit(0, (), np.zeros(values.shape[0], dtype=float), 0.0, 0)
    observations = tuple(
        ActiveObservation(links[link_index], chosen_modes[mode_index])
        for link_index, mode_index in best_choice
    )
    upper = float(root_bound if truncated else best_score)
    return InformationPricingResult(
        target=int(q),
        fusion=int(fusion),
        observations=observations,
        scenario_information=tuple(float(value) for value in best_info),
        robust_information=float(np.min(best_info)),
        objective=float(best_score),
        energy=float(best_energy),
        remote_reports=int(best_remote),
        explored_nodes=int(explored),
        exact=not truncated,
        upper_bound=upper,
        total_feasible_links=len(feasible),
        retained_candidate_links=len(links),
        certificate_scope=certificate_scope,
        candidate_links=tuple(links),
    )
