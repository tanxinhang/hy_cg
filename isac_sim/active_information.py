"""Information-consistent active observation design for the V1.5 path.

This module is deliberately separate from the frozen V1.4 detector path.  It
turns a passive bistatic link ``(i,j,q)`` into an active observation
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

    @property
    def certificate_gap(self) -> float:
        return float(max(self.upper_bound - self.objective, 0.0))


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
    all_links = feasible_links_for_target(cfg, base, coarse_tables, q, plan)
    if not all_links:
        scenario_count = len(active.aspect_angles_deg) if active.aspect_enable else 1
        return InformationPricingResult(
            q, fusion, (), (0.0,) * scenario_count, 0.0, 0.0, 0.0, 0,
            1, True, 0.0,
        )

    all_values = observation_information_matrix(
        cfg, base, coarse_tables, refined_tables, q, fusion, all_links, chosen_modes
    )
    # Shortlist physical links by their best robust mode, retaining mode choice
    # for the certified combinatorial search rather than collapsing it early.
    robust_by_link_mode = np.min(all_values, axis=0)
    link_rank = np.max(robust_by_link_mode, axis=1)
    keep = np.argsort(-link_rank, kind="stable")[: active.max_candidates_per_pair]
    links = [all_links[int(index)] for index in keep]
    values = all_values[:, keep, :]
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
    )
