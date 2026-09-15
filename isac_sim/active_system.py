"""Fleet-level V1.5-System master for mixed-mode active columns.

Columns are evaluated by the mixed-mode exact-LLR detector under a full-load
sensing-power envelope.  The envelope recomputes both desired echoes and
cross-UAV leakage, giving a conservative column value that is independent of
which other columns the master selects.  This preserves a linear, certifiable
global master while exposing rather than ignoring the power externality.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .active_information import (
    ActiveObservation,
    SensingMode,
    active_observation_gammas,
    configured_sensing_modes,
    evaluate_active_detection,
    price_active_information_bundle,
    received_information,
)
from .config import Config, Link, validate_config
from .model import BaseGains, LinkTables, compute_link_tables
from .reporting import ReportingPlan, is_local_observation, report_chi


@dataclass(frozen=True)
class ActiveColumn:
    target: int
    fusion: int
    observations: tuple[ActiveObservation, ...]
    scenario_information: tuple[float, ...]
    scenario_pd: tuple[float, ...]
    scenario_pfa: tuple[float, ...]
    energy_by_tx: tuple[float, ...]
    tx_load: tuple[int, ...]
    receiver_load: tuple[int, ...]
    remote_reports: int
    cpu_cycles: float

    @property
    def robust_pd(self) -> float:
        return float(min(self.scenario_pd, default=0.0))

    @property
    def robust_information(self) -> float:
        return float(min(self.scenario_information, default=0.0))

    @property
    def processing_load(self) -> int:
        return len(self.observations)

    @property
    def total_energy(self) -> float:
        return float(sum(self.energy_by_tx))


@dataclass(frozen=True)
class GlobalActiveMasterResult:
    columns: tuple[ActiveColumn, ...]
    objective: dict[str, float]
    candidate_column_count: int
    exact_over_columns: bool
    power_externality_model: str = "full_load_envelope"


def active_observation_cpu_cycles(cfg: Config, observation: ActiveObservation) -> float:
    """Matched-filter, exact-LLR, and optional DD-refinement work."""
    active = cfg.active_sensing
    return float(
        observation.mode.looks * (
            active.matched_filter_cycles_per_look
            + active.llr_cycles_per_look
        )
        + (active.dd_refine_cycles if observation.mode.refined else 0.0)
    )


def active_column_cpu_cycles(
    cfg: Config, observations: Sequence[ActiveObservation]
) -> float:
    if not observations:
        return 0.0
    size = len(observations)
    return float(
        cfg.fusion.cpu_fixed_cycles
        + cfg.fusion.cpu_per_observation_cycles * size
        + cfg.fusion.cpu_cubic_cycles * size ** 3
        + sum(active_observation_cpu_cycles(cfg, obs) for obs in observations)
    )


def _make_active_column(
    cfg: Config,
    base: BaseGains,
    envelope_coarse: LinkTables,
    envelope_refined: LinkTables,
    q: int,
    fusion: int,
    observations: Sequence[ActiveObservation],
    reference_scales: np.ndarray,
    *,
    calibration_samples: int,
    evaluation_samples: int,
    seed: int,
) -> ActiveColumn:
    chosen = tuple(observations)
    M = cfg.scale.M
    energy = np.zeros(M, dtype=float)
    tx_load = np.zeros(M, dtype=int)
    receiver = np.zeros(M, dtype=int)
    plan = ReportingPlan(
        mode="explicit", f_q=np.full(cfg.scale.Q, fusion, dtype=int)
    )
    remote = 0
    for observation in chosen:
        i, j = observation.link
        energy[i] += observation.mode.energy
        tx_load[i] += 1
        receiver[j] += 1
        remote += int(not is_local_observation(plan, observation.link, q))

    scenario_count = (
        len(cfg.active_sensing.aspect_angles_deg)
        if cfg.active_sensing.aspect_enable else 1
    )
    if chosen:
        detection = evaluate_active_detection(
            cfg, base, envelope_coarse, envelope_refined, q, fusion, chosen,
            calibration_samples=calibration_samples,
            evaluation_samples=evaluation_samples,
            seed=seed,
            transmitter_reference_scales=reference_scales,
        )
        gammas = active_observation_gammas(
            cfg, base, envelope_coarse, envelope_refined, q, chosen,
            reference_scales,
        )
        information = np.zeros(gammas.shape[0], dtype=float)
        for index, observation in enumerate(chosen):
            chi = float(np.clip(
                report_chi(envelope_coarse, plan, observation.link, q), 0.0, 1.0
            ))
            information += received_information(
                gammas[:, index], observation.mode.looks, chi,
                cfg.active_sensing.information_metric,
            )
        scenario_pd = detection.scenario_pd
        scenario_pfa = detection.scenario_pfa
        scenario_information = tuple(float(x) for x in information)
    else:
        scenario_pd = (0.0,) * scenario_count
        scenario_pfa = (float(cfg.detect.Pfa_target),) * scenario_count
        scenario_information = (0.0,) * scenario_count
    return ActiveColumn(
        target=q,
        fusion=fusion,
        observations=chosen,
        scenario_information=scenario_information,
        scenario_pd=scenario_pd,
        scenario_pfa=scenario_pfa,
        energy_by_tx=tuple(float(x) for x in energy),
        tx_load=tuple(int(x) for x in tx_load),
        receiver_load=tuple(int(x) for x in receiver),
        remote_reports=int(remote),
        cpu_cycles=active_column_cpu_cycles(cfg, chosen),
    )


def generate_active_columns(
    cfg: Config,
    base: BaseGains,
    coarse_tables: LinkTables,
    refined_tables: LinkTables,
    *,
    modes: Sequence[SensingMode] | None = None,
    candidate_limit: int = 4,
    allowed_fusions: dict[int, Sequence[int]] | None = None,
    calibration_samples: int = 512,
    evaluation_samples: int = 1024,
    seed: int = 0xC0115,
) -> list[ActiveColumn]:
    """Generate active columns and exact-LLR values for the global master."""
    validate_config(cfg)
    chosen_modes = tuple(modes or configured_sensing_modes(cfg))
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    reference = np.full(
        cfg.scale.M, max(mode.power_scale for mode in chosen_modes), dtype=float
    )
    envelope_coarse = compute_link_tables(
        cfg, base, sensing_power_scale_by_uav=reference
    )
    envelope_refined = compute_link_tables(
        cfg, base, dd_gain=base.eta_fine,
        sensing_power_scale_by_uav=reference,
    )
    columns: list[ActiveColumn] = []
    K = max(int(cfg.selector.max_links_per_target), 0)
    for q in range(cfg.scale.Q):
        fusions = (
            range(cfg.scale.M) if allowed_fusions is None
            else sorted(set(int(f) for f in allowed_fusions[q]))
        )
        for fusion in fusions:
            if fusion < 0 or fusion >= cfg.scale.M:
                raise ValueError(f"invalid fusion UAV {fusion} for target {q}")
            priced = price_active_information_bundle(
                cfg, base, coarse_tables, refined_tables, q, fusion,
                modes=chosen_modes,
            )
            candidates = list(priced.candidate_links)[:candidate_limit]
            choices = range(-1, len(chosen_modes))
            for assignment in itertools.product(choices, repeat=len(candidates)):
                selected = [
                    ActiveObservation(candidates[index], chosen_modes[mode_index])
                    for index, mode_index in enumerate(assignment)
                    if mode_index >= 0
                ]
                if len(selected) > K:
                    continue
                if sum(obs.mode.energy for obs in selected) > (
                    cfg.active_sensing.energy_budget_per_target + 1e-12
                ):
                    continue
                column = _make_active_column(
                    cfg, base, envelope_coarse, envelope_refined, q, fusion,
                    selected, reference,
                    calibration_samples=calibration_samples,
                    evaluation_samples=evaluation_samples,
                    seed=seed,
                )
                columns.append(column)
    return columns


def solve_global_active_master(
    cfg: Config,
    columns: Sequence[ActiveColumn],
    *,
    lex_tolerance: float = 1e-4,
) -> GlobalActiveMasterResult:
    """Solve the fleet-level lexicographic active-column master exactly."""
    validate_config(cfg)
    pool = list(columns)
    Q, M, B = cfg.scale.Q, cfg.scale.M, len(pool)
    if any(not any(column.target == q for column in pool) for q in range(Q)):
        raise ValueError("active column pool must cover every target")
    n = B + Q + 1
    dmax = B + Q
    lower = np.zeros(n)
    upper = np.ones(n)
    upper[B:] = max(float(cfg.detect.pd_required), 1.0)
    integrality = np.zeros(n, dtype=int)
    integrality[:B] = 1
    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    def add(coeff: np.ndarray, lb: float = -np.inf, ub: float = np.inf) -> None:
        rows.append(coeff); lbs.append(lb); ubs.append(ub)

    for q in range(Q):
        row = np.zeros(n)
        for b, column in enumerate(pool): row[b] = float(column.target == q)
        add(row, 1.0, 1.0)
        row = np.zeros(n)
        for b, column in enumerate(pool):
            if column.target == q: row[b] = -column.robust_pd
        row[B + q] = -1.0
        add(row, ub=-float(cfg.detect.pd_required))
        row = np.zeros(n); row[B + q] = 1.0; row[dmax] = -1.0
        add(row, ub=0.0)

    if cfg.fusion.max_targets_per_uav >= 0:
        for fusion in range(M):
            row = np.zeros(n)
            for b, column in enumerate(pool): row[b] = float(column.fusion == fusion)
            add(row, ub=float(cfg.fusion.max_targets_per_uav))
    for i in range(M):
        row = np.zeros(n)
        for b, column in enumerate(pool): row[b] = column.energy_by_tx[i]
        add(row, ub=float(cfg.active_sensing.energy_budget_per_uav))
        if cfg.active_sensing.max_tx_observations_per_uav >= 0:
            row = np.zeros(n)
            for b, column in enumerate(pool): row[b] = column.tx_load[i]
            add(row, ub=float(cfg.active_sensing.max_tx_observations_per_uav))
    if cfg.selector.max_observations_per_receiver >= 0:
        for receiver in range(M):
            row = np.zeros(n)
            for b, column in enumerate(pool): row[b] = column.receiver_load[receiver]
            add(row, ub=float(cfg.selector.max_observations_per_receiver))
    cpu_budget = (
        cfg.fusion.cpu_rate_cycles_per_s * cfg.fusion.processing_window_s
        if cfg.fusion.cpu_rate_cycles_per_s > 0.0 else None
    )
    for fusion in range(M):
        if cfg.selector.max_observations_per_fusion_uav >= 0:
            row = np.zeros(n)
            for b, column in enumerate(pool):
                row[b] = column.processing_load if column.fusion == fusion else 0.0
            add(row, ub=float(cfg.selector.max_observations_per_fusion_uav))
        if cpu_budget is not None:
            row = np.zeros(n)
            for b, column in enumerate(pool):
                row[b] = column.cpu_cycles if column.fusion == fusion else 0.0
            add(row, ub=float(cpu_budget))
    if cfg.selector.max_remote_reports >= 0:
        row = np.zeros(n)
        for b, column in enumerate(pool): row[b] = column.remote_reports
        add(row, ub=float(cfg.selector.max_remote_reports))
    if cfg.selector.max_total_links >= 0:
        row = np.zeros(n)
        for b, column in enumerate(pool): row[b] = column.processing_load
        add(row, ub=float(cfg.selector.max_total_links))

    objectives: list[np.ndarray] = []
    objective = np.zeros(n); objective[dmax] = 1.0; objectives.append(objective)
    objective = np.zeros(n)
    objective[:B] = [
        max(float(cfg.detect.pd_required) - column.robust_pd, 0.0)
        for column in pool
    ]
    objectives.append(objective)
    for attribute in ("total_energy", "remote_reports", "cpu_cycles"):
        objective = np.zeros(n)
        objective[:B] = [float(getattr(column, attribute)) for column in pool]
        objectives.append(objective)

    solution = None
    optima: list[float] = []
    for stage, objective in enumerate(objectives, start=1):
        constraint = LinearConstraint(np.vstack(rows), np.asarray(lbs), np.asarray(ubs))
        result = milp(
            objective, integrality=integrality, bounds=Bounds(lower, upper),
            constraints=constraint,
            options={"presolve": False, "mip_rel_gap": 0.0},
        )
        if not result.success or result.x is None:
            raise ValueError(f"global active master stage {stage} failed: {result.message}")
        solution = result.x
        selected_indices = [index for index in range(B) if result.x[index] > 0.5]
        if stage == 1:
            selected_pd = [pool[index].robust_pd for index in selected_indices]
            optimum = max(
                max(float(cfg.detect.pd_required) - pd, 0.0) for pd in selected_pd
            )
        else:
            optimum = float(sum(objective[index] for index in selected_indices))
        optima.append(optimum)
        tolerance = lex_tolerance * max(1.0, abs(optimum))
        if stage == 1:
            minimum_pd = float(cfg.detect.pd_required) - optimum - tolerance
            for q in range(Q):
                row = np.zeros(n)
                for b, column in enumerate(pool):
                    if column.target == q:
                        row[b] = -column.robust_pd
                add(row, ub=-minimum_pd)
        else:
            add(objective.copy(), ub=optimum + tolerance)
    assert solution is not None
    chosen = tuple(pool[index] for index in range(B) if solution[index] > 0.5)
    if len(chosen) != Q:
        raise RuntimeError("global active master returned a non-integral assignment")
    robust_pd = np.asarray([column.robust_pd for column in chosen])
    deficits = np.maximum(float(cfg.detect.pd_required) - robust_pd, 0.0)
    return GlobalActiveMasterResult(
        columns=chosen,
        objective={
            "worst_detection_deficit": float(np.max(deficits)),
            "total_detection_deficit": float(np.sum(deficits)),
            "total_energy": float(sum(column.total_energy for column in chosen)),
            "remote_reports": float(sum(column.remote_reports for column in chosen)),
            "cpu_cycles": float(sum(column.cpu_cycles for column in chosen)),
            "worst_pd": float(np.min(robust_pd)),
            "mean_pd": float(np.mean(robust_pd)),
            "max_pfa": float(max(max(column.scenario_pfa) for column in chosen)),
            "mean_pfa": float(np.mean([
                value for column in chosen for value in column.scenario_pfa
            ])),
        },
        candidate_column_count=B,
        exact_over_columns=True,
    )
