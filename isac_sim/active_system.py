"""Fleet-level resource master for mixed-mode active-evidence columns.

Columns are evaluated by the mixed-mode exact-LLR detector under a full-load
sensing-power envelope.  The envelope recomputes both desired echoes and
cross-UAV leakage, giving a conservative column value that is independent of
which other columns the master selects.  This preserves a linear, certifiable
global master while exposing rather than ignoring the power externality.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
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
    # ``scenario_information`` is the evidence that reaches the final fusion
    # node.  The generated quantity removes report erasures while preserving
    # the sensing action, so their difference isolates transport loss.
    scenario_generated_information: tuple[float, ...] = ()
    scenario_network_information_loss: tuple[float, ...] = ()
    # Observation processing runs at each receiver; only aggregation runs at
    # the final fusion UAV.  Keeping ``cpu_cycles`` preserves the public total.
    receiver_cpu_cycles: tuple[float, ...] = ()
    fusion_cpu_cycles: float = 0.0
    local_aggregation_cpu_cycles: tuple[float, ...] = ()
    transport_mode: str = "direct_llr"
    fusion_inputs: int = 0

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

    @property
    def network_evidence_loss(self) -> float:
        """Worst-scenario KL/Jeffreys information lost during reporting."""
        return float(max(self.scenario_network_information_loss, default=0.0))

    @property
    def robust_evidence_retention(self) -> float:
        """Minimum scenario-wise received/generated evidence ratio."""
        if not self.scenario_generated_information:
            return 1.0
        ratios = []
        for generated, received in zip(
            self.scenario_generated_information, self.scenario_information
        ):
            ratios.append(1.0 if generated <= 1e-15 else received / generated)
        return float(np.clip(min(ratios, default=1.0), 0.0, 1.0))


@dataclass(frozen=True)
class GlobalActiveMasterResult:
    columns: tuple[ActiveColumn, ...]
    objective: dict[str, float]
    candidate_column_count: int
    exact_over_columns: bool
    power_externality_model: str = "full_load_envelope"


@dataclass(frozen=True)
class ActiveTransportHeadroom:
    """Fixed-acquisition comparison between current and lossless reporting."""

    current: ActiveColumn
    lossless_report: ActiveColumn

    @property
    def worst_pd_headroom(self) -> float:
        return float(self.lossless_report.robust_pd - self.current.robust_pd)

    @property
    def mean_pd_headroom(self) -> float:
        return float(
            np.mean(self.lossless_report.scenario_pd)
            - np.mean(self.current.scenario_pd)
        )


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
    """Total receiver-side observation and final-fusion processing cycles."""
    if not observations:
        return 0.0
    return float(
        sum(active_observation_cpu_cycles(cfg, obs) for obs in observations)
        + active_fusion_cpu_cycles(cfg, len(observations))
    )


def active_fusion_cpu_cycles(cfg: Config, observation_count: int) -> float:
    """Aggregation cycles executed only by the selected final fusion UAV."""
    if observation_count <= 0:
        return 0.0
    size = int(observation_count)
    return float(
        cfg.fusion.cpu_fixed_cycles
        + cfg.fusion.cpu_per_observation_cycles * size
        + cfg.fusion.cpu_cubic_cycles * size ** 3
    )


def active_receiver_cpu_cycles(
    cfg: Config, observations: Sequence[ActiveObservation]
) -> tuple[float, ...]:
    """Per-UAV cycles for matched filtering, exact LLRs, and DD refinement."""
    cycles = np.zeros(cfg.scale.M, dtype=float)
    for observation in observations:
        cycles[observation.link[1]] += active_observation_cpu_cycles(cfg, observation)
    return tuple(float(value) for value in cycles)


def active_local_aggregation_cpu_cycles(
    cfg: Config, observations: Sequence[ActiveObservation]
) -> tuple[float, ...]:
    """Per-receiver scalar additions used to collapse local exact LLRs."""
    counts = np.zeros(cfg.scale.M, dtype=int)
    for observation in observations:
        counts[observation.link[1]] += 1
    per_addition = float(cfg.fusion.cpu_per_observation_cycles)
    return tuple(
        per_addition * max(int(count) - 1, 0) for count in counts
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
    transport_mode: str = "direct_llr",
) -> ActiveColumn:
    chosen = tuple(observations)
    M = cfg.scale.M
    energy = np.zeros(M, dtype=float)
    tx_load = np.zeros(M, dtype=int)
    receiver = np.zeros(M, dtype=int)
    plan = ReportingPlan(
        mode="explicit", f_q=np.full(cfg.scale.Q, fusion, dtype=int)
    )
    if transport_mode not in {"direct_llr", "receiver_local_llr"}:
        raise ValueError(
            "transport_mode must be 'direct_llr' or 'receiver_local_llr'"
        )
    remote_receivers: set[int] = set()
    remote = 0
    for observation in chosen:
        i, j = observation.link
        energy[i] += observation.mode.energy
        tx_load[i] += 1
        receiver[j] += 1
        if not is_local_observation(plan, observation.link, q):
            remote += 1
            remote_receivers.add(j)
    if transport_mode == "receiver_local_llr":
        remote = len(remote_receivers)
        fusion_input_count = len({obs.link[1] for obs in chosen})
    else:
        fusion_input_count = len(chosen)

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
            transport_mode=transport_mode,
        )
        gammas = active_observation_gammas(
            cfg, base, envelope_coarse, envelope_refined, q, chosen,
            reference_scales,
        )
        information = np.zeros(gammas.shape[0], dtype=float)
        generated_information = np.zeros(gammas.shape[0], dtype=float)
        for index, observation in enumerate(chosen):
            chi = float(np.clip(
                report_chi(envelope_coarse, plan, observation.link, q), 0.0, 1.0
            ))
            information += received_information(
                gammas[:, index], observation.mode.looks, chi,
                cfg.active_sensing.information_metric,
            )
            generated_information += received_information(
                gammas[:, index], observation.mode.looks, 1.0,
                cfg.active_sensing.information_metric,
            )
        scenario_pd = detection.scenario_pd
        scenario_pfa = detection.scenario_pfa
        scenario_information = tuple(float(x) for x in information)
        scenario_generated_information = tuple(
            float(x) for x in generated_information
        )
        scenario_network_information_loss = tuple(
            float(x) for x in np.maximum(generated_information - information, 0.0)
        )
    else:
        scenario_pd = (0.0,) * scenario_count
        scenario_pfa = (float(cfg.detect.Pfa_target),) * scenario_count
        scenario_information = (0.0,) * scenario_count
        scenario_generated_information = (0.0,) * scenario_count
        scenario_network_information_loss = (0.0,) * scenario_count
    observation_cpu = np.asarray(
        active_receiver_cpu_cycles(cfg, chosen), dtype=float
    )
    local_aggregation_cpu = (
        np.asarray(active_local_aggregation_cpu_cycles(cfg, chosen), dtype=float)
        if transport_mode == "receiver_local_llr"
        else np.zeros(M, dtype=float)
    )
    receiver_cpu = tuple(float(value) for value in (
        observation_cpu + local_aggregation_cpu
    ))
    fusion_cpu = active_fusion_cpu_cycles(cfg, fusion_input_count)
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
        cpu_cycles=float(sum(receiver_cpu) + fusion_cpu),
        scenario_generated_information=scenario_generated_information,
        scenario_network_information_loss=scenario_network_information_loss,
        receiver_cpu_cycles=receiver_cpu,
        fusion_cpu_cycles=fusion_cpu,
        local_aggregation_cpu_cycles=tuple(
            float(value) for value in local_aggregation_cpu
        ),
        transport_mode=transport_mode,
        fusion_inputs=fusion_input_count,
    )


def evaluate_active_transport_headroom(
    cfg: Config,
    base: BaseGains,
    envelope_coarse: LinkTables,
    envelope_refined: LinkTables,
    q: int,
    fusion: int,
    observations: Sequence[ActiveObservation],
    reference_scales: np.ndarray,
    *,
    calibration_samples: int = 2048,
    evaluation_samples: int = 4096,
    seed: int = 0x10A55,
    transport_mode: str = "direct_llr",
) -> ActiveTransportHeadroom:
    """Evaluate report-only headroom with the sensing bundle held fixed.

    The counterfactual changes only report success probabilities to one.  The
    same seed and fusion-independent physical random stream ensure that the
    comparison does not resample the echo or change acquisition decisions.
    """
    current = _make_active_column(
        cfg, base, envelope_coarse, envelope_refined, q, fusion, observations,
        reference_scales,
        calibration_samples=calibration_samples,
        evaluation_samples=evaluation_samples,
        seed=seed,
        transport_mode=transport_mode,
    )
    lossless_coarse = replace(
        envelope_coarse,
        chi_comm=np.ones_like(envelope_coarse.chi_comm, dtype=float),
    )
    lossless_refined = replace(
        envelope_refined,
        chi_comm=np.ones_like(envelope_refined.chi_comm, dtype=float),
    )
    lossless = _make_active_column(
        cfg, base, lossless_coarse, lossless_refined, q, fusion, observations,
        reference_scales,
        calibration_samples=calibration_samples,
        evaluation_samples=evaluation_samples,
        seed=seed,
        transport_mode=transport_mode,
    )
    return ActiveTransportHeadroom(current=current, lossless_report=lossless)


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
    include_priced_rescue_column: bool = True,
    transport_mode: str = "direct_llr",
) -> list[ActiveColumn]:
    """Generate active columns and exact-LLR values for the global master.

    Exhaustive mode enumeration is limited to ``candidate_limit`` physical
    links.  Independently, the information-pricing oracle's complete selected
    bundle is injected as a rescue column.  This preserves a strong weak-target
    action when pricing uses a wider candidate pool than enumeration.
    """
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
            generated_signatures: set[tuple[ActiveObservation, ...]] = set()
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
                    transport_mode=transport_mode,
                )
                columns.append(column)
                generated_signatures.add(tuple(selected))
            rescue = tuple(priced.observations)
            if (
                include_priced_rescue_column
                and rescue
                and rescue not in generated_signatures
                and len(rescue) <= K
                and sum(obs.mode.energy for obs in rescue)
                <= cfg.active_sensing.energy_budget_per_target + 1e-12
            ):
                columns.append(_make_active_column(
                    cfg, base, envelope_coarse, envelope_refined, q, fusion,
                    rescue, reference,
                    calibration_samples=calibration_samples,
                    evaluation_samples=evaluation_samples,
                    seed=seed,
                    transport_mode=transport_mode,
                ))
    return columns


def screen_fusion_candidates(
    cfg: Config,
    base: BaseGains,
    coarse_tables: LinkTables,
    refined_tables: LinkTables,
    *,
    modes: Sequence[SensingMode] | None = None,
    information_limit: int = 3,
) -> dict[int, tuple[int, ...]]:
    """Analytically shortlist final fusion UAVs before Monte Carlo columns.

    For each target, the shortlist retains the strongest fusion destinations
    by robust received information and one communication-locality anchor (the
    priced bundle requiring the fewest remote reports).  This is a screening
    rule, not a certificate over discarded fusion destinations.
    """
    validate_config(cfg)
    if information_limit < 1:
        raise ValueError("information_limit must be positive")
    chosen_modes = tuple(modes or configured_sensing_modes(cfg))
    screened: dict[int, tuple[int, ...]] = {}
    for q in range(cfg.scale.Q):
        priced = [
            price_active_information_bundle(
                cfg, base, coarse_tables, refined_tables, q, fusion,
                modes=chosen_modes,
            )
            for fusion in range(cfg.scale.M)
        ]
        ranked = sorted(
            priced,
            key=lambda item: (-item.robust_information, item.remote_reports, item.fusion),
        )
        retained = {item.fusion for item in ranked[:information_limit]}
        locality_anchor = min(
            priced,
            key=lambda item: (item.remote_reports, -item.robust_information, item.fusion),
        )
        retained.add(locality_anchor.fusion)
        screened[q] = tuple(sorted(retained))
    return screened


def solve_global_active_master(
    cfg: Config,
    columns: Sequence[ActiveColumn],
    *,
    lex_tolerance: float = 1e-4,
    pd_design_target: float | None = None,
) -> GlobalActiveMasterResult:
    """Solve the fleet-level lexicographic active-column master exactly."""
    validate_config(cfg)
    design_target = (
        float(cfg.detect.pd_required)
        if pd_design_target is None else float(pd_design_target)
    )
    if not float(cfg.detect.pd_required) <= design_target <= 1.0:
        raise ValueError("pd_design_target must lie in [pd_required, 1]")
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
        add(row, ub=-design_target)
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
    def cpu_at_uav(column: ActiveColumn, uav: int) -> float:
        if len(column.receiver_cpu_cycles) == M:
            receiver_cycles = column.receiver_cpu_cycles[uav]
            fusion_cycles = column.fusion_cpu_cycles if column.fusion == uav else 0.0
            return float(receiver_cycles + fusion_cycles)
        # Compatibility for externally constructed pre-upgrade columns.
        return float(column.cpu_cycles if column.fusion == uav else 0.0)

    for fusion in range(M):
        if cfg.selector.max_observations_per_fusion_uav >= 0:
            row = np.zeros(n)
            for b, column in enumerate(pool):
                row[b] = column.processing_load if column.fusion == fusion else 0.0
            add(row, ub=float(cfg.selector.max_observations_per_fusion_uav))
        if cpu_budget is not None:
            row = np.zeros(n)
            for b, column in enumerate(pool):
                row[b] = cpu_at_uav(column, fusion)
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
        max(design_target - column.robust_pd, 0.0)
        for column in pool
    ]
    objectives.append(objective)
    # Detection fairness is primary.  Among statistically equivalent columns,
    # preserve detector-relevant evidence before trading energy/report/compute.
    for attribute in (
        "network_evidence_loss", "total_energy", "remote_reports", "cpu_cycles"
    ):
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
                max(design_target - pd, 0.0) for pd in selected_pd
            )
        else:
            optimum = float(sum(objective[index] for index in selected_indices))
        optima.append(optimum)
        tolerance = lex_tolerance * max(1.0, abs(optimum))
        if stage == 1:
            minimum_pd = design_target - optimum - tolerance
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
    receiver_cpu_total = float(sum(
        sum(column.receiver_cpu_cycles)
        if len(column.receiver_cpu_cycles) == M else 0.0
        for column in chosen
    ))
    fusion_cpu_total = float(sum(
        column.fusion_cpu_cycles
        if len(column.receiver_cpu_cycles) == M else column.cpu_cycles
        for column in chosen
    ))
    per_uav_cpu = [sum(cpu_at_uav(column, uav) for column in chosen) for uav in range(M)]
    return GlobalActiveMasterResult(
        columns=chosen,
        objective={
            "worst_detection_deficit": float(np.max(deficits)),
            "total_detection_deficit": float(np.sum(deficits)),
            "pd_design_target": design_target,
            "network_evidence_loss": float(sum(
                column.network_evidence_loss for column in chosen
            )),
            "worst_evidence_retention": float(min(
                (column.robust_evidence_retention for column in chosen), default=1.0
            )),
            "total_energy": float(sum(column.total_energy for column in chosen)),
            "remote_reports": float(sum(column.remote_reports for column in chosen)),
            "cpu_cycles": float(sum(column.cpu_cycles for column in chosen)),
            "receiver_cpu_cycles": receiver_cpu_total,
            "fusion_cpu_cycles": fusion_cpu_total,
            "max_uav_cpu_cycles": float(max(per_uav_cpu, default=0.0)),
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
