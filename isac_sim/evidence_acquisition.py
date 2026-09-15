"""Unified active-evidence acquisition for cooperative OTFS-ISAC sensing.

The public entry point in this module joins three layers that were previously
exposed separately: detector-consistent information pricing, mixed-mode
active-column construction, and fleet-wide resource allocation.  Information
is used to generate candidate evidence; calibrated worst-scenario detection
probability remains the operational decision metric.

The returned scope is deliberately explicit.  The local pricing oracle may be
exact over its declared candidate set, and the global MILP is exact over the
generated columns, but neither statement is a certificate over the complete
fleet-wide active-action universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .active_information import SensingMode, configured_sensing_modes
from .active_system import (
    ActiveColumn,
    GlobalActiveMasterResult,
    generate_active_columns,
    screen_fusion_candidates,
    solve_global_active_master,
)
from .config import Config, validate_config
from .model import BaseGains, LinkTables


@dataclass(frozen=True)
class EvidenceAcquisitionScope:
    """Scientific and algorithmic boundary of one solved acquisition problem."""

    information_metric: str
    operational_metric: str
    uncertainty_model: str
    power_externality_model: str
    local_pricing_certificate: str
    global_certificate: str


@dataclass(frozen=True)
class EvidenceAcquisitionResult:
    """Candidate pool, selected schedule, and its declared certificate scope."""

    candidate_columns: tuple[ActiveColumn, ...]
    schedule: GlobalActiveMasterResult
    scope: EvidenceAcquisitionScope

    @property
    def selected_columns(self) -> tuple[ActiveColumn, ...]:
        return self.schedule.columns

    @property
    def objective(self) -> dict[str, float]:
        return self.schedule.objective


def solve_active_evidence_acquisition(
    cfg: Config,
    base: BaseGains,
    coarse_tables: LinkTables,
    refined_tables: LinkTables,
    *,
    modes: Sequence[SensingMode] | None = None,
    candidate_limit: int = 4,
    allowed_fusions: dict[int, Sequence[int]] | None = None,
    fusion_candidate_limit: int | None = None,
    calibration_samples: int = 512,
    evaluation_samples: int = 1024,
    seed: int = 0xE71D3,
    lex_tolerance: float = 1e-4,
) -> EvidenceAcquisitionResult:
    """Acquire, transport, and schedule detector-relevant evidence.

    Candidate observations are priced with KL/Jeffreys information under the
    declared aspect scenarios.  Each resulting mixed-mode column is evaluated
    with the exact LLR and true erasures under the conservative full-load
    interference envelope.  The global master then selects one column per
    target subject to sensing, reporting, receiver, computation, and energy
    constraints, using calibrated worst-scenario ``P_D`` as its first-order
    operational objective.
    """
    validate_config(cfg)
    if not cfg.active_sensing.enable:
        raise ValueError("active evidence acquisition requires active_sensing.enable=True")
    if cfg.detect.soft_stat_model != "llr":
        raise ValueError("active evidence acquisition requires detect.soft_stat_model='llr'")
    if cfg.detect.comm_error_model != "erasure":
        raise ValueError("active evidence acquisition requires detect.comm_error_model='erasure'")

    chosen_modes = tuple(modes) if modes is not None else configured_sensing_modes(cfg)
    if allowed_fusions is not None and fusion_candidate_limit is not None:
        raise ValueError(
            "allowed_fusions and fusion_candidate_limit are mutually exclusive"
        )
    screened_fusions = allowed_fusions
    if fusion_candidate_limit is not None:
        screened_fusions = screen_fusion_candidates(
            cfg,
            base,
            coarse_tables,
            refined_tables,
            modes=chosen_modes,
            information_limit=fusion_candidate_limit,
        )
    columns = tuple(generate_active_columns(
        cfg,
        base,
        coarse_tables,
        refined_tables,
        modes=chosen_modes,
        candidate_limit=candidate_limit,
        allowed_fusions=screened_fusions,
        calibration_samples=calibration_samples,
        evaluation_samples=evaluation_samples,
        seed=seed,
    ))
    schedule = solve_global_active_master(
        cfg, columns, lex_tolerance=lex_tolerance
    )
    uncertainty = (
        "finite_aspect_scenarios"
        if cfg.active_sensing.aspect_enable
        else "single_nominal_scenario"
    )
    local_scope = {
        "full": "complete_pool_when_within_declared_limit",
        "scenario_union": "scenario_complementarity_shortlist",
        "robust_singleton": "robust_singleton_shortlist",
    }[cfg.active_sensing.candidate_strategy]
    return EvidenceAcquisitionResult(
        candidate_columns=columns,
        schedule=schedule,
        scope=EvidenceAcquisitionScope(
            information_metric=cfg.active_sensing.information_metric,
            operational_metric="worst_scenario_pd_at_calibrated_pfa",
            uncertainty_model=uncertainty,
            power_externality_model=schedule.power_externality_model,
            local_pricing_certificate=local_scope,
            global_certificate="exact_over_generated_columns",
        ),
    )
