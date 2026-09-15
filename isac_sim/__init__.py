"""Simplified Lagrangian-guided DOTFS-ISAC cooperative-sensing simulator.

The package replaces the former single-file prototype
(``lagrangian_dotfs_isac_simplified_v10_paper_plots.py``) with a small set of
single-responsibility modules:

    config       configuration groups and dotted-path overrides
    model        geometry, channel gains and per-link quantity tables
    dd           DD-domain leakage kernels and the C2F (coarse-to-fine)
                 eta^c / eta^loc / eta^f gain model
    waveform     physical OTFS PSF kernel for end-to-end validation
    prior        target-state prior-uncertainty utilities
    fusion       soft-information fusion, deflection and target priority
    selection    the proposed Lagrangian rule, C2F and the baselines
    simulate     Monte-Carlo execution and metric aggregation
    experiments  sweep / ablation definitions (data, not flags)
    report       console summary, CSV and LaTeX tables
    plotting     figures
    evidence_acquisition
                 detector-consistent active observation generation and
                 fleet-wide resource scheduling
    cli          the unified command-line interface
    naming       display order and human-readable labels

Typical use::

    from isac_sim import Config, run_simulation
    summary = run_simulation(Config())

or from the shell::

    python -m isac_sim --mode ablation --mc 100 --out runs/
    python -m isac_sim --mode c2f --mc 200
    python -m isac_sim --mode prior-sweep --mc 50
    python -m isac_sim --mode waveform-check
"""

from __future__ import annotations

from .config import (
    PRESETS,
    Config,
    apply_overrides,
    apply_preset,
    default_config,
    iter_leaf_paths,
)
from .experiments import ABLATION_VARIANTS, DD_VARIANTS, EXPERIMENTS
from .model import (
    BaseGains,
    LinkTables,
    build_base_gains,
    compute_link_tables,
    denominator_guard,
    generate_geometry,
    rescale_sensing_tables_for_rcs,
)
from .oracle import greedy_objective, oracle_exhaustive
from .prior import perturbed_geometry, predicted_geometry
from .reporting import ReportingPlan, assign_fusion_nodes
from .selection import C2F_METHODS, METHODS
from .bundle_master import (
    BundleMasterResult,
    ObservationBundle,
    generate_restricted_bundles,
    joint_bundle_column_generation,
    joint_bundle_restricted_master,
    rcs_robust_bundle_column_generation,
    solve_restricted_bundle_master,
    bundle_cpu_cycles,
)
from .fusion_headroom import FusionHeadroom, fusion_headroom_diagnostic
from .simulate import run_one_trial, run_simulation, summarize
from .theory import (
    curvature_reference_bound,
    empirical_curvature,
    greedy_guarantee,
    same_objective_oracle,
    submodularity_audit,
    task_objective,
)
from .waveform import psf_local_capture, psf_main_bin, sweep_compare_analytic_vs_psf
from .active_system import (
    ActiveColumn,
    ActiveTransportHeadroom,
    GlobalActiveMasterResult,
    active_column_cpu_cycles,
    active_fusion_cpu_cycles,
    active_local_aggregation_cpu_cycles,
    active_observation_cpu_cycles,
    active_receiver_cpu_cycles,
    evaluate_active_transport_headroom,
    generate_active_columns,
    screen_fusion_candidates,
    solve_global_active_master,
)
from .low_rcs_rescue import (
    DetectableRcsBracket,
    RcsOperatingPoint,
    bracket_minimum_detectable_rcs,
)
from .active_statistics import paired_cluster_summary
from .coherent_oracle import (
    CoherentOracleInformation,
    coherent_group_gamma,
    coherent_oracle_information,
    evaluate_coherent_tx_detection_oracle,
)
from .scientific_gates import (
    CoherentPowerOracle,
    CommonLatentLlr,
    LocalLlrAggregation,
    PhysicalHeadroomDashboard,
    ReliabilityProtectionPlan,
    SymmetricComplementarityResult,
    aggregate_partitioned_llrs,
    allocate_reliability_protection,
    aggregation_reliability_threshold,
    coherent_power_oracle,
    common_latent_gaussian_llrs,
    erasure_received_kl,
    evidence_delivery_variance,
    gaussian_phase_coherence,
    hypothesis_dependent_erasure_kl,
    lower_tail_detection_summary,
    swerling_information,
    symmetric_complementarity_oracle,
)
from .evidence_acquisition import (
    EvidenceAcquisitionResult,
    EvidenceAcquisitionScope,
    solve_active_evidence_acquisition,
)
from .active_information import (
    ActiveDetectionResult,
    ActiveObservation,
    InformationPricingResult,
    SensingMode,
    aspect_scenario_factors,
    active_candidate_links,
    configured_sensing_modes,
    evaluate_active_detection,
    price_active_information_bundle,
    received_information,
)

__all__ = [
    "Config",
    "default_config",
    "apply_overrides",
    "apply_preset",
    "PRESETS",
    "iter_leaf_paths",
    "run_simulation",
    "run_one_trial",
    "summarize",
    "generate_geometry",
    "build_base_gains",
    "compute_link_tables",
    "denominator_guard",
    "rescale_sensing_tables_for_rcs",
    "BaseGains",
    "LinkTables",
    "METHODS",
    "C2F_METHODS",
    "ObservationBundle",
    "BundleMasterResult",
    "generate_restricted_bundles",
    "joint_bundle_column_generation",
    "rcs_robust_bundle_column_generation",
    "solve_restricted_bundle_master",
    "joint_bundle_restricted_master",
    "bundle_cpu_cycles",
    "FusionHeadroom",
    "fusion_headroom_diagnostic",
    "EXPERIMENTS",
    "ABLATION_VARIANTS",
    "DD_VARIANTS",
    "oracle_exhaustive",
    "greedy_objective",
    "same_objective_oracle",
    "task_objective",
    "submodularity_audit",
    "empirical_curvature",
    "greedy_guarantee",
    "curvature_reference_bound",
    "perturbed_geometry",
    "predicted_geometry",
    "ReportingPlan",
    "assign_fusion_nodes",
    "psf_main_bin",
    "psf_local_capture",
    "sweep_compare_analytic_vs_psf",
    "ActiveObservation",
    "ActiveDetectionResult",
    "InformationPricingResult",
    "SensingMode",
    "aspect_scenario_factors",
    "active_candidate_links",
    "configured_sensing_modes",
    "evaluate_active_detection",
    "price_active_information_bundle",
    "received_information",
    "ActiveColumn",
    "ActiveTransportHeadroom",
    "GlobalActiveMasterResult",
    "active_column_cpu_cycles",
    "active_fusion_cpu_cycles",
    "active_local_aggregation_cpu_cycles",
    "active_observation_cpu_cycles",
    "active_receiver_cpu_cycles",
    "evaluate_active_transport_headroom",
    "generate_active_columns",
    "screen_fusion_candidates",
    "solve_global_active_master",
    "DetectableRcsBracket",
    "RcsOperatingPoint",
    "bracket_minimum_detectable_rcs",
    "paired_cluster_summary",
    "CoherentOracleInformation",
    "coherent_group_gamma",
    "coherent_oracle_information",
    "evaluate_coherent_tx_detection_oracle",
    "CoherentPowerOracle",
    "CommonLatentLlr",
    "LocalLlrAggregation",
    "PhysicalHeadroomDashboard",
    "ReliabilityProtectionPlan",
    "SymmetricComplementarityResult",
    "aggregate_partitioned_llrs",
    "allocate_reliability_protection",
    "aggregation_reliability_threshold",
    "coherent_power_oracle",
    "common_latent_gaussian_llrs",
    "erasure_received_kl",
    "evidence_delivery_variance",
    "gaussian_phase_coherence",
    "hypothesis_dependent_erasure_kl",
    "lower_tail_detection_summary",
    "swerling_information",
    "symmetric_complementarity_oracle",
    "EvidenceAcquisitionResult",
    "EvidenceAcquisitionScope",
    "solve_active_evidence_acquisition",
]

__version__ = "1.6.0"
