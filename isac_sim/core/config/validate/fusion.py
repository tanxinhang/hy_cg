"""融合规则、CPU 预算与各容量上限。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from isac_sim.core.config.run import Config


def check_fusion(cfg: Config) -> None:
    """融合规则、CPU 预算与各容量上限。"""
    if cfg.fusion.rule.lower() not in {
        "max_in_rate", "max_min_rate", "nearest_target", "nearest_centroid",
        "nearest_target_capacitated",
        "capacitated_value", "capacitated_pd_lookahead",
    }:
        raise ValueError(
            f"Unknown fusion.rule={cfg.fusion.rule!r}; expected 'max_in_rate', "
            "'max_min_rate', 'nearest_target', 'nearest_target_capacitated', "
            "'capacitated_value', or 'capacitated_pd_lookahead'"
        )
    if cfg.fusion.max_targets_per_uav < -1 or cfg.fusion.max_targets_per_uav == 0:
        raise ValueError("fusion.max_targets_per_uav must be -1 or positive")
    if not math.isfinite(cfg.fusion.processing_window_s) or cfg.fusion.processing_window_s <= 0.0:
        raise ValueError("fusion.processing_window_s must be positive")
    if (
        not math.isfinite(cfg.fusion.cpu_rate_cycles_per_s)
        or cfg.fusion.cpu_rate_cycles_per_s == 0.0
    ):
        raise ValueError("fusion.cpu_rate_cycles_per_s must be negative (off) or positive")
    cpu_costs = (
        cfg.fusion.cpu_fixed_cycles,
        cfg.fusion.cpu_per_observation_cycles,
        cfg.fusion.cpu_cubic_cycles,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in cpu_costs):
        raise ValueError("fusion CPU cost coefficients must be finite and non-negative")
    if cfg.prior.search_gate_sigma < 0:
        raise ValueError("prior.search_gate_sigma must be non-negative")
    if not 0.0 < cfg.prior.robust_position_confidence < 1.0:
        raise ValueError("prior.robust_position_confidence must lie in (0, 1)")
    if cfg.run.workers < 1:
        raise ValueError("run.workers must be at least one")
    if cfg.selector.max_local_observations_per_target < -1:
        raise ValueError(
            "selector.max_local_observations_per_target must be -1 or non-negative"
        )
    if cfg.selector.max_remote_reports < -1:
        raise ValueError("selector.max_remote_reports must be -1 or non-negative")
    if cfg.selector.max_observations_per_receiver < -1:
        raise ValueError(
            "selector.max_observations_per_receiver must be -1 or non-negative"
        )
    if cfg.selector.max_observations_per_fusion_uav < -1:
        raise ValueError(
            "selector.max_observations_per_fusion_uav must be -1 or non-negative"
        )
    if cfg.selector.bundle_shortlist_per_type < 1:
        raise ValueError("selector.bundle_shortlist_per_type must be positive")
    if cfg.selector.bundle_cg_max_iterations < 1:
        raise ValueError("selector.bundle_cg_max_iterations must be positive")
    if cfg.selector.bundle_pricing_tolerance < 0.0:
        raise ValueError("selector.bundle_pricing_tolerance must be non-negative")
    if cfg.selector.bundle_exact_pricing_max_candidates < 0:
        raise ValueError(
            "selector.bundle_exact_pricing_max_candidates must be non-negative"
        )
