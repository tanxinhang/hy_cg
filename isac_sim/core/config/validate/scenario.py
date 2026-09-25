"""规模、检测样本数与链路相关系数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from isac_sim.core.config.run import Config


def check_scenario(cfg: Config) -> None:
    """规模、检测样本数与链路相关系数。"""
    if cfg.scale.M < 2:
        raise ValueError("scale.M must be at least two")
    if cfg.scale.Q < 1:
        raise ValueError("scale.Q must be at least one")
    if (not math.isfinite(cfg.geometry.min_uav_separation_m)
            or cfg.geometry.min_uav_separation_m < 0.0):
        raise ValueError(
            "geometry.min_uav_separation_m must be finite and non-negative"
        )
    if cfg.run.num_mc < 1:
        raise ValueError("run.num_mc must be at least one")
    if not math.isfinite(cfg.detect.Pfa_target) or not 0.0 < cfg.detect.Pfa_target < 1.0:
        raise ValueError("detect.Pfa_target must lie in (0, 1)")
    if cfg.detect.n_looks < 1:
        raise ValueError("detect.n_looks must be at least one")
    if cfg.detect.sensing_dwell_s is not None:
        from isac_sim.core.config.timing import looks_from_dwell
        looks_from_dwell(cfg, cfg.detect.sensing_dwell_s)
    if cfg.detect.target_response_model not in (
        "unit_phase", "deterministic_unknown", "swerling1_shared",
        "swerling2_fast",
    ):
        raise ValueError("unknown detect.target_response_model")
    if cfg.detect.num_h1_per_target < 1:
        raise ValueError("detect.num_h1_per_target must be at least one")
    if cfg.detect.num_false_per_target < 1:
        raise ValueError("detect.num_false_per_target must be at least one")
    corr_values = (
        cfg.corr.rho_tx,
        cfg.corr.rho_rx,
        cfg.corr.rho_target,
        cfg.corr.rho_dd,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in corr_values):
        raise ValueError("correlation coefficients must be finite and non-negative")
    if sum(corr_values) > 1.0 + 1e-12:
        raise ValueError("correlation coefficients must sum to at most one")
