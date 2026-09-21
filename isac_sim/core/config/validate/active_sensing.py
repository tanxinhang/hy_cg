"""主动感知的模态表、能量预算、候选族与信息度量。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math
from numbers import Integral

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from isac_sim.core.config.run import Config


def check_active_sensing(cfg: Config) -> None:
    """主动感知的模态表、能量预算、候选族与信息度量。"""
    active = cfg.active_sensing
    mode_lengths = {
        len(active.mode_names), len(active.power_scales),
        len(active.looks), len(active.refined),
    }
    if len(mode_lengths) != 1 or not active.mode_names:
        raise ValueError("active sensing mode fields must have equal non-zero lengths")
    if not all(isinstance(name, str) and name.strip() for name in active.mode_names):
        raise ValueError("active sensing mode names must be non-empty strings")
    if len(set(active.mode_names)) != len(active.mode_names):
        raise ValueError("active sensing mode names must be unique")
    if not all(math.isfinite(value) and value > 0.0 for value in active.power_scales):
        raise ValueError("active sensing power scales must be finite and positive")
    if not all(
        isinstance(value, Integral) and not isinstance(value, bool) and value >= 1
        for value in active.looks
    ):
        raise ValueError("active sensing look counts must be positive integers")
    if not all(isinstance(value, bool) for value in active.refined):
        raise ValueError("active sensing refinement flags must be Boolean")
    if (
        not math.isfinite(active.energy_budget_per_target)
        or active.energy_budget_per_target <= 0.0
    ):
        raise ValueError("active sensing energy budget must be finite and positive")
    if (
        not math.isfinite(active.energy_budget_per_uav)
        or active.energy_budget_per_uav <= 0.0
    ):
        raise ValueError("active sensing per-UAV energy budget must be finite and positive")
    if active.max_tx_observations_per_uav < -1:
        raise ValueError("active sensing transmitter cap must be -1 or non-negative")
    if not all(
        math.isfinite(value) and value >= 0.0 for value in (
            active.matched_filter_cycles_per_look,
            active.llr_cycles_per_look,
            active.dd_refine_cycles,
        )
    ):
        raise ValueError("active sensing computation costs must be finite and non-negative")
    if active.max_candidates_per_pair < 1 or active.branch_node_limit < 1:
        raise ValueError("active sensing search limits must be positive")
    if active.candidate_strategy not in {"scenario_union", "robust_singleton", "full"}:
        raise ValueError(
            "active sensing candidate strategy must be scenario_union, robust_singleton, or full"
        )
    if (
        active.scenario_topk_per_scenario < 0
        or active.complementary_pair_topk < 0
        or active.complete_pool_max_links < 1
    ):
        raise ValueError("active sensing candidate-family limits are invalid")
    if active.information_metric not in {"forward_kl", "jeffreys"}:
        raise ValueError("active sensing information metric must be 'forward_kl' or 'jeffreys'")
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (active.energy_price, active.report_price)
    ):
        raise ValueError("active sensing prices must be finite and non-negative")
    if not active.aspect_angles_deg or not all(
        math.isfinite(value) for value in active.aspect_angles_deg
    ):
        raise ValueError("active sensing aspect angles must be finite and non-empty")
    if not math.isfinite(active.aspect_floor) or not 0.0 < active.aspect_floor <= 1.0:
        raise ValueError("active sensing aspect floor must lie in (0, 1]")
