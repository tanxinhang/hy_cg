"""目标先验、接收端对消参数、选择器评分口径与检测门限。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from isac_sim.core.config.run import Config


def check_proposal(cfg: Config) -> None:
    """目标先验、接收端对消参数、选择器评分口径与检测门限。"""
    if cfg.prior.scheduler_rcs.lower() not in {"realized", "mean"}:
        raise ValueError(
            f"Unknown prior.scheduler_rcs={cfg.prior.scheduler_rcs!r}; "
            "expected 'realized' or 'mean'"
        )
    if not 0.0 < cfg.prior.rcs_lower_factor <= 1.0:
        raise ValueError("prior.rcs_lower_factor must lie in (0, 1]")
    if int(cfg.cancellation.n_cpi) < 1:
        raise ValueError("cancellation.n_cpi must be positive")
    if str(cfg.cancellation.residual_accounting) not in {"measured", "structural"}:
        raise ValueError(
            "cancellation.residual_accounting must be 'measured' or 'structural'"
        )
    if cfg.cancellation.adaptive_soft_enable:
        grid = tuple(float(v) for v in cfg.cancellation.adaptive_soft_mu_grid)
        if not grid or any(not math.isfinite(v) or v < 0.0 for v in grid):
            raise ValueError(
                "adaptive soft mu grid must be finite, non-empty, and non-negative"
            )
        slack = float(cfg.cancellation.adaptive_soft_risk_slack)
        if not math.isfinite(slack) or not 0.0 <= slack <= 1.0:
            raise ValueError("adaptive soft risk slack must lie in [0, 1]")
        quantile = float(cfg.cancellation.adaptive_soft_residual_quantile)
        if not math.isfinite(quantile) or not 0.5 <= quantile < 1.0:
            raise ValueError(
                "adaptive soft residual quantile must lie in [0.5, 1)"
            )
    if int(cfg.cancellation.tangent_order) not in (0, 1):
        raise ValueError(
            "cancellation.tangent_order currently supports only 0 or 1"
        )
    if int(cfg.cancellation.interference_tangent_order) not in (0, 1):
        raise ValueError(
            "cancellation.interference_tangent_order currently supports only 0 or 1"
        )
    if not math.isfinite(cfg.cancellation.tangent_step_bins) or (
        cfg.cancellation.tangent_step_bins <= 0.0
    ):
        raise ValueError("cancellation.tangent_step_bins must be positive")
    for _name in ("direct_estimation_sigma_delay_bins",
                  "direct_estimation_sigma_doppler_bins"):
        _v = float(getattr(cfg.cancellation, _name))
        if not math.isfinite(_v) or _v < 0.0:
            raise ValueError(f"cancellation.{_name} must be finite and non-negative")
    if cfg.selector.score_mode.lower() not in {"first_order", "exact_utility", "detector_pd"}:
        raise ValueError(
            f"Unknown selector.score_mode={cfg.selector.score_mode!r}; "
            "expected 'first_order', 'exact_utility', or 'detector_pd'"
        )
    if not 0.0 < cfg.detect.pd_required <= 1.0:
        raise ValueError("detect.pd_required must lie in (0, 1]")
    if not 0.0 < cfg.detect.weak_pd_required <= 1.0:
        raise ValueError("detect.weak_pd_required must lie in (0, 1]")
    if cfg.detect.target_rcs <= 0.0:
        raise ValueError("detect.target_rcs must be positive and is measured in m^2")
