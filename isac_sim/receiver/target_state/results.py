"""共享目标偏移 MAP 的结果容器。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TargetStateMAPResult", "CrossfitTargetStateResult"]

_NAN = float("nan")


@dataclass(frozen=True)
class TargetStateMAPResult:
    """一次共享偏移 MAP 的结果。

    ``raw_delta_xy_m``（搜索器的原始输出）与 ``delta_xy_m``（门控后真正应用的）
    **必须分开**：否则分不清"搜索错了"还是"门控拒绝了"。阶段 A 门控恒接受，两
    者相等；阶段 D 上门控后会不等。

    诊断量（``gain_over_zero`` / ``peak_gap`` / ``boundary_hit`` /
    ``receiver_support``）只由 ``gated_topk`` 填：算 ``objective_at_zero`` 需要
    一次额外候选评价，而 ``coarse_to_fine`` 的评价集合是 bit-exact 冻结的（铁律 1），
    所以那里这些字段留在默认值。
    """

    target: int
    delta_xy_m: np.ndarray  # shape (2,)
    estimated_position_m: np.ndarray  # shape (3,)
    objective: float
    prior_penalty: float
    receiver_gain: np.ndarray  # 每个接收机的匹配收益 G_j
    converged: bool
    evaluations: int  # 去重后的候选数
    solver: str = "coarse_to_fine"
    accepted: bool = True
    raw_delta_xy_m: np.ndarray | None = None
    objective_at_zero: float = _NAN
    objective_second: float = _NAN
    gain_over_zero: float = _NAN
    peak_gap: float = _NAN
    boundary_hit: bool = False
    boundary_expanded: bool = False
    boundary_unresolved: bool = False
    receiver_support: int = 0
    screen_receiver_count: int = 0
    coarse_candidates: int = 0
    full_candidates: int = 0
    receiver_evaluations: int = 0


@dataclass(frozen=True)
class CrossfitTargetStateResult:
    """双折 cross-fit 的结果（方案 §3）。"""

    target: int
    statistic: float  # T_CF = T_A + T_B
    fold_statistics: tuple[float, float]
    delta_a_xy_m: np.ndarray  # 在 CPI A 上估出的偏移（用在 B 上）
    delta_b_xy_m: np.ndarray  # 在 CPI B 上估出的偏移（用在 A 上）
    objective_a: float
    objective_b: float
    evaluations: int
    seconds: float
    converged: bool = True
    accepted_a: bool = True
    accepted_b: bool = True
    raw_delta_a_xy_m: np.ndarray | None = None
    raw_delta_b_xy_m: np.ndarray | None = None
