"""唯一 production path。

正式系统链路（论文系统口径，固定不变）::

    generate scenario
          -> build belief
          -> receiver certificate        (Module A: belief-aware TP-UIC)
          -> receiver-aware link quality
          -> C2F selection               (Module B: receiver-aware cooperative selection)
          -> soft fusion                 (Module C: detection / fusion)
          -> detection evaluation

**本模块是正式系统的唯一入口。** 实验脚本一律调用 :func:`run_proposed_trial` 或
:func:`run_baseline_trial`，不得再各自拼装一遍流程。

proposed 与 baseline **共用同一条 pipeline**，差异只发生在三个可替换位：
``receiver`` / ``selector`` / ``fusion``。这保证比较时不会出现
"proposed 用真实 TP-UIC、baseline 用固定 kappa" 这类口径错配。

本模块刻意**不重新实现**任何数学：它委托 :func:`isac_sim.experiments.flow.simulate.run_one_trial`，
因此在默认配置下与既有结果逐位一致。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from isac_sim.core.config import Config
from experiments.flow.simulate import MethodResult, run_one_trial

# ---------------------------------------------------------------- 常量

#: 正式 proposed 方法名。这是论文主方法，也是 paired 比较的锚。
PROPOSED_METHOD: str = "proposed_c2f"

#: 属于 proposed 家族的方法（ablations 可在此范围内变体）。
PROPOSED_METHODS: Tuple[str, ...] = (
    "proposed_c2f",
    "proposed_c2f_pd",
    "proposed_c2f_full",
    "proposed_c2f_full_pd",
    "proposed_c2f_adaptive",
    "proposed_c2f_adaptive_pd",
    "proposed_c2f_adaptive_pd_distributed",
    "proposed_c2f_adaptive_pd_robust",
    "proposed_c2f_adaptive_pd_calibrated",
    "proposed_c2f_adaptive_pd_fusion_polish",
)

#: 正式链路的七个阶段。invariant 测试用它来断言 production path 没被绕过。
PIPELINE_STAGES: Tuple[str, ...] = (
    "generate_scenario",
    "build_belief",
    "receiver_certificate",
    "receiver_aware_link_quality",
    "c2f_selection",
    "soft_fusion",
    "detection_evaluation",
)


# ---------------------------------------------------------------- 入口


def run_trial(
    cfg: Config,
    trial_index: int,
    method: str = PROPOSED_METHOD,
) -> MethodResult:
    """在一个共享几何上跑**单个**方法的一次 trial。

    这是 proposed 与全部 baseline 的共同底座。
    """
    results: Dict[str, MethodResult] = run_one_trial(
        cfg, trial_index, methods=[method]
    )
    return results[method]


def run_proposed_trial(
    cfg: Config,
    trial_index: int,
    method: str = PROPOSED_METHOD,
) -> MethodResult:
    """**唯一正式 proposed 实现。**

    Parameters
    ----------
    cfg
        完整配置。主口径预设为 ``target-local-v1`` / ``paper-canonical``。
    trial_index
        Monte-Carlo trial 下标（配对比较依赖它与 seed 共同决定 RNG）。
    method
        proposed 家族内的方法名；默认 :data:`PROPOSED_METHOD`。
        传入非 proposed 家族的名字会报错 —— proposed 的语义不属于本函数。
    """
    if method not in PROPOSED_METHODS:
        raise ValueError(
            f"{method!r} 不是 proposed 家族方法；"
            f"baseline 请改用 run_baseline_trial()。"
        )
    return run_trial(cfg, trial_index, method=method)


def run_baseline_trial(
    cfg: Config,
    trial_index: int,
    method: str,
) -> MethodResult:
    """跑一个 baseline。与 proposed **共用同一条 pipeline**。

    差异只来自 ``method`` 指定的 selector / fusion，receiver 物理模型保持一致。
    """
    if method in PROPOSED_METHODS:
        raise ValueError(
            f"{method!r} 属于 proposed 家族；请改用 run_proposed_trial()。"
        )
    return run_trial(cfg, trial_index, method=method)


def run_comparison(
    cfg: Config,
    trial_index: int,
    methods: Optional[Sequence[str]] = None,
) -> Dict[str, MethodResult]:
    """在同一几何上跑一组方法（配对比较）。

    所有方法共享同一个 geometry、同一条 detector 随机流，
    因此跨方法的差值是**配对**的，可直接做边际差与 CI。
    """
    roster: Optional[List[str]] = list(methods) if methods is not None else None
    return run_one_trial(cfg, trial_index, methods=roster)


__all__ = [
    "PROPOSED_METHOD",
    "PROPOSED_METHODS",
    "PIPELINE_STAGES",
    "run_trial",
    "run_proposed_trial",
    "run_baseline_trial",
    "run_comparison",
]
