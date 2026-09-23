"""接收端侧入参的校验与"感知块能否复用"的判定。

这里的每条检查都对应一种**静默失效**：开关拼错被当成 "off"、同时给了残余
比例与残余功率、或在接收机模型不一致时误触发快速路径 —— 任一发生，返回的
表都会报出"假设"的数字，而调用方毫无察觉。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isac_sim.core.config import Config


@dataclass
class ReceiverInputs:
    """解析后的接收端侧入参。"""

    #: 目标存活因子（``eta_surv``），形状 (M,) 或 (M,Q)；None = 分子逐位不变
    retention: np.ndarray | None
    #: 是否使用了"算法型"接收机（而非冻结常数）
    algorithm_receiver: bool
    #: 感知块能否从 ``reuse_from`` 原样拷贝（见下方判定条件）
    can_reuse_sensing: bool


def resolve_inputs(
    cfg: Config,
    *,
    dd_gain: np.ndarray | None,
    active_tx_mask: np.ndarray | None,
    reuse_from: object | None,
    sensing_power_scale_by_uav: np.ndarray | None,
    residual_fraction_by_receiver: np.ndarray | None,
    residual_power_by_receiver_target: np.ndarray | None,
    target_retention_by_receiver: np.ndarray | None,
) -> ReceiverInputs:
    """校验入参并判定快速路径。校验一律前置，不放在分支里面。"""
    M, Q = cfg.scale.M, cfg.scale.Q
    r = cfg.radio

    # ---- 接收端对消：还是那个冻结常数吗？ --------------------------------
    # 只要不是常数 —— 实测的逐接收机比例，或解析桥 —— 就改变了**感知**块，
    # 于是下面的 can_reuse_sensing 必须为假：快速路径是把 reuse_from 的感知量
    # 原样拷过来的，而那张表是用它自己那次调用的接收机模型建的。
    # 两者混用是最糟的静默失效 —— 调用方传了实测比例，返回的表却报常数的数字，
    # 而且什么异常都不会抛。校验放在前面而不是分支里面，也是这个原因。
    if cfg.cancellation.mode not in ("off", "predict", "measure"):
        # 开关拼错绝不能当成 "off" 读：静默退回冻结常数是这个文件里最贵的
        # 失效模式 —— 跑出来的东西看着"配好了算法"，报的却是假设的数字。
        raise ValueError(
            "Unknown cancellation.mode=%r; expected 'off', 'predict' or 'measure'"
            % (cfg.cancellation.mode,)
        )
    if residual_fraction_by_receiver is not None and residual_power_by_receiver_target is not None:
        raise ValueError(
            "pass either residual_fraction_by_receiver or "
            "residual_power_by_receiver_target, not both"
        )
    if (
        cfg.cancellation.mode == "measure"
        and residual_fraction_by_receiver is None
        and residual_power_by_receiver_target is None
    ):
        raise ValueError(
            "cancellation.mode='measure' needs the per-trial geometry to run the "
            "estimator, which compute_link_tables does not receive. Pass "
            "residual_power_by_receiver_target=... from "
            "measure_receiver_context(...).as_residual_power(), or use the "
            "legacy residual_fraction_by_receiver bridge."
        )

    # ---- 分子的另一半：同一仿射映射下的回波存活率 ------------------------
    retention = None
    if target_retention_by_receiver is not None:
        retention = np.asarray(target_retention_by_receiver, dtype=float)
        if retention.shape not in ((M,), (M, Q)):
            raise ValueError(
                "target_retention_by_receiver must have shape (%d,) or (%d, %d), "
                "got %s" % (M, M, Q, np.shape(target_retention_by_receiver))
            )
        if not np.all(np.isfinite(retention)) or np.any(retention < 0.0):
            raise ValueError("target_retention_by_receiver must be finite and non-negative")
        if np.any(retention > 1.0):
            # 存活率大于 1 不是接收机挣到的增益：联合阶段会把"投影到该目标块上的
            # 其它一切"留在那里，所以这个比值可以超过 1（实测 1.001）。宁可拒绝
            # 也不要静默裁剪 —— 传进这种值的人几乎一定是漏了 as_retention()，
            # 接下来会把检测统计量乘上一个假象。
            raise ValueError(
                "target_retention_by_receiver must not exceed 1.0; max is %.6f. "
                "Use measure_receiver_context(...).as_retention(), which clamps."
                % float(np.max(retention))
            )

    algorithm_receiver = (
        residual_fraction_by_receiver is not None
        or residual_power_by_receiver_target is not None
        or target_retention_by_receiver is not None
        or cfg.cancellation.mode == "predict"
    )

    # 感知量本来与通信干扰无关，例外是 "reliable_comm_assisted" 功率模型 ——
    # 那里有效感知功率被 chi_comm 抬高了。
    # 感知量依赖 dd_gain，所以只有在调用方没有覆盖它时快速路径才成立
    # （active_set 的复评会传 dd_gain=None 并复用选择阶段的感知块）。
    can_reuse_sensing = (
        reuse_from is not None
        and r.P_sense_by_uav is None
        and r.P_comm_by_uav is None
        and r.rho_by_uav is None
        and sensing_power_scale_by_uav is None
        and dd_gain is None
        and not algorithm_receiver
        and r.isac_power_model != "reliable_comm_assisted"
        # 在 active-set 耦合下感知分母里含"当前在发的上报载荷"，必须重建。
        # 正交上报在感知观测期间没有载荷项，所以它的感知块与调度无关、可以复用。
        and not (
            cfg.interference.coupling == "shared_spectrum"
            and active_tx_mask is not None
        )
    )
    return ReceiverInputs(
        retention=retention,
        algorithm_receiver=algorithm_receiver,
        can_reuse_sensing=can_reuse_sensing,
    )
