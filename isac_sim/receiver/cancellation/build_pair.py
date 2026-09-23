"""cancellation 积木：成对假设观测。"""

from __future__ import annotations

from dataclasses import replace
from isac_sim.core.config import Config
import math
import numpy as np
from typing import Tuple

from isac_sim.receiver.cancellation.build import build_observation
from isac_sim.receiver.cancellation.containers import Observation
from isac_sim.receiver.cancellation.dictionaries import target_dictionary

def build_observation_pair(
    cfg: Config,
    geom_true,
    geom_belief,
    base,
    receiver: int,
    *,
    rng: np.random.Generator,
    direct_error_rng: np.random.Generator | None = None,
    sense_power: np.ndarray,
    radiated_power: np.ndarray,
    processing_gain: float,
    hw_gain: float,
    exclude_target: int,
    active_mask: np.ndarray | None = None,
    weak_index: int | None = None,
    share_noise: bool = False,
) -> Tuple[Observation, Observation]:
    """``(H1, H0)`` —— 一份几何、一份直达场、两个假设。

    调用 :func:`build_observation` 两次**不能**产生配对。每次调用都会消耗
    ``rng`` 去抽取直达相位 ``h_true``、回波相位 ``alpha_true`` 与噪声，于是
    第二次调用拿到的是一个**不同的**直达分量：在 600 m 场景的 trial 0 上实测，
    ``||x_direct||^2`` 从 ``5.59e-10`` 变成了 ``8.75e-10``。两个统计量随后之所以
    不同，是因为照射变了，而不是因为被测回波被移除了，于是基于它们建起来的
    每一个检测指标都塌到 ``P_D ~ P_FA`` —— 这正是观察到的事实（H1/H0 的统计量
    比值停在 1.00--1.15，而物理上承诺的是 ``+17 dB``）。

    因此 H0 是**由 H1 派生**出来的：把被测目标的列置零，于是两个观测按构造共享
    ``x_direct``、``X``、``A``，以及其它每个目标的回波。只有噪声被重抽（默认
    独立），这正是标定在 H0 池上的 CFAR 门限所需要的；当想要的是配对差而不是
    门限时，用 ``share_noise=True`` 复用 H1 的那次噪声抽样。

    其它目标的回波是**有意**留在 H0 里的：在本场景中，被测目标只是 ``Q`` 个
    散射体之一，其余的都是干扰，把它们移除会美化检测器。
    """
    obs1 = build_observation(
        cfg, geom_true, geom_belief, base, receiver,
        rng=rng, direct_error_rng=direct_error_rng,
        sense_power=sense_power, radiated_power=radiated_power,
        processing_gain=processing_gain, hw_gain=hw_gain,
        active_mask=active_mask, include_echo=True,
        weak_index=int(exclude_target) if weak_index is None else int(weak_index),
    )
    # ``Observation`` 把真值字典隐含地留在 ``targets`` 里，好让类型保持轻量；
    # 重建它既是确定性的也是廉价的。这是构造被测回波的唯一办法，因为信念字典
    # 会用错偏移。
    A_true = target_dictionary(
        cfg, obs1.targets, covariance_expanded=False
    )
    ids = np.asarray(obs1.A_true_target_ids)
    drop = ids == int(exclude_target)
    if not drop.any():
        raise ValueError(
            "exclude_target %d has no column in the true dictionary -- H0 "
            "would silently equal H1 and every detection metric would be "
            "meaningless" % int(exclude_target)
        )
    s_h0 = obs1.s_target - A_true[:, drop] @ obs1.alpha_true[drop]
    alpha_h0 = obs1.alpha_true.copy()
    alpha_h0[drop] = 0.0

    if share_noise:
        noise = obs1.y - obs1.x_direct - obs1.s_target
    else:
        n_bins = int(obs1.y.size)
        noise = (rng.normal(size=n_bins) + 1j * rng.normal(size=n_bins)) * math.sqrt(
            obs1.sigma2 / 2.0
        )
    obs0 = replace(
        obs1,
        y=obs1.x_direct + s_h0 + noise,
        s_target=s_h0,
        alpha_true=alpha_h0,
    )
    return obs1, obs0
