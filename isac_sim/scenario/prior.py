"""目标状态先验扰动。

一位审稿人问了一个显然但尖锐的问题：``D_q`` 以及每条链路 ``(i, j, q)`` 在 DD
域上的可行性，都依赖**被预测的**目标状态 ``p_q, v_q`` —— 而这个预测本身来自
跟踪器，跟踪器又是由我们正在调度的那些链路喂出来的。论文把这一层留白了。

本模块给出 ``prior-sweep`` 实验所用的轻量、OTFS 忠实的目标先验不确定度模型。
它移植/改编自 ``CodeCg/gate_otfs_collision/experiments.py``
（``predict_tracks``、``predict_tracks_with_crn_noise``、
``perturb_scene_prediction``），改成作用在
:class:`isac_sim.sensing.model.Geometry` 上的纯函数。

支持两种框架：

* **鲁棒性**（``pos_sigma_m > 0`` 或 ``vel_sigma_mps > 0``）。整个仿真器消费
  **被扰动的**目标状态，于是结果回答的是"先验错了之后 P_D 掉多少"。这是工程
  实践者会问的问题，也正是 ``prior-sweep`` 模式所报的东西。
* **先预测后行动**（由 :func:`predicted_geometry` 提供）。给定跟踪器输出，给出
  被预测的状态。尚未接进仿真循环，但可供下游实验用来固定检测侧的几何。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model import Geometry


def perturbed_geometry(
    cfg: Config,
    geom: Geometry,
    sigma_pos_m: float,
    sigma_vel_mps: float,
    rng: np.random.Generator,
) -> Geometry:
    """返回 ``geom`` 的一份拷贝，其目标状态被高斯噪声扰动过。

    UAV 状态不动 —— 只扰动目标，因为调度侧的不确定度就在那里。
    ``sigma_pos_m = 0`` 或 ``sigma_vel_mps = 0`` 给出该轴上未扰动的状态，且
    **不消耗**任何 RNG 抽样（这对逐位一致很重要）。
    """
    if sigma_pos_m <= 0 and sigma_vel_mps <= 0:
        return Geometry(
            p_uav=geom.p_uav.copy(),
            v_uav=geom.v_uav.copy(),
            p_tgt=geom.p_tgt.copy(),
            v_tgt=geom.v_tgt.copy(),
        )

    p_tgt = geom.p_tgt.copy()
    v_tgt = geom.v_tgt.copy()
    if sigma_pos_m > 0:
        # 在三维里扰动，但把目标保持在标称高度上（``v_tgt`` 本来也基本是二维的：
        # ``cfg.geometry`` 把目标建在零高度并让它们在平面内运动）。
        eps = rng.normal(0.0, sigma_pos_m, size=p_tgt.shape)
        p_tgt = p_tgt + eps
        p_tgt[:, 2] = geom.p_tgt[:, 2]
    if sigma_vel_mps > 0:
        eps = rng.normal(0.0, sigma_vel_mps, size=v_tgt.shape)
        v_tgt = v_tgt + eps
        v_tgt[:, 2] = 0.0
    return Geometry(
        p_uav=geom.p_uav.copy(),
        v_uav=geom.v_uav.copy(),
        p_tgt=p_tgt,
        v_tgt=v_tgt,
    )


def predicted_geometry(
    cfg: Config,
    geom: Geometry,
    rng: np.random.Generator,
    sigma_pos_m: float | None = None,
    sigma_vel_mps: float | None = None,
) -> Geometry:
    """匀速、KF 风格的预测。

    ``p_pred = p + v * dt`` 且 ``v_pred = v``，再叠加幅度可配的高斯扰动。
    ``dt`` 取自 ``cfg.prior.dt_s``，两个 sigma 默认取 ``cfg.prior.sigma_pos_m``
    与 ``cfg.prior.sigma_vel_mps``。
    """
    sigma_p = cfg.prior.sigma_pos_m if sigma_pos_m is None else sigma_pos_m
    sigma_v = cfg.prior.sigma_vel_mps if sigma_vel_mps is None else sigma_vel_mps
    dt = cfg.prior.dt_s

    p_pred = geom.p_tgt + geom.v_tgt * dt
    v_pred = geom.v_tgt.copy()

    if sigma_p > 0:
        p_pred = p_pred + rng.normal(0.0, sigma_p, size=p_pred.shape)
        p_pred[:, 2] = geom.p_tgt[:, 2]
    if sigma_v > 0:
        v_pred = v_pred + rng.normal(0.0, sigma_v, size=v_pred.shape)
        v_pred[:, 2] = 0.0

    return Geometry(
        p_uav=geom.p_uav.copy(),
        v_uav=geom.v_uav.copy(),
        p_tgt=p_pred,
        v_tgt=v_pred,
    )


def crn_noise_pair(
    track_pos: np.ndarray,
    track_vel: np.ndarray,
    sigma_pos_m: float,
    sigma_vel_mps: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """公共随机数（CRN）扰动的原语。

    返回给定标准差的 ``(pos_eps, vel_eps)``。两个方法若从同一个 RNG 消费同一批
    原语，就看到完全相同的实现 —— 这正是让配对比较的置信区间收紧的 CRN 技巧。
    与兄弟包 ``gate_otfs_collision`` 里的 ``predict_tracks_with_crn_noise`` 一致。
    """
    pos_eps = rng.normal(0.0, 1.0, size=track_pos.shape) if sigma_pos_m > 0 else np.zeros_like(track_pos)
    vel_eps = rng.normal(0.0, 1.0, size=track_vel.shape) if sigma_vel_mps > 0 else np.zeros_like(track_vel)
    return pos_eps, vel_eps
