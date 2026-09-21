"""cancellation 积木：观测构造的共享上下文。"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.receiver.cancellation.steering import _u_of


class BuildContext:
    """构造一次观测所需的共享量。

    把 ``build_observation`` 的十余个入参收敛成一个对象，这样「直达源列表 /
    目标源列表 / 目标归属」三块可以各自独立成积木，而不必各自重复接收
    十几个位置参数。字段即原函数体里的局部变量，语义一一对应。
    """

    def __init__(
        self,
        cfg: Config,
        geom_true,
        geom_belief,
        base,
        receiver: int,
        *,
        sense_power: np.ndarray,
        radiated_power: np.ndarray,
        processing_gain: float,
        hw_gain: float,
        active_mask: np.ndarray | None = None,
        base_belief=None,
    ) -> None:
        self.cfg = cfg
        self.geom_true = geom_true
        self.geom_belief = geom_belief
        self.base = base
        self.base_belief = base_belief
        self.receiver = receiver
        self.sense_power = sense_power
        self.radiated_power = radiated_power
        self.processing_gain = float(processing_gain)
        self.hw_gain = float(hw_gain)
        M = cfg.scale.M
        self.active_mask = (
            np.ones(M, dtype=bool) if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        # 方位只在该接收机带孔径时才有意义。目标与直达共用同一套
        # 信念/真值划分：信念字典是接收机能 steer 的方向，真值字典是回波
        # 真正所在的方向 —— 这正是实验要定价的不对称。
        self.m_rx = int(cfg.aperture.m_rx) if cfg.aperture.enable else 1
        self.axis = int(cfg.aperture.axis)
        self.p_rx = None
        self.u_tgt_true = None
        self.u_tgt_belief = None
        if self.m_rx > 1:
            self.p_rx = np.asarray(geom_true.p_uav[receiver], dtype=float)[:3]
            self.u_tgt_true = _u_of(geom_true.p_tgt, self.p_rx, cfg.scale.Q, self.axis)
            self.u_tgt_belief = _u_of(
                geom_belief.p_tgt, self.p_rx, cfg.scale.Q, self.axis
            )
