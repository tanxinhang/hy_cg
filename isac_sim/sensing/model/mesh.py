"""UAV-UAV 直连网格：距离、连通掩码与直连增益。

⚠️ 本函数是随机数契约的**第一段**：阴影衰落 ``rng.normal`` → 莱斯增益。
"""
from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model.containers import BaseGains, Geometry
from isac_sim.sensing.model.mathkit import path_gain, rician_power_gain


def build_uav_mesh(cfg: Config, geom: Geometry, rng: np.random.Generator, channel: BaseGains | None):
    """建 UAV-UAV 直连网格。

    给了 ``channel`` 就复用它已实现的物理量：UAV 间衰落是物理量，
    不能因为跟踪器的 belief 变了就重抽。
    """
    M = cfg.scale.M
    d = cfg.detect
    if channel is not None:
        return channel.d_uu, channel.edge_mask, channel.direct_gain
    diff_uu = geom.p_uav[:, None, :] - geom.p_uav[None, :, :]
    d_uu = np.linalg.norm(diff_uu, axis=-1)
    edge_mask = (d_uu <= cfg.geometry.comm_range) & (~np.eye(M, dtype=bool))

    direct_gain = path_gain(d_uu, cfg)
    direct_gain *= 10.0 ** (rng.normal(0.0, d.shadow_std_db, size=(M, M)) / 10.0)
    direct_gain *= rician_power_gain((M, M), d.rician_K_db, rng)
    np.fill_diagonal(direct_gain, 0.0)
    return d_uu, edge_mask, direct_gain
