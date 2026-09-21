"""基础增益装配。

⚠️ **随机数调用顺序是逐位契约**，``build_base_gains`` 只负责按序串联：

1. :func:`~isac_sim.sensing.model.mesh.build_uav_mesh` —— 阴影衰落 → 莱斯增益
2. :func:`~isac_sim.sensing.model.bistatic._bistatic_fields` —— 共享 RCS → 逐链路 RCS
3. 碰撞计数与精化损耗：不抽随机数

自身不做任何计算，因此把任一段换成等价实现时，
**只要它消耗随机数的次序不变**，结果就逐位保持。
"""
from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model.bistatic import _bistatic_fields
from isac_sim.sensing.model.collision import build_dd_collision_count
from isac_sim.sensing.model.containers import BaseGains, Geometry
from isac_sim.sensing.model.eta import build_eta_arrays
from isac_sim.sensing.model.mesh import build_uav_mesh


def build_base_gains(
    cfg: Config,
    geom: Geometry,
    rng: np.random.Generator,
    channel: BaseGains | None = None,
    rcs_view: str = "realized",
) -> BaseGains:
    """为给定几何建基础增益。

    给了 ``channel`` 就复用它已验证的 UAV-UAV 物理实现。在
    ``rcs_view="realized"`` 下它同时提供目标 RCS（旧行为）；在
    ``rcs_view="mean"`` 下目标增益用配置的均值 RCS，于是调度器在感知之前
    看不到本 CPI 的 RCS 实现值。
    """
    d_uu, edge_mask, direct_gain = build_uav_mesh(cfg, geom, rng, channel)
    f = _bistatic_fields(cfg, geom, rng, channel, rcs_view)
    collision = build_dd_collision_count(cfg, geom, f.valid_dd, f.delay_bin, f.doppler_bin)
    eta_loc, eta_fine = build_eta_arrays(
        cfg, f.valid_dd, f.l_float_grid, f.k_float_grid, f.dd_frac_loss
    )
    return BaseGains(
        edge_mask=edge_mask,
        direct_gain=direct_gain,
        d_uu=d_uu,
        d_uav_tgt=f.d_uav_tgt,
        tau=f.tau,
        doppler=f.doppler,
        delay_bin=f.delay_bin,
        doppler_bin=f.doppler_bin,
        valid_dd=f.valid_dd,
        dd_frac_loss=f.dd_frac_loss,
        dd_collision_count=collision,
        target_gain=f.target_gain,
        geom_factor=f.geom_factor,
        aspect_azimuth=f.aspect_azimuth,
        rcs_fluct=f.rcs_fluct,
        eta_loc=eta_loc,
        eta_fine=eta_fine,
        delay_frac=f.delay_frac_full,
        doppler_frac=f.doppler_frac_full,
    )
