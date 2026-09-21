"""部署几何与运动学采样。"""

from __future__ import annotations

import numpy as np
from isac_sim.core.config import Config
from isac_sim.sensing.model.containers import Geometry


# ==========================================================================
# Geometry sampling
# ==========================================================================
def generate_geometry(cfg: Config, rng: np.random.Generator) -> Geometry:
    g = cfg.geometry
    M, Q = cfg.scale.M, cfg.scale.Q

    p_uav = np.zeros((M, 3))
    p_uav[:, :2] = rng.uniform(0.0, g.area_xy, size=(M, 2))
    p_uav[:, 2] = rng.uniform(g.h_uav_min, g.h_uav_max, size=M)

    p_tgt = np.zeros((Q, 3))
    p_tgt[:, :2] = rng.uniform(0.0, g.area_xy, size=(Q, 2))
    p_tgt[:, 2] = rng.uniform(g.h_target_min, g.h_target_max, size=Q)

    v_uav = np.zeros((M, 3))
    sp_u = rng.uniform(g.uav_speed_min, g.uav_speed_max, size=M)
    ang_u = rng.uniform(0.0, 2.0 * np.pi, size=M)
    v_uav[:, 0] = sp_u * np.cos(ang_u)
    v_uav[:, 1] = sp_u * np.sin(ang_u)

    v_tgt = np.zeros((Q, 3))
    sp_t = rng.uniform(g.target_speed_min, g.target_speed_max, size=Q)
    ang_t = rng.uniform(0.0, 2.0 * np.pi, size=Q)
    v_tgt[:, 0] = sp_t * np.cos(ang_t)
    v_tgt[:, 1] = sp_t * np.sin(ang_t)

    geom = Geometry(p_uav=p_uav, v_uav=v_uav, p_tgt=p_tgt, v_tgt=v_tgt)

    # 目标状态的先验扰动。两个 sigma 都为零时模拟器原样使用这套几何、随机流
    # 不变 —— 于是默认配置的运行与 parity 金标值保持逐位一致。
    if cfg.prior.sigma_pos_m > 0 or cfg.prior.sigma_vel_mps > 0:
        from isac_sim.scenario.prior import perturbed_geometry

        geom = perturbed_geometry(
            cfg,
            geom,
            cfg.prior.sigma_pos_m,
            cfg.prior.sigma_vel_mps,
            rng,
        )

    return geom
