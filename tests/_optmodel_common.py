"""优化模型一致性门禁的共用脚手架（非测试文件，pytest 不收集）。

场景取 600 m / RCS 0.1 / M=5 / Q=3（主工作点的小规模版），种子固定，
因此四个条款文件跑的是**同一个世界**，跨文件的数字可以直接互相引用。
"""
from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.core.config.presets.successors import HEADLINE_RELEASE_PRESET
from isac_sim.cooperation.reporting import assign_fusion_nodes
from isac_sim.receiver.cancellation.sources import TargetSource
from isac_sim.sensing.model import (
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)


def make_cfg(preset: str = HEADLINE_RELEASE_PRESET, **overrides) -> Config:
    """主工作点的小规模版：600 m / RCS 0.1 / M=5 / Q=3。"""
    cfg = apply_preset(Config(), preset)
    base = {
        "scale.M": 5,
        "scale.Q": 3,
        "geometry.area_xy": 600.0,
        "detect.target_rcs": 0.1,
        "run.seed": 2026,
        "run.verbose": False,
    }
    base.update(overrides)
    return apply_overrides(cfg, base)


def make_world(cfg: Config, seed: int = 7):
    """一次部署：真实几何 + 信念侧链路表 + 融合落点。"""
    rng = np.random.default_rng([cfg.run.seed, seed])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    tables = compute_link_tables(cfg, base)
    plan = assign_fusion_nodes(cfg, base, tables, geom=geom)
    return geom, base, tables, plan


def make_sources(n_target: int = 5, n_uav: int = 2):
    """构造一组回波源，用来单独测保护侧变量 ``z``。"""
    return [
        TargetSource(
            uav=i,
            target=t,
            power=1.0 / (1.0 + i + t),
            doppler_bin=0.1 * (i + 1),
            delay_bin=0.2 * (t + 1),
        )
        for t in range(n_target)
        for i in range(n_uav)
    ]
