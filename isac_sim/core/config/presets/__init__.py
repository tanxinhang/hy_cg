"""命名配置束（preset）：基础字典 + 派生链 + 应用入口。"""

from __future__ import annotations

from isac_sim.core.config.overrides import apply_overrides
from isac_sim.core.config.presets.base import PRESETS
from isac_sim.core.config.presets.successors import HEADLINE_RELEASE_PRESET
from isac_sim.core.config.run import Config

__all__ = ["PRESETS", "HEADLINE_RELEASE_PRESET", "apply_preset"]


def apply_preset(cfg: Config, name: str) -> Config:
    """返回应用了指定 preset 的 ``cfg``。

    >>> cfg = apply_preset(Config(), "isac-consistent")
    """
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return apply_overrides(cfg, PRESETS[name])
