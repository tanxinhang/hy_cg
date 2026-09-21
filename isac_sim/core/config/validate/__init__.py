"""配置一致性校验：按物理/算法段拆开，由 :func:`validate_config` 统一调用。

每个 ``check_*`` 只负责一段配置的合法性。**调用顺序是行为的一部分** ——
原实现是一个 282 行的长函数，出错时"先报哪一条"由语句顺序决定，
这里必须逐条保持，否则同一份非法配置会报出不同的错误。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from isac_sim.core.config.validate.active_sensing import check_active_sensing
from isac_sim.core.config.validate.comm import check_comm
from isac_sim.core.config.validate.fusion import check_fusion
from isac_sim.core.config.validate.proposal import check_proposal
from isac_sim.core.config.validate.radio import check_radio
from isac_sim.core.config.validate.scenario import check_scenario

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from isac_sim.core.config.run import Config

__all__ = ["validate_config"]


def validate_config(cfg: Config) -> None:
    """拒绝内部不自洽的物理模型组合。

    历史 preset 仍然可跑，但新配置在 MAC 与载荷干扰假设描述的不是
    同一个系统时会直接失败。
    """
    check_scenario(cfg)
    check_active_sensing(cfg)
    check_comm(cfg)
    check_proposal(cfg)
    check_radio(cfg)
    check_fusion(cfg)
