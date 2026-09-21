"""通信口径（MAC / 干扰模型）与门控、协同的互锁条件。"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from isac_sim.core.config.run import Config


def check_comm(cfg: Config) -> None:
    """通信口径（MAC / 干扰模型）与门控、协同的互锁条件。"""
    interference_model = cfg.comm.interference_model.lower()
    mac_model = cfg.comm.mac_model.lower()
    if interference_model not in {"full_concurrent", "active_set", "orthogonal"}:
        raise ValueError(
            f"Unknown comm.interference_model={cfg.comm.interference_model!r}; "
            "expected 'full_concurrent', 'active_set', or 'orthogonal'"
        )
    if mac_model not in {"serial", "parallel", "slot"}:
        raise ValueError(
            f"Unknown comm.mac_model={cfg.comm.mac_model!r}; "
            "expected 'serial', 'parallel', or 'slot'"
        )
    if interference_model == "orthogonal" and mac_model != "serial":
        raise ValueError(
            "comm.interference_model='orthogonal' requires comm.mac_model='serial'"
        )
    if cfg.interference.sense_gate_by_active_tx:
        # ``sense_gate_by_active_tx`` 声明的是"不在 active_tx_mask 里的节点什么都不
        # 辐射"。这句话只有在**正交**口径下才是完整的：那个口径下感知观测期间不发
        # 任何上报载荷，于是 ``compute_link_tables`` 会门控干扰场（自回波门控修好
        # 之后，也门控观测本身），但从不门控想要的通信信号。在"载荷并发"口径下，
        # 被静默的上报者会丢掉干扰却保住信号，也就是调度读起来会比它实际能做到的
        # 更好。把门控绑到自洽的那个口径上，才让 ``active_tx_mask`` 只
        # 表示**一件事** —— 辐射中的照射机集合 —— 而不是随调用方变成两件事。
        if interference_model != "orthogonal":
            raise ValueError(
                "interference.sense_gate_by_active_tx=True requires "
                "comm.interference_model='orthogonal' (got "
                f"{cfg.comm.interference_model!r}): under a concurrent-payload "
                "口径 the gate silences interference but not the communication "
                "signal, so it cannot describe a radiation set."
            )
        if cfg.interference.coupling != "shared_spectrum":
            raise ValueError(
                "interference.sense_gate_by_active_tx=True requires "
                "interference.coupling='shared_spectrum' (got "
                f"{cfg.interference.coupling!r}); under the legacy coupling the "
                "mask is not consulted at all, so the gate would be a silent "
                "no-op."
            )
    if cfg.coordination.enable:
        # 门控才是让掩码具有物理意义的东西；没有它，``compute_link_tables``
        # 永不查看 ``active_tx_mask``，协同回路就会对着一个没变的分母反复重选 ——
        # 一个静默空操作，而它已经产出一个无法上报的数字。
        if not cfg.interference.sense_gate_by_active_tx:
            raise ValueError(
                "coordination.enable=True requires "
                "interference.sense_gate_by_active_tx=True; otherwise "
                "active_tx_mask is ignored by compute_link_tables and the "
                "coordination fixed point cannot change anything."
            )
        if cfg.coordination.rounds < 1:
            raise ValueError("coordination.rounds must be at least one")
