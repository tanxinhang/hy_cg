"""cancellation 积木：接收机上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from isac_sim.core.config import Config
import numpy as np

@dataclass(frozen=True, eq=False)
class ReceiverContext:
    """TP-UIC 被允许知道什么 —— 由调用方**声明**，而不是猜出来。

    早先的测量是自己从 ``Config`` 重建世界的：它把真值几何当作信念（一个它从未
    声称拥有的完美跟踪器）、把感知功率默认成 ``rho * P_default``，并且忽略了
    激活掩码、硬件增益与协同调度。这些默认每一个都对应一台**与 trial 实际运行
    不同的**接收机，于是测出来的深度描述的是调度器并没有在用的那台机器。

    除了显式的 :meth:`from_trial` 辅助方法之外，这里没有任何东西是从 ``cfg``
    推断出来的 —— 那个辅助方法接收的正是 trial 交给
    :func:`model.compute_link_tables` 的同一批对象。
    """

    cfg: Config
    geom_true: object
    geom_belief: object
    base: object
    sense_power: np.ndarray
    radiated_power: np.ndarray
    processing_gain: float
    hw_gain: float
    # 真值回波与接收机字典不必共享同一份目标增益视图。``base`` 为了兼容仍是
    # 真值侧的对象；当给出了本字段时，它的目标增益只缩放**信念**字典。两者
    # 分开显式化，可防止一个信念模式接收机拿真值 RCS/幅度去给它的受保护
    # 目标排序。
    base_belief: object | None = None
    active_mask: np.ndarray | None = None
    arm: str = "tp_uic_full"

    @classmethod
    def from_trial(
        cls,
        cfg: Config,
        geom_true,
        geom_belief,
        base,
        *,
        sense_power: np.ndarray | None = None,
        radiated_power: np.ndarray | None = None,
        processing_gain: float | None = None,
        hw_gain: float | None = None,
        base_belief=None,
        active_mask: np.ndarray | None = None,
        arm: str = "tp_uic_full",
    ) -> "ReceiverContext":
        """从一次生产 trial 已经持有的对象构造上下文。

        ``sense_power`` 默认为 ``rho * P_default``（被动释放模型），
        ``radiated_power`` 默认为**仅感知场**，与 ``model.compute_link_tables``
        在 orthogonal 报告模型下称为 ``P_rad_sense`` 的量一致。跑激活集干扰或
        协同的调用方必须两个都显式传入 —— 这正是本类的全部要点。
        """
        m = int(cfg.scale.M)
        if sense_power is None:
            sense_power = np.full(m, cfg.radio.rho * cfg.radio.P_default)
        if radiated_power is None:
            radiated_power = np.asarray(sense_power, dtype=float).copy()
        if processing_gain is None:
            processing_gain = float(cfg.waveform.N * cfg.waveform.L)
        if hw_gain is None:
            from isac_sim.sensing.model import radar_hardware_gain

            hw_gain = float(radar_hardware_gain(cfg))
        return cls(
            cfg=cfg,
            geom_true=geom_true,
            geom_belief=geom_belief,
            base=base,
            sense_power=np.asarray(sense_power, dtype=float),
            radiated_power=np.asarray(radiated_power, dtype=float),
            processing_gain=float(processing_gain),
            hw_gain=float(hw_gain),
            base_belief=base_belief,
            active_mask=active_mask,
            arm=str(arm),
        )

    @property
    def belief_is_truth(self) -> bool:
        """接收机是否拿到了真值当它自己的信念。

        判**同一性**而非相等：两个分别构造出来的几何可以在数值上相等，却仍是
        不同的抽样。想要**完美跟踪器**的调用方必须靠传同一个对象来说明，而本
        属性正是让结果文件能区分 ``kappa_oracle-belief`` 与
        ``kappa_actual-belief``、而不是把它们折进同一列好看的数字里的东西。
        """
        return self.geom_belief is self.geom_true
