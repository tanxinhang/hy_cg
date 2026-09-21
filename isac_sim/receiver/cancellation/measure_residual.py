"""cancellation 积木：残余分数测量。"""

from __future__ import annotations

from isac_sim.core.config import Config
import numpy as np
from typing import Sequence, Tuple

from isac_sim.receiver.cancellation.context import ReceiverContext
from isac_sim.receiver.cancellation.measure import measure_receiver_context

def measure_residual_fraction(
    cfg: Config,
    geom,
    base,
    *,
    rng: np.random.Generator,
    geom_belief=None,
    active_mask: np.ndarray | None = None,
    arm: str = "tp_uic_full",
    sense_power: np.ndarray | None = None,
    processing_gain: float | None = None,
    hw_gain: float | None = None,
    receivers: Sequence[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """逐接收机的 ``(fraction, kappa_db)``，由**真实估计器实测**得到。

    ``fraction[j] = I_res / I_in`` 是接收机 ``j`` 在该 ``arm`` 下的值，也就是
    ``model.compute_link_tables`` 去乘 ``I_sense_field[j]`` 的那个量。它与被删掉的
    对消常数同量纲，因此是一个可直插的替代品 —— 这正是 :func:`predict_cancellation`
    返回**分数**而不是 dB 的全部原因。

    .. warning::
       ``geom_belief=None`` 表示 **oracle belief**：接收机拿到真值目标状态作为
       它的字典。那是一个**完美跟踪器**的上界，不是对声明系统的测量 ——
       config 之所以写明 ``prior.belief_sigma_pos_m``/``belief_sigma_vel_mps``，
       正是因为跟踪器并不完美；在 ``None`` 下测出的深度必须标成
       ``kappa_oracle``，永远不能当作系统的深度来引用。生产调用方应传入调度器
       所看到的那份 ``geom_belief``（``BeliefState.as_geometry``），于是 trial
       的两半共享同一个信念。若还需要完整的接收机产物（回波存活率、残差
       协方差），请用 :func:`measure_receiver_context`。

    为什么调度器必须用这个而不是预测值：预测的留存率是一个维度比例
    （``_protection_leakage_fraction``），而本模块在 600 m 场景上的实测是输入
    直达场的约 17%，比例说的却是 3.1% —— 乐观了 5 倍，因为直达核与目标核占据
    **同一个**紧凑的 DD 区域。预测是规划用的数；这个才是交付物。

    没有激活照射源的接收机其 ``I_in = 0``，此时任何分数给出的残差都一样（零），
    dB 值则未定义；这些条目报成 ``1.0`` 与 ``inf``，这样调用方把两者相乘时
    永远不会产生一个静默的 ``0 * inf``。
    """
    ctx = ReceiverContext.from_trial(
        cfg, geom, geom if geom_belief is None else geom_belief, base,
        sense_power=sense_power, processing_gain=processing_gain,
        hw_gain=hw_gain, active_mask=active_mask, arm=arm,
    )
    measured = measure_receiver_context(ctx, rng=rng, receivers=receivers)
    return measured.fraction, measured.kappa_db
