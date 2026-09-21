"""cancellation 积木：目标源列表（真值 / 信念两版）。"""

from __future__ import annotations

from typing import List, Tuple

from isac_sim.receiver.cancellation.link_jacobians import (
    target_bearing_jacobian,
    target_bearing_std,
    target_link_jacobians,
    target_link_std_bins,
)
from isac_sim.receiver.cancellation.link_offsets import target_link_offset
from isac_sim.receiver.cancellation.sources import TargetSource


def build_target_sources(ctx) -> Tuple[List[TargetSource], List[TargetSource], List[int]]:
    """构造目标源的真值版与信念版，并返回逐列的真值目标编号。

    两版唯一的差别是几何：真值版用 ``geom_true`` 的 DD 偏移与真实增益，
    信念版用 ``geom_belief`` 的偏移、信念增益，并额外携带链路标准差与
    状态雅可比（接收机对自己信念不确定性的量化）。
    """
    gain_belief_table = (
        ctx.base.target_gain if ctx.base_belief is None else ctx.base_belief.target_gain
    )
    targets_true: List[TargetSource] = []
    targets_belief: List[TargetSource] = []
    true_ids: List[int] = []
    for i in range(ctx.cfg.scale.M):
        if i == ctx.receiver or not ctx.active_mask[i]:
            continue
        for q in range(ctx.cfg.scale.Q):
            gain_true = float(ctx.base.target_gain[i, ctx.receiver, q])
            gain_belief = float(gain_belief_table[i, ctx.receiver, q])
            if gain_true <= 0.0 and gain_belief <= 0.0:
                continue
            k_t, l_t = target_link_offset(ctx.cfg, ctx.geom_true, i, ctx.receiver, q)
            k_b, l_b = target_link_offset(ctx.cfg, ctx.geom_belief, i, ctx.receiver, q)
            scale = float(ctx.processing_gain) * float(ctx.hw_gain)
            power_true = float(ctx.sense_power[i]) * max(gain_true, 0.0) * scale
            power_belief = float(ctx.sense_power[i]) * max(gain_belief, 0.0) * scale
            sigma_k, sigma_l = target_link_std_bins(ctx.cfg, ctx.geom_belief, i, ctx.receiver, q)
            sigma_u = target_bearing_std(ctx.cfg, ctx.geom_belief, ctx.receiver, q)
            jac_k, jac_l = target_link_jacobians(ctx.cfg, ctx.geom_belief, i, ctx.receiver, q)
            jac_u = target_bearing_jacobian(ctx.cfg, ctx.geom_belief, ctx.receiver, q)
            targets_true.append(TargetSource(
                i, q, power_true, k_t, l_t,
                u=(float(ctx.u_tgt_true[q]) if ctx.m_rx > 1 else 0.0)))
            targets_belief.append(TargetSource(
                i, q, power_belief, k_b, l_b,
                u=(float(ctx.u_tgt_belief[q]) if ctx.m_rx > 1 else 0.0),
                sigma_doppler_bin=sigma_k,
                sigma_delay_bin=sigma_l,
                sigma_bearing_u=sigma_u,
                jacobian_doppler_state=jac_k,
                jacobian_delay_state=jac_l,
                jacobian_bearing_state=jac_u))
            true_ids.append(int(q))
    return targets_true, targets_belief, true_ids
