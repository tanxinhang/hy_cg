"""cancellation 积木：基础臂的装配（no_ic → targeted_tpuic_full）。"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from isac_sim.receiver.cancellation.arm import _Arm
from isac_sim.receiver.cancellation.arm_operator import run_arm
from isac_sim.receiver.cancellation.manifold import orthonormalise
from isac_sim.receiver.cancellation.protection import protected_target_ids

# 装配剪枝用的依赖表：``only=X`` 时要一并装配的前置臂（会被 X 读到的中间结果）。
_ARM_DEPS: Dict[str, Tuple[str, ...]] = {
    # The adaptive arm is selected against the hard full TP-UIC arm and must
    # retain that reference when production measurement prunes unused arms.
    "adaptive_soft_tpuic": ("tp_uic_full",),
}


def gate_candidates(ctx, stage1_residual) -> List[int]:
    """能量门：逐目标块算 ``||U_q^H r1||^2 / rank``，超过门限的留下。

    这个门**总是**被求值 —— 它的读数是整个消融的出处 —— 但只有
    ``statistic`` 策略会照它行动。
    """
    obs, A = ctx.obs, ctx.A
    ids_here = None if obs.A_target_ids is None else np.asarray(obs.A_target_ids)
    gate: List[int] = []
    if ids_here is None or not A.shape[1]:
        return gate
    for q in sorted(set(int(v) for v in ids_here)):
        block = orthonormalise(A[:, np.flatnonzero(ids_here == q)])
        if not block.rank:
            continue
        projected = block.U.conj().T @ stage1_residual
        level = float(np.vdot(projected, projected).real) / block.rank
        if level > ctx.threshold:
            gate.append(int(q))
    return gate


def supported_candidates(ctx, gate: List[int]) -> Tuple[int, ...]:
    """stage 2 的支撑集：声明式（受保护集）或统计式（能量门）。"""
    obs = ctx.obs
    ids_here = None if obs.A_target_ids is None else np.asarray(obs.A_target_ids)
    if ctx.candidate_policy == "statistic":
        supported = list(gate)
    else:
        shielded = (
            protected_target_ids(ctx.cfg, obs.targets_belief or obs.targets)
            if obs.protected_targets is None else obs.protected_targets
        )
        supported = (
            [] if ids_here is None
            else sorted(q for q in set(int(v) for v in ids_here) if q in shielded)
        )
    cols: List[int] = []
    for q in supported:
        cols.extend(int(c) for c in np.flatnonzero(ids_here == q))
    return tuple(cols), tuple(supported)


def build_basic_arms(ctx, parts: Dict[str, _Arm], only: str | None = None):
    """装配除自适应软臂之外的所有臂，并返回门/支撑集信息供打分阶段使用。

    ``only`` 不是一个新的算法开关，而是一条**装配剪枝**：生产接线只读一个臂
    （``tp_uic_full``），但这里原本会装配八个。测量阶段跑满八臂是纯浪费 ——
    剪掉的是**没有被读取的量**，所以被测臂的数值必须逐位不变（由
    ``tests/test_tpuic_production_wire.py`` 钉住）。

    ``only`` 的臂若依赖另一个臂的中间结果，依赖会被自动保留；
    剪枝只砍无依赖的。默认 ``protected_only`` 的 ``tp_uic_full``
    直接由保护声明得到候选集，不依赖 Stage 1。
    ``None``（默认）装配全部，行为与改动前逐位一致。
    """
    cfg, obs = ctx.cfg, ctx.obs
    zero = np.zeros(ctx.n_bins, dtype=complex)
    wanted = None if only is None else {only, *_ARM_DEPS.get(only, ())}
    if wanted is None or "no_ic" in wanted:
        parts["no_ic"] = _Arm("no_ic", zero.copy(), zero.copy(), zero.copy(),
                              np.zeros(ctx.X.shape[1], dtype=complex),
                              np.zeros(ctx.X.shape[1]), ctx.empty, predict="none")
    # ⚠️ 顺序必须与原实现一致：``run_arm`` 会往 ``ctx`` 上挂中间结果，调换顺序
    # 可能改变数值。剪枝只允许**跳过**臂，不允许**重排**臂。
    if wanted is None or "plain_ls" in wanted:
        parts["plain_ls"] = run_arm(ctx, "plain_ls", ctx.empty, None)
    if wanted is None or "ridge_ls" in wanted:
        parts["ridge_ls"] = run_arm(ctx, "ridge_ls", ctx.empty, ctx.prior)
    if wanted is None or "protected_ls" in wanted:
        parts["protected_ls"] = run_arm(ctx, "protected_ls", ctx.belief, None)
    # Stage 1 只用于全臂消融，或者给旧 ``statistic`` 策略产生门。
    # 默认 ``protected_only`` 的生产 full 臂从原始 y 重新做联合 MAP，
    # 所以对 Stage 1 的数值输出没有依赖。
    need_stage1 = wanted is None or ctx.candidate_policy == "statistic"
    if need_stage1:
        parts["tp_uic_stage1"] = run_arm(
            ctx, "tp_uic_stage1", ctx.belief, ctx.prior
        )
    targeted = ctx.weak_block if ctx.weak_block is not None else ctx.belief
    if wanted is None or "targeted_tpuic_stage1" in wanted:
        parts["targeted_tpuic_stage1"] = run_arm(ctx, "targeted_tpuic_stage1", targeted, ctx.prior)
    if wanted is None or "soft_tpuic" in wanted:
        parts["soft_tpuic"] = run_arm(
            ctx, "soft_tpuic", targeted, ctx.prior,
            soft_mu=float(cfg.cancellation.soft_protection_mu),
        )
    gate: List[int] = []
    if need_stage1:
        stage1 = parts["tp_uic_stage1"]
        residual = ctx.y - (stage1.sub_direct + stage1.sub_target + stage1.sub_noise)
        gate = gate_candidates(ctx, residual)
    candidates, supported = supported_candidates(ctx, gate)
    if wanted is None or "tp_uic_full" in wanted:
        parts["tp_uic_full"] = run_arm(ctx, "tp_uic_full", ctx.belief, ctx.prior, candidates)
    ids_here = None if obs.A_target_ids is None else np.asarray(obs.A_target_ids)
    targeted_candidates: Tuple[int, ...] = ()
    if ids_here is not None and ctx.A.shape[1] > 0:
        targeted_candidates = tuple(int(c) for c in np.flatnonzero(ids_here == ctx.wt))
    if wanted is None or "targeted_tpuic_full" in wanted:
        parts["targeted_tpuic_full"] = run_arm(
            ctx, "targeted_tpuic_full", targeted, ctx.prior, targeted_candidates
        )
    return gate, supported
