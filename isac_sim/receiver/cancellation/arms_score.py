"""cancellation 积木：把各臂的中间结果打分成 ``CancellationResult``。"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from isac_sim.receiver.cancellation.arm import _Arm
from isac_sim.receiver.cancellation.arm_ctx import ArmContext
from isac_sim.receiver.cancellation.arms_predict import predicted_powers
from isac_sim.receiver.cancellation.result import CancellationResult


def score_arms(ctx: ArmContext, parts: Dict[str, _Arm],
               gate: Tuple[int, ...], supported: Tuple[int, ...]
               ) -> Dict[str, CancellationResult]:
    """逐臂打分：残差、生存率、噪声增强、目标条件化存活率。"""
    out: Dict[str, CancellationResult] = {}
    s_energy = float(np.vdot(ctx.s, ctx.s).real)
    n_energy = float(np.vdot(ctx.n, ctx.n).real)
    for name, arm in parts.items():
        residual = ctx.y - (arm.sub_direct + arm.sub_target + arm.sub_noise)
        retained, pred_estimate, i_res_pred = predicted_powers(ctx, arm, ctx.prior)
        structural = float(np.vdot(ctx.x - arm.sub_direct, ctx.x - arm.sub_direct).real)
        estimation = float(np.vdot(arm.sub_noise, arm.sub_noise).real)
        # 残余干扰的记账口径（``cancellation.residual_accounting``）。
        # ``measured``（默认）：两项都记，逐位不变。``structural``：只记结构残差
        # —— ``estimation`` 是被减掉的噪声而非残余干扰（实测占 ||n||^2 的
        # 0.12%，残差保留 99.93%）。解析侧同步只取 ``retained``，否则
        # ``calibration_error_db`` 会在两个口径之间比。
        only_structural = (
            str(ctx.cfg.cancellation.residual_accounting) == "structural"
        )
        i_res = structural if only_structural else structural + estimation
        if i_res_pred is None:
            i_res_pred = i_res
        elif only_structural and arm.predict == "estimator" and retained > 0.0:
            # 候选分支把结构保留并进了后验协方差（那里 ``retained == 0``），
            # 那种情形下解析侧拆不出独立结构项，裁剪只会造出 NaN ⇒ 不裁。
            i_res_pred = retained
        if ctx.weak_block is not None and ctx.weak_block.rank:
            projected = ctx.weak_block.U.conj().T @ residual
            statistic = float(np.vdot(projected, projected).real) / ctx.weak_block.rank
        else:
            statistic = 0.0
        if estimation <= 0.0 or n_energy <= 0.0:
            # 一个什么都没减的臂也就没有减掉噪声：报 -inf，而不是让 EPS 守卫
            # 凭空造出一个 -22 dB 的读数。
            noise_db = float("-inf")
        else:
            noise_db = float(10.0 * math.log10(estimation / n_energy))
        # 两个比值都只对**零**分母做守卫，不对 EPS 做守卫。``EPS = 1e-12`` 比本
        # 场景下一个真实回波能量还大（实测：单目标 trial 上 1.21e-13 W），所以
        # ``max(s_energy, EPS)`` 会把分母换成一个大八倍的值，给 ``no_ic`` 臂报出
        # ``eta_survive = 0.1209`` —— 而它的存活率按定义是 1.0。同一个坑在
        # :func:`_positive_sigma` 上有记录；它能在这里存活下来，是因为旧的回波
        # 生成器（每目标三个单位模散射体）把 ``s_energy`` 抬到了 EPS 之上。
        if s_energy > 0.0:
            projected_s = ctx.belief.project(ctx.s)
            raw_protect = float(np.vdot(projected_s, projected_s).real / s_energy)
            # An orthogonal projection cannot increase energy.  Roundoff can
            # nevertheless produce values such as 1.000000000000001.
            eta_protect = float(np.clip(raw_protect, 0.0, 1.0))
            left_s = ctx.s - arm.sub_target
            eta_survive = float(np.vdot(left_s, left_s).real / s_energy)
        else:
            # 观测里没有回波：两个比值都未定义，此时一个约定比一个会静默污染
            # 中位数的 NaN 更安全。
            eta_protect = eta_survive = 0.0
        # 被测目标自己的存活率。``sub_target_q`` 对非估计器臂
        # （``no_ic`` / ``perfect_channel``）是 ``None``：它们
        # 只是缩放或减掉**直达**场，从不碰回波，所以目标整体存活。在那里报字段
        # 默认值 0.0 会被读成"回波被摧毁了"。
        s_q_energy = eta_survive_q = 0.0
        if ctx.s_q is not None:
            s_q_energy = float(np.vdot(ctx.s_q, ctx.s_q).real)
            if s_q_energy > 0.0:
                removed_q = arm.sub_target_q
                if removed_q is None:
                    removed_q = np.zeros_like(ctx.s_q)
                left = ctx.s_q - removed_q
                eta_survive_q = float(np.vdot(left, left).real / s_q_energy)
        out[name] = CancellationResult(
            name=name,
            residual=residual,
            h_hat=arm.h_hat,
            c_h_diag=np.asarray(arm.c_diag, dtype=float),
            i_in=float(np.vdot(ctx.x, ctx.x).real),
            i_res=i_res,
            i_res_structural=structural,
            i_res_estimate=estimation,
            i_res_pred=i_res_pred,
            i_res_moment_mean=float(arm.i_res_moment_mean),
            i_res_pred_var=float(arm.i_res_moment_var),
            i_in_pred=ctx.i_in_pred,
            i_res_retained=retained,
            i_res_pred_estimate=pred_estimate,
            eta_protect=eta_protect,
            eta_survive=eta_survive,
            eta_survive_q=eta_survive_q,
            eta_survive_pred_q=float(arm.eta_survive_pred_q),
            eta_survive_risk_q=float(arm.eta_survive_risk_q),
            s_q_energy=s_q_energy,
            soft_mu=arm.soft_mu,
            noise_enhance_db=noise_db,
            protect_dim=int(arm.subspace.rank),
            n_coefficients=int(ctx.X.shape[1]),
            residual_direct_gram=arm.residual_direct_gram,
            residual_noise_eigenvalues=arm.residual_noise_eigenvalues,
            matched_stat=statistic,
            candidates=arm.candidates,
            # 只有 ``tp_uic_full`` 有候选阶段，所以只有它有门可报。把这些记在
            # 其它臂上会暗示它们跑了一个从未有过的门。
            gate_targets=gate if name == "tp_uic_full" else (),
            supported_targets=supported if name == "tp_uic_full" else (),
        )
    return out
