"""cancellation 积木：受保护目标集合与保护基。"""

from __future__ import annotations

from isac_sim.core.config import Config
import numpy as np
from typing import Dict, List, Sequence

from isac_sim.receiver.cancellation.manifold import kernel_vector, orthonormalise, tangent_columns
from isac_sim.receiver.cancellation.sources import TargetSource
from isac_sim.receiver.cancellation.steering import _n_obs, lift_dictionary

def protected_target_ids(cfg: Config, sources: Sequence[TargetSource]) -> "frozenset[int]":
    """保护基在本接收机上覆盖了哪些目标。

    从 :func:`_protection_basis` 里抽出来，是因为 stage 2 需要同一个答案，但用途
    不同。保护预算花在 ``max_protected_targets`` 个**最弱**的回波上，而 stage 2
    的联合支撑集被**声明为**就是这一个集合（:func:`cancellation_arms`，
    ``candidate_policy="protected_only"``）：stage 1 有意保留其能量的那些回波，
    恰好就是 stage 2 必须显式建模的那些；而唯独**不能**用来找出它们的一条规则，
    就是"对 stage-1 残差做能量检验" —— 那个残差按构造就保留了它们。
    """
    c = cfg.cancellation
    if not c.protect_targets or not sources:
        return frozenset()
    budget = int(c.max_protected_targets)
    if budget <= 0:
        # ``0`` 表示保护每一个信念回波；``_protection_leakage_fraction``
        # 记录了这是为消融保留的无约束变体。
        return frozenset(int(s.target) for s in sources)
    by_target: Dict[int, List[TargetSource]] = {}
    for src in sources:
        by_target.setdefault(int(src.target), []).append(src)
    ranked = sorted(by_target, key=lambda q: sum(s.power for s in by_target[q]))
    return frozenset(int(q) for q in ranked[:budget])


def _protection_basis(
    cfg: Config,
    sources: Sequence[TargetSource],
    n_bins: int,
    protected_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """``U_j = orth([J_1, ..., J_Q])`` —— 模块 M2。

    保护是按**目标**预算的，不是按回波路径：源按目标编号分组，算出每个目标在
    本接收机上的回波总功率，然后保护 ``max_protected_targets`` 个最弱目标在
    **所有**照射路径上的分量。``max_protected_targets = 0`` 保护一切，保留它
    只是因为第一个实验需要这一臂来展示代价。
    """
    c = cfg.cancellation
    # 两个提前出口都必须**直接返回**。零列数组才是正确的答案（空基在下游能被
    # 干净地拼接掉）；构造出来之后继续往下走则不是 —— 对过滤后的块列表做
    # ``np.concatenate`` 要么抛异常，要么静默产出一个"保护预算从未授权过"的基。
    # 标准场景（15 架 UAV、10 个目标、protect_targets=True）永远走不到这两个
    # 分支，所以这个错误躲过了每一次全场景测试 —— 它只在
    # ``protect_targets=False``、源列表为空、或掩码把所有源都滤掉时才触发。
    if not c.protect_targets or not sources:
        return np.zeros((_n_obs(cfg), 0), dtype=complex)

    wanted = (
        protected_target_ids(cfg, sources)
        if protected_ids is None
        else frozenset(int(q) for q in protected_ids)
    )
    sources = [s for s in sources if int(s.target) in wanted]
    if not sources:
        return np.zeros((_n_obs(cfg), 0), dtype=complex)

    lifted_blocks: List[np.ndarray] = []
    z95 = 1.6448536269514722
    for s in sources:
        block = tangent_columns(
            cfg, s.doppler_bin, s.delay_bin,
            int(c.tangent_order), float(c.tangent_step_bins),
        )
        bearings = [float(s.u)] * int(block.shape[1])
        if c.covariance_protection:
            sigma_k = max(float(s.sigma_doppler_bin or 0.0), 0.0)
            sigma_l = max(float(s.sigma_delay_bin or 0.0), 0.0)
            sigma_u = max(float(s.sigma_bearing_u or 0.0), 0.0)
            extra: List[np.ndarray] = []
            extra_bearings: List[float] = []
            for sign in (1.0, -1.0):
                if sigma_k > 0.0:
                    extra.append(kernel_vector(
                        cfg, float(s.doppler_bin) + sign * z95 * sigma_k,
                        float(s.delay_bin),
                    ))
                    extra_bearings.append(float(s.u))
                if sigma_l > 0.0:
                    extra.append(kernel_vector(
                        cfg, float(s.doppler_bin),
                        float(s.delay_bin) + sign * z95 * sigma_l,
                    ))
                    extra_bearings.append(float(s.u))
                if sigma_u > 0.0:
                    extra.append(kernel_vector(
                        cfg, float(s.doppler_bin), float(s.delay_bin)
                    ))
                    extra_bearings.append(float(np.clip(
                        float(s.u) + sign * z95 * sigma_u, -1.0, 1.0
                    )))
            if extra:
                block = np.column_stack([block, *extra])
                bearings.extend(extra_bearings)
        # **先**升维再正交化：空间导向会改变列内积，所以一个已经正交化的 DD 基
        # 不能简单地被升维。
        lifted_blocks.append(lift_dictionary(cfg, block, bearings))
    merged = np.concatenate(lifted_blocks, axis=1)
    return orthonormalise(merged).U


def _noise_power(cfg: Config) -> float:
    from isac_sim.sensing.model import noise_power

    return float(noise_power(cfg))
