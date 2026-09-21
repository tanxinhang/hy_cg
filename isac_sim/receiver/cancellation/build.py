"""cancellation 积木：观测构造。"""

from __future__ import annotations

import math

import numpy as np

from isac_sim.core.config import Config
from isac_sim.receiver.cancellation.build_ctx import BuildContext
from isac_sim.receiver.cancellation.build_direct import (
    build_direct_sources,
    perturb_direct_sources,
)
from isac_sim.receiver.cancellation.build_ids import build_belief_ids, build_true_ids
from isac_sim.receiver.cancellation.build_targets import build_target_sources
from isac_sim.receiver.cancellation.containers import Observation
from isac_sim.receiver.cancellation.dictionaries import direct_dictionary, target_dictionary
from isac_sim.receiver.cancellation.protection import _noise_power, _protection_basis
from isac_sim.receiver.cancellation.steering import _n_obs


def build_observation(
    cfg: Config,
    geom_true,
    geom_belief,
    base,
    receiver: int,
    *,
    rng: np.random.Generator,
    sense_power: np.ndarray,
    radiated_power: np.ndarray,
    processing_gain: float,
    hw_gain: float,
    active_mask: np.ndarray | None = None,
    base_belief=None,
    include_echo: bool = True,
    exclude_target: int | None = None,
    weak_index: int = 0,
    protected_targets: frozenset[int] | None = None,
) -> Observation:
    """由与链路表**同一份**几何，为一个接收机装配式 (1) 的观测。

    ``geom_belief`` 可以等于 ``geom_true``（完美跟踪器）。直达路径永不依赖
    信念 —— 只有目标字典和保护基依赖，这正是实验要定价的那条不对称。

    子积木：直达源列表、目标源列表、列归属表分别在
    :mod:`build_direct` / :mod:`build_targets` / :mod:`build_ids`。
    """
    ctx = BuildContext(
        cfg, geom_true, geom_belief, base, receiver,
        sense_power=sense_power, radiated_power=radiated_power,
        processing_gain=processing_gain, hw_gain=hw_gain,
        active_mask=active_mask, base_belief=base_belief,
    )
    n_bins = _n_obs(cfg)   # DD 格数 x 阵元数（DD-only 模型下就是 K）

    # ``direct`` 是**真值**源列表；``direct_est`` 是接收机以为的那一份，用来造
    # 干扰字典。两者之差就是直连参数估计误差 delta（见
    # ``cancellation.direct_estimation_sigma_*_bins``）。门关着时两者是同一个
    # 列表对象，下面的分支因此完全不执行。
    direct = build_direct_sources(ctx)
    direct_est = perturb_direct_sources(cfg, direct, rng=rng)
    targets_true, targets_belief, true_ids = build_target_sources(ctx)

    X = direct_dictionary(cfg, direct_est)
    A = target_dictionary(cfg, targets_belief) if cfg.cancellation.protect_targets else \
        np.zeros((_n_obs(cfg), 0), dtype=complex)
    A_true = target_dictionary(cfg, targets_true, covariance_expanded=False)

    # 真值只落在每个块的**中心列**上。切向列是真系数为零的额外回归量，拟合它们
    # 只能增加估计方差 —— 这恰是式 (3) 要测量的代价。这样写真值才能保持比较
    # 公平：分数 DD 基是一个建模选择，不是免费信息。
    #
    # 这条规则对**回波**字典与直达字典同等适用，而回波侧直到 2026-09-19 才被
    # 修正：此前系数是在整列集合上抽取的，于是每个切向列都带一个单位模真散射体。
    # 在 ``tangent_order = 1`` 下生成的回波就成了
    # ``a_0 a + a_tau d_tau a + a_nu d_nu a``，三个单位模系数 —— 模型说的是一个
    # 散射体，生成的是三个，被测的就不是那个模型了。检测器测到了这个代价：在
    # ``perfect_channel + truth``（无信念误差、无估计误差、只有生成的回波）下，
    # 解析 CFAR 电平给出 ``P_FA = 0.64`` 而非 ``0.05``，因为 H0 里仍含着那个
    # 被杂波投影宣布为不存在的切向分量。两个对"谁承载信号"说法不一致的向量
    # 无法比较，所以现在两者按同一方式构造。
    n_basis = 1 + 2 * int(cfg.cancellation.interference_tangent_order)
    h_true = np.zeros(X.shape[1], dtype=complex)
    if X.shape[1]:
        centre = np.arange(0, X.shape[1], n_basis)
        h_true[centre] = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=centre.size))
    n_tgt_basis = 1 + 2 * int(cfg.cancellation.tangent_order)
    alpha_true = np.zeros(A_true.shape[1], dtype=complex)
    if alpha_true.size:
        centre_t = np.arange(0, alpha_true.size, n_tgt_basis)
        alpha_true[centre_t] = np.exp(
            1j * rng.uniform(0.0, 2.0 * np.pi, size=centre_t.size)
        )
    ids_true, centres_true = build_true_ids(true_ids, n_tgt_basis)
    ids_belief, centres_belief = build_belief_ids(cfg, targets_belief)

    # 直连场永远由**真值**源生成：接收机估错了自己的字典，不会改变物理上到达
    # 天线口的那个场。字典 X 与真值字典只在 delta > 0 时才不同，那时直连场就有
    # 一部分落在 span(X) 之外 —— 那一部分正是"消不掉的干扰"。
    X_true = X if direct_est is direct else direct_dictionary(cfg, direct)
    x_direct = X_true @ h_true
    if include_echo:
        s_target = A_true @ alpha_true
    else:
        s_target = np.zeros(_n_obs(cfg), dtype=complex)
    if exclude_target is not None:
        drop = ids_true == int(exclude_target)
        s_target = s_target - A_true[:, drop] @ alpha_true[drop]
        alpha_true = alpha_true.copy()
        alpha_true[drop] = 0.0
    sigma2 = _noise_power(cfg)
    noise = (rng.normal(size=n_bins) + 1j * rng.normal(size=n_bins)) * math.sqrt(sigma2 / 2.0)

    basis_belief = _protection_basis(
        cfg, targets_belief, n_bins, protected_ids=protected_targets
    )
    basis_truth = _protection_basis(cfg, targets_true, n_bins)

    return Observation(
        y=x_direct + s_target + noise,
        X=X,
        A=A,
        x_direct=x_direct,
        s_target=s_target,
        h_true=h_true,
        alpha_true=alpha_true,
        sigma2=sigma2,
        direct=direct,
        targets=targets_true,
        targets_belief=targets_belief,
        basis_belief=basis_belief,
        basis_truth=basis_truth,
        weak_index=int(weak_index),
        A_target_ids=ids_belief,
        A_true_target_ids=ids_true,
        A_centre_mask=centres_belief,
        A_true_centre_mask=centres_true,
        protected_targets=protected_targets,
    )
