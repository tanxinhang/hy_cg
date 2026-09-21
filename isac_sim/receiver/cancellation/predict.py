"""cancellation 积木：解析对消深度预测。"""

from __future__ import annotations

from isac_sim.core.config import Config
import numpy as np
from typing import Tuple

from isac_sim.receiver.cancellation.constants import EPS
from isac_sim.receiver.cancellation.protection import _noise_power

def predict_cancellation(
    cfg: Config,
    i_sense_field: np.ndarray,
    n_illuminators: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """逐接收机的"残余直达场"与"保护留存"预测。

    这是调度器会去调的 ``C_TP-UIC`` 接口。两项：

    *留存* —— 落在目标子空间里的那部分直达能量是被**有意**保留的，因为减掉它
    就等于连目标证据一起减掉。它上界是"保护秩 / 观测维度"，即
    ``rho_prot = rank / K``；实验测量的是实现值，而本函数返回的是那个界 ——
    界才是接收机**能承诺**的那部分。

    *估计* —— 每个 CPI 拟合 ``d`` 个复系数，残余是 ``d * sigma_eff^2``，其中
    ``sigma_eff^2 = n0 / n_cpi``，于是残余占输入场的比例是
    ``d * n0 / (n_cpi * I_sense)``。

    两者都以**输入直达场的比例**返回，于是调用方可以直接写
    ``I_res = fraction * I_sense_field``，不必改动
    ``model.compute_link_tables`` 现有的分母结构。
    """
    c = cfg.cancellation
    d_basis = 1 + 2 * int(c.interference_tangent_order)
    d = np.maximum(np.asarray(n_illuminators, dtype=float), 1.0) * d_basis
    n0 = _noise_power(cfg)
    n_cpi = max(int(c.n_cpi), 1)
    i_sense = np.maximum(np.asarray(i_sense_field, dtype=float), EPS)
    estimate = (d * n0 / n_cpi) / i_sense
    retained = np.full_like(estimate, _protection_leakage_fraction(cfg))
    return estimate + retained, retained


def _protection_leakage_fraction(cfg: Config) -> float:
    """被保护的观测占比的**维度比例估计**。

    ``n_targets * (M - 1) * (1 + 2*order) / K``。这是接收机在看到场景之前唯一
    能写下来的数，而它**不是**一个安全的界：它假设保护子空间与干扰子空间处于
    一般位置。它们并不。600 m 场景上实测比例读 3.1%，而实现出来的留存是输入
    直达场的 17% —— 乐观了约 5 倍 —— 因为直达核与目标核占据**同一个**紧凑的
    DD 区域（整个 600 m 场景大致只跨 4 个延迟格 × 10 个多普勒格，于是 4096 个
    格大部分是空的，被占用的那些则是共享的）。

    只能用于规划。调度器必须消费**实测**的 ``I_res``（或拟合到它的预测），
    永远不要消费这个数。
    """
    c = cfg.cancellation
    if not c.protect_targets:
        return 0.0
    budget = int(c.max_protected_targets)
    n_targets = cfg.scale.Q if budget <= 0 else min(budget, cfg.scale.Q)
    n_bins = float(cfg.waveform.N * cfg.waveform.L)
    worst = float(n_targets * (cfg.scale.M - 1) * (1 + 2 * int(c.tangent_order)))
    return min(1.0, worst / n_bins)


def kappa_from_budget(
    cfg: Config, i_sense_field: np.ndarray, n_illuminators: np.ndarray
) -> np.ndarray:
    """接收机设计所蕴含的有效对消深度，单位 dB。"""
    fraction, _ = predict_cancellation(cfg, i_sense_field, n_illuminators)
    depth = -10.0 * np.log10(np.clip(fraction, EPS, None))
    return np.minimum(depth, float(cfg.cancellation.hw_ceiling_db))
