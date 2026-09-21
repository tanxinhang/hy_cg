"""按匹配滤波输出的精确投影抽样本，并施加上报成功/失败信道。

⚠️ **随机数调用顺序是契约的一部分**：本模块内 ``rng`` 的调用次序必须与
原始实现一致（``cn()`` ×5 → ``rng.random`` ×2 → ``rng.normal`` ×2），
否则逐位门禁失效。改动任何一行前先确认这一点。
"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config


def draw_matched_filter_energy(
    rng: np.random.Generator,
    *,
    n_trials: int,
    n_looks: int,
    desired_variance: float,
    background_variance: float,
    interference_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(x0, x1)``：H0/H1 下每个 look 的匹配滤波能量和。

    目标的复系数与接收噪声逐 look 抽取；干扰按同样的复高斯投影到窗口上 ——
    所以这是真实的匹配滤波投影，而不是把干扰当成额外的白噪声。
    """
    shape = (n_trials, n_looks)

    def cn() -> np.ndarray:
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)

    noise0, noise1 = cn(), cn()
    interference0 = np.sqrt(interference_variance) * cn()
    interference1 = np.sqrt(interference_variance) * cn()
    target = np.sqrt(desired_variance) * cn()
    z0 = (noise0 + interference0) / np.sqrt(background_variance)
    z1 = (noise1 + interference1 + target) / np.sqrt(background_variance)
    x0 = np.sum(np.abs(z0) ** 2, axis=1)
    x1 = np.sum(np.abs(z1) ** 2, axis=1)
    return x0, x1


def apply_report_channel(
    rng: np.random.Generator,
    *,
    n_trials: int,
    report_success: float,
    llr0: np.ndarray,
    llr1: np.ndarray,
    failure_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """以 ``report_success`` 的概率保留 LLR，否则替换为零均值高斯失效代理。

    ``report_success == 1`` 时**不消耗任何随机数** —— 这是刻意的，
    保证「理想信道」这一档的可复现性不受无关取值影响。
    """
    if report_success >= 1.0:
        return llr0, llr1
    success0 = rng.random(n_trials) < report_success
    success1 = rng.random(n_trials) < report_success
    failure0 = rng.normal(0.0, np.sqrt(max(failure_variance, 0.0)), n_trials)
    failure1 = rng.normal(0.0, np.sqrt(max(failure_variance, 0.0)), n_trials)
    return np.where(success0, llr0, failure0), np.where(success1, llr1, failure1)
