"""卡方分位数与 CFAR 门限：纯 numpy，不引入 scipy。

``T_q`` 的自由度恒为 ``2 * rank(B_q)``（一个复方向贡献两个实自由度），于是
CDF 收敛成有限和、分位数可以用二分稳定求出。一个"只是想设门限"的检测器不该
为此拉进 scipy —— 发布路径上并没有这个依赖。
"""
from __future__ import annotations

import math


def _chi2_cdf_even(x: float, half: int) -> float:
    """``P(chi^2_{2*half} <= x)``：Erlang 分布的 CDF，闭式有限和。"""
    if x <= 0.0 or half <= 0:
        return 0.0
    h = 0.5 * float(x)
    term = 1.0
    total = 1.0
    for j in range(1, int(half)):
        term *= h / j
        total += term
    return float(1.0 - math.exp(-h) * total)


def _chi2_quantile_even(p: float, half: int) -> float:
    """:func:`_chi2_cdf_even` 的逆：函数单调，二分即可，不需要 scipy。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly inside (0, 1), got %r" % (p,))
    lo, hi = 0.0, 2.0 * float(max(half, 1))
    for _ in range(64):
        if _chi2_cdf_even(hi, half) >= p:
            break
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _chi2_cdf_even(mid, half) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def glrt_threshold(p_fa: float, dof_real: int) -> float:
    """``T = (1/2) chi^2_{dof_real}`` 在给定 ``C_res`` 下的 CFAR 电平。

    ``T_q`` 是白化后的单位方差复残差在 ``dof_real / 2`` 维子空间上的能量，
    因此 ``2 T_q`` 服从自由度 ``dof_real`` 的卡方分布，且
    ``E[T_q] = dof_real / 2``。

    电平取 ``0.5 * chi^2_{dof}(1 - p_fa)``。这个式子里每个因子都各自出过一次
    bug：门限加在**折半后**的统计量 ``T`` 上，而辅助函数返回的是 ``2 T`` 的
    分位数；分位数必须取 ``1 - p_fa`` 而不是 ``p_fa`` —— 放在原假设的 5% 分位
    上会以 95% 的概率误报，漏掉 0.5 则让门限高 8 dB。两个方向都由
    ``test_threshold_achieves_the_requested_false_alarm_rate`` 钉住。
    """
    if dof_real <= 0:
        return float("inf")
    if dof_real % 2:
        raise ValueError(
            "dof must be even (a complex direction gives two real ones), got %d"
            % dof_real
        )
    return 0.5 * _chi2_quantile_even(1.0 - p_fa, dof_real // 2)


def glrt_p_value(statistic: float, dof_real: int) -> float:
    """给定 ``C_res`` 下的 ``P(T >= statistic | H0)``。"""
    if dof_real <= 0:
        return 1.0
    return float(
        max(0.0, 1.0 - _chi2_cdf_even(2.0 * float(statistic), dof_real // 2))
    )
