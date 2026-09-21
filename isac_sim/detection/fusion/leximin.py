"""leximin 势：max-min 的字典序细化，贪心可用版本。

为什么不能直接用 min
--------------------
纯 ``U = min_q P_D,q`` 在**增量贪心**下会死锁，这不是实现细节而是结构性的：

    空集 P_D = [0.050, 0.050, 0.050]
    给 q0 加一条链路 -> [0.259, 0.050, 0.050]
    U: 0.050 -> 0.050      边际增益 **恰好为 0**

只要还有别的并列最弱目标，抬任意一个都不改变 min，于是每一步的
``Delta = U(S u (i,j,q)) - U(S)`` 都是 0，选择器一条链路都不选。实测
（600 m / RCS 0.1 / M=5 / Q=3，5 个 seed）：开 ``maxmin_objective`` 后选中
链路数 12.4 -> **0.0**，最差目标 P_D 0.317 -> 0.050（等于 P_FA）。
调 epsilon、调门限都救不了 —— min 在这个集合函数上的边际就是测不到。

leximin：把 min 细化，而不是软化
--------------------------------
把 P_D **升序**排好，按位置给递减的几何权重：

    U = sum_k rho^k * P_D,(k)        (P_D,(0) <= P_D,(1) <= ...)

``rho -> 0`` 时它就是 ``min``；``rho > 0`` 时第二个、第三个最弱的目标也带得动
U，于是边际恒正，贪心第一步就能动。关键是它不是"回避"max-min：

* 它是 **Schur-凹**的（升序向量上的递减权重和），因此严格满足 Pigou–Dalton
  转移原则 —— 把 P_D 从弱目标挪给强目标必然使 U 下降；
* 它按字典序细化 min：先比最小的，最小的相同再比次小的。这正是提案 §20
  的字典序精神，只不过把"次目标"从"上报开销"换成"次弱目标"；
* 最弱目标的权重是 1，次弱是 ``rho`` —— ``rho=0.1`` 时弱目标的边际比强目标
  强一个数量级，"盯住最弱目标"这件事没有被稀释。

``rho`` 是这个势**唯一**的自由参数，且它有明确极限含义（0 = 真 max-min），
不是"为什么 lambda=0.005"那类无解释常数。

被最小化的量仍然是 min
----------------------
注意分工：本模块只解决"贪心走得动"。提案 (P1) 里那个被接受的 ``t`` 仍然是
:func:`isac_sim.detection.fusion.maxmin.worst_case_pd` 给出的精确
``min_q P_D,q`` —— 两者不能混：势用来排序候选，Phi 用来判接受。
"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model import EPS

#: 判定"并列第几弱"的浮点容差。
_TIE_TOL = 1e-12


def leximin_potential(cfg: Config, predicted_pd: np.ndarray, rho: float) -> float:
    r"""升序加权和 ``sum_k rho^k * P_D,(k)``（先按 ``pd_required`` 封顶）。"""
    pd = np.asarray(predicted_pd, dtype=float)
    if pd.size == 0:
        return 0.0
    req = max(float(cfg.detect.pd_required), EPS)
    capped = np.minimum(pd, req)
    weights = float(rho) ** np.arange(capped.size, dtype=float)
    return float(np.dot(weights, np.sort(capped)))


def leximin_gradient(cfg: Config, predicted_pd: np.ndarray, rho: float) -> np.ndarray:
    r"""``dU/dP_D,q``：第 k 弱的目标拿到 ``rho^k``，并列则均分该段权重。

    返回向量与 ``predicted_pd`` 同形；已封顶（达到 ``pd_required``）的目标
    梯度为 0 —— 封顶之后给它们再多的观测也不会提升任何一个弱目标。
    """
    pd = np.asarray(predicted_pd, dtype=float)
    req = max(float(cfg.detect.pd_required), EPS)
    capped = np.minimum(pd, req)
    n = capped.size
    grad = np.zeros(n, dtype=float)
    if n == 0:
        return grad

    order = np.argsort(capped, kind="stable")
    sorted_vals = capped[order]
    weights = float(rho) ** np.arange(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(sorted_vals[j + 1] - sorted_vals[i]) <= _TIE_TOL:
            j += 1
        share = float(np.mean(weights[i : j + 1]))
        grad[order[i : j + 1]] = share
        i = j + 1

    # 封顶的目标不再有边际。
    grad[pd >= req] = 0.0
    return grad
