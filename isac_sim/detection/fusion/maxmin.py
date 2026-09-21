"""最差目标（max-min）目标函数：提案 (P0) / (P1) 的 epigraph 形式。

提案 §5 要求论文主问题是

    max_{x, z}  min_q  P_D,q(x, z)

而不是一个把各目标混起来的加权和。这里实现它的目标值与次梯度，
由 ``selector.maxmin_objective`` 门控：**默认关闭**，于是冻结的发布路径
逐位不变（铁律：新能力必须 config 门控）。

为什么不是 softmin
------------------
``softmin``（-Q*tau*log sum exp(-P_D/tau)）是一个平滑代理，不是 epigraph
max-min。实测（见 ``tests/test_optimization_model_contract.py``）它在
``tau=0.10`` 下让 ``U([.70,.70,.70]) < U([.99,.99,.65])`` —— 最弱目标更差的
方案反而胜出，正是提案 §5 想避免的"容忍某个目标掉队"。调 ``softmin_tau``
治不了：平滑代理在有限温度上永远不是 min。**min 就是 min**，本模块直接用它。

饱和（saturate）
----------------
与既有口径一致，``P_D,q`` 先按 ``detect.pd_required`` 封顶。理由不是数学的
而是工程的：一旦每个目标都达到设计点，max-min 的值就等于设计点，边际为零，
贪婪自然停下，不会为了数值上微不足道的增益继续消耗上报预算。提案 §5 的
``t`` 本身也以 1 为上界，所以封顶不改变优化问题的含义。
"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model import EPS

#: 判定"第几个目标是最小的"时允许的浮点容差（次梯度支撑集用）。
_TIE_TOL = 1e-12


def worst_case_pd(cfg: Config, predicted_pd: np.ndarray) -> float:
    r"""``Phi = min_q min(P_D,q, P_D,req)`` —— 最差目标保证检测概率。

    这就是提案 (P1) 里的那个 ``t``：不是代理、不是上界，就是被最小化的量
    本身。空目标集返回 ``0.0``（没有任何观测 ⇒ 没有任何保证）。
    """
    pd = np.asarray(predicted_pd, dtype=float)
    if pd.size == 0:
        return 0.0
    req = max(float(cfg.detect.pd_required), EPS)
    return float(np.min(np.minimum(pd, req)))


def maxmin_support(cfg: Config, predicted_pd: np.ndarray) -> np.ndarray:
    """次梯度的支撑集：所有并列最小、且尚未达到设计点的目标，均分权重。

    ``min`` 不可微，其（超）次梯度在最小值处取遍单纯形。这里取**均分**这一
    个合法元素：

    * 只用并列最小的目标 —— 改进一个已经不是瓶颈的目标对 ``Phi`` 没有贡献；
    * 已经达到 ``pd_required`` 的目标被排除 —— 封顶之后它们的边际为零，
      继续给它们资源不会提升最差目标；
    * 并列时均分 —— 让贪婪在平局面前不偏向字典序靠前的目标。

    返回的权重和为 1（支撑集非空时），可直接乘 ``dP_D/dD`` 得到
    ``dPhi/dD_q``。
    """
    pd = np.asarray(predicted_pd, dtype=float)
    req = max(float(cfg.detect.pd_required), EPS)
    capped = np.minimum(pd, req)
    below = pd < req
    if pd.size == 0 or not bool(np.any(below)):
        return np.zeros_like(pd, dtype=float)
    floor = float(np.min(capped[below]))
    support = below & (capped <= floor + _TIE_TOL)
    n = int(np.count_nonzero(support))
    if n == 0:
        return np.zeros_like(pd, dtype=float)
    return support.astype(float) / float(n)
