"""TP-UIC 生产接线的 trial 级证书上下文。

生产链路有六处 ``compute_link_tables`` 调用点（粗/细、调度/评估、
真值/信念、active_set 重建）。逐个传参既易漏，又最容易造成**两个世界**：
一处注入了实测对消、另一处仍用冻结常数，链路照样跑通、数字照样合理，
但接收机模型自相矛盾。这正是 ``tools/run_receiver_closed_loop.py``
存在的理由。

这里改用 contextvar：一个 trial 只测量一次接收机，把证书广播给该 trial
内**所有**链路表调用点，调用点签名不变。

门控关闭（默认）时上下文恒为空 ⇒ 逐位不变。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

_ACTIVE: ContextVar[Optional["TrialCertificate"]] = ContextVar(
    "tpuic_trial_certificate", default=None
)


@dataclass(frozen=True)
class TrialCertificate:
    """一次 trial 的接收机实测证书。

    ``fraction[j] = I_res / I_in`` 是接收机 j 的**实测**残余比例，
    直接替代冻结常数 ``kappa_dc``；``retention[j, q]`` 是被测目标 q 的
    回波存活率，走分子。两者必须成对给出：只给分母等于把对消器夺走的
    回波能量仍记在目标头上（半闭环高估）。
    """

    #: 逐接收机残余比例，形状 ``(M,)``
    fraction: np.ndarray
    #: 逐(接收机,目标)回波存活率，形状 ``(M,)`` 或 ``(M,Q)``
    retention: np.ndarray


@contextmanager
def trial_certificate(cert: TrialCertificate) -> Iterator[TrialCertificate]:
    """把一个 trial 的证书绑定到当前上下文。"""
    token = _ACTIVE.set(cert)
    try:
        yield cert
    finally:
        _ACTIVE.reset(token)


def current_trial_certificate() -> Optional[TrialCertificate]:
    """当前 trial 的证书；门控关闭时为 ``None``。"""
    return _ACTIVE.get()


def resolve_injection(
    residual_fraction_by_receiver: Optional[np.ndarray],
    target_retention_by_receiver: Optional[np.ndarray],
) -> tuple:
    """两个注入参数都省略时，从上下文回填。

    **显式传参优先**：调用方已经给出任何一个，就原样返回 —— 广播不能覆盖
    调用方的显式意图。门控关闭时上下文恒为空，两个名字保持 ``None``，
    于是走回冻结常数，逐位不变。
    """
    if residual_fraction_by_receiver is None and target_retention_by_receiver is None:
        cert = _ACTIVE.get()
        if cert is not None:
            return cert.fraction, cert.retention
    return residual_fraction_by_receiver, target_retention_by_receiver
