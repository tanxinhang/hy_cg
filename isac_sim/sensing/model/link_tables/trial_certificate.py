"""TP-UIC 生产接线的 trial 级 nominal receiver-state 上下文。

生产链路有六处 ``compute_link_tables`` 调用点（粗/细、调度/评估、
真值/信念、active_set 重建）。逐个传参既易漏，又最容易造成**两个世界**：
一处注入了实测对消、另一处仍用冻结常数，链路照样跑通、数字照样合理，
但接收机模型自相矛盾。这正是 ``tools/run_receiver_closed_loop.py``
存在的理由。

这里改用 contextvar：一个 trial 只测量一次接收机，把 nominal state 广播给该 trial
内**所有**链路表调用点，调用点签名不变。

门控关闭（默认）时上下文恒为空 ⇒ 逐位不变。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

_ACTIVE: ContextVar[Optional["TrialReceiverState"]] = ContextVar(
    "tpuic_trial_certificate", default=None
)


@dataclass(frozen=True)
class TrialReceiverState:
    """一次 trial 的 nominal 接收机状态，不承载风险保证。

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
def trial_receiver_state(
    state: TrialReceiverState,
) -> Iterator[TrialReceiverState]:
    """把 nominal receiver state 绑定到当前 trial 上下文。"""
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        _ACTIVE.reset(token)


def current_trial_receiver_state() -> Optional[TrialReceiverState]:
    """当前 trial 的 nominal receiver state；门控关闭时为 ``None``。"""
    return _ACTIVE.get()


# Compatibility only.  New code must use the receiver-state names: this
# context feeds continuous link calculations and therefore is not a risk
# certificate.  Keeping aliases avoids changing frozen numerical paths.
TrialCertificate = TrialReceiverState
trial_certificate = trial_receiver_state
current_trial_certificate = current_trial_receiver_state


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
