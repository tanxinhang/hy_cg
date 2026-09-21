"""certificate（自 ``isac_sim/types.py`` 拆出）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from isac_sim.types.fields import CERTIFICATE_SOURCE_FIELDS


@dataclass(frozen=True)
class CertificateView:
    """调度器真正看到的东西：两个标量。"""

    residual_power: float   # I_res_jq
    target_retention: float  # eta_jq


def certificate_view(result: Any) -> CertificateView:
    """把 :class:`CancellationResult` 投影成调度器视图。

    少了任何一个源字段都会抛 :class:`AttributeError` —— 宁可炸，也不要静默
    把诊断量当接口量用。
    """
    return CertificateView(
        residual_power=float(getattr(result, CERTIFICATE_SOURCE_FIELDS["residual_power"])),
        target_retention=float(getattr(result, CERTIFICATE_SOURCE_FIELDS["target_retention"])),
    )


def assert_certificate_contract(result: Any) -> None:
    """校验证书对象满足契约：两个源字段存在且为标量。"""
    for name, src in CERTIFICATE_SOURCE_FIELDS.items():
        if not hasattr(result, src):
            raise AttributeError(
                f"证书契约破坏：{type(result).__name__} 缺少 {name} 的源字段 {src!r}"
            )
        value = getattr(result, src)
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"证书契约破坏：{src!r} 必须是标量，实际是 {type(value).__name__}"
            )
