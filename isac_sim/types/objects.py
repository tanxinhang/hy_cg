"""objects（自 ``isac_sim/types.py`` 拆出）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from isac_sim.core.config import Link


__all__ = [
    "Link",
    "Scenario",
    "BeliefState",
    "ReceiverCertificate",
    "SelectionResult",
    "DetectionResult",
    "ReceiverDiagnostics",
    "CertificateView",
    "CERTIFICATE_FIELDS",
    "CERTIFICATE_SOURCE_FIELDS",
    "DIAGNOSTIC_FIELDS",
    "certificate_view",
    "assert_certificate_contract",
]


Scenario = Tuple["Geometry", "BaseGains", "LinkTables"]


ReceiverCertificate = "CancellationResult"


SelectionResult = Dict[int, List[Link]]


DetectionResult = "MethodResult"


ReceiverDiagnostics = Dict[str, Any]
