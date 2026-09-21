"""连续 DD 可辨识性审计的结果容器。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class IdentifiabilityAudit:
    """``rho`` 的三个口径：在网格上、在流形上、以及离开网格之后。"""

    arm: str
    target: int
    dictionary: str
    rho_on_grid: np.ndarray
    rho_manifold: np.ndarray
    rho_manifold2: np.ndarray
    xi_rel_on_grid: float
    xi_rel_manifold: float
    n_templates: int
    n_nuisance_on_grid: int
    n_nuisance_manifold: int
    ncp_unit_on_grid: float
    ncp_unit_manifold: float
    dof_on_grid: int
    dof_manifold: int
    off_grid: Dict[Tuple[float, float], np.ndarray] = field(default_factory=dict)

    @property
    def rho_min_on_grid(self) -> float:
        """网格口径下最坏模板的可辨识度。"""
        return float(np.min(self.rho_on_grid)) if self.rho_on_grid.size else float("nan")

    @property
    def rho_min_manifold(self) -> float:
        """一阶流形口径下最坏模板的可辨识度。"""
        return (
            float(np.min(self.rho_manifold)) if self.rho_manifold.size else float("nan")
        )

    @property
    def rho_min_manifold2(self) -> float:
        """二阶流形口径下最坏模板的可辨识度。"""
        return (
            float(np.min(self.rho_manifold2)) if self.rho_manifold2.size else float("nan")
        )

    @property
    def off_grid_min(self) -> float:
        """所有离网偏移里最差的那个 ``rho``。"""
        vals = [float(np.min(v)) for v in self.off_grid.values() if v.size]
        return float(np.min(vals)) if vals else float("nan")

    @property
    def off_grid_median(self) -> float:
        """离网偏移下 ``rho`` 的中位水平。"""
        vals = [float(np.median(v)) for v in self.off_grid.values() if v.size]
        return float(np.median(vals)) if vals else float("nan")

    @property
    def off_grid_max(self) -> float:
        """离网偏移下 ``rho`` 的最好情况。"""
        vals = [float(np.max(v)) for v in self.off_grid.values() if v.size]
        return float(np.max(vals)) if vals else float("nan")
