"""同一 DD 格内的遮蔽计数。

阵列启用时用 |A(du)|^2 做**软计数**：单阵元时 |A|^2 恰为 1，
所以软计数在整数情形下就是发布模型用的整数计数 —— 阵列只会去掉
它真正能分辨的遮蔽，没有需要调的硬门限。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model.containers import Geometry


def build_dd_collision_count(
    cfg: Config, geom: Geometry, valid_dd, delay_bin, doppler_bin
) -> np.ndarray:
    """每个回波被同格邻居遮蔽的程度。"""
    M, Q = cfg.scale.M, cfg.scale.Q
    from isac_sim.sensing.aperture import bearings as _bearings, masking_fraction as _masking

    dd_collision_count = np.ones((M, M, Q), dtype=float)
    if cfg.dd.enable_dd_collision_penalty:
        use_aperture = bool(cfg.aperture.enable) and int(cfg.aperture.m_rx) > 1
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                bins: Dict[Tuple[int, int], List[int]] = {}
                for q in range(Q):
                    if not valid_dd[i, j, q]:
                        continue
                    key = (int(delay_bin[i, j, q]), int(doppler_bin[i, j, q]))
                    bins.setdefault(key, []).append(q)
                if use_aperture:
                    us = _bearings(geom, j, Q, int(cfg.aperture.axis))
                    m_rx = int(cfg.aperture.m_rx)
                for qs in bins.values():
                    for q in qs:
                        if not use_aperture:
                            dd_collision_count[i, j, q] = float(len(qs))
                            continue
                        # 软计数：每个同格邻居按 |A(du)|^2 而不是"整个自己"
                        # 来遮蔽这个回波。单阵元时 |A|^2 恰为 1，所以它就**是**
                        # 发布模型用的整数计数 —— 阵列只会去掉它真正能分辨的
                        # 遮蔽，没有需要调的硬门限。
                        masked = 1.0
                        for r in qs:
                            if r == q:
                                continue
                            masked += _masking(m_rx, float(us[r] - us[q]))
                        dd_collision_count[i, j, q] = float(masked)
    return dd_collision_count
