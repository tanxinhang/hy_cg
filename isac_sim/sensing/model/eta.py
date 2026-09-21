"""C2F 粗/细 DD 增益数组（论文式 coarse/fine_dd_gain）。

``refine`` 关闭时两个数组都等于 ``dd_frac_loss``，已有路径因此逐位不变。
"""
from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config


def build_eta_arrays(cfg: Config, valid_dd, l_float_grid, k_float_grid, dd_frac_loss):
    """返回 ``(eta_loc, eta_fine)``。"""
    # C2F DD 细化（论文式 coarse/fine_dd_gain）。细化关闭时两个数组都等于
    # ``dd_frac_loss``，已有路径因此逐位不变。计算很便宜：每条链路
    # 2 * (2W+1)^2 次 Dirichlet 求值，泄漏分布缓存在一张 1e-3 网格上。
    from isac_sim.sensing.dd import eta_local_array, eta_fine_array

    if cfg.refine.enable or cfg.refine.apply_to_all:
        eta_loc = eta_local_array(
            cfg,
            np.where(valid_dd, l_float_grid, np.nan),
            np.where(valid_dd, k_float_grid, np.nan),
        )
        eta_fine = eta_fine_array(cfg, dd_frac_loss, eta_loc)
    else:
        eta_loc = dd_frac_loss.copy()
        eta_fine = dd_frac_loss.copy()
    return eta_loc, eta_fine
