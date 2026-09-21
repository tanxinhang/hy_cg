"""DD 域增益（自 ``isac_sim/sensing/dd.py`` 拆出）。"""

from __future__ import annotations

import numpy as np
from isac_sim.core.config import Config

from isac_sim.sensing.dd.kernels import leakage_1d, window_bins


def eta_coarse(delay_frac: float, doppler_frac: float) -> float:
    """粗粒度的主 bin DD 增益。

    与 ``isac_sim.sensing.model.build_base_gains`` 已经在用的 ``dd_frac_loss``
    一致，所以两个定义可以互换。
    """
    return float((np.sinc(delay_frac) ** 2) * (np.sinc(doppler_frac) ** 2))


def eta_local_dirichlet(
    N_k: int,
    N_l: int,
    centre_k: float,
    centre_l: float,
    half_w: int,
) -> float:
    """用 Dirichlet 泄漏算局部窗内**可恢复**的 DD 能量。

    源中心在本函数之前一直保持分数形式。只有窗口被取整到物理 bin 索引，这与粗
    增益对自身偏移的取整方式一致（也与兄弟实现 ``omega_dd_dirichlet`` 里
    "被保护窗口"的约定一致）。
    """
    p_k = leakage_1d(N_k, centre_k)
    p_l = leakage_1d(N_l, centre_l)
    inds = window_bins(centre_k, centre_l, half_w, N_k, N_l)
    e = 0.0
    for k, l in inds:
        e += float(p_k[k] * p_l[l])
    return float(min(max(e, 0.0), 1.0))


def eta_local_sinc(
    delay_frac: float,
    doppler_frac: float,
    half_w: int,
) -> float:
    """用连续 sinc 窗算可恢复的 DD 能量。

    ``delay_frac`` / ``doppler_frac`` 是带符号的分数偏移
    （``l_float - round(l_float)``）；窗口和遍历整数偏移
    ``Delta in [-half_w, +half_w]``。返回值落在 ``[0, 1]``。
    """
    total = 0.0
    for dk in range(-half_w, half_w + 1):
        for dl in range(-half_w, half_w + 1):
            total += float((np.sinc(dl - delay_frac) ** 2) * (np.sinc(dk - doppler_frac) ** 2))
    return float(min(max(total, 0.0), 1.0))


def eta_local(
    cfg: Config,
    delay_frac: float,
    doppler_frac: float,
    l_float: float,
    k_float: float,
) -> float:
    """局部 ``(2W+1) x (2W+1)`` 窗内可恢复的 DD 能量。

    核由 ``cfg.refine.window_kernel`` 选择：

    * ``"dirichlet"``（默认）用周期 Dirichlet 泄漏 :func:`eta_local_dirichlet`
      和**分数**中心 ``l_float`` / ``k_float``。
    * ``"sinc"`` 用连续 sinc 窗 :func:`eta_local_sinc` 和**带符号的**分数偏移。

    当 ``half_w == 0`` 时，sinc 与粗增益精确一致；有限网格的 Dirichlet 版本随
    网格增大趋近于它。
    """
    r = cfg.refine
    if r.window_kernel.lower() == "sinc":
        return eta_local_sinc(delay_frac, doppler_frac, r.half_width)
    if r.window_kernel.lower() == "dirichlet":
        return eta_local_dirichlet(
            cfg.waveform.N,
            cfg.waveform.L,
            k_float,
            l_float,
            r.half_width,
        )
    raise ValueError(f"Unknown refine.window_kernel={r.window_kernel!r}")


def eta_refine(
    cfg: Config,
    eta_c: float,
    eta_loc: float,
) -> float:
    """细粒度 DD 增益。

    * ``cfg.refine.mode = "interp"``（旧口径）：式 (fine_dd_gain)
      ``min{1, max[eta_min, eta_c + kappa_dd * (eta_loc - eta_c)]}``。
    * ``cfg.refine.mode = "window"``：精细估计器解出了分数延迟-多普勒偏移，因此
      精确恢复局部窗能量，即 ``eta^f = eta^loc``。没有自由参数，``kappa_dd`` /
      ``eta_min`` 这两条启发式随之消失。
    """
    r = cfg.refine
    if getattr(r, "mode", "interp").lower() == "window":
        return float(min(1.0, max(0.0, eta_loc)))
    f = eta_c + r.kappa_dd * (eta_loc - eta_c)
    return float(min(1.0, max(r.eta_min, f)))
