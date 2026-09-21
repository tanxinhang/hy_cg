"""把目标/干扰/杂波/多径的 PSF 投影到同一个局部窗口上。

窗口由 ``cfg.refine.half_width`` 决定。所有投影都以**单位能量的模板** ``h_unit`` 为基准，
因此量纲一致、可以互相比较。
"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.sensing.model import EPS
from isac_sim.sensing.waveform.dd_transform import full_otfs_kernel

#: 未分辨干扰体（离目标最近的几个 DD 位移），用于构造真实的泄漏而非正交的第二个目标。
INTERFERER_SHIFTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0), (0.43, -0.31), (-0.37, 0.52), (0.61, 0.44),
)

#: 杂波环采样的 DD 位移（固定 8 个方位，环半径 0.75）。
CLUTTER_RADIUS = 0.75
CLUTTER_DIRECTIONS = 8

#: 多径分量：(功率占比, DD 位移)。
MULTIPATH_TAPS: tuple[tuple[float, tuple[float, float]], ...] = (
    (0.65, (0.22, 0.35)),
    (0.35, (-0.41, 0.18)),
)


def local_indices(cfg: Config) -> list[tuple[int, int]]:
    """局部窗口内的 (delay, doppler) 索引（按模 N / L 回绕）。"""
    W = max(int(cfg.refine.half_width), 0)
    return [
        (dk % cfg.waveform.N, dl % cfg.waveform.L)
        for dk in range(-W, W + 1)
        for dl in range(-W, W + 1)
    ]


def _projection(cfg: Config, indices, h_unit: np.ndarray, k: float, l: float) -> float:
    """把某个 DD 位置上的 PSF 投影到模板上，返回投影功率。"""
    kernel = full_otfs_kernel(
        cfg.waveform.N, cfg.waveform.L, k, l,
        float(cfg.waveform.delta_f), int(round(cfg.waveform.fc / 1e6)),
    )
    vector = np.asarray([kernel[a, b] for a, b in indices], dtype=complex)
    return float(np.abs(np.vdot(h_unit, vector)) ** 2)


def psf_projections(
    cfg: Config,
    *,
    target_offset: tuple[float, float],
    interferer_offset: tuple[float, float],
    sync_error: tuple[float, float],
    num_interferers: int,
) -> dict[str, float]:
    """计算目标捕获能量与各类干扰的投影功率。

    理想目标 PSF 与**有同步误差**的模板 PSF 分别生成：前者给出窗口内真实能量，
    后者决定匹配滤波实际捕获了多少 —— 两者的差就是同步误差的代价。
    """
    target_k, target_l = map(float, target_offset)
    interferer_k, interferer_l = map(float, interferer_offset)
    sync_k, sync_l = map(float, sync_error)
    indices = local_indices(cfg)

    captured_kernel = full_otfs_kernel(
        cfg.waveform.N, cfg.waveform.L, target_k, target_l,
        float(cfg.waveform.delta_f), int(round(cfg.waveform.fc / 1e6)),
    )
    h_true = np.asarray([captured_kernel[a, b] for a, b in indices], dtype=complex)
    template_kernel = full_otfs_kernel(
        cfg.waveform.N, cfg.waveform.L, target_k + sync_k, target_l + sync_l,
        float(cfg.waveform.delta_f), int(round(cfg.waveform.fc / 1e6)),
    )
    h_template = np.asarray([template_kernel[a, b] for a, b in indices], dtype=complex)
    template_energy = float(np.vdot(h_template, h_template).real)
    h_unit = h_template / np.sqrt(max(template_energy, EPS))

    local_window_energy = float(np.vdot(h_true, h_true).real)
    captured_energy = float(np.abs(np.vdot(h_unit, h_true)) ** 2)

    n_interferers = min(max(int(num_interferers), 1), len(INTERFERER_SHIFTS))
    leakage_terms = [
        _projection(cfg, indices, h_unit, interferer_k + sk, interferer_l + sl)
        for sk, sl in INTERFERER_SHIFTS[:n_interferers]
    ]
    leakage_projection = float(np.mean(leakage_terms))

    clutter_terms = [
        _projection(
            cfg, indices, h_unit,
            target_k + CLUTTER_RADIUS * np.cos(angle),
            target_l + CLUTTER_RADIUS * np.sin(angle),
        )
        for angle in np.linspace(0.0, 2.0 * np.pi, CLUTTER_DIRECTIONS, endpoint=False)
    ]
    clutter_projection = float(np.mean(clutter_terms))

    multipath_projection = 0.0
    for power, (shift_k, shift_l) in MULTIPATH_TAPS:
        multipath_projection += power * _projection(
            cfg, indices, h_unit, target_k + shift_k, target_l + shift_l
        )

    return {
        "target_k": target_k,
        "target_l": target_l,
        "interferer_k": interferer_k,
        "interferer_l": interferer_l,
        "sync_error_k": sync_k,
        "sync_error_l": sync_l,
        "num_interferers": float(n_interferers),
        "captured_energy": captured_energy,
        "local_window_energy": local_window_energy,
        "leakage_projection": leakage_projection,
        "clutter_projection": clutter_projection,
        "multipath_projection": multipath_projection,
    }
