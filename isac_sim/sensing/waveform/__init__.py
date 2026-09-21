"""波形层：OTFS 变换原语、物理 PSF 度量，以及波形驱动的检测自校验。

拆自原 ``sensing/waveform.py``（462 行、含一个 251 行的巨型函数），
对外导入路径不变：

    dd_transform      时频/DD 变换与全 PSF 核（缓存）
    psf               主瓣能量与局部窗口捕获能量
    sweep             解析 eta 与物理 PSF 的逐点扫描
    projection        目标/干扰/杂波/多径的 PSF 投影
    budget            -> 有效检测信噪比
    sampling          匹配滤波抽样 + 上报信道
    analytic          解析矩与 Cornish--Fisher 门限
    exact_reference   Erlang 精确参照与二分标定
    detection_check   装配以上各步的自校验入口
"""

from isac_sim.sensing.model import EPS
from isac_sim.sensing.waveform.analytic import (
    failure_variance,
    fused_moments_and_threshold,
    scaled_llr,
)
from isac_sim.sensing.waveform.budget import interference_variance, link_budget
from isac_sim.sensing.waveform.dd_transform import (
    apply_fractional_delay_doppler,
    full_otfs_kernel,
    otfs_demodulate,
    otfs_modulate,
)
from isac_sim.sensing.waveform.detection_check import waveform_llr_detection_check
from isac_sim.sensing.waveform.exact_reference import (
    calibrated_singleton_threshold,
    erlang_sf,
    exact_singleton_reference,
)
from isac_sim.sensing.waveform.projection import psf_projections
from isac_sim.sensing.waveform.psf import psf_local_capture, psf_main_bin
from isac_sim.sensing.waveform.sampling import (
    apply_report_channel,
    draw_matched_filter_energy,
)
from isac_sim.sensing.waveform.sweep import sweep_compare_analytic_vs_psf

__all__ = [
    "EPS",
    "otfs_modulate",
    "otfs_demodulate",
    "apply_fractional_delay_doppler",
    "full_otfs_kernel",
    "psf_main_bin",
    "psf_local_capture",
    "sweep_compare_analytic_vs_psf",
    "waveform_llr_detection_check",
    "psf_projections",
    "link_budget",
    "interference_variance",
    "draw_matched_filter_energy",
    "apply_report_channel",
    "scaled_llr",
    "failure_variance",
    "fused_moments_and_threshold",
    "erlang_sf",
    "exact_singleton_reference",
    "calibrated_singleton_threshold",
]
