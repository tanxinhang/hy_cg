"""链路模型：几何、信道增益与由它们导出的逐链路量表。

这是 v10 原型信道层的忠实移植：算术与随机抽取的**次序**都没改，
所以运行结果保持逐位兼容。

分包依据
--------
``model.py`` 原本是一个 1091 行的文件，里面既有初等数学，又有
563 行的 ``compute_link_tables``。按"一个文件、一个积木"拆开：

``mathkit``      波长 / 带宽 / 噪声功率 / 雷达硬件增益 / 分母守卫
``statistics``   Q 函数 / 门限反解 / 检测概率 / 二项置信区间
``containers``   ``Geometry`` / ``BaseGains`` / ``BistaticFields`` / ``LinkTables``
``mesh``         UAV-UAV 直连网格        ← 随机数契约第 1 段
``bistatic``     逐 (i,j,q) 的双站量      ← 随机数契约第 2 段
``collision``    同格遮蔽计数
``eta``          C2F 粗/细 DD 增益数组
``base_gains``   ``build_base_gains``：按序串联上面四段
``geometry``     ``generate_geometry``
``link_tables``  ``compute_link_tables``：通信 / 感知 SINR 表

⚠️ ``build_base_gains`` 的随机数调用顺序是逐位契约，见
:mod:`isac_sim.sensing.model.base_gains` 的模块文档。
"""

from __future__ import annotations

from isac_sim.sensing.model.base_gains import build_base_gains
from isac_sim.sensing.model.bistatic import _bistatic_fields
from isac_sim.sensing.model.collision import build_dd_collision_count
from isac_sim.sensing.model.containers import (
    BaseGains,
    BistaticFields,
    Geometry,
    LinkTables,
    packet_bits_for_target,
    rescale_sensing_tables_for_rcs,
)
from isac_sim.sensing.model.eta import build_eta_arrays
from isac_sim.sensing.model.geometry import generate_geometry
from isac_sim.sensing.model.link_tables import compute_link_tables
from isac_sim.sensing.model.mathkit import (
    EPS,
    bandwidth,
    denominator_guard,
    noise_power,
    path_gain,
    radar_hardware_gain,
    rician_power_gain,
    wavelength,
)
from isac_sim.sensing.model.mesh import build_uav_mesh
from isac_sim.sensing.model.statistics import (
    binomial_ci95,
    d_pd_d_D,
    normal_pdf,
    pd_from_deflection,
    qfunc,
    threshold_from_pfa,
)

__all__ = [
    # 初等量
    "EPS",
    "wavelength",
    "bandwidth",
    "noise_power",
    "radar_hardware_gain",
    "denominator_guard",
    "path_gain",
    "rician_power_gain",
    # 统计
    "qfunc",
    "normal_pdf",
    "threshold_from_pfa",
    "pd_from_deflection",
    "d_pd_d_D",
    "binomial_ci95",
    # 容器
    "Geometry",
    "BaseGains",
    "BistaticFields",
    "LinkTables",
    "rescale_sensing_tables_for_rcs",
    "packet_bits_for_target",
    # 装配
    "build_uav_mesh",
    "_bistatic_fields",
    "build_dd_collision_count",
    "build_eta_arrays",
    "build_base_gains",
    "generate_geometry",
    "compute_link_tables",
]
