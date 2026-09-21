"""DOTFS-ISAC 协同感知仿真的配置源。

设计说明
--------
最初的单文件原型把约 60 个字段和一个约 30 个开关的 CLI 全部塞进一个扁平的
``SimConfig``，其中大多数只是为了描述**实验变体**（关掉哪个惩罚项、扫哪条
鲁棒性轴……）。这里按含义拆开：

* :mod:`~isac_sim.core.config.physical` / :mod:`~isac_sim.core.config.radio` /
  :mod:`~isac_sim.core.config.comm` / :mod:`~isac_sim.core.config.detection`
  等描述**一个物理设定** —— 也就是审稿人会在系统模型表里查到的那些量。
* :mod:`~isac_sim.core.config.selector` 保存所提链路选择规则的参数。
* :mod:`~isac_sim.core.config.run` 保存 Monte-Carlo / 可复现性 / 上报开关。

实验变体**故意不做成**额外的顶层开关，而是
:mod:`experiments.flow.sweeps` 里的点分路径覆盖字典，经
:func:`apply_overrides` 施加。因此新增一个消融永远不用改这个包。

每个字段的语义都保持 v10 原型不变，以便数字可复现；改的只是**组织方式**。

包内布局
--------
``aliases``        共用小类型别名
``physical``       规模、部署盒、波形与波形损伤
``radio``          接收机射频与雷达链路预算
``comm``           通信口径
``interference``   直连干扰口径与对消深度常数
``cancellation``   接收端干扰消除（TP-UIC）
``aperture``       接收阵列角度维
``detection``      检测器与 DD 格
``fusion``         软融合与链路相关性
``prior``          目标先验与调度器 belief 口径
``selector``       DD 细化与链路选择规则
``coordination``   多机协同回合与主动感知
``run``            Monte-Carlo 开关与根配置 ``Config``
``presets``        命名配置束（基础字典 + 派生链）
``validate``       一致性校验
``overrides``      点分路径覆盖机制
"""

from __future__ import annotations

from isac_sim.core.config.aliases import (
    CommErrorModel,
    IsacPowerModel,
    Link,
)
from isac_sim.core.config.aperture import Aperture
from isac_sim.core.config.cancellation import Cancellation
from isac_sim.core.config.comm import CommCfg
from isac_sim.core.config.coordination import ActiveSensing, Coordination
from isac_sim.core.config.detection import DD, Detect
from isac_sim.core.config.fusion import Corr, Fusion
from isac_sim.core.config.interference import Interference
from isac_sim.core.config.overrides import (
    _coerce,
    _resolve,
    apply_overrides,
    iter_leaf_paths,
)
from isac_sim.core.config.physical import (
    Geometry,
    Scale,
    Waveform,
    WaveformImpairments,
)
from isac_sim.core.config.presets import (
    HEADLINE_RELEASE_PRESET,
    PRESETS,
    apply_preset,
)
from isac_sim.core.config.prior import Prior
from isac_sim.core.config.radio import Radio
from isac_sim.core.config.run import Config, Run, default_config
from isac_sim.core.config.selector import Refine, Selector
from isac_sim.core.config.validate import validate_config

__all__ = [
    # 类型别名
    "Link",
    "CommErrorModel",
    "IsacPowerModel",
    # 物理设定
    "Scale",
    "Geometry",
    "Waveform",
    "WaveformImpairments",
    "Radio",
    "CommCfg",
    "Interference",
    "Fusion",
    "Corr",
    "ActiveSensing",
    "Detect",
    "Prior",
    "DD",
    "Refine",
    "Selector",
    "Coordination",
    "Cancellation",
    "Aperture",
    "Run",
    # 根配置
    "Config",
    "default_config",
    # 命名配置束
    "PRESETS",
    "HEADLINE_RELEASE_PRESET",
    "apply_preset",
    # 校验与覆盖
    "validate_config",
    "apply_overrides",
    "iter_leaf_paths",
]
