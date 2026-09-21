"""Monte-Carlo 运行开关与根配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from isac_sim.core.config.aperture import Aperture
from isac_sim.core.config.cancellation import Cancellation
from isac_sim.core.config.comm import CommCfg
from isac_sim.core.config.coordination import ActiveSensing, Coordination
from isac_sim.core.config.detection import DD, Detect
from isac_sim.core.config.fusion import Corr, Fusion
from isac_sim.core.config.interference import Interference
from isac_sim.core.config.physical import Geometry, Scale, Waveform, WaveformImpairments
from isac_sim.core.config.prior import Prior
from isac_sim.core.config.radio import Radio
from isac_sim.core.config.selector import Refine, Selector


@dataclass
class Run:
    """Monte-Carlo, reproducibility and reporting knobs."""

    num_mc: int = 200
    seed: int = 2026
    verbose: bool = True
    # 独立 trial 可以在不同进程里求值。结果按 trial 序号顺序消费，
    # 所以改这个值不改变随机流，也不改变数值输出。
    workers: int = 1
    # 墙钟指标有意做成 opt-in：它们不随 worker 数确定，不能污染 V1 汇总。
    record_runtime: bool = False

# --------------------------------------------------------------------------
# 根配置
# --------------------------------------------------------------------------
@dataclass
class Config:
    scale: Scale = field(default_factory=Scale)
    geometry: Geometry = field(default_factory=Geometry)
    waveform: Waveform = field(default_factory=Waveform)
    waveform_impairments: WaveformImpairments = field(default_factory=WaveformImpairments)
    radio: Radio = field(default_factory=Radio)
    comm: CommCfg = field(default_factory=CommCfg)
    detect: Detect = field(default_factory=Detect)
    dd: DD = field(default_factory=DD)
    refine: Refine = field(default_factory=Refine)
    prior: Prior = field(default_factory=Prior)
    selector: Selector = field(default_factory=Selector)
    run: Run = field(default_factory=Run)
    fusion: Fusion = field(default_factory=Fusion)
    corr: Corr = field(default_factory=Corr)
    active_sensing: ActiveSensing = field(default_factory=ActiveSensing)
    interference: Interference = field(default_factory=Interference)
    coordination: Coordination = field(default_factory=Coordination)
    cancellation: Cancellation = field(default_factory=Cancellation)
    aperture: Aperture = field(default_factory=Aperture)

    # 模型代码用的向后兼容别名，好让公式读起来与原型一致。
    # 它们是 property，不是存储字段。
    @property
    def M(self) -> int:
        return self.scale.M

    @property
    def Q(self) -> int:
        return self.scale.Q

def default_config() -> Config:
    """Return the canonical default configuration."""
    return Config()
