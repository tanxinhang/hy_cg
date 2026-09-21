"""命名配置束：基础 preset 字典。"""

from __future__ import annotations

from typing import Any, Dict


# --------------------------------------------------------------------------
# 命名配置束
# --------------------------------------------------------------------------
# 一个 preset 是一组**自洽**的覆盖，把模拟器从冻结的旧抽象切到物理自洽的模型。
# preset 有意不做成 CLI 开关：它们就是普通覆盖字典，因此也能被组合进实验变体。
PRESETS: Dict[str, Dict[str, Any]] = {
    # 冻结的修正前模型：尺度盲的 SINR 守卫加上解耦的感知残余地板。它已不再是默认。
    #
    # 老实说明契约：本 preset 只钉住下面两个键，因此它只恢复**干扰 / SINR 守卫**
    # 那一层，别的都不恢复。它**不**钉检测器侧模型 —— 那部分是在本 preset 写完之后
    # 才修订的（全方差公式的组间项从偏转分母移到了 H1 方差，且
    # ``comm_error_model`` 的取值由 ``erasure`` 改成 ``gaussian_replacement``）。
    # 因此那次修订之前产出的历史 CSV —— 包括已归档的
    # ``.workbuddy/baseline_historical_603b61d/main/main.csv`` —— **无法**只靠本
    # preset 复现。冻结回归基线 ``.workbuddy/baseline/main/main.csv`` 已于
    # 2026-09-16 针对当前模型重新冻结，本 preset 就是对着它校验的。
    "legacy": {
        "interference.coupling": "legacy",
        "radio.eps_mode": "legacy",
    },
    # 修正后的耦合模型，配历史上那条"选择后 active-set"的敏感性路径。它是
    # paper-canonical MAC 的一个**消融**，不是替代。当某实验需要一边改变
    # active-set 假设、一边对未来的默认值改动免疫时用它。
    "isac-consistent": {
        "interference.coupling": "shared_spectrum",
        "radio.eps_mode": "noise_relative",
        "comm.interference_model": "active_set",
    },
    # 同上，但用严格（由波形导出）的软统计量与有限块长可靠度模型。
    "isac-consistent-strict": {
        "interference.coupling": "shared_spectrum",
        "radio.eps_mode": "noise_relative",
        "comm.interference_model": "active_set",
        "detect.soft_stat_model": "llr",
        "comm.reliability_model": "fbl",
        "fusion.mode": "explicit",
    },
    # 论文图表/表格的唯一事实来源。上报包正交，而感知波形持续辐射；这既保住了
    # 定义性的 ISAC 耦合，又不需要一个内生 active-set 不动点。legacy/active-set
    # 变体仍作为消融保留。
    "paper-canonical": {
        "geometry.uav_speed_min": 30.0,
        "geometry.uav_speed_max": 60.0,
        "geometry.target_speed_min": 50.0,
        "geometry.target_speed_max": 90.0,
        "interference.coupling": "shared_spectrum",
        "radio.eps_mode": "noise_relative",
        "comm.interference_model": "orthogonal",
        "comm.mac_model": "serial",
        "comm.reliability_model": "fbl",
        "comm.latency_model": "blocklength",
        "comm.enforce_chi_min": True,
        "fusion.mode": "explicit",
        "detect.soft_stat_model": "llr",
        "detect.rcs_model": "mean",
        "prior.belief_mode": True,
        "prior.scheduler_rcs": "mean",
        "refine.enable": True,
        "refine.mode": "window",
        "selector.score_mode": "exact_utility",
        "selector.stop_at_D_min": False,
    },
}
