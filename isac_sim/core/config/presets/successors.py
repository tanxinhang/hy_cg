"""派生 preset：在基础 preset 上逐级继承得到的研究口径。"""

from __future__ import annotations

from isac_sim.core.config.presets.base import PRESETS


# 冻结的 V1 候选。它有意继承论文发布版的每一项物理、检测、细化与选择器设置，
# 只改"目标专属融合落点"这一条规则。把这个继承关系保持成**可执行**的，是为了
# 防止将来改论文 preset 时意外造出第二套没有记录的实验协议。
PRESETS["target-local-v1"] = {
    **PRESETS["paper-canonical"],
    "fusion.rule": "nearest_target",
}

# 预算驱动的后继版。数值资源预算有意**不**烤进 preset：实验必须把它们当作物理场景
# 输入报出来，而不是在扫完一圈之后挑一个好点。
PRESETS["capacitated-target-fusion-v1.1"] = {
    **PRESETS["target-local-v1"],
    "fusion.rule": "capacitated_value",
    "detect.comm_error_model": "erasure",
    "selector.use_delay_price": False,
    "selector.lambda_c": 0.0,
}

# 只改优化的后继版：继承完整的 V1.1 物理/检测模型，去掉实验性的强制锚点限制。
PRESETS["joint-bundle-v1.2"] = {
    **PRESETS["capacitated-target-fusion-v1.1"],
    "selector.require_local_anchor": False,
}

# 把历史上 50 m^2 的默认值代数分解成"小目标 RCS + 显式雷达硬件预算"。
# 这是一座标定桥，不是被独立验证过的硬件设计：
# 0.1 m^2 * 10^(27/10) = 50.12 m^2。
PRESETS["small-uav-link-budget-bridge"] = {
    **PRESETS["paper-canonical"],
    "detect.target_rcs": 0.1,
    "radio.radar_tx_gain_dbi": 16.0,
    "radio.radar_rx_gain_dbi": 16.0,
    "radio.radar_system_loss_db": 5.0,
}

# 三个有物理命名的小型 UAV 场景尺度。它们直接继承 paper-canonical，因此**不带**
# 雷达硬件预算（radar_net_gain_db = 0 dB：0 dBi 发 / 0 dBi 收 / 0 dB 损耗）。
# 那 27 dB 预算只属于 small-uav-link-budget-bridge，有意不在这里共享。
# 变的只有几何、连通性与目标类均值 RCS。
# 观测角起伏保持关闭，等它的角度规律对着实测标定好再说。
PRESETS["small-uav-dense-s1"] = {
    **PRESETS["paper-canonical"],
    "geometry.area_xy": 1000.0,
    "geometry.h_uav_min": 200.0,
    "geometry.h_uav_max": 500.0,
    "geometry.h_target_min": 200.0,
    "geometry.h_target_max": 500.0,
    "geometry.comm_range": 1200.0,
    "detect.target_rcs": 0.1,
}

# 800 m 水平部署所需的紧凑型标称 RCS 场景。
# 这是区域的边长，不是硬性的或固定的双站基线距离。
PRESETS["small-uav-compact-800m"] = {
    **PRESETS["paper-canonical"],
    "geometry.area_xy": 800.0,
    "geometry.h_uav_min": 200.0,
    "geometry.h_uav_max": 500.0,
    "geometry.h_target_min": 200.0,
    "geometry.h_target_max": 500.0,
    "geometry.comm_range": 1000.0,
    "detect.target_rcs": 0.05,
}

PRESETS["small-uav-nominal-s2"] = {
    **PRESETS["paper-canonical"],
    "geometry.area_xy": 2000.0,
    "geometry.h_uav_min": 300.0,
    "geometry.h_uav_max": 600.0,
    "geometry.h_target_min": 200.0,
    "geometry.h_target_max": 800.0,
    "geometry.comm_range": 1800.0,
    "detect.target_rcs": 0.05,
}

PRESETS["small-uav-sparse-s3"] = {
    **PRESETS["paper-canonical"],
    "geometry.area_xy": 4000.0,
    "geometry.h_uav_min": 500.0,
    "geometry.h_uav_max": 1200.0,
    "geometry.h_target_min": 500.0,
    "geometry.h_target_max": 1200.0,
    "geometry.comm_range": 2500.0,
    "detect.target_rcs": 0.02,
}

# 第一段波形标定阶段用的隔离后继协议。它有意继承冻结的 V1 工作点，只改波形
# 适配器开关与确定性门限标定的分辨率。在实验给出显式、可审计的场景之前，损伤量
# 保持为零；本 preset 绝不可用来覆盖 V1 结果。
PRESETS["target-local-waveform-v2-phase1"] = {
    **PRESETS["target-local-v1"],
    "waveform_impairments.enable": True,
    "detect.fused_calibration_samples": 16_384,
}

# 当前唯一被允许承载头条结论的稿件/结果身份。V1.1 在其预注册测试通过之前仍是
# 一个带门控的局部后继版；脚本与审计可以 import 这个常数，而不用猜。
HEADLINE_RELEASE_PRESET = "target-local-v1"
