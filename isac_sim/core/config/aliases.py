"""配置模块共用的小类型别名。"""

from __future__ import annotations

from typing import Literal, Tuple


# --------------------------------------------------------------------------
# 包内共用的类型别名
# --------------------------------------------------------------------------
Link = Tuple[int, int]

# NOTE 实验方法名册**不属于积木库** —— 它只是实验记账（名字清单 + 展示顺序 +
# 方法名到策略的映射），已整体搬到 `experiments/methods.py`。本模块只保留
# 真正的物理/算法开关类型：下面的通信误差模型与 ISAC 功率模型，两者的取值
# 都在 sensing/ 与 detection/ 里被真实分支读取。
# `tests/test_method_roster.py` 保证这边不会再长出方法名。

CommErrorModel = Literal["erasure", "gaussian_replacement", "flip", "biased"]

IsacPowerModel = Literal["sensing_only", "joint_waveform", "reliable_comm_assisted"]
