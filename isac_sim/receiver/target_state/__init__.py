"""共享目标位置偏移的 cross-fitted MAP（第一版：只做二维水平偏移）。

第一版只引入**一个**新变量：所有接收机共有的目标水平位置偏移
``delta = [dx, dy]``。真值几何、直达场、噪声、其它目标的状态、UAV 阵形一律
不动 —— 见 :mod:`apply` 的字段白名单。

统计模型（与方案 §2 一一对应）：对一个 CPI 上第 ``j`` 个接收机，TP-UIC 之后
的白化残差 ``z_j = C_j^{-1/2} r_j``；给定候选偏移后重新生成目标字典
``B_j(delta)``，目标匹配收益 ``G_j(delta) = z_j^H P_{B_j(delta)} z_j``
（投影前已投影掉其它目标的中心列）。共享 MAP 是

    delta_hat = argmax  sum_j G_j(delta) - 0.5 delta^T P_xy^{-1} delta.

积木：:mod:`geometry`（偏移 -> 几何 -> 源）、:mod:`apply`（只换接收机可知的
字典）、:mod:`scorer`（缓存白化量的逐候选打分）、:mod:`fit`（coarse-to-fine
求解）、:mod:`crossfit`（双折）、:mod:`results`（容器）。
"""
from isac_sim.receiver.target_state.apply import apply_target_offset
from isac_sim.receiver.target_state.crossfit import crossfit_target_state_score
from isac_sim.receiver.target_state.fit import (
    fit_shared_target_offset,
    make_receiver_views,
)
from isac_sim.receiver.target_state.geometry import (
    shift_belief_geometry,
    shifted_target_sources,
)
from isac_sim.receiver.target_state.results import (
    CrossfitTargetStateResult,
    TargetStateMAPResult,
)
from isac_sim.receiver.target_state.views import (
    ReceiverView,
    make_receiver_views,
)
from isac_sim.receiver.target_state.scorer import (
    TargetStateScorer,
    build_scorer,
    neighbourhood_energy,
)

__all__ = [
    "TargetStateMAPResult",
    "CrossfitTargetStateResult",
    "TargetStateScorer",
    "shift_belief_geometry",
    "shifted_target_sources",
    "apply_target_offset",
    "build_scorer",
    "neighbourhood_energy",
    "make_receiver_views",
    "fit_shared_target_offset",
    "crossfit_target_state_score",
]
