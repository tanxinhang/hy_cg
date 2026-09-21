"""TP-UIC V1：目标保持、不确定性可感知的干扰对消 —— 积木转出口。

链路表里的直连残余曾经是一个常数 ``kappa_dc``，而它是场景提出的**需求**
而非某个接收机的描述：600 m / RCS 0.1 的几何要 52.2 dB，发布值 40 dB 让残余
高出处理后回波 10.2 dB。那个常数已从配置中删除 —— 没有接收机实现撑着它，
它却撑着 SINR 的整个分母。本包把残余从"假设的常数"变成**可执行估计器的输出**：
不传入实测残余就是**没有对消**。

信号模型（``K = N*L`` 网格上的复向量）::

    y_j = X_j h_j + A_j alpha_j + n_j

四块积木
--------
**M1 干扰字典**（:mod:`dictionaries`）：每个激活照射源的直达响应，可带局部分数
DD 切向基。**M2 目标切向保护**（:mod:`manifold`、:mod:`protection`）：保护每个
信念目标的中心响应**及其一阶导数列**，取 ``U = orth([J_1..J_Q])``、``P = U U^H``、
``M = I - P`` 把观测劈成"可能含目标证据"与"可用于干扰学习"两部分。
**M3 受保护 MAP**（:mod:`protect`）：``h_hat = (X^H M X + sigma^2 R_h^-1)^-1 X^H M y``，
并保守相减 ``r1 = y - M X h_hat``；因 ``U^H M = 0`` 有不变式 ``P r1 = P y``。
**M4 不确定性核算**（:mod:`accounting`）：
``I_res = tr(P X R_h X^H P) + tr(M X C_h X^H M)``，于是"对消没做完"与"算法失败"
不再是一句话。

约定：``索引 = 多普勒格 * L + 延迟格``；``sigma^2`` 是**每格**噪声功率；
``R_h`` 逐系数对角。``(M, Q)`` 形状的量是「接收机 × 目标」口径。
"""
from __future__ import annotations

from isac_sim.receiver.cancellation.constants import EPS
from isac_sim.receiver.cancellation.sources import DirectSource, TargetSource
from isac_sim.receiver.cancellation.containers import Observation
from isac_sim.receiver.cancellation.result import CancellationResult
from isac_sim.receiver.cancellation.prior_quantile import (
    _residual_quantile_from_descriptors, residual_power_prior_quantile)
from isac_sim.receiver.cancellation.link_offsets import (
    dd_offset_from_physical, direct_link_offset, target_link_offset, wavelength)
from isac_sim.receiver.cancellation.link_jacobians import (
    target_bearing_jacobian, target_bearing_std, target_link_jacobians,
    target_link_std_bins)
from isac_sim.receiver.cancellation.manifold import (
    Subspace, kernel_vector, orthonormalise, tangent_columns)
from isac_sim.receiver.cancellation.steering import (
    _n_obs, _u_of, lift_dictionary, steering_derivative, steering_vector)
from isac_sim.receiver.cancellation.dictionaries import (
    _source_bearings, direct_dictionary, target_dictionary)
from isac_sim.receiver.cancellation.protect import protected_map, soft_protected_map
from isac_sim.receiver.cancellation.accounting import (
    residual_accounting, soft_residual_accounting)
from isac_sim.receiver.cancellation.joint import (
    _positive_sigma, joint_interference_covariance, joint_refine,
    marginalized_interference_map)
from isac_sim.receiver.cancellation.arm import _Arm
from isac_sim.receiver.cancellation.arms import cancellation_arms
from isac_sim.receiver.cancellation.build import build_observation
from isac_sim.receiver.cancellation.build_pair import build_observation_pair
from isac_sim.receiver.cancellation.protection import (
    _noise_power, _protection_basis, protected_target_ids)
from isac_sim.receiver.cancellation.predict import (
    _protection_leakage_fraction, kappa_from_budget, predict_cancellation)
from isac_sim.receiver.cancellation.context import ReceiverContext
from isac_sim.receiver.cancellation.measurement import ReceiverMeasurement
from isac_sim.receiver.cancellation.measure import measure_receiver_context
from isac_sim.receiver.cancellation.measure_residual import measure_residual_fraction

__all__ = [
    "EPS", "DirectSource", "TargetSource", "Observation", "CancellationResult",
    "residual_power_prior_quantile", "_residual_quantile_from_descriptors",
    "wavelength", "dd_offset_from_physical", "direct_link_offset",
    "target_link_offset", "target_link_jacobians", "target_link_std_bins",
    "target_bearing_std", "target_bearing_jacobian", "kernel_vector",
    "tangent_columns", "Subspace", "orthonormalise", "_u_of", "_n_obs",
    "steering_vector", "steering_derivative", "lift_dictionary",
    "_source_bearings", "direct_dictionary", "target_dictionary",
    "protected_map", "soft_protected_map", "soft_residual_accounting",
    "residual_accounting", "joint_refine", "joint_interference_covariance",
    "marginalized_interference_map",
    "_positive_sigma", "_Arm", "cancellation_arms", "build_observation",
    "build_observation_pair", "protected_target_ids", "_protection_basis",
    "_noise_power", "predict_cancellation", "_protection_leakage_fraction",
    "kappa_from_budget", "ReceiverContext", "ReceiverMeasurement",
    "measure_receiver_context", "measure_residual_fraction",
]
