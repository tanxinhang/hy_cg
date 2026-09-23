"""TP-UIC V1.1：在 TP-UIC 残余上做**目标条件化**的白化 GLRT。

为什么 V1 的检测器答不了这个问题
--------------------------------
V1 用**整块**被保护子空间的能量给每条臂打分::

    T = ||U_w^H r||^2 / rank(U_w)

这在 600 m 场景里问的是"这个 DD 小块里有没有能量"。那里十个散射体挤在少数
几个延迟格里，于是丢掉目标 ``q`` 之后还有九条回波留在**同一个**子空间里，
``T(H1) ~ T(H0)`` 是构造使然 —— 实测只差 ``+0.06 .. +0.52 dB``，而物理上承诺
的是 ``+17 dB``。坏掉的那一环是检测器，不是抵消器。

V1.1 换的是**问题**，抵消器一动不动::

    H0,q:  r = A_-q a_-q + v
    H1,q:  r = A_q a_q + A_-q a_-q + v,        v ~ CN(0, C_res)

其中 ``(r, C_res)`` 由抵消器交出来。白化、投影掉其他目标的流形，只检验目标
``q`` 剩下的那部分::

    r~    = C_res^{-1/2} r
    A~_x  = C_res^{-1/2} T_e A_x
    P_-q  = I - A~_-q A~_-q^dagger
    B_q   = P_-q C_res^{-1/2} T_e A_q
    T_q   = r~^H P_{B_q} r~                                              (1)

于是形式化的断言从"这块地盘有没有被占"变成"给定另外九个目标和残余干扰，目标
``q`` 还提不提供证据"。论文因此可以陈述这样一条链::

    TP-UIC -> (r, C_res, eta_surv) -> whiten -> nuisance projection -> T_q

即抵消器不只是减掉功率，它还**导出**检测器需要的统计量，两者是协同设计而不是
事后拼接。

残余协方差：精确且便宜
----------------------
记 ``T_d`` 为作用在直连场上的传递、``T_e`` 为作用在回波上的传递::

    r = T_d x + T_e s + (I - F) n - c + (M X) e

其中 ``e`` 是**基于参考**估计的系数误差。估计器臂 ``T_e = T_d = I - F``；
oracle 臂 ``T_e = I``（回波原封不动通过）且 ``T_d = direct_gain I``。把直联系数
取成方差为对角阵 ``R_diag`` 的高斯量（信念式 (3) 已经对它积分过了），干扰加噪声
的协方差是::

    C_res = sigma^2 I + T_d X R_diag X^H T_d^H + (M X) C_h (M X)^H / n_cpi
          = sigma^2 I + U S U^H,   U = [B, T_d X diag(sqrt R), M X L]      (2)

即**单位阵加低秩**，于是 ``C_res^{-1/2}`` 可以通过对 ``span(U)`` 上约 60 x 60 的
限制做特征分解来精确施加：永远不构造 ``K x K`` 矩阵，这才是 4096 个 bin 上白化
付得起的原因。

``sigma^2 I`` 这个地板是值得声明的建模决定。在自己的观测上拟合系数的抵消器会
连被拟合子空间里的噪声一起去掉，它的残余协方差在 ``span(M X)`` 上是**奇异**的
（此处 4096 维中的 14 维）—— 一个检测器绝不能白化进去的零空间。
``cancellation.n_cpi`` 已经声明系数来自参考预算，按这个读法系数误差就是独立
噪声，地板真正满秩，参考预算也才总算能出现在检测指标里（V1 测过它买不到抵消
深度，因为保留项占主导；在这里它买到的是标定，而标定正是检测器真正消耗的量）。

``direct_variance="prior"`` 用 ``cancellation.prior_variance``，即已发布的信念
``sigma_h^2 = 1``，与观测构造器所抽取的单位幅度直联系数一致。``"zero"`` 是"接收
机完全信任自己的估计"这一消融臂；它在检测上付出的代价是残余协方差的标定误差，
而这个误差被上报而不是被藏起来。

可辨识性：测量，而不是断言
--------------------------
``rho_c = ||P_-q C^{-1/2} a_c||^2 / ||C^{-1/2} a_c||^2`` 是一个模板自身白化匹配
滤波能量中**逃过干扰投影**的比例，即 ``1 - rho`` 是被掩蔽的比例。它无标度、
落在 ``[0, 1]``，两端由测试钉住：模板落在其他目标的张成空间里时 ``rho = 0``，
与之正交时 ``rho = 1``。``xi_rel_q`` 是整块模板上的矩阵版本；把模板**移出** DD
网格就得到连续 DD 审计 —— 若 ``rho`` 在每一个分数偏移下都崩塌，那么共位是
本征的，答案就是更高的物理分辨率，而不是更好的抵消器。

本模块**不做**什么
------------------
* 不碰 ``model.py``，不改任何已发布的数字。默认路径
  （``cancellation.enable = False``）上这里什么都不会被导入。
* 不重新调抵消器。V1 已冻结，本模块只消费它。
* 不为 OTFS SIC 声称新意。这里能拿出来的是：抵消深度**和**残余统计量出自同一个
  算法，且有一个目标条件化检测器把两者都消费掉。
"""
from isac_sim.receiver.cancellation_glrt.arm_plan import ARM_ORDER, ArmPlan
from isac_sim.receiver.cancellation_glrt.arm_plan import _belief_subspace, _empty_subspace
from isac_sim.receiver.cancellation_glrt.arm_table import arm_plans
from isac_sim.receiver.cancellation_glrt.audit import identifiability_audit
from isac_sim.receiver.cancellation_glrt.belief_error import (
    _belief_bin_sigmas,
    _belief_error_factor,
)
from isac_sim.receiver.cancellation_glrt.chi2 import (
    _chi2_cdf_even,
    _chi2_quantile_even,
    glrt_p_value,
    glrt_threshold,
)
from isac_sim.receiver.cancellation_glrt.covariance import LowRankCovariance
from isac_sim.receiver.cancellation_glrt.glrt import target_conditioned_glrt
from isac_sim.receiver.cancellation_glrt.neighbourhood import target_neighbourhood_glrt
from isac_sim.receiver.cancellation_glrt.ident import IdentifiabilityAudit
from isac_sim.receiver.cancellation_glrt.information import (DetectionInformation,
    StochasticDetectionInformation, detection_information, stochastic_detection_information)
from isac_sim.receiver.cancellation_glrt.linalg import (
    _generalised_min_eigen,
    _numerical_rank,
    _project_out,
    _projector_energy,
    _psd_sqrt,
)
from isac_sim.receiver.cancellation_glrt.masking import MaskingCurve, masking_curve
from isac_sim.receiver.cancellation_glrt.null_threshold import _null_cov_threshold
from isac_sim.receiver.cancellation_glrt.priors import (
    _direct_prior,
    _estimator_covariance,
    _oracle_gain,
)
from isac_sim.receiver.cancellation_glrt.probe import (
    _low_rank_form,
    _probe_basis,
    _removed_vector,
)
from isac_sim.receiver.cancellation_glrt.residual_form import ResidualModel
from isac_sim.receiver.cancellation_glrt.residual_model import residual_model
from isac_sim.receiver.cancellation_glrt.restrict import restrict_to_target
from isac_sim.receiver.cancellation_glrt.target_glrt import TargetGLRT
from isac_sim.receiver.cancellation_glrt.target_glrt import (
    _centre_mask,
    _dictionary,
    _dictionary_centre_mask,
    _manifold_columns,
)

__all__ = [
    "ARM_ORDER",
    "ArmPlan",
    "IdentifiabilityAudit",
    "DetectionInformation", "StochasticDetectionInformation", "detection_information", "stochastic_detection_information",
    "LowRankCovariance",
    "MaskingCurve",
    "ResidualModel",
    "TargetGLRT",
    "arm_plans",
    "glrt_p_value",
    "glrt_threshold",
    "identifiability_audit",
    "masking_curve",
    "residual_model",
    "restrict_to_target",
    "target_conditioned_glrt",
    "target_neighbourhood_glrt",
]
