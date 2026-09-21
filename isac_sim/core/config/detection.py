"""检测器：检验统计量口径与虚警工作点。"""

from __future__ import annotations

from dataclasses import dataclass
from isac_sim.core.config.aliases import CommErrorModel


@dataclass
class Detect:
    """感知侧检测模型与软信息统计量。"""

    Pfa_target: float = 0.05
    D_min: float = 3.0
    # 选择器与字典序 oracle 用的逐目标设计点。它与"用来冻结资源面预算的
    # 经验性弱目标工作要求"是两回事。
    pd_required: float = 0.95
    weak_pd_required: float = 0.80
    # --- 软统计量模型 -----------------------------------------------------
    # "gaussian"：旧的 ``mu = kappa_mu * log(1 + gamma)``，配一个手工设定的
    #     ``soft_mu_scale``。它没有任何推导 —— 只编码了"感知 SINR 越高、
    #     软均值应当越大"。
    # "llr"：     软统计量就是时延-多普勒匹配滤波 / 局部能量输出的
    #     **中心化局部对数似然比**。对复高斯噪声下独立快起伏的 look，
    #     积分格能量服从 Gamma 分布，于是指数分布，于是
    #         ell  = -ln(1+gamma) + x * gamma/(1+gamma),  x = |z|^2 / sigma_n^2
    #         delta = E1[ell] - E0[ell] = gamma^2/(1+gamma)
    #         sigma0^2 = Var0[ell]      = gamma^2/(1+gamma)^2
    #         D_10   = D_KL(p1||p0)     = gamma - ln(1+gamma)
    #     也就是说**一个自由参数都没有**，逐链路的信息增益就是字面意义上的
    #     库尔贝克-莱布勒散度。
    soft_stat_model: str = "gaussian"
    # 一个 CPI 内非相干积累的 OTFS 帧数。只被 ``soft_stat_model="llr"`` 使用，
    # 那里的单链路偏转是 ``L * gamma^2``。这是物理参数（CPI 长度），不是旋钮。
    n_looks: int = 16
    soft_mu_scale: float = 8.0
    soft_sigma0: float = 1.0
    soft_sigma_floor: float = 0.25
    soft_error_sigma_scale: float = 3.0
    # 真实删除模型下"通用标定多报检测器"用的确定性 Monte-Carlo 求积点数。
    fused_calibration_samples: int = 8192
    # 已发布 ``gaussian_replacement`` 上报信道的实验性精确混合标定。False 保持
    # 冻结的 Cornish–Fisher 路径逐位不变；True 则对实现的中心化 Gamma /
    # 高斯替代混合做确定性求积。
    exact_gaussian_replacement_threshold: bool = False
    # Monte-Carlo 检测器使用的环境级污染。
    enable_comm_error_pollution: bool = True
    # ``gaussian_replacement`` 是已发布的 V1 替代模型：失败包被零均值不确定性
    # 替换。``erasure`` 是真删除，贡献精确的零统计量。
    comm_error_model: CommErrorModel = "gaussian_replacement"
    soft_error_flip_scale: float = 1.0
    soft_error_bias_scale: float = 0.5
    h0_error_bias_scale: float = 0.0
    # 每个目标独立的 H1 噪声/上报实现次数。取 1 保持已发布的 Monte-Carlo 协议；
    # 取更大值用于固定几何与被选集下的条件 P_D 审计。
    num_h1_per_target: int = 1
    num_false_per_target: int = 30
    # 信道抽象
    path_loss_exp: float = 2.0
    shadow_std_db: float = 1.0
    rician_K_db: float = 15.0
    target_rcs: float = 50.0
    sensing_processing_gain: float | None = None
    # --- 目标 RCS 模型 ----------------------------------------------------
    # "iid":        每个 (i,j,q) 各自抽一次指数起伏，于是同一个目标对每个双站对
    #               看起来都是不同的物体（旧行为，为可比性保留）。
    # "swerling1":  每个 (目标, CPI) 抽一次指数实现，被所有双站对共享 ——
    #               物理上正确的那种"隐变量目标"读法。可选再乘上一个双站
    #               观测角因子 g(theta_i, theta_j)。
    # "mean":       链路预算里用均值 RCS。这是局部独立 look 的 LLR 的正确搭档：
    #               它的 H1 能量分布已经把 RCS 起伏边缘化掉了，在这里再抽一次
    #               RCS 等于重复计入。
    rcs_model: str = "iid"
    rcs_aspect_enable: bool = False


@dataclass
class DD:
    """OTFS 时延-多普勒的有效性格、分数损耗与碰撞机制。"""

    use_otfs_bin_validity: bool = True
    enable_dd_fractional_penalty: bool = True
    enable_dd_collision_penalty: bool = True
    dd_collision_alpha: float = 1.0
