"""接收机射频参数（噪声系数、参考增益、雷达链路预算）。"""

from __future__ import annotations

from dataclasses import dataclass
from isac_sim.core.config.aliases import IsacPowerModel


@dataclass
class Radio:
    """发射功率、噪声地板、ISAC 功率划分与残余干扰。"""

    P_default: float = 1.0
    # 新主线直接以两类物理功率为变量。二者同时为 None 时才使用下面的
    # P_by_uav/rho_by_uav 兼容路径；显式向量允许为零，以表达 sensing-only、
    # reporting-only 与关闭的 UAV，而不再用 epsilon 功率伪装。
    P_sense_by_uav: tuple[float, ...] | None = None
    P_comm_by_uav: tuple[float, ...] | None = None
    # 可选逐 UAV 总功率。None 保持历史上的统一 P_default。
    P_by_uav: tuple[float, ...] | None = None
    rho: float = 0.80  # 联合波形中的感知功率占比
    # 可选的逐 UAV 感知占比；每架 UAV 的总功率仍是 P_default 瓦。
    # 取 None 精确保持历史上的均匀划分。
    rho_by_uav: tuple[float, ...] | None = None
    noise_psd_dbm_hz: float = -174.0
    noise_figure_db: float = 7.0
    # sensing_only            : 只有 rho*P 贡献感知回波
    # joint_waveform          : 整个 P 都贡献
    # reliable_comm_assisted  : rho*P 再加上按可靠度加权的通信部分
    isac_power_model: IsacPowerModel = "sensing_only"
    # 自干扰 / 直连对消误差 / 多机干扰的残余地板。
    # 只被 ``interference.coupling="legacy"`` 使用；耦合模型改用共享干扰场构造
    # 感知干扰。
    residual_self_factor: float = 1e-14
    residual_direct_factor: float = 1e-4
    residual_multi_uav_factor: float = 1e-10
    rinr_sigma_factor: float = 0.15

    # --- 雷达硬件链路预算 -------------------------------------------------
    # 期望双站回波路径上的定向增益。这些量在历史上是缺失的，结果逼迫
    # target_rcs 去吸收整份天线/系统预算。取值为 dB/dBi，统一换算一次为
    # G_hw = 10**((G_tx + G_rx - L_sys)/10)。默认全零，从而精确保持每个已发布
    # 结果。
    radar_tx_gain_dbi: float = 0.0
    radar_rx_gain_dbi: float = 0.0
    radar_system_loss_db: float = 0.0
    # 可选的净增益抽象，只用于预注册的敏感性扫描。一旦设定就覆盖上面的分量求和。
    # 这避免在平台天线方案尚未确定之前先编一个收发增益拆分。
    radar_net_gain_db: float | None = None

    # --- SINR 分母守卫 ----------------------------------------------------
    # 每个 SINR 都是 ``signal / (n0 + interference + guard)``。守卫的唯一作用是
    # 避免除零，所以它相对噪声功率**必须**可忽略。历史常数
    # ``model.EPS = 1e-12`` 不是：在默认射频设置下 ``n0 = 3.83e-14 W``，
    # 也就是守卫是噪声地板的 26 倍，悄悄把每个感知 SINR 压低 14.33 dB ——
    # 这也是把感知干扰项整个盖掉的原因。
    #   "noise_relative"：guard = n0 * 10**(eps_rel_db/10)，即固定在噪声地板
    #                     下方若干 dB，任何功率尺度下都正确。**模型修正后的默认。**
    #   "legacy"：        guard = 1e-12。尺度盲；只为让修正前的冻结结果还能通过
    #                     ``PRESETS["legacy"]`` 复现而保留。
    eps_mode: str = "noise_relative"
    eps_rel_db: float = -30.0
