"""通信侧口径：MAC、干扰模型、可靠性与时延模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CommCfg:
    """UAV 间通信约束与打包。"""

    R_min: float = 2.0e5
    chi_min: float = 0.30
    enforce_chi_min: bool = False
    comm_leakage_from_sensing: float = 0.05
    comm_direct_leakage_factor: float = 0.0
    K_candidates: int = 4
    b_d: float = 160.0
    # 通信干扰项怎么构成。
    # "full_concurrent"：假设每架 UAV 都按满通信功率发射。
    #     保守最坏情况；与选了哪些链路无关。
    # "active_set"：选择后的敏感性模型。最终评估时只有落在被选中上报集里的
    #     发射机贡献载荷干扰。候选选择仍从保守表出发，所以它被有意标为消融，
    #     而不是一个自洽的内生干扰优化器。
    # "orthogonal"：上报载荷严格时分/频分正交，于是任何别的上报载荷都不会干扰
    #     某条上报链路。持续辐射的感知波形泄漏仍然存在。这是会议版所用的
    #     ``mac_model="serial"`` 的自洽搭档。
    interference_model: str = "full_concurrent"

    # --- 上报可靠度模型 ---------------------------------------------------
    # "heuristic"： chi = gamma / (gamma + gamma_req)。一个单调光滑映射，
    #     没有概率含义（旧行为，为与冻结结果逐位可比而保留）。
    # "fbl"：       有限块长误包率
    #                   eps ~ Q( (C(gamma) - k/n) / sqrt(V(gamma)/n) ),
    #               chi = 1 - eps。这给 chi 一个真正的**包成功概率**含义，
    #               也让载荷大小 k 与块长 n 成为一等参数。
    reliability_model: str = "heuristic"
    # 一个软信息上报包占用的信道使用数（块长）。只被
    # ``reliability_model="fbl"`` 使用。
    n_block: int = 2048
    # 时延记账。
    # "payload"：     T = k / R    （旧口径：时延跟着链路速率走）。
    # "blocklength"： T = n / B    （块在带宽 B 上占 n 个信道使用，
    #                 所以时延与速率无关）。
    latency_model: str = "payload"
    # --- 上报 MAC ---------------------------------------------------------
    # "serial"：  T = sum_l B_l / R_l      （严格时分）。
    # "parallel"：T = max_l B_l / R_l      （所有上报并发）。
    # "slot"：    冲突图着色（共享发射端或接收端算冲突）；同一时隙内的上报并发，
    #             时隙之间串行。在 "slot" 下，一条链路的通信 SINR 会被重算，
    #             **只**把同时隙的发射机算作干扰，于是干扰模型与时延模型描述的
    #             是同一个 MAC。
    #
    # "serial" / "parallel" 保持旧行为：它们时延跟随 MAC，但 SINR 用的是
    # ``interference_model``（默认 "full_concurrent"）。自洽的选择是 "slot"。
    mac_model: str = "serial"
