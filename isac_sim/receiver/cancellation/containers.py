"""cancellation 积木：观测容器。"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import numpy as np
from typing import List

from isac_sim.receiver.cancellation.sources import DirectSource, TargetSource

@dataclass
class Observation:
    """DD 域观测，以及一个对消器所可能知道的一切。"""

    y: np.ndarray  # (K,) 复向量
    X: np.ndarray  # (K, d) 直达字典
    A: np.ndarray  # (K, Q) 信念目标字典
    x_direct: np.ndarray  # (K,) 真值直达分量（仅诊断用）
    s_target: np.ndarray  # (K,) 真值回波分量（仅诊断用）
    h_true: np.ndarray  # (d,) 真值直达系数
    alpha_true: np.ndarray  # (Q,) 真值回波系数
    sigma2: float
    direct: List[DirectSource] = field(default_factory=list)
    # Receiver-side direct-path parameter estimates used to build ``X``.
    # ``direct`` remains the truth-only diagnostic list.  Keeping the estimated
    # sources is necessary for a deployable residual covariance: DD mismatch
    # must be linearised around what the receiver knows, not around truth.
    direct_est: List[DirectSource] | None = None
    targets: List[TargetSource] = field(default_factory=list)
    # **信念侧**的目标源。上面的 ``targets`` 装的是真值源，所以没有这个字段，
    # 调用方就无法从目标的一个子集重建信念侧字典 —— 而这正是 V1.1 连续 DD
    # 可辨识性审计所需要的（它问的是：一旦把其它目标的流形当作 nuisance，
    # 某个目标的**局部流形**是否仍然可分）。默认 ``None``，于是每个既有的构造点
    # 与每条已记录的结果都继续有效。
    targets_belief: List[TargetSource] | None = None
    # 保护基：接收机只用信念那一支。真值基的存在是为了让实验能给信念误差定价。
    basis_belief: np.ndarray | None = None
    basis_truth: np.ndarray | None = None
    # 哪一个目标在接受检验（"弱目标"）。统计量是在属于该目标的**整块列**上构成的，
    # 因为一个目标被多架 UAV 照射，单列模板会低估它的可检测性。
    weak_index: int = 0
    # ``A`` 与 ``A_true`` 每一列的目标编号，于是实验可以选中某个目标的块、
    # 剔除它的回波来构造 H0 观测，而无需重建几何。
    A_target_ids: np.ndarray | None = None
    A_true_target_ids: np.ndarray | None = None
    # 物理中心的出处必须显式化，因为实验性的信念字典可能追加数目不定的协方差
    # sigma 列；此时"每第 (1+2*order) 列"这种固定规则就是错的。
    A_centre_mask: np.ndarray | None = None
    A_true_centre_mask: np.ndarray | None = None
    # Explicit receiver-side protection decision. ``None`` preserves the
    # historical weakest-target policy; a frozenset is the binary z_j vector.
    protected_targets: frozenset[int] | None = None
