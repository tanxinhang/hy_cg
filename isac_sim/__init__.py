"""``isac_sim`` —— 积木库。

这个包**只放可自由组合的原子件**，不放链路流程：

======================  ==================================================
``core``                配置数据类、校验、预设与覆盖；展示命名
``scenario``            目标状态先验与调度器的 belief 视图
``sensing``             几何 / 信道 / 干扰 / DD 核 / 链路质量
``receiver``            接收端干扰消除与检测统计量（Module A / C）
``cooperation``         观测选择的原子约束、代价与回报预算
``detection``           软融合、LLR、相关与 oracle
``types``               层间数据对象契约
======================  ==================================================

设计约束（每次改动都要守住）
----------------------------
* **本包不依赖** ``experiments/`` / ``audits/`` / ``tools/`` —— 依赖方向只能从外向内。
* **本包不 import 任何 IO**（文件读写、绘图、argparse）。这些属于 ``experiments/app``。
* **层内模块之间只用绝对导入**，层边界在每一条 import 上可见。
* 本包**跑不动任何链路**：链路（MC trial、sweep、baseline 对照）在
  ``experiments/flow`` 里被拼装。

用法::

    from isac_sim.core.config import Config, apply_preset
    from isac_sim.sensing.model import build_base_gains, compute_link_tables
    from isac_sim.receiver.cancellation import CancellationResult

运行整条链路请用仓库根的 ``run_isac_sim.py``（薄入口，逻辑在
``experiments/app/cli.py``）::

    python run_isac_sim.py --help
"""

from __future__ import annotations

#: 积木库的物理/算法层，由内向外。后一层可以依赖前一层，反向依赖即违规。
LAYERS: tuple[str, ...] = (
    "core",
    "scenario",
    "sensing",
    "receiver",
    "detection",
    "cooperation",
)

__version__ = "1.7.0"

__all__ = ["LAYERS", "__version__"]
