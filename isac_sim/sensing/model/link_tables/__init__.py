"""逐链路通信 / 感知 SINR 表（原 ``model.compute_link_tables``，563 行单体）。

拆包依据
--------
一个 563 行的函数意味着 11 个职责揉在一起。按"改一段不用读全篇"拆开：

``inputs``             接收端入参校验 + "感知块能否复用"的判定
``power``              功率划分与 n0 / eps_den / B / gamma_req
``fields``             干扰场（通信与感知共用同一并发发射集）+ 协同门控
``residual``           对消深度：绝对残余功率 / 实测比例 / 解析桥 / 冻结常数
``comm``               通信 SINR、速率、可靠度、上报可行性
``pair_terms``         逐 (i,j) 的残余、RINR、sigma0、有效感知功率
``sensing``            逐 (i,j,q) 的感知 SINR 与软统计量矩
``compute_link_tables``装配入口（公开 API 与其文档在这里）

本函数**不抽随机数**，所以拆分不涉及逐位随机流契约；等价性由整表逐字段
md5 比对保证。
"""

from __future__ import annotations

from isac_sim.sensing.model.link_tables.comm import comm_tables
from isac_sim.sensing.model.link_tables.compute_link_tables import compute_link_tables
from isac_sim.sensing.model.link_tables.fields import InterferenceFields, interference_fields
from isac_sim.sensing.model.link_tables.inputs import ReceiverInputs, resolve_inputs
from isac_sim.sensing.model.link_tables.pair_terms import PairTerms, pair_terms
from isac_sim.sensing.model.link_tables.power import PowerSplit, power_split
from isac_sim.sensing.model.link_tables.residual import ResidualModel, resolve_residual
from isac_sim.sensing.model.link_tables.sensing import SensingTables, sensing_tables

__all__ = [
    "compute_link_tables",
    "resolve_inputs",
    "ReceiverInputs",
    "power_split",
    "PowerSplit",
    "interference_fields",
    "InterferenceFields",
    "resolve_residual",
    "ResidualModel",
    "comm_tables",
    "pair_terms",
    "PairTerms",
    "sensing_tables",
    "SensingTables",
]
