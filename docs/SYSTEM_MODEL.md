# SYSTEM_MODEL

> STATUS: **AUTHORITATIVE** — 本文件是系统口径的唯一事实来源。
> 权威顺序：`SYSTEM_MODEL → THEORY → ALGORITHM → CODE → EXPERIMENT`。
> 任何 `V*.md` 或探索性文档都**不是**事实来源；它们全部归档在 `docs/archive/`。

---

## 1. 一句话口径

面向低 RCS 多目标的分布式多 UAV OTFS-ISAC 协同检测，利用目标 belief 指导接收端目标保持干扰消除，并将接收端残余干扰与目标保真能力反馈给协同观测选择和软信息融合。

问题类型是 **predicted-belief-assisted confirmation / re-detection**，在此之下研究 **receiver-aware cooperative sensing**。

## 2. 正式链路

```
Belief → TP-UIC receiver → Receiver certificate → Cooperative observation selection → Soft fusion → P_D
             ↑                                                                              │
             └────────────────── P_D deficit → Protection priority ────────────────────────┘
```

理论问题：

```
max_{x,z} min_q  P_D,q
s.t.  sensing + reporting + receiver processing + protection budgets
```

## 3. 不是什么

当前系统**不是**：全波形级 OTFS 收发机联合优化 / blind initial acquisition / tracking 系统 /
beam·power·report·fusion·CPI 全变量联合优化 / DRL·MASAC 资源控制系统 / centralized global optimizer。

## 4. 三个一级算法模块

| 模块 | 职责 | 输入 → 输出 | 代码位置 |
|---|---|---|---|
| **A. Belief-aware Receiver** | 清完干扰后还剩多少干扰、目标还剩多少 | `((ξ̂_q, P_q), X_j, receiver context)` → `C_jq = (Ī_res_jq, η_jq)` | `isac_sim/receiver/cancellation.py`、`cancellation_glrt.py`、`scenario/belief.py` |
| **B. Receiver-aware Cooperative Selection** | 在资源约束下选最有检测价值的双站观测 | `C_jq` → `γ_jq` → `x_ijq`；主求解器 **C2F adaptive greedy** | 积木 `isac_sim/cooperation/primitives.py`；求解器 `experiments/selection.py`（`select_c2f*`）、`coordination.py` |
| **C. Detection / Fusion** | 融合并给出检测概率 | `S_q` → `L_q^fuse` → `P_D,q, P_FA,q`，detector-consistent | `isac_sim/detection/fusion.py`、`llr.py` |

**非正式实现**（experimental，不得进生产路径）：soft-μ、covariance protection、tangent variants、
risk polish、fusion polish、bundle master、power policy。

## 5. 唯一 production path

**`experiments/flow/pipeline.py`** 是正式流程的唯一入口（2026-09-20 起它已移出
`isac_sim/` —— 积木库不承载链路）。仓库根的 `run_isac_sim.py` 是薄入口。

```
generate scenario → build belief → receiver certificate
  → receiver-aware link quality → C2F selection → soft fusion → detection evaluation
```

| API | 用途 |
|---|---|
| `run_proposed_trial(cfg, trial_index)` | **唯一正式 proposed 实现**，方法固定为 `proposed_c2f` |
| `run_baseline_trial(cfg, trial_index, method)` | baseline，与 proposed **共用同一 pipeline** |
| `run_trial(cfg, trial_index, method)` | 共同底座 |
| `run_comparison(cfg, trial_index, methods)` | 配对比较（共享几何与 detector 流） |

proposed 与 baseline 的差异**只发生在** `receiver / selector / fusion` 三个可替换位，
因此不会出现"proposed 用真实 TP-UIC、baseline 用固定 κ"这类口径错配。

⚠️ 历史上 `run_tpuic_production.py`、`run_receiver_closed_loop.py`、`run_joint_tpuic_coordination.py`
等多个脚本都自称 production —— **一律以 `experiments.flow.pipeline.run_proposed_trial` 为准**。
`tests/test_layer_boundaries.py::test_single_production_path` 会检查这个唯一性。

## 6. 数据对象契约

`isac_sim/types.py`。核心系统只允许这几种对象流动：

```
Scenario → BeliefState → ReceiverCertificate → SelectionResult → DetectionResult
```

**`ReceiverCertificate` 固定为 `C_jq = (I_res_jq, η_jq)`，只有两个量。**
`kappa / rho / mu / rank / gate_target / support / covariance_eig` 等一律属于
`ReceiverDiagnostics`，仅供审计与出图，**不得**进入 scheduler 核心接口。

⚠️ **实测缺口**：`CancellationResult` 现在是约 25 个字段的富对象，其中调度器只该读两个。
所以 `types.py` 不再假装它是两字段的，而是提供**显式投影**：

```python
from isac_sim.types import certificate_view
view = certificate_view(result)      # -> CertificateView(residual_power, target_retention)
```

源字段名写在 `CERTIFICATE_SOURCE_FIELDS`（`residual_power ← i_res`、`target_retention ← eta_survive`），
`tests/test_layer_boundaries.py` 会核对它真的存在于 `CancellationResult` 上。
把 `CancellationResult` 本身收敛成两字段证书是**语义改动**，必须走 release-identity
流程，不属分层阶段。（此处原写"碰 96 冻结键里的 `direct_cancellation_db`"——
该键已于 2026-09-20 随 κ 一起退役，冻结键现为 **95**。）

## 7. 主工作点与关键参数

| 项 | 值 |
|---|---|
| 主工作点 | **600 m / RCS 0.1 m²** |
| 主口径预设 | `target-local-v1`（= `paper-canonical` + `fusion.rule="nearest_target"`） |
| 干扰/时序口径 | `comm.interference_model = "orthogonal"` |
| Γ_dp（直连对消深度） | **发布锁 40 dB**，已停止优化 |
| 冻结键 / 可标定键 | 96 / 18（见 `V1_STABLE_RELEASE.md`） |
| 契约运行时 | CPython 3.11.0 / numpy 2.2.6（逐位门禁只在此运行时下有效） |

## 8. 代码地图（2026-09-20 切层后）

`isac_sim/` 是**积木库**：只放可自由组合的原子件，不放链路流程、不做 IO。
链接在 `experiments/` 里被拼装 —— **积木库自己跑不动任何链路**，这是刻意的。

```
isac_sim/                       ★ 积木库（无流程、无 IO、无反向依赖）
├── core/           config, naming                    —— 配置源（96 冻结键）与展示契约
├── scenario/       belief, prior                     —— 场景与调度器 belief 视图
├── sensing/        model, dd, waveform, aperture, soft_channel, fbl
├── receiver/       cancellation, cancellation_glrt   —— Module A / C 统计量
├── cooperation/    primitives, reporting             —— Module B 的原子约束/代价/回报预算
├── detection/      fusion, llr, corr, oracle         —— Module C
├── types.py        —— 层间数据对象契约（含证书投影，见 §6）
└── __init__.py     —— 只有 docstring + LAYERS + __version__，**不做任何再导出**

experiments/                    流程与编排（依赖 isac_sim，反向不允许）
├── methods.py           方法名册（字符串清单，不是积木）
├── selection.py         C2F 求解器族
├── coordination.py      协同回合
├── flow/
│   ├── pipeline.py      ★ 唯一 production path
│   ├── simulate.py      MC trial 编排
│   └── sweeps.py        sweep / ablation 定义
├── app/                 cli.py, __main__.py, report.py, plotting.py
└── exploratory/         bundle_master.py, fusion_polish.py

audits/                         审计（非核心，不参与正式系统）
├── theory.py            次模性 / 曲率审计
└── packetization.py     上报负载反事实审计

run_isac_sim.py                 仓库根薄入口 → experiments.app.cli
```

**分层是"可执行的"，不是靠约定**。四条边界由
`tests/test_layer_boundaries.py`（8 条）与 `tests/test_no_undefined_names.py` 守住：

| 边界 | 判据 |
|---|---|
| 依赖方向 | `isac_sim` 不得 import `experiments` / `audits` / `tools` / `tests` |
| 无 IO | `isac_sim` 不得 import `argparse` / `matplotlib` / `plotly` / `csv`，不得调 `open()` / `print()` |
| 无旧路径 | 全仓不得出现 `isac_sim.<旧扁平名>` 形式的导入 |
| 全绝对导入 | `isac_sim` / `experiments` / `audits` 内一律绝对导入，层边界在每条 import 上可见 |
| 层序不退化 | 子包之间的向上依赖不得新增（当前 5 条已知的登记在 `KNOWN_UPWARD_EDGES`） |
| 单一 production path | `def run_proposed_trial` 只能出现在 `experiments/flow/pipeline.py` |

⚠️ **2026-09-20 之前的分层是"假的"**：`isac_sim/__init__.py` 用 `sys.modules`
别名把 235 处旧扁平路径兜住，写错层**不会报错**。兼容层已删除，别名机制不再存在。

⚠️ **子包之间还不是 DAG**。实测存在 5 条向上依赖
（`scenario→sensing`、`sensing→{receiver,detection,cooperation}`、`detection→cooperation`）。
收敛它们是下一阶段的事；当前只用不变式**冻结现状**，防止继续恶化。

⚠️ 契约守卫 `tools/check_contract_refs.py` 的引用已同步到新路径与新行号
（其中 5 条 `sensing/model.py` 的区间漂移是**既有债务**，非本次引入）。

`tools/` 是外围脚本，`results/` 是实验产物，均**不属于**正式系统代码。
