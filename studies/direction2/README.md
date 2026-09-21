# 方向 2（协作融合）——探究产物归档

> **本目录只装方向 2**（协作融合：弱目标 / RCS 稳健选择 / 节点功率 / 融合阈值）。
> 与 `studies/direction1/`（TP-UIC 本体）完全隔离。开始时间：2026-09-21。

---

## 1. 一句话总判

**方向 2 只有一个可迁移的检出杠杆：把已实现的 `geometry_robust_base`（belief 位置误差球上的
增益下界）接到主链路做选择视野。生产路径 n=12 配对实测 ΔP_D = +0.1250（模型版 δ=0，κ=59.6）
/ +0.1583（δ 敏感性版 δ=3e-3，κ=49.3），P_FA 几乎中性 —— **但代价是上报比特 ×2.5 / ×2.0
（`pd_per_mbit` 腰斩），J7 不通过：它是"用上报换检出"，不是免费增益。**

**次优项 `links12`（每目标链路上限 6→12）在 structural 口径下退化为空操作**
（+0.0083 / +0.0000）—— 高 κ 下调度器每目标只选 2.6 条，离上限还远。它只在
**低 κ（measured ≈37 dB）** 下有效（+0.0500）。⇒ **两个杠杆的有效区间由 κ 决定，方向相反。**

**三项判死**：融合阈值（精确标定 ≡ Cornish–Fisher，**逐位等价**）、
节点功率（`sensing_power_scale_by_uav` 死参数，零处传非 None）、
融合权重（按方法名硬编码，不是 config 旋钮）。

⚠️ **口径决定结论，而且决定推荐**：低 κ 下推荐 `links12`（+0.050，代价 +25% 比特），
高 κ 下推荐 `robust`（+0.125~+0.158，代价 +100~146% 比特）。**报收益必须先报 κ 区间**。
（三口径与判据见 `docs/AUDIT_DIRECTION2_ATTRIBUTION.md` §14 与 §16。）

> **🔴 MC=120 补记（默认 `measured` 口径，§10）**：把样本量提到 120 后，
> `robust` 的 ΔP_D 掉到 **+0.0150（z=1.30，不可分辨）** —— n=12 的 +0.125~+0.158 是
> **structural 高 κ 口径**下的值，两者不冲突，但**默认口径下 `robust` 不是检出杠杆**，
> 它确定的是**上报 −41% / 效率 +76%**。默认口径下唯一三项判据全过的是
> **`robust_links12`（+0.0283, z=2.37, 效率 +52%）**。

---

## 2. 目录结构

| 子目录 | 内容 |
|---|---|
| `README.md` | 本文件：汇总索引（唯一入口） |
| `scripts/` | 一次性探针（可复现，不是门禁，跑它们不产生 FAIL） |
| `data/` | 实测产物（`.csv` / `.json`），与脚本一一对应 |
| `docs/` | 审计报告（判据与原始证据） |

---

## 3. ⚠️ 口径声明（读数字前必看）

本目录的 P_D **不是生产发布值**，不可与 `0.4300`（`production_wire` 全闭环）并列。
差异来源：探针走 `select_lagrangian` + 单套 coarse 链路表，**没有**生产链路的两样东西：

1. `_trial_certificate_scope` 的 contextvar 证书广播（生产用它在选择侧也带上实测残余分数）；
2. `refine.enable=True` 下的 C2F 细表（`select_c2f_adaptive` + `eta_fine`）。

⇒ **只有配对 ΔP_D 可迁移**；绝对值只能在本目录内部横比。

**两个世界**：`paper-canonical` 开 `prior.belief_mode=True`，生产是**双世界**
（`base_belief/tables_belief` 给调度，`base_truth/tables_truth` 给检测，
且只有 belief 引导的 DD 窗真正捕获真值 bin 的链路才进检测器）。
只建一套表的扫描 = **静默跑在"完美先验"世界里**，结论不可迁移。

---

## 4. 判据（跑之前钉死）

| 编号 | 判据 | 阈值 |
|---|---|---|
| J1 | 配对（同几何 / 同 belief 抽样 / 同残余分数 / 同检测流） | 必须 |
| J2 | 可分辨性 | \|ΔP_D\| ≥ **0.10**（MC=20 的 P_D SE≈0.05） |
| J3 | 诚实性 | 各臂 P_FA 极差 ≤ 0.01 |
| J4 | 空操作自查 | 翻转前确认该键与 preset 取值不同 |
| J5 | 稀缺性自查 | 融合类改动只在预算稀缺处才有意义 |

---

## 5. 数据资产清单

| 脚本 | 产物 | 结论 |
|---|---|---|
| `diag_fusion2_attribution.py`（默认 profile） | `data/diag_attribution/` | 单世界口径：融合侧杠杆全零（⚠️ 口径错误，仅留痕） |
| `diag_fusion2_attribution.py --profile ablate` | `data/diag_attribution_ablate/` | 单世界口径：融合机制活但已饱和（⚠️ 同上） |
| `diag_fusion2_belief.py`（默认 profile） | `data/diag_belief/` | ★ 双世界口径：6 臂 n=20 主扫描 |
| `diag_fusion2_belief.py --profile focus` | `data/diag_belief_focus/` | ★ MC=40 定论：阈值零杠杆、加链路 +0.042 |
| `diag_fusion2_belief.py --profile twoversion` | `data/diag_belief_twoversion/` | 模型版 / δ 敏感性版（**诊断探针口径**） |
| `run_robust_verdict.py --accounting measured` | `data/robust_verdict_mc12/` | ★ MC=12 生产路径，低 κ 档 |
| `run_robust_verdict.py --accounting structural --delta 0` | `data/robust_verdict_struct_d0/` | ★★ **模型版（上界）**，κ=59.6 |
| `run_robust_verdict.py --accounting structural --delta 0.003` | `data/robust_verdict_struct_d3/` | ★★ **δ 敏感性版**，κ=49.3 |
| `run_robust_verdict.py --share-cert 0` | `data/robust_verdict_nocache/` | 共享证书的逐位无损自证（n=2） |
| `run_robust_verdict.py --trials 1`（structural 冒烟） | `data/smoke_struct_d0/` / `data/smoke_struct_d3/` | 验证切口径确实改 κ（n=1） |
| `run_robust_verdict.py`（默认 out，n=2） | `data/robust_verdict/` | ⚠️ **已被 `robust_verdict_mc12` 取代**，仅留痕（879 B） |

---

## 6. ★ 核心数字（配对 ΔP_D，n=20 / 40）

### 6.1 measured 口径（MC=40，`data/diag_belief_focus/`）

| 臂 | P_D | ΔP_D | se | ΔP_FA |
|---|---:|---:|---:|---:|
| baseline | 0.3833 | — | — | — |
| `links12` | 0.4250 | +0.0417 | 0.0213 | +0.0024 |
| `robust` | 0.4333 | +0.0500 | 0.0388 | +0.0046 |
| `exact_thr` | 0.3833 | **+0.0000** | 0 | −0.0003 |

### 6.2 structural 口径：模型版 / δ 敏感性版（MC=20，`data/diag_belief_twoversion/`）

⚠️ **本小节是诊断探针口径**（`select_lagrangian` + 单套粗表，baseline 链路只 14.70 条），
不是生产路径 ⇒ **数字不可与 §10 并列**。生产口径的两版见 §10。

| 臂 | 口径 | P_D | ΔP_D | se | ΔP_FA |
|---|---|---:|---:|---:|---:|
| baseline | measured | 0.4000 | — | — | — |
| `links12` | measured | 0.4500 | +0.0500 | 0.0273 | −0.0003 |
| `links12` | structural δ=0（**模型版 = 上界**） | 0.6167 | **+0.2167** | 0.0775 | −0.0000 |
| `links12` | structural δ=3e-3 | 0.5500 | +0.1500 | 0.0854 | −0.0025 |
| `robust` | structural δ=0（**模型版 = 上界**） | 0.7333 | **+0.3333** | 0.0725 | +0.0003 |
| `robust` | structural δ=3e-3 | 0.6500 | +0.2500 | 0.0720 | +0.0081 |

⚠️ δ=0 是字典完备的构造产物 ⇒ **上界，不可达**；δ=3e-3 是未标定扫描档（超标 1.86×）
⇒ 报数必须写成"若 δ=X 则 P_D=Y"，**不得称"真实版"**。

---

## 7. ★ A 步：robust 门控已接线（默认关，逐位不变）

| 项 | 内容 |
|---|---|
| 门控键 | `prior.robust_geometry_for_scheduler: bool = False`（新增**未登记**键 ⇒ free 改动） |
| 接线点 | `experiments/flow/simulate.py` `run_one_trial` 的 **belief 分支**，作用于 `base_belief` |
| 语义 | 调度器（选择 / 融合节点分配 / 预算 / 弱目标定义）只看**鲁棒化**增益下界；检测侧 `base_truth` 不动 |
| 恒等条件 | `belief_sigma_pos_m=0`、`belief_mode=False`、门控关 ⇒ **逐位相同** |
| 断言 | `tests/test_direction2_robust_gate.py`（9 tests，1.3 s） |

⚠️ **它只在 `run_one_trial` 的 belief 分支生效**：直接调 `run_method_on_trial` 不走这个门控。

### 7.1 ⚠️ 主链路默认没有直连对消（本轮最硬的发现）

`kappa_dc` 只来自 `_trial_certificate_scope`，而它受 `cancellation.production_wire`
门控 —— **默认 False ⇒ `kappa_dc = 1.0`（直连场全额存活）**。40 dB 常数被物理删除后
没有替代品，于是默认主链路在 600 m / RCS 0.1 下：

```
D_fuse ≈ 1e-6 … 1e-4   （正常量级是几十）      ⇒ P_D = 0.0000
```

第一版定论扫描就踩了这个坑：4 个臂全 0.0000、链路锁在上限 60、上报 0 bit
（不是"融合饱和"，是**根本没有可用信噪比**）。

⇒ **定论扫描必须显式开 `cancellation.production_wire=True`**（实测 κ）。
`run_robust_verdict.py --wire 1` 已把它做成默认，并用一个"每 trial 一次测量、
跨臂共享"的缓存把 4× 的测量成本降回 1×（语义与生产一致：生产也是一个 trial
一次测量、所有方法共享）。

---

## 8. 审计报告索引（`docs/`）

| 文档 | 内容 |
|---|---|
| `AUDIT_DIRECTION2_ATTRIBUTION.md` | 判据（J1–J5）+ 现状盘点（preset 已设）+ 三类实测 + 判决 |

## 9. B 步：MC=120 生产路径定论 + 代价维度

`scripts/run_robust_verdict.py` 走 **`run_one_trial`（生产路径）**：belief 双世界 +
证书作用域 + C2F 细表 + 融合节点分配 + 弱目标定义全部在位 ⇒ `baseline` 与发布
数字同口径（n=2 冒烟 baseline P_D = 0.4500，发布 0.4300）。

配对：公共随机数（`default_rng([seed, trial])`），每 trial 四臂共享几何与 belief。
代价维度：`overhead_bits` / `overhead_delay_s` / 链路数 / `pd_per_mbit`。

（MC=120 结果见 §10。）

---

## 10. ★★★ MC=120 正式定论（`measured` 口径，`data/robust_verdict_mc120/`）

7041 s（58.7 s/trial）。baseline P_D = **0.3942**（发布 0.4300 同量级，略低因实测
κ≈37 dB < 原常数 40 dB）。κ 列未落盘（该 run 启动早于 `--accounting` 接线），
κ≈37 dB 取方向 1 端到端实测值。

| 臂 | P_D | ΔP_D | se | z | ΔP_FA | 链路 | kbit | `pd/Mbit` | Δ弱目标 | J6 | J7 | J10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| baseline | 0.3942 | — | — | — | — | 46.6 | 19.02 | 20.7 | — | — | — | — |
| `robust` | 0.4092 | +0.0150 | 0.0115 | **1.30** | −0.0005 | 59.8 | **11.24** | **36.4** | −0.0417 | ❌ | ✅ | ✅ |
| `links12` | 0.4092 | +0.0150 | 0.0064 | **2.33** | +0.0006 | 59.5 | 24.64 | 16.6 | +0.0250 | ✅ | ❌ | ✅ |
| `robust_links12` | 0.4225 | **+0.0283** | 0.0120 | **2.37** | −0.0001 | 60.0 | 13.38 | **31.6** | −0.0500 | ✅ | ✅ | ✅ |

**三条硬结论**

1. **`robust` 单独在 MC=120 下不可分辨**（z=1.30 < 1.96）。它的确定收益是**上报**：
   kbit 19.02→11.24（**−41%**）、`pd/Mbit` 20.7→36.4（**+76%**），P_D 只是方向为正。
   ⇒ 在默认 `measured` 口径下，`robust` 是**效率杠杆，不是检出杠杆**。
2. **`links12` 是检出杠杆但更贵**：z=2.33 过 J6，代价 kbit +30%、效率 −20%（J7 ❌）。
3. **`robust_links12` 是唯一三项全过的臂**：+0.0283（z=2.37）、P_FA 中性、
   效率 +52%。且 **ΔP_D ≈ 两臂之和**（0.0283 ≈ 0.0150+0.0150）⇒ **两机制可加，
   不抵消** —— 修正 n=4 时"robust 把 links12 抵消掉"的读法（那是噪声）。
   ⚠️ 唯一代价：弱目标 Δ = −0.0500，**正好压在 J10 线上**。

**J9 不触发**（3 臂中 2 臂过 J6）⇒ 方向 2 在默认口径有 MC=120 可分辨收益，但**只有
+0.015~+0.028**，比方向 1 的记账修正（baseline 0.394→0.667~0.742）小一个数量级。

---

## 11. ★★ MC=12 生产路径三口径定论（`data/robust_verdict_*`）

`scripts/run_robust_verdict.py` 走 **`run_one_trial`（生产路径）**：belief 双世界 +
证书作用域（实测 κ）+ C2F 细表 + 融合节点分配 + 弱目标定义全部在位。公共随机数配对，
三档口径**同 seed=10917 前缀** ⇒ 可逐 trial 配对。共享证书已用 `--share-cert 0` 自证逐位无损。

| 口径 | κ 中位 | 臂 | P_D | ΔP_D | z | P_FA | 链路 | kbit | `pd/mbit` | 弱目标 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `measured` δ=0 | ≈37 | baseline | 0.3333 | — | — | 0.0561 | 49.25 | 17.71 | 18.8 | 0.25 |
| | | `robust` | 0.3500 | +0.0167 | 0.35 | 0.0525 | 60.0 | **9.49** | 36.9 | 0.08 |
| | | `links12` | 0.3833 | +0.0500 | 2.56 | 0.0602 | 59.4 | 22.19 | 17.3 | 0.33 |
| **`structural` δ=0** | **59.6** | baseline | 0.7417 | — | — | 0.0514 | 25.67 | 10.08 | 73.6 | 0.750 |
| | | **`robust`** | **0.8667** | **+0.1250** | **4.48** | 0.0527 | 54.58 | 24.80 | 34.9 | 0.667 |
| | | `links12` | 0.7500 | +0.0083 | 0.56 | 0.0516 | 38.00 | 15.41 | 48.7 | 0.750 |
| **`structural` δ=3e-3** | **49.3** | baseline | 0.6667 | — | — | 0.0475 | 30.92 | 11.09 | 60.1 | 0.500 |
| | | **`robust`** | **0.8250** | **+0.1583** | **3.98** | 0.0541 | 56.92 | 22.29 | 37.0 | 0.583 |
| | | `links12` | 0.6667 | **+0.0000** | 0.00 | 0.0473 | 44.75 | 17.28 | 38.6 | 0.500 |

**判决**：`robust` ✅J6（两版）／❌J7（效率降 38–53%）｜`links12` ⚠️ 只在 measured 有效，
structural 下空操作｜J8 ✅ `robust` 胜出｜J9 不触发｜J10 `robust` δ=0 略负（−0.083）。

⚠️ §6.2 的 +0.3333 / +0.2167 是**诊断探针口径**（`select_lagrangian` + 单套粗表 +
baseline 只 14.7 条），**不可与本表并列**。本表才是生产口径。

### 11.1 κ 依赖是真实的，不是 MC 噪声（CI 不重叠）

`robust` 的 ΔP_D 95% CI：

| 口径 | κ | n | ΔP_D | 95% CI |
|---|---:|---:|---:|---|
| `measured` δ=0 | ≈37 | **120** | +0.0150 | **[−0.0076, +0.0376]** |
| `structural` δ=3e-3 | 49.3 | 12 | +0.1583 | [+0.080, +0.236] |
| `structural` δ=0 | 59.6 | 12 | +0.1250 | [+0.070, +0.180] |

⇒ **低 κ 与大 κ 的置信区间完全不重叠**，样本量还是低 κ 那边大 10 倍 ⇒ 效应量差异
（0.015 vs 0.125~0.158）不是抽样噪声。"κ 决定方向 2 哪个杠杆有效"是**已证实的律**。

⚠️ 但 `structural` 两栏只有 **n=12**，未达 J6 的 MC=120 定论规模；z=4.0~4.5 是
n=12 的 z，不可与 n=120 的 z 直接比大小（只比效应量与 CI）。

---

## 12. 下一步

1. ~~接 robust 门控~~ ✅ §7（默认关）。
2. ~~两版数字（模型版 / δ 敏感性版）~~ ✅ §11。
3. ~~MC=120 生产路径定论~~ ✅ §10。
4. **待用户裁决**（三选一）：
   - **A** `robust_links12` 做默认 ⇒ B 类，动发布数字，+0.028 P_D / 效率 +52%，
     代价弱目标 −0.05（压线）。
   - **B** 保持默认关，`robust` 当**效率选项**按需开（默认口径下省 41% 上报，
     P_D 不可分辨）。**推荐** —— 因为方向 1 的记账修正量级（+0.27~+0.35）比它大 10×。
   - **C** 转方向 3（联合闭环）。
5. 若采纳方向 1 的 `structural` 默认口径，则 §10（measured）作废，改用 §11：
   那时 `robust` 是**真检出杠杆**（+0.125~+0.158, z≈4.0–4.5），但效率 −38~−59%。

`tests/test_direction2_fusion_audit.py`（11 tests，<1 s，全量 696 passed / 7 xfailed）：
preset 已设的 4 个门、节点功率零调用（死参数）、融合权重签名只有 `method`、
`geometry_robust_base` 的下界性 / 置信度单调 / belief 关闭时恒等。
