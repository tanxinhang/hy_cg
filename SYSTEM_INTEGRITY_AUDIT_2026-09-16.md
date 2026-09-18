# 系统完整性审计：系统模型 / 算法 / 理论

**日期**：2026-09-16
**对象**：`isac_sim/`（V1 稳定版本，发布口径 `target-local-v1`）
**可执行证据**：`tools/audit_system_integrity.py`

```
python tools/audit_system_integrity.py          # 结构性 + 理论不变量（快）
python tools/audit_system_integrity.py --deep   # 追加系统模型数值探针
```

当前结果：**10 PASS / 11 FAIL**。下面每条 FAIL 都是实测出来的数，不是读代码读出来的判断。
唯一被信任的东西是 assert；读代码得到的印象不算证据。

---

## 0. 结论摘要

| 层 | 最严重的一条 | 性质 |
|---|---|---|
| **系统模型** | 发布工作点是**噪声受限**（rinr = −2.8 dB），而「干扰受限」是整个性能杠杆路线图的前提 | 前提错误，需改写结论 |
| **算法** | 选择器永远看不到 `active_tx_mask`，感知 SINR 在选择期是**几何常数** | 能力缺失，需在论文声明 |
| **理论** | V1 预设**永远进不到**论文描述的精确 LLR 标定阈值分支，代码用 `return` 静默退回 Cornish–Fisher | 理论与发布口径不一致 |

三条 Chebyshev 式纠正（都是本次审计自己推翻的）——先列在这里，因为它们推翻的是既有文档：

1. **「干扰受限」是几何条件量，不是系统属性。** 实测 rinr 随部署尺度单调变化：
   4000 m **−2.8 dB**（发布工作点）/ 2000 m +0.6 / 1000 m +5.0 / 800 m +6.3 / 400 m **+10.6 dB**。
   既有文档记录的「+9.9 dB」是在 **400 m 压缩几何**下测的，被外推到了 4 km 发布工作点。
2. **贪心可逆性：主路径不可逆，但之前那条（来自 `select_c2f_adaptive`）判断错位。**
   `_greedy_lagrangian` 内确实只 append 无 swap；唯一的 swap 在 `select_c2f_adaptive` 的
   **后处理 polish** 里，不在发布路径上。
3. **`geom_factor` 不是「重构丢掉的物理」。** 它在归档脚本里只进旧的手设 β 权重，
   而 β 已被派生的 `selection_utility` 取代（`llr.py:65`、`selection.py:290` 有文档）。
   它现在是**死字段**，属清理项，不是物理错误。

---

## 1. 系统模型层

### P0 — 「干扰受限」前提在发布工作点不成立（`C3` / `C4`）

```
median rinr vs footprint:  4000m -2.8dB | 2000m +0.6 | 1000m +5.0 | 800m +6.3 | 400m +10.6
```

感知分母是 `(n0 + residual_total + eps_den) * (1 + waveform_inr)`（`model.py:791-794`），
`rinr = residual_total/(n0+eps_den)`（`model.py:753`）**正是**这个分母里的干扰/噪声比。
在 `direct_cancellation_db=40`、面积 4000 m 的发布几何下它只有 **−2.8 dB**：残留在噪声底以下。

**为什么这条贵**：整套「四族杠杆」结论（`PERFORMANCE_OPTIMIZATION_ROADMAP.md`）建立在
`r ≫ 1` 上。实测的那条 `r ≈ +10 dB` 来自 400 m 压缩几何。在该处成立的分母类饱和结论
（`P_default×100` 只买 0.31 dB）**不能直接搬到发布工作点**——那里 `r < 1`，
分母类旋钮恰恰是复活的那一族。

**必需的处置（二选一，不能都不做）**
- 所有功率/杠杆结论必须显式绑定部署尺度陈述；或
- 重做杠杆排序，并说明发布工作点其实**不受残留干扰主导**。

### P1 — 干扰口径枚举会被静默忽略（`A1` / `A2`）

`config.py:987` 校验时 `.lower()` 归一化，但**没有写回**；`model.py:614` 用 `==` 精确比较。

- `interference_model="Orthogonal"`：通过校验，实测与 `full_concurrent` **逐位相同**，
  与 `orthogonal` 不同。正交假设被无声丢弃，退化成最坏情况。
- `interference_model="active_set"` 但没传 `active_tx_mask`：实测结果与 `full_concurrent`
  **逐位相同**。因为 `model.py:617-621` 的分支实际由 `active_tx_mask` 决定，枚举值只是装饰。

**根因**：同一个语义 dispersed 在两处，一处宽松一处严格；且缺少终局 `else: raise`。
这是最贵的一类错误——不崩、不报警，只是把 Distinct 的物理假设换成另一个。

**修法**：`model.py:613` 前加 `assert c.interference_model in {...}`，并把三者间的不变量写成断言。

### P1 — 死配置项（口径条件性）

实测这**三个键都能改变物理量** → `residual_direct_factor` / `residual_multi_uav_factor` /
`comm_direct_leakage_factor` 当前都能改到数（不是死键）。但这两个键在
`interference.coupling="shared_spectrum"`（发布口径）下**恒不被读**——注释在
`config.py:147-149` 已写明「only used by legacy」。属文档已声明，风险可控。

真正的清理项是 `geom_factor`（见 0.3）：算了、存了、无人消费。

### PASS — 数值保护项已修好（`B1`）

`eps_mode="noise_relative"` 下 guard = 1e-3·n0 → 只抬 **0.0043 dB**；
未知模式**抛 `ValueError`** 而不是回退。这一层是对的，历史 26×n0（−14.3 dB）的问题已关闭。

### PASS — 表健康（`B2`）

`gamma_comm` / `gamma_sense` / `rinr` / `raw_gamma_sense` 全部有限且非负。
`max/min` 归约吞 NaN 的风险在当前工作点没有兑现。

### 提示 — 带宽约定（`G3`，当前 PASS 但是脆的）

`bandwidth() = N·delta_f`（`model.py:31`），而 OTFS 帧用 `l_float = tau·L·delta_f`（`model.py:400`）。
只有因为当前 **N == L == 64** 才无事。任何单独改 N 或 L 的扫描都会静默重定义噪声底。
**建议加 `assert N == L` 或显式 `bandwidth_override`**，趁它现在还是 PASS。

---

## 2. 算法层

### P0 — 选择器对干扰是盲的（`F1`）

`selection.py` 全文不出现 `active_tx_mask`；掩码在 `simulate.py` 里**选择提交之后**才生成。
默认 `interference.sense_gate_by_active_tx=False`（`config.py:304`）。

**后果**：选择期内感知 SINR 是几何常数 ⇒ 干扰规避项**不存在于**目标函数里 ⇒
任何选择器（含 oracle）都无法规避干扰源。这不是调参问题，是自由度缺失。
`selection.py` 里就算加了极端厉害的调度规则，也拿不到这块钱。

**必需的处置**：要么把 `active_tx_mask` 送进候选评估（代价：多次建表），
要么在论文里明确写「本工作不含干扰感知调度」。不能默认读者知道。

### P0 — 贪心主路径不可逆（`F2`）

`_greedy_lagrangian`（`selection.py:329-506`）只有 append/add，无 remove/swap。
唯一的 swap 在 `select_c2f_adaptive` 的后处理 polish（`selection.py:840`），**不在发布路径**上。

**后果**：每次 commit 永久生效 ⇒ 结果是「首次通过」的结果，不是局部最优。
把 greedy 输出与 oracle（`oracle.py`）比对得出的「贪心损失」因而被**系统性高估**：
它混入了一部分「因为没有 swap 而丢掉的」而不是「因为贪心本性丢掉的」。
论文里任何「greedy gap」的结论都受影响。

### P1 — 迭代预算是写死的（`F3`）

```
power_joint.py:rounds=2   power_c2f.py:rounds=2   power_c2f.py:passes=0
power_c2f_conservative.py:rounds=2   joint_polish.py:passes=2
```

没有一处来自收敛性推导。项目自己的文档已承认（`SCHEDULING_CONVERGENCE_AND_FUSION_UPGRADE.md:145`
标「固定 2，无收敛判据」）。

**判据**：唯一可靠的是「多给几轮是否还改进」。跑 `rounds ∈ {1,2,3,4,6}`，
记录 objective **和成本**，首次出现在 target 不再变化的 R 才是经验收敛轮数。
若实测 2 就是收敛点，那也要**用实测结果**写进论文，而不是靠字面量。

### P1 — 变体标签会说谎（`H1` / `H2`）

- `ABLATION_VARIANTS["full"] = {}`、`DD_VARIANTS["full_dd"] = {}`（`experiments.py:40,48`）
  是空覆盖，完全继承调用方配置。一旦用 `--set` 污染，"full" 行就不再是 full。
  同文件 `experiments.py:1164-1177` 的 interference 变体**显式钉死两个键**并注释
  「否则标签会说谎」——明知问题，但只修了那一处。
- `PRESETS["legacy"]` 只钉 2 键。注释（`config.py:716-726`）已诚实声明无法用它复现历史 CSV，
  这个属于「已知道但不设门禁」，建议加运行期警告。

### P2 — 已实现但无人调用的能力

`distributed_bids` 唯一入口在 `EXPERIMENTAL_METHODS`（被 `selection.py:77-79` 排除出默认名单），
`experiments.py` 21 个实验零引用 ⇒ 已发布结果里 `bid_rounds` 恒 0。
`llr.draw_llr_erased`、`llr.optimal_fusion_weight` 全仓库（含 tests）零调用。
整条主动证据栈只在 `__init__.py` 重导出，CLI/实验均无入口。

**风险不是性能，是「能力表」与「可复现脚本」不一致**：论文写了、代码有、但跑不出来。

---

## 3. 理论层

### P0 — 发布口径永远进不到论文描述的精确检测分支（`D1`）

```
V1 preset resolves to:  soft_stat_model='llr'  comm_error_model='gaussian_replacement'
```

`fusion.py:351-356` 要求三者全成立才走精确路径：`soft_stat_model=="llr"`、
`comm_error_model=="erasure"`、`not corr.enable`。V1 预设满足第一条和第三条，
**第二条靠 dataclass 默认值继承，值是 `gaussian_replacement`**，于是 `return fallback`。

**关键不是「没走精确路径」，是它用 `return` 而不是 `raise` 表达不兼容。**
结果：论文写的是精确 LLR–擦除分位数检测器，实际跑的是 Cornish–Fisher 矩匹配阈值，
而这条偏差**不会出现在任何日志里**。

**处置（必须选一）**
- 若要精确：像 v1.1 那样在预设里显式钉 `detect.comm_error_model="erasure"`
  （`config.py:794` 已经这么做了，V1 没跟上）；或
- 若要 CF：把论文的检测器描述改成实际执行的 CF 阈值，并把精确分支标注为「未来工作」。

### P1 — 同一个互斥在两处用不同严重度表达（`D2`）

`corr.enable × exact_llr_sum`：`fusion.py:325` **抛异常**，`fusion.py:354-356` **静默退回 CF**。
前者拒绝，后者换算法。两个路径产出的数字不可比，而它们用同一个配置项进入。

另有判据不一致：`fused_h0_variance` 的 corr 分支要求 `base is not None`，
而 `fused_h0_skewness`（`fusion.py:294`）**不要求** ⇒ 同一配置下 H0 方差走独立模型、
H0 偏度走相关模型，一个统计量两种 factorization。

### PASS — 理论闭式是自洽的（`G1` / `G2`）

逐位成立（残差 0.00e+00）：Jeffreys = δ = KLD + reverse KLD；deflection δ²/v₀ = L·γ²；
最优融合权 δ/v₀ = 1+γ。H1 与 H0 方差相异（真的二元假设检验，没有退化）。
有限块长色散 `V = (1−(1+γ)⁻²)(log₂e)²` 与正态近似完全一致——**这部分可以对审稿人拍胸脯**。

### 理论 GAP（对论文的实际影响）

| 类 | 项 | 处置 |
|---|---|---|
| GAP-A | 补充证明未进编译产物（`ReproducibilitySupplement.tex` 未被 `\input`） | 要么 input，要么删正文引用 |
| GAP-A | `coherent_oracle.py` 整模块在主文档链 0 命中 | 论文别宣称，或补正文 |
| GAP-A | `corr.py` 观测相关性全实现，主文档 0 命中 | 同上 |
| GAP-B | `comm.reliability_model="heuristic"` 绕过论文的 FBL 公式，无 k/n | 必须声明或在发布口径禁掉 |
| GAP-B | `detect.soft_mu_scale=8.0`、`soft_error_sigma_scale=3.0` 是**隐藏自由尺度** | 审稿人会问这两个数从哪来 |
| GAP-B | `refine.mode="interp"` + `enable=False` 与正文「无插值系数」正面冲突 | 改默认或改正文 |
| GAP-B | `dd_collision_alpha` 论文只写 α=1 | 声明为固定或加消融 |
| GAP-B | 上界诊断（`coherent_oracle` / `fusion_headroom`）**强制 erasure**，与发布 a=9 不同协议 | 其 P_D 不可与正文数字并列 |

**上界可被违反**：`fusion_headroom.py:33-34` 里 `ksafe` 侧用边际贪心、`k2` 侧用穷举，
贪心先锁最优单链路会错过最优二元组 ⇒ 这个「headroom」**可以为负**。上界诊断不成立时不会报错。

---

## 4. 建议的处置顺序

先修 **会被静默吞掉** 的（它们不崩、不报警，只让结论建立在不同物理上）：

1. `model.py:613` 前加口径枚举硬断言 → 关掉 A1/A2
2. 决定 D1 的走精确还是走 CF，二选一并同步论文 → 关掉理论与实现的最大不一致
3. 修 `tools/audit_coupling_changes.py:255,333`（见下）→ 关掉 H3
4. 初始化 Rinr 前提：补 C4 扫描到杠杆文档，把结论改成几何条件陈述
5. `bandwidth` N/L 加断言（趁它还是 PASS）
6. F1/F2 是**论文声明问题**，改代码成本高（多次建表 / 加 swap move），先改措辞
7. `geom_factor`、`EXPERIMENTAL_METHODS` 死能力等属清理，放到最后一个 PR

### ~~顺手就能修的两行~~ → **已撤回：这是审计脚本自己的假失败**

初版 H3 用**行正则**匹配 `^\s*apply_(preset|overrides)\(`，报出两条"丢弃返回值"：
`tools/audit_coupling_changes.py:255` 与 `:333`。**两条都是假失败**，已用 AST 版判据推翻：

- `:255` —— `apply_overrides(...)` 是作为**实参**传进 `compute_link_tables`，返回值被正常使用；
- `:333` —— `apply_preset(Config(), "nope")` 在 `try` 里做**负向测试**（只关心它抛 `KeyError`）。

修正后的判据：只认**语句级**且值被丢弃的调用，且处在 `try` 块内的降为 WARN。
现在 H3 = PASS。**教训（第 3 次了）：行正则会把"调用出现在表达式上下文"误判成失败，
这种假失败会让人去修一段本来正确的代码——比漏报更有害。**

---

## 5. 审计脚本本身学到的东西（防自欺）

本次我自己的脚本先踩了两个坑，都是「审计脚本会骗人」的实例，已修：

1. **丢弃返回值** → 所有 deep-check 在**默认配置**而不是 V1 口径上跑，差一点把
   「comm_error_model=gaussian_replacement」误读成「预设写错了」。
2. **作用域过宽的正则给出假 PASS** → 文件级搜 `.remove(` 命中了
   `select_c2f_adaptive` 的 polish，把「贪心可逆」判成 PASS。改成切
   `_greedy_lagrangian` 函数体后才是真结论（FAIL）。

⇒ 修一条 FAIL 之前，先确认是**被测物**错了还是**断言**错了；**不要**为了让审计变绿而放宽判据。
