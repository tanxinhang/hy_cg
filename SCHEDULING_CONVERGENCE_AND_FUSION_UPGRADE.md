# 调度回合与融合升级：现状、缺口与路线

三个问题：融合策略怎么升级、通信与感知的时序该选哪种、需要多少回合收敛。
本文先把代码事实核清楚，再给路线。所有数字标了出处，可以独立复算。

---

## 1. 融合栈有四个层次，"融合策略"必须指明改哪一层

代码里 `fusion` 这个词出现在四个互不相同的位置，改的层次不同，收益完全不同：

| 层 | 实现位置 | 可选项 | 默认 |
|---|---|---|---|
| ① 局部软统计量 | `detect.soft_stat_model` | `gaussian` / `llr` | `gaussian` |
| ② 跨观测加权 | `fusion.compute_weights` | `equal` / `deflection` / `exact_llr_sum` | `deflection` |
| ③ 判决阈值 | `fusion.calibrated_fused_threshold` | 正态 / Cornish–Fisher / 精确 LLR 混合分位 | CF 修正 |
| ④ 融合点选择 | `fusion.mode` × `fusion.rule` | `tx` / `explicit`；`max_in_rate` / `max_min_rate` / `nearest_target` / `nearest_target_capacitated` / `capacitated_value` | `tx` / `max_in_rate` |
| ⑤ 报告计划精修 | `fusion_polish.py`、`joint_polish.py`、`bundle_master.py` | 最小报告数 / 邻域搜索 / 列生成+整数主问题 | **未启用** |

论文当前写的是第 ④ 层：`eq:nearest_target_fusion` 用"离预测位置最近的 UAV"，并且
正文自己写明 *"a low-complexity placement heuristic, not a jointly optimal fusion
solution"*（`ProposedMethod.tex:13`）。所以"融合策略有待升级"在论文里是被承认的，
问题是要选对升级点。

### 三个真实缺口（不是"还可以更好"，是逻辑上不自洽）

**(a) 相关模型与精确统计量互斥。**
`fusion.py:324-326`：

```python
if statistic_mode == "exact_llr_sum":
    if cfg.corr.enable and len(links) > 1:
        raise ValueError("exact LLR sum currently requires independent observations")
```

即"最准的局部统计量"（LLR 精确和）和"最准的相关模型"（Σ⁻¹ 加权）**不能同时开**。
多 UAV 双站网络恰恰必然相关（`corr.py` 开头列了五个相关源：共发射机、共接收机、
共目标起伏、DD 邻域泄漏、同一散射体）。这不是参数没调好，是两个特性写在了互斥分支里。

**(b) 相关一旦开启，阈值退回旧规则。**
`fusion.py:350-356`：

```python
if (cfg.detect.soft_stat_model.lower() != "llr"
        or cfg.detect.comm_error_model != "erasure"
        or (cfg.corr.enable and len(links) > 1)):
    return float(fallback)   # Cornish-Fisher
```

注释理由是"其联合高阶分布未被当前相关抽象识别"。但代价是：**加权用了 Σ⁻¹（二阶），
阈值却用独立假设下的 CF 修正（三阶）** —— 同一份统计量在权重和阈值两处用了不一致的
分布假设。既然 `wᵀs` 的 H0 方差已经是 `wᵀΣw`，可以做二次型的精确/近似分布，
而不是退回独立假设。

**(c) H0 用混合采样、H1 用矩匹配，两者不对称。**
`calibrated_fused_threshold` 用确定性 MC 积分精确刻画 H0（含包错原子），
但 H1 一侧 `predicted_pd_for_links` 仍走 `mom.m1/v1` 矩匹配。检测概率由两侧共同决定，
一侧精确一侧近似，误差方向不可控。

### 实验证据：融合层的两个开关现在是负收益或零收益

`results_v1_lowrcs_cheap/trials.csv`（400 m / RCS 0.05，MC=100，审计口径）：

| cell | 改动 | P_D | 相对 base |
|---|---|---|---|
| base | — | 0.491 | — |
| corr | `corr.enable=True` | 0.481 | **−0.010** |
| capacitated | `fusion.rule=nearest_target_capacitated` | 0.491 | 0.000 |
| budget | 报告/观测上限放宽 | 0.509 | +0.018 |

**注意这个口径警告**：这批数据跑在 `interference_model='orthogonal'`，而 `joint_polish`
/`fusion_polish`/`power_tables` 三个精修模块的入口都硬断言 `orthogonal`
（`joint_polish.py:32`、`fusion_polish.py:51`、`power_tables.py:23`）。
换句话说，**第 ⑤ 层（真正的融合优化）在当前扫描口径下能跑，但论文口径下会直接抛错**。
`corr` 的负收益也必须在 `active_set` 下重测才能采信——这个结论与 ρ 的教训同源：
口径不同，杠杆的符号都可能翻转。

---

## 2. 时序：论文的自洽模型是"感知并发 + 报告正交"，`active_set` 是消融

`comm.interference_model` 表面是干扰记账，实际是**调度时序假设**：

| 取值 | 时序含义 | 代码注释原文 |
|---|---|---|
| `orthogonal` | 报告载荷时频正交（串行），感知波形持续并发 | "self-consistent companion of `mac_model='serial'` used by the conference release"（`config.py:205-208`） |
| `active_set` | 仅当选报告机在评估期辐射 | "deliberately labelled **an ablation** rather than a self-consistent endogenous-interference optimizer"（`config.py:200-204`） |
| `full_concurrent` | 所有 UAV 全程满功率辐射 | "Conservative worst case"（`config.py:198-199`） |

论文 `SystemModel.tex:45` 写的是 *"Reports do not mutually interfere; sensing
transmissions remain concurrent."* —— 这正是 `orthogonal`。而主结果
（`tools/rerun_paper.py`）钉的是 `active_set`。

所以回答"先通信还是先感知"这个问题，代码给出的答案是：

- **不是二选一，也不完全是"一起做"**；当前架构是**感知在时间上铺满整个 CPI，
  报告在剩余维度上正交复用**。两者不是先后关系，是两个不同的资源维度。
- `MAC` 侧还有第三个轴：`comm.mac_model ∈ {serial, parallel, slot}`，它决定报告之间
  如何复用时间（串行 / 并行 / 冲突图着色分槽）。`slot` 是唯一"MAC 与干扰模型描述
  同一件事"的自洽选项（`config.py:239-241`），默认 `serial`。

### 真正缺失的变量：时间划分

系统里有**功率**划分 `radio.rho`（感知功率占比，默认 0.80），有**报告时间**复用
（`mac_model`），但**没有"感知 vs 通信的时间比例"这个变量**。
`radio.isac_power_model ∈ {sensing_only, joint_waveform, reliable_comm_assisted}`
三个取值全是功率域假设，没有时间域对应物。

这意味着"先感知后通信 / 先通信后感知"在当前模型里**无法表达**，只能通过
`interference_model` 隐式近似。要让"动态调整"成为可优化对象，需要补一个
时间划分变量（例如感知占 CPI 的比例 τ），它与 ρ 正交：

- ρ 调的是**同一时刻两种功能的功率分配**（`P_sense + P_comm ≡ P_default`，
  所以 ρ 不改变干扰场，见 `RHO_COORDINATION_LEVER.md`）
- τ 调的是**两种功能各占多少时间**（改变的是能量与 CPI 长度，而不是瞬时功率）

---

## 3. 回合与收敛：当前是单圈闭环，"回合"不在系统层

这是最关键的一条。`belief.py` 的模块文档写的是

> Truth vs belief: the **predict -> schedule -> sense -> update loop**.

但实际实现是：

- `simulate.py:962` 只调用了一次 `BeliefState.from_truth(cfg, geom, rng)`；
- 全文**没有任何第二次调度**（grep `rounds` / `replan` / `second` 在 `simulate.py` 无命中）；
- `belief.py:43` 的原文是 *"``Q_cv`` are provided so a downstream tracker update
  **can** close the loop"* —— 预留了能力，闭环**没有接上**。

所以：**系统当前跑的是 1 个回合（单次调度 → 单次判决），不存在"收敛"问题，
因为没有反馈路径。**

### 内部的"轮"是贪心记账，不是闭环迭代

系统里能数出四组"轮"，语义完全不同，混在一句话里说会出错：

| 计数 | 含义 | 停止判据 | 实测（见 §4） |
|---|---|---|---|
| `coarse_rounds` | 粗选阶段每轮 commit **1 条**观测 | 边际增益 ≤ 0 或达 `max_total_links` | ~51 |
| `fine_rounds` | 细选重放后的最终链路数 | 同上 | ~47–50 |
| `bid_rounds` | 分布式协商轮数（仅 `distributed_bids=True`） | 同上 | 见 §4 |
| `bundle_pricing_iterations` | 列生成轮数 | 无新列 或 达 `bundle_cg_max_iterations=8` | ≤ 16（2 stage × 8） |
| `power_joint rounds` | 外层功率–拓扑交替 epoch | **固定 2，无收敛判据** | 2（写死） |

两个观察：

1. **`bid_rounds` 从未被启用过。** `distributed_bids=True` 只在
   `proposed_c2f_adaptive_pd_distributed` 分支成立，而这个方法名在
   `experiments.py` 里**一次都没有被引用**。所有已发布结果的 `bid_rounds` 恒为 0。
   系统的多轮协商能力已经实现，但从未参与过任何实验。
2. **只有外层交替缺收敛判据。** 其余四个都有自然停止条件（固定点 / 无新列 /
   边际非正），唯独 `optimize_power_joint` 的 `rounds=2` 是硬编码的
   （`tools/audit_v1_lowrcs_sweep.py:76` `POWER_ROUNDS = 2`）。
   如果实测发现第 2 轮就没改进，这个 "2" 应该换成固定点判据并如实报告。

---

## 4. 实测回合数

`tools/probe_convergence.py`（本文件配套，自带轨迹钩子）。
配置：审计基线 `tools.audit_v1_exact_budget.config`，seed 10919，报告上限 8，
`gaussian_replacement`，RCS 0.05。三种时序在同一份几何上配对求解。

### 4.1 内层 greedy：约 51 轮 commit 到自然停止

48 个 cell（3 时序 × 2 区域 × 8 trial），trial 内配对。轮数与干扰模型**完全无关**
（同一几何的候选集与目标数决定轮数，不由时序决定），所以下表按区域合并：

| 区域 | `coarse_rounds` | `fine_rounds` | 90% 增益 | 99% 增益 | 99.9% 增益 |
|---|---|---|---|---|---|
| 400 m | 51.6 | 48.6–49.8 | 18.6 | 39.6 | 47.4 |
| 600 m | 49.6 | 45.9–46.4 | 16.9 | 37.6 | 45.8 |

读法：**每轮 commit 1 条观测**，所以"轮数 = 链路数"。停止判据是边际增益 ≤ 0
（自然停止，未触及 `max_total_links=60`）。前 18 轮拿到 90% 的增益，
之后 20 轮换 9 个百分点的收敛尾巴，最后 8 轮只值 0.1%。

**这对工程的含义**：如果只关心 90% 的最优性，18 轮就够了；要 99% 需要 38 轮。
当前实现是跑满到自然停止（51 轮），属于"不计代价求最优"。

### 4.2 分布式协商：轮数翻倍，消息减少约 38 倍

`distributed_bids=True` 让每个目标每轮先出一价、再统一 commit：

| 区域 | `bid_rounds` | 集中式消息 | 分布式消息 | 比值 | P_D（集中/分布） |
|---|---|---|---|---|---|
| 400 m | 101.2–102.4 | 29 695 / 30 386 | 795–801 | 37–38× | 0.425 / 0.425 |
| 600 m | 96.5–97.0 | 31 095 / 31 769 | 770–772 | 40–41× | 0.362 / 0.362 |

**协商轮数正好是 commit 轮数的两倍**（每轮每个目标一个 bid）。但消息量降到 1/38，
**且 P_D 逐位相同**（0.425 vs 0.425，0.362 vs 0.362）。

注：这里的"消息"口径是代码自带的 `coordination_messages`——集中式记账为打分次数
（`score_evaluations`），分布式记账为实际出价消息数（`bid_messages`）。两者不是同一
种消息，所以这个倍数应读作"两种记账口径的比值"，而不是严格意义上的协议开销比。
但结论方向是稳的：**同样的解可以用远少的外部通信换到**，代价是协商轮数翻倍。

而这个能力**至今没被任何实验调用过**（§3），所以论文里没有任何一处报过它。

### 4.3 时序：正交（时分）比并发高约 0.038 的 P_D

同一份几何、同样 8 个 trial、同样 RCS 0.05：

| 时序 | 400 m | 600 m | 相对正交 |
|---|---|---|---|
| `orthogonal`（感知并发 + 报告正交串行） | **0.463** | **0.400** | — |
| `active_set`（仅当选报告机在评估期辐射） | 0.425 | 0.362 | **−0.038** |
| `full_concurrent`（所有 UAV 全程辐射） | 0.425 | 0.362 | **−0.038** |

这就是"先通信 / 先感知 / 一起做"的量化答案：**把两种功能在时间上分开
（正交报告）比同时辐射高约 8.9%（400 m）到 10.5%（600 m）的 P_D**。
`active_set` 与 `full_concurrent` 在这一档场景下无法区分。

而主结果（`tools/rerun_paper.py`）钉的是 `active_set`——即论文目前报告的是
**并发口径下的较低性能**，且该口径在配置里被标注为消融。这是一个方向性的选择，
不是数值错误，但必须在正文里说清。

（`convergence.csv` 的 `*_mean_target_pd` 是单 trial 的目标级平均，不是
worst-target；worst 需要先逐目标跨 trial 平均再取最小，本探针未产出。）

### 4.4 外层功率–拓扑交替：2 个 epoch 确实够

`POWER_ROUNDS = 2` 是硬编码的，所以单独验证它是否足够
（`tools/probe_power_rounds.py`，400 m / RCS 0.05 / orthogonal）：

| epochs | 目标函数 | 相对 1 轮的边际 | accept 次数 | 建表次数 | 耗时 |
|---|---|---|---|---|---|
| 1 | −2.61551 | — | 14 | 98 | 17.2 s |
| 2 | −2.50831 | **+0.10720** | 16 | 188 | 23.5 s |
| 3 | −2.50831 | **0.00000** | 16 | 278 | 29.6 s |
| 4 | −2.50831 | 0.00000 | 16 | 368 | 35.8 s |
| 6 | −2.50831 | 0.00000 | 16 | 548 | 47.9 s |
| 8 | −2.50831 | 0.00000 | 16 | 728 | 61.1 s |

**收敛在第 2 轮，第 3 轮起零增益。** 所以 `rounds=2` 是正确取值，不必改。

这里有一个方法论教训值得记下：单看 `objective_trace` 的**最后一个增量大于平均值**
（0.0294 > 0.0263）会得出"远未收敛"的结论——我一开始就是这么判断的，**是错的**。
原因很简单：收敛后 trace 不再增长，那最后一个增量永远是"收敛前最后一次接受"，
它当然可能大于全程平均。**判断收敛只能看"再多给轮数是否还改进"，
不能看 trace 内部的增量分布。**

---

## 5. 路线建议

**融合升级（按性价比排序）**

1. **先修不自洽，再谈升级**：解除 (a) 的互斥——让 Σ⁻¹ 加权与 LLR 精确统计量共存;
   相关开启时把阈值从 CF 升级为 `wᵀΣw` 的二次型分布；(c) 的 H1 也换成同一混合模型。
   这三项是"把现有部件装对"，不引入新物理假设，属于审稿人会直接问的点。
2. **融合点选择从启发式换成优化**：`bundle_master.py` 的列生成 + 整数主问题
   已经实现了，但只在 V1.2 变体里，主结果用的还是 `nearest_target`。
   把 `fusion.rule` 的默认从 `nearest_target` 升级为优化解，是把
   `ProposedMethod.tex:13` 那句自我限定删掉的唯一途径。
3. **口径统一后再评估 `corr`**：它现在 −0.010，但测在 `orthogonal`。
   与 ρ 同源的教训——先重测，再结论。

**时序**

4. **不要把"先通信 / 先感知"当成二选一**。当前架构是感知铺满、报告正交复用；
   补一个**时间划分变量 τ**（与 ρ 正交）后，"动态调整"才是可优化的对象，
   在此之前"动态调度"没有可调的自由度。
5. **`active_set` 的口径身份要写清**：配置注释说它是消融，论文正文却当主口径。
   要么把主结果切到 `orthogonal`（自洽），要么在正文里明说 `active_set` 的
   保守性来源。这属于必须消除的表述矛盾。

**回合**

6. **闭环要么接上，要么别在文档里叫 loop**。当前 `belief.py` 的
   "predict→schedule→sense→update" 只有前三步。若要做动态调整，
   最小实现是：判决后更新 belief → 用新 belief 重跑一次选择器 → 报告
   P_D 是否提升。这就是"回合"的真正定义，也才有"几回合收敛"可答。
7. **外层交替的 `rounds=2` 经实测是正确的**（§4.4 第 3 轮零增益），无需改数值。
   但它仍是硬编码：`rounds=0` 或场景变化后没有人会发现它不够。建议换成固定点
   判据（本轮无 accept 即停），数值上仍等价于 2，但会随场景自适应。

---

## 6. 顺带发现的独立问题：基线回归门禁不通过

跑 `tools/parity_check.py --baseline` 时（原本是为了确认本次改动没有破坏
bit-exact），发现**门禁本来就是红的**：

```
legacy baseline  fields=297   max|diff|=1.65333e+18 MISMATCH
    worst field: ('T_mean_ms', 224.0769233912711, 1.6533333333333332e+18)
```

定位到唯一失控的字段是 `raw_sense_sinr` 的 `T_mean_ms`：
基线 224.08 ms，当前 **6.4e17 ms**（mc=2）/ 1.65e18（mc=12），
而同一份输出里 `selected_rate_mean_mbps` 仍是正常的 0.364（基线 0.474）、
`B_mean_bits` 22080（基线 22027）。

即：**速率正常、比特数正常，但延迟炸了 15 个数量级** ——
`T = Σ B_l/R_l` 中至少有一条被选中链路的 `R_l` 塌到接近 0
（`raw_sense_sinr` 会选到 SINR ≤ 0、`log2(1+SINR) ≤ 0` 的腿），
其余 8 个方法的延迟偏差都在 1 倍以内，属于 mc 规模差异导致的正常统计波动。

**归属已排除本次改动**（这是重点，因为项目铁律是 legacy preset 必须逐位复现基线）：

- 本次只改了 `select_c2f_adaptive`（15 行，纯新增诊断分支，默认 `trajectory=None`）；
- `raw_sense_sinr` 走 `select_topk_baseline`（`selection.py:930+`），**不经过**
  `select_c2f_adaptive`；
- `git stash` 掉本次改动后重跑，`raw_sense_sinr` 的 T_mean_ms 仍是 **6.4e17**；
- 同一条件下"本次改动版"与"HEAD 版"的 `T_mean_ms` **逐位相同**。

所以这是一个**先于本次会话存在的、量纲级的回归漂移**，且正好落在
`verify_baseline_freeze` 那条长期红色的门禁上。建议单独立项归因
（`baseline-drift-attribution` 的流程正好适用），不要在本文档里顺手改。

证据文件：`parity_after_selection_hook.log`。

---

## 复算入口

```bash
# 内层轮数 / 三时序对比 / 外层 trace（约 6 分钟，mc=8，两区域）
E:/anaconda/3_11_python/python.exe tools/probe_convergence.py \
    --mc 8 --areas 400 600 --rcs 0.05 \
    --models orthogonal active_set full_concurrent \
    --power --out results_v1_convergence

# 单独验证外层交替需要几个 epoch（约 3 分钟）
E:/anaconda/3_11_python/python.exe tools/probe_power_rounds.py \
    --area 400 --rcs 0.05 --rounds 1 2 3 4 6 8
```
产出 `convergence.csv`（每 trial 每时序的轮数 / P_D / 开销）、
`trajectory.csv`（逐轮 utility，用于算 90/99/99.9% 收敛点）、
`power_trace.csv`（外层交替的 objective_trace 与逐轮增量）。

`select_c2f_adaptive` 新增了可选的 `trajectory` 参数（默认 `None`）。它只在传入列表时
记录每轮 commit 后的 utility，不参与任何决策，默认路径逐位不变（已用 `git stash`
对照验证：改动前后 `T_mean_ms` 全部逐位相同）。
