# 功率分配与节点间协调：哪些"功率类"旋钮真的有杠杆

日期：2026-09-16。缘起：对"为什么用功率扫描而不是自适应优化"的追问，以及随之而来的
**"功率增加虽然抬高干扰，但节点间可以协调，协调没有被建模吗？"**

结论先行：**协调已经建模了，但协调真正能作用的那条轴，被一个口径错配藏起来了。**
本文修正我此前"功率无效，所以低 RCS 没救"的说法——那句话只在**总功率**这一个维度
成立，被我不当地推广到了"分配"维度。

> **⚠️ 口径更正（2026-09-16 晚，冻结前复核）**：本文件初版把 `active_set` 标成"论文口径"、
> 把 `orthogonal` 标成"审计基线"，**两个标签方向反了**。实际铁证：`results_target_local_v1/main/config.json`
> 记 `comm.interference_model = "orthogonal"`，`paper-canonical` preset 亦钉 `orthogonal`
> （`config.py:754-775`），正文 `SystemModel.tex:45` 写的是 "Reports do not mutually interfere;
> sensing transmissions remain concurrent"（= 正交串行上报）。`active_set` 属 `isac-consistent`
> preset，是**消融**。下文所有"论文口径/审计口径"的字样已按此更正；**数值本身未变**（表格是实测）。

## 0. 一句话结论

在 400 m / RCS 0.05 m² 下，三条功率类轴相隔一个数量级：

| 轴 | 实测感知 SINR 增益 | 结论 |
|---|---|---|
| 总功率 `radio.P_default` ×2 | **+0.20 dB** | 真无效（原结论成立） |
| 感知/上报分配 `radio.rho` 全幅 0.2→0.95 | **+6.77 dB** | **强杠杆（此前遗漏）** |
| 调度门控（半数发射机静默，`sense_gate_by_active_tx`） | **+2.77 dB** | 有效，但默认关闭 |

**而 ρ 的杠杆大小完全由 `comm.interference_model` 决定**：论文主结果口径 `orthogonal`
下只有 1.48 dB，消融口径 `active_set` 下是 6.77 dB。**低 RCS 优化跑的就是论文口径**，
所以 `optimize_power_joint` 感知不到 ρ 的收益，把 ρ 从默认 0.80 一路压到 **0.432**——
在 `active_set` 消融口径下这是**反向优化**：固定 ρ=0.41 的 P_D 是 0.315，而不优化的
ρ=0.80 是 0.425。论文口径下 ρ=0.41 的 P_D **未实测**（正交口径杠杆只有 1.48 dB，
代价应当小得多），不可直接引用这组 delta。

## 1. 机制：ρ 在共享干扰场里被抵消，但不在回波里

`isac_sim/model.py:580-588` 与 `:626`：

```
P        = P_default                每架 UAV 的总功率
P_sense  = rho     * P_default      感知分量
P_comm   = (1-rho) * P_default      上报分量
I_sense_field = (P_sense + P_comm) @ G_direct = P_default * Σ_k G_direct[k, j]
```

因为 `P_sense + P_comm ≡ P_default`（逐 UAV），**ρ 从干扰场里整体抵消掉了**。
期望回波没有这个抵消：

```
echo[i] = P_sense[i] * target_gain * G_proc * G_hw = rho_i * P_default * target_gain * ...
```

于是

```
SINR_sense ≈ rho_i * P_default * G_echo / (P_default * G_intf * kappa_dc + n0)
           = rho_i * G_echo / (G_intf * kappa_dc + n0 / P_default)
```

两个直接推论：

1. 干扰主导时（当前几何的近远比 ≈ 41 dB，`kappa_dc` 后仍远高于噪声），
   `SINR ∝ rho_i`，**与 `P_default` 无关**。这就是总功率 ×2 只有 +0.20 dB 的原因。
   注意机制不是"干扰涨得比信号快"——对 `P_default` 两者都是线性，而是**同倍放大、比值不变**。
2. `SINR` 对 ρ 是**线性**的，而 ρ 不进入干扰场，所以 ρ 是**免费**的感知增益方向。
   ⚠️ 该"免费"只在 `active_set` / `full_concurrent` 记账下成立；`orthogonal`
   （= 论文主结果口径）下干扰场同样 ∝ ρ，杠杆被抹平到 +1.48 dB。

## 2. 口径分歧：`orthogonal` 把 ρ 的杠杆抹平了

`comm.interference_model` 决定感知观测期间谁在辐射（`model.py:614-622`）：

| 取值 | `P_rad_sense` | 物理含义 | 谁在用 |
|---|---|---|---|
| `full_concurrent` | `P_sense + P_comm` = `P_default` | 所有 UAV 同时满功率辐射 | `Config()` 默认 |
| **`orthogonal`** | `P_sense` = `rho * P_default` | 感知观测期间 payload 静默 | **论文主结果口径 `target-local-v1` / `paper-canonical`** |
| **`active_set`** | `P_sense + P_comm` = `P_default` | 同上（无 mask 时） | **消融口径 `isac-consistent`** |

在 `orthogonal` 下干扰场**正比于 ρ**，于是信号与干扰同步放大，ρ 的杠杆被抵消；
在 `active_set` 下干扰场与 ρ 无关，ρ 直接变成 SINR。**两种口径下 ρ 的敏感性符号相反。**

实测（`results_v1_rho_coordination/`，n=32 = 2 区域 × 2 RCS × 8 trial，解析值已与
`compute_link_tables` 逐位对齐，delta = 0.0 dB）：

| ρ | `active_set` P=1 W | `active_set` P=2 W | `orthogonal` P=1 W | `orthogonal` P=2 W |
|---|---|---|---|---|
| 0.2 | −20.34 | −20.14 | −14.85 | −14.19 |
| 0.4 | −17.33 | −17.13 | −13.99 | −13.58 |
| 0.6 | −15.57 | −15.37 | −13.65 | −13.35 |
| 0.8 | −14.32 | −14.12 | −13.46 | −13.22 |
| 0.95 | −13.57 | −13.37 | −13.37 | −13.16 |
| **ρ 全幅增益** | **+6.77** | +6.77 | **+1.48** | +1.03 |
| **P ×2 增益** | **+0.20** | — | +0.64 | — |

两个值得记的细节：ρ → 1 时 `P_comm → 0`，两种口径的干扰场都收敛到 `P_default`
（或 `P_sense`），所以最右列两行几乎重合；`orthogonal` 在低 ρ 端远好于 `active_set`
（−14.85 vs −20.34），因为 payload 不落在感知观测里。

## 3. 协调的第二条轴：调度门控

`interference.sense_gate_by_active_tx`（默认 `False`）决定**调度器能否影响感知干扰场**。
开关关闭时 `I_sense` 是几何常数，**任何选择器都无法规避干扰源**——这正是"协调"在
感知侧缺位的地方。开启后以活跃发射机占比门控（`results_v1_rho_coordination/`，Block B）：

| 配置 | 感知 SINR |
|---|---|
| `gate=False`, 活跃 100% | −14.32 |
| `gate=True`, 活跃 100% | −14.32 |
| `gate=True`, **活跃 50%** | **−11.55（+2.77 dB）** |

**这是乐观上界**：被静默的 UAV 只从干扰场里移除，其回波贡献仍被保留。真实调度器
同时会失去那架 UAV 的观测，所以 +2.77 dB 不可直接兑现，只说明**干扰侧的可省空间**。

## 4. ρ 的增益能否传导到 P_D

`results_v1_rho_pd_check/`（400 m / RCS 0.05 m²，MC=40，8 workers，审计协议配对）：

| cell | ρ | `interference_model` | pd | worst-target | weak | reports |
|---|---|---|---|---|---|---|
| `orth_rho80` | 0.80 | `orthogonal` | 0.463 | 0.350 | 0.550 | 7.3 |
| `as_rho41` | 0.41 | `active_set` | **0.315** | 0.250 | 0.375 | 3.4 |
| `as_rho80` | 0.80 | `active_set` | 0.425 | 0.375 | 0.475 | 2.9 |
| `as_rho95` | 0.95 | `active_set` | 0.463 | 0.375 | 0.575 | 1.4 |
| `as_rho95_looks64` | 0.95 | `active_set` | **0.578** | **0.500** | 0.625 | 1.7 |
| `as_rho95_maxmin` | 0.95 | `active_set` | 0.527 | 0.450 | 0.600 | **8.0** |

传递是真实的：**ρ 0.41 → 0.95 把 pd 从 0.315 拉到 0.463（+0.148）**。但注意代价：
`reports` 从 3.4 掉到 1.4——ρ 抬高即 `P_comm` 压低，上报能力被拿去换感知。这符合
第 1 节的公式：ρ 不是免费午餐，它把预算从通信挪到感知。

**仍未达到 0.80。** 组合起来更远一些（`looks64` 叠加到 0.578），但 δ 依旧可观。
这说明 ρ 是此前被完全遗漏的一块，却不是全部。

## 5. 为什么自适应优化器输出 ρ = 0.432

`results_v1_lowrcs_power/trials.csv` 里 `maxmin_power`（MC=60 未跑满）：
`rho_mean = 0.432`，`rho_std = 0.335`（逐 UAV 0.30–0.62），`power_seconds = 49.1 s/trial`。

它**不是在偷懒，而是在论文口径下做了理性决策**：

1. 扫描用的 `config()` 就是 `target-local-v1` → `orthogonal`（= 论文主结果口径）；
2. 在 `orthogonal` 下，ρ 从 0.43 提到 0.95 只换来约 +0.6 dB 感知 SINR（第 2 节表）；
3. 而抬高 ρ 会增大感知波形泄漏、压低通信质量——优化器的联合效用目标里通信那一半
   是净损失；
4. 于是它理性地**把 ρ 压到 0.43 保通信**。

这个解在 `orthogonal`（论文口径）下是最优附近；**换到 `active_set` 消融口径下就是次优**：
固定 ρ=0.41 的 pd (0.315) 反而低于不优化的 ρ=0.80 (0.425)。
⇒ 结论不是"优化器错了"，而是**"ρ 这个旋钮的回报取决于口径，不能跨口径搬运结论"**。

## 6. 结论与建议

1. **`rho` 必须重新纳入旋钮清单**，且应作为**首要**算法旋钮之一。此前它被我按"功率类"
   一律排除，是错的。
2. **低 RCS 扫描跑的就是论文口径，结论可直接引用**。`results_v1_lowrcs_cheap/` 与
   `results_v1_lowrcs_power/` 全部是 `orthogonal` = `target-local-v1` = 论文主结果口径，
   其中的 `corr`、`capacitated`、`power2` 等零/负增益结论**在该口径下成立，可以写进论文**。
   需要补跑的是**另一侧**：若要展示 ρ 的协调价值，得在 `active_set` 消融口径下跑对照。
3. **`sense_gate_by_active_tx` 值得作为一个正式消融轴**（配合 `direct_cancellation_db`），
   但必须先给它补一个"静默即失去该照明机观测"的诚实版本，否则 +2.77 dB 会被高估。
4. **优化器应当至少在两种口径下各跑一次**，用差值量化口径敏感度，而不是只在一个口径下
   标定。49 s/trial 的代价只花在一种口径上，就无法回答"ρ 的回报是不是被口径吃掉了"。
5. 仍未闭环：ρ 的缺口补完后 pd ≈ 0.58，离 0.80 还有距离。链路预算仍是主项（`shortfall`
   探针给出的 4.3–7.9 dB 感知 SINR 缺口），ρ 是其中被漏掉的一块，不是全部。

## 7. 复现

```bash
# 解析诊断（秒级）：rho x P_default x interference_model，含与 compute_link_tables 的交叉校验
E:/anaconda/3_11_python/python.exe tools/probe_rho_coordination.py \
    --mc 8 --areas 400 600 --rcs 0.05 0.2 --out results_v1_rho_coordination

# MC 传导验证：新增 rho / active_set cells
E:/anaconda/3_11_python/python.exe tools/audit_v1_lowrcs_sweep.py \
    --mc 40 --workers 8 --areas 400 --rcs 0.05 \
    --cells orth_rho80 as_rho41 as_rho80 as_rho95 as_rho95_looks64 as_rho95_maxmin \
    --out results_v1_rho_pd_check
```

`tools/probe_rho_coordination.py` 自带一致性闸门：解析式与 `compute_link_tables` 在
`rho=0.80, P_default=1.0` 下必须给出相同的中位感知 SINR（两个模型都要对上），
否则脚本输出非零 delta 说明解析实现有误，结论不可用。
