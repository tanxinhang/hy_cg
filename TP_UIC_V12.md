# TP-UIC V1.2 — 修生成契约、修候选门、把消除器接回生产链路

> 承接 `TP_UIC_V11.md`。本版**不动检测器定义、不动消除器数学**，做的是三件被审计逐条点名的事
> （P0-1/P0-2/P0-4/P0-3），外加一个在修 P0-1 时才暴露出来的会计缺陷。
> 生效日期 **2026-09-19**。凡引用 V1/V1.1 的**数字**，一律以本文件为准。

---

## 一、五处改动（三处改结论，两处只改可读性）

| # | 位置 | 改什么 | 影响 |
| --- | --- | --- | --- |
| **P0-1** | `cancellation.py:build_observation` | 真值系数只落在每个 source block 的**中心列**，切向列真值恒零 | **改数字**：回波能量降 **12.39 dB** |
| **P0-2** | `tools/run_tp_uic_v11.py` 新增 D 段 | `perfect_channel + truth + 无 belief 误差` 的 H0 单元实验 | 新增证据，不改既有列 |
| **P0-4** | `cancellation.py:cancellation_arms` | stage-2 联合支撑从「能量门限」改为**声明式保护集** | **改数字**：支撑 9→3、C_IC +5.4 dB |
| **会计缺陷** | `cancellation.py:cancellation_arms` | `eta_survive/eta_protect` 分母守卫 `max(s_energy, EPS)` → 纯比值 | **改数字**：`no_ic` 的 η 从 0.1209 恢复 1.0000 |
| **provenance** | `tools/run_tp_uic_v11.py:148` | `"receiver"` 列曾写入 target id | 只改出处，不改结果 |
| **守卫** | `model.compute_link_tables` | 算法型接收机下**禁用 `reuse_from` 快路径**；未知 `mode` **抛异常** | 默认路径逐位不变；堵掉静默回落 |
| 另附 | `cancellation.py` 新增 `eta_survive_q` | 被测目标**自身**的守恒率（整场口径会随支撑集大小漂移） | 新增列，不改既有列 |

`reuse_from` 那条守卫值得单独说：快路径会把 `reuse_from` 的**感知块逐位复制**过去，
而那张表是按它自己那次调用的接收机模型建的。不堵的话，调用方传了实测比例、拿回的却是
**常数那套数字**，且**什么都不报**——这类静默最贵。同理未知 `mode` 必须抛异常，
否则拼错的值会安安静静地跑出冻结常数（与全仓库"字符串开关不回落"的约定一致）。

### P0-1 的量级：不是笔误，是 12.4 dB

审计的判断正确且可量化。旧代码把单位模随机系数赋给 `A_true` 的**每一列**（含
`2 × tangent_order` 个切向列），于是每个目标实际写入 3 条散射回波。实算同一几何下的回波能量：

```
||A_true @ alpha||^2   中心列真值 1.0313e-11 ｜ 全列真值 1.7878e-10 ｜ 比值 17.34x = +12.39 dB
```

切向列是有限差分方向，范数比中心列大一个量级（`1/step`，`step = 0.05`），所以能量几乎全落在
检测器**没有建模**的那些列上。检测器的 nuisance 块只取中心列（`centre_only=True`），
H0 里于是留有它声明不存在的东西 —— 这正是解析门限失效的机理。

### P0-4：把「假装是检验的常数」改成「声明的规则」

旧候选门用 `||U_q^H r1||^2 / rank(U_q) > threshold` 选 stage-2 支撑。问题不在阈值高低，而在
**这个统计量的水平不由数据决定**：不变式 `P r1 = P y` 说明受保护子空间里的残余**就是** `y` 的
受保护部分，它在**两个假设下都**含直连场的保留量。一个水平由假设决定的检验无法被标定，
它每个 trial 都会选中每个受保护目标，而它打印的门限与虚警率无关。

因此默认口径改为 **`protected_only`**：联合支撑**就是**保护集本身（接收机自己已经声明为
「携带目标证据」的那几个回波），这是一个**声明的设计规则**，不是检验。旧口径保留为
`--candidate-policy statistic`，并且**两个口径的选择都被记录下来**（`gate_targets` /
`supported_targets`），所以「旧门控到底选了什么」是结果文件里的一个数，不是注释里的一句话。

**没有**提供第三种口径（门限 ∩ ¬保护集）：单目标场景下所有目标都被保护，该口径会让 stage-2
支撑变空、`tp_uic_full` 退化成 `tp_uic_stage1`。删掉机制的口径不该叫口径 —— 想删 joint
阶段，现成的消融臂就是 `tp_uic_stage1`。

### 顺带修掉的会计缺陷（同类陷阱的第二次出现）

```python
eta_survive = ||s - f(s)||^2 / max(s_energy, EPS)      # EPS = 1e-12
```

本场景真实回波能量 **1.21e-13 W**，比 `EPS` 还小，于是分母被替换成一个**大 8 倍**的值：
`no_ic` 臂（按定义 `f = 0`）的 η 实测 **0.1209** 而不是 1.0。`_positive_sigma` 的 docstring
里已经记过同一个陷阱（噪声功率 3.83e-14 比 EPS 小 26 倍），它在这里能活下来，只是因为
**旧的错误回波把 `s_energy` 抬到了 EPS 之上**，把它掩盖了。现在改成纯比值 + 零分母约定。

---

## 二、P0-2：oracle 闭合 —— 解析门限现在是 CFAR 门限

三个可归因的解释（`C_res` 形状 / belief 误差 / **信号模型错**）互相纠缠，是 V1.1 无法收口的
原因。D 段把前两个**按构造**拿掉：`perfect_channel` 精确减去直连场，`truth` 字典精确陈述其余
九个目标并精确投影掉，无 belief 误差。剩下的 `T_H0` 只能是 `(1/2)χ²_dof`。

| arm | thr | T_H0 | T_H0 / E[T_H0] | T_H1 | **P_FA** | P_D | C_res rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| perfect channel | 20.67 | 13.59 | 0.971 | 13.78 | **0.053** | 0.067 | 0 |
| no-IC | 20.67 | 13.79 | 0.985 | 13.28 | 0.080 | 0.053 | 14 |
| TP-UIC stage-1 | 20.67 | 13.78 | 0.992 | 13.81 | 0.067 | 0.067 | 28 |

**P_FA = 0.053，`T_H0` 落在其声明均值的 3% 以内**（150 trials）。V1.1 在 truth 配置下读到
`P_FA = 0.40–0.70`。⇒ §5.6 记的「解析门限未标定」**主要是生成模型错**，不是 `prior_variance`
的形状问题：一个被 `+12.4 dB` 切向能量污染、且这些能量恰好落在 nuisance 块声明为空的子空间里的
H0，任何 `C_res` 都救不回来。

`no_ic`（0.080）与 `stage-1`（0.067）残留的小超额与它们的低秩项（rank 14 / 28）一致，属
「建模地板」而非失配：`C_res` 把它们拟合到的系数误差建模成来自**独立参考预算**的噪声，
真实残余里还带着自拟合的那一份。这一项仍然需要标定，量级从「1.5–2×」降到了「1.1–1.6×」。

---

## 三、结果（`results_tp_uic_v12/`，由 `tools/summarise_tpuic_v11.py` 直接生成）

口径：`paper-canonical`，600 m，RCS 0.1 m²，seed 2026，receiver rule = median，`p_fa = 0.05`，
`n_cpi = 1`，`max_protected_targets = 3`，dictionary = belief，`--trials 16 --audit-trials 5
--single-trials 40 --oracle-trials 150`。

### 3.1 A 段：多目标 matched H1/H0（belief 字典）

| arm | C_IC (dB) | eta_surv | **eta_q** | rho_w | T_H1 | thr | P_D | P_FA | AUC | ncp_best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-IC | 0.00 | 1.000 | 1.000 | 0.0080 | 18.00 | 20.67 | 0.312 | 0.250 | 0.547 | 0.039 |
| fixed 40 dB | 40.00 | 1.000 | 1.000 | 0.0109 | 19.98 | 20.67 | 0.438 | 0.438 | 0.523 | 0.045 |
| plain LS | 35.34 | 0.700 | 0.968 | 0.0073 | 18.43 | 20.67 | 0.375 | 0.438 | 0.535 | 0.035 |
| ridge LS | 35.38 | 0.700 | 0.968 | 0.0077 | 16.93 | 20.67 | 0.312 | 0.375 | 0.480 | 0.037 |
| protected LS | 8.02 | 0.779 | 0.986 | 0.0063 | 17.70 | 20.67 | 0.250 | 0.125 | 0.559 | 0.029 |
| TP-UIC stage-1 | 8.02 | 0.781 | 0.986 | 0.0063 | 17.61 | 20.67 | 0.250 | 0.188 | 0.543 | 0.029 |
| **TP-UIC full** | 35.29 | 0.705 | **0.981** | 0.0077 | 16.85 | 20.67 | 0.250 | 0.312 | 0.480 | 0.037 |
| perfect channel | inf | 1.000 | 1.000 | 0.0109 | 19.59 | 20.67 | 0.438 | 0.438 | 0.523 | 0.045 |

`eta_q` 是**被测目标自身**的守恒率，`eta_surv` 是**整场十个回波**的平均。两者必须分开看：
`eta_surv` 会随 stage-2 支撑里放了几个目标而变化（见 3.4 的消融），而审计的 Q1 问的是前者。

### 3.2 C 段：单目标机制验证

| arm | C_IC (dB) | eta_surv | T_H1 | thr | P_D | ncp_best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no-IC | 0.00 | 1.000 | 15.13 | 20.67 | 0.225 | 2.210 |
| plain LS | 35.56 | 0.965 | 15.06 | 20.67 | 0.225 | 2.210 |
| protected LS / stage-1 | 12.15 | 0.983 / 0.984 | 15.50 / 15.48 | 20.67 | 0.125 | 2.054 |
| **TP-UIC full** | 35.55 | 0.977 | 15.63 | 20.67 | 0.275 | **2.230** |
| perfect channel | inf | 1.000 | 15.68 | 20.67 | 0.250 | 2.383 |

### 3.3 B / D 段：几何结论**逐位未变**，但**解释必须改**

| arm | n=0 | n=1 | n=2 | n=3 | n=5 | n=9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no-IC (belief) | 1.000 | 0.381 | 0.187 | 0.065 | 0.044 | 0.017 |
| plain LS (belief) | 1.000 | 0.381 | 0.187 | 0.065 | 0.039 | 0.016 |
| TP-UIC full (belief) | 1.000 | 0.381 | 0.185 | 0.065 | 0.043 | 0.016 |
| perfect channel (belief) | 1.000 | 0.399 | 0.190 | 0.068 | 0.045 | 0.018 |

可辨识性审计：`no_IC` 的 `rho_grid = 1.83e-04`、`rho_manifold = 3.26e-14`、
`off_grid_median = 2.65e-15` —— 与 V1.1 表 **完全相同到有效位**。

这就是审计预判的那件事：`rho / xi_rel / masking` 只依赖模板与 `C_res`，**不依赖 `alpha_true`**，
所以「多目标 DD 子空间严重重叠」这个几何发现不会被生成器修正抹掉；被修正抹掉的是
「严重程度」和「它是否让 GLRT 接近随机猜测」这两句话里的**噪声水平**。答案：几何重叠仍在，
GLRT 的 AUC 仍在 0.48–0.56（in T 表），**依然不可分**（Q2/Q3 结论不变）。

> ⚠️ **措辞修正（2026-09-19，依据 `ANGULAR_IDENTIFIABILITY_PROBE.md`）**
>
> 上表的 ρ 是 **DD-only 接收机抽象下的重叠**，**不是物理本征重叠**。同一场景、同一 ρ 定义，
> 把目标流形换成 `a_DD ⊗ a_arr`（半波长 ULA）后，真实场景 `rho`（24 trials，见
> `ANGULAR_IDENTIFIABILITY_PROBE.md` §4.3；**2026-09-19 用修好的估计器重跑**）：
>
> | 接收阵元数 | 口径 | `rho_median`（典型模板） | `rho_weighted`（功率加权） |
> | --- | ---: | ---: | ---: |
> | 1（DD-only） | — | **4.10e-04** | 0.228 |
> | 4 | 8 cm | **0.742** | 0.730 |
> | 8 | 18 cm | **0.941** | 0.872 |
> | 16 | 38 cm | **0.978** | 0.933 |
>
> ⇒ 「10 个目标压在少数 DD 格上」应读作「在**当前的二维接收机抽象**下压在少数格上」，
> 而不是「物理上不可辨识」。Q2/Q3 的结论不变（它们测的是**当前系统**的表现，
> 不是物理极限），但支撑它们的那句**原因**变了：瓶颈是接收机有没有空间维，不是 DD 分辨率。
>
> ⚠️ 口径提醒：**用 `rho_median` 说这句话，不要用 `rho_weighted`**——M=1 的加权均值已高达
> 0.228，因为少数功率大、DD 上孤立的目标把均值拉起来了；典型模板的中位数才是 4.1e-04。
> （本表上一版 0.470/0.656/0.800 由探针的共轭缺陷产生，已作废，详见探针文档 §4.5。）

### 3.4 消融：`eta_surv` 的漂移由支撑集大小解释

`--candidate-policy statistic`（旧口径），其余完全相同：

| 口径 | 支撑目标数 | C_IC (dB) | eta_surv | T_H1 | AUC | arms 段耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `protected_only`（默认） | 3 | 35.29 | 0.705 | 16.85 | 0.480 | 128 s |
| `statistic`（旧） | 9 | 29.93 | 0.877 | 16.46 | 0.477 | **738 s** |

⇒ 旧口径把 10 个目标里的 9 个放进联合支撑，只换来整场保留率高 17 个百分点，
代价是 **C_IC 掉 5.4 dB**，而检测（AUC / P_D）没有任何改善；选择器侧还贵 5.8 倍。
两口径下 `plain LS` 都是 0.6999/0.700 ⇒ 差异确实来自支撑集，不是消除器变好了。

---

## 四、P0-3：消除器接回生产链路（第一套端到端数字）

### 4.1 接了什么

`model.compute_link_tables(..., residual_fraction_by_receiver=...)` 现在接受**逐接收机**的
残余比例，替换标量 `kappa_dc = 10^(-interference.direct_cancellation_db/10)`：
`residual_direct = frac[j] * I_sense_field[j]`。默认 `None` ⇒ 逐位不变（门禁 2/2b 实测 0 差异）。
数据来源是新的 `cancellation.measure_residual_fraction(cfg, geom, base)`：它在每个接收机上
**真的跑一遍**消除器，回报 `I_res / I_in`（与 `i_res/i_in` 同尺度，所以是 drop-in）。

`cancellation.mode` 的语义同步明确化：`"off"` = 冻结常数；`"predict"` = 解析桥
`predict_cancellation`；`"measure"` **被显式拒绝**（测量需要逐 trial 几何，而
`compute_link_tables` 拿不到它），错误信息直接告诉你改用哪个函数、传哪个参数。

### 4.2 结果（`results_tp_uic_production/`，24 trials，同一几何/选择器/检测器）

| receiver | kappa 中位 | kappa 最小 | kappa 最大 | rinr (dB) | P_D | P_FA | links |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `constant`（冻结 40 dB） | 40.00 | 40.00 | 40.00 | 8.47 | **0.6167** | 0.0499 | 33.5 |
| `measured`（TP-UIC full） | **37.14** | 32.37 | 41.93 | 11.46 | **0.5458** | 0.0508 | 37.5 |

读法（三条，都很关键）：

1. **可执行接收机交出 37.1 dB 中位**（跨 trial 32.4–41.9 dB），比冻结常数低 **~2.9 dB**，
   比 `KAPPA_DERIVATION.md` 算出的场景需求 **52.25 dB 低 ~15 dB**。
2. 端到端 `P_D` 从 **0.6167 掉到 0.5458**（−0.071，P_FA 相同 ≈0.05，rinr +3 dB，选择器
   多挑 4 条链路）。⇒ **低 RCS 缺口没有被 TP-UIC 关上**，而且常数在这里是**偏乐观**的。
3. **解析桥 `mode="predict"` 完全不能用**：同一几何下 median `rinr` 从 8.38 跳到 **2513**
   （+24.8 dB），`gamma_sense` 掉 24 dB。因为它把「保护子空间的直连保留量」当成残余下限
   （维度比 3.1%），而 `tp_uic_full` 的 stage-2 会把这些目标显式建模、把那份直连能量也压掉，
   实测量级 ~1e-4。⇒ 预测既不是上界也不是下界，**只能用于规划**，这与模块 docstring 的
   告警一致，而现在有数字了。

### 4.3 这批数字**不能**用来宣称什么

`constant` 列是**理想化假设**，不是可比算法臂（`TP_UIC_V11.md` §8 已定）。两列应读作
「在两种接收机声明下，整条链值多少」，只有 `measured` 一列背后有可执行接收机。

### 4.4 还没做的接线

`simulate.run_trial` 的 trial 循环**没有**自动测量并下发该数组：它从 `base` 建表，拿不到几何，
而把数组贯穿进去要改 ~10 个 `compute_link_tables` 调用点（含 `reuse_from` 快路径与协调重算）。
今天的工具走的是「同一 trial 内自建两张表 + 同一选择器 + 同一检测器」的等价路径，
差别只在这一条：**生产 MC 的单次运行里还没有自动生效**。这是下一步，不是已完成的声明。

---

## 五、结论相对 V1.1 的净变化

| 问题 | V1.1 | V1.2 | 变化 |
| --- | --- | --- | --- |
| **Q1** TP-UIC 是否减少弱目标损伤 | 整场 η：0.756 → **0.938**（−24.4% → −6.2%） | **逐目标 η_q**：plain LS 0.968 → TP-UIC full **0.981**（−3.2% → −1.9%） | **方向不变、量级大幅缩小**；且必须用 `eta_q` |
| **Q2** 保留能否兑现为 P_D | 不能（AUC 0.51–0.55） | 不能（AUC 0.48–0.56） | **不变** |
| **Q3** 失败来自对消还是可辨识性 | 可辨识性（DD 分辨率） | 同，且 `rho` 表**逐位相同** | **不变，且更硬** |
| CFAR 门限是否可用 | 解析门限 `P_FA` 0.40–0.94，原因不明 | oracle 下 **0.053**；belief 下 0.125–0.438 | **缺口归因完成**，belief 误差是剩下的主因 |
| 单目标是否「可辨识就兑现」 | P_D 0.825（所有臂） | P_D 0.125–0.275（`ncp_best` 2.210 不变） | **旧 0.825 是 +12.4 dB 虚假回波撑起来的** |
| 生产链路用什么接收机 | 标量 κ=40 dB | 实测 37.1 dB ⇒ `P_D` −0.071 | **首次有端到端数字** |

**Q1 的正确表述**（这是本版最重要的一句更正）：

> 在修正后的生成模型下，`600 m / RCS 0.1` 工作点上 plain LS 只吃掉被测弱目标 **3.2%** 的回波，
> TP-UIC full 把损伤压到 **1.9%**（`truth` 字典下 4.1% → 1.1%）。**保护机制真实存在、方向正确、
> 且在完美跟踪器下依然成立**（所以不是 belief 假象）；但它的绝对量级比 V1.1 报的小一个档次——
> 旧口径的「吃掉 24%」里，绝大部分是被错误注入的切向分量造成的，而「压到 6%」是用旧能量门限
> 把 9 个目标都塞进联合支撑换来的。

---

## 六、门禁与复现

```
PY=E:/anaconda/3_11_python/python.exe

# 主表（本文件 3.1–3.3）
$PY tools/run_tp_uic_v11.py --trials 16 --audit-trials 5 --single-trials 40 \
    --receiver-rule median --oracle-trials 150 --out results_tp_uic_v12
# 消融（3.4）
$PY tools/run_tp_uic_v11.py --trials 16 --audit-trials 5 --single-trials 40 \
    --receiver-rule median --skip oracle --candidate-policy statistic \
    --out results_tp_uic_v12_ablation
# truth / 无 belief 误差对照
$PY tools/run_tp_uic_v11.py --trials 10 --audit-trials 4 --single-trials 20 \
    --dictionary truth --no-belief-error --skip oracle --out results_tp_uic_v12_truth
# 生产端到端（4.2）
$PY tools/run_tpuic_production.py --trials 24 --out results_tp_uic_production
# 表
$PY tools/summarise_tpuic_v11.py results_tp_uic_v12
```

门禁（`PY` 必须是 py3.11，见 `ENV_NOTES.md`）：

| 门禁 | 命令 | 结果 |
| --- | --- | --- |
| 1 发布身份 | `tools/check_release_identity.py --check` | CLEAN，96 冻结键 / 0 违规 |
| 2 重构口径逐位 | `tools/parity_check.py --baseline --strict` | CLEAN，0 差异 |
| 2b 发布口径逐位 | `tools/parity_check.py --release-baseline --strict` | CLEAN，1653 格 / 0 差异 |
| 3 契约引用 | `tools/check_contract_refs.py` | CLEAN，32/32（本次抓到并修好 **3 条漂移**） |
| 4 单测 | `pytest tests/ -q` | **228 passed**（较 V1.1 的 215 增 13 条） |

新增单测：`tests/test_cancellation_tp_uic.py`（7 条）：

* `test_generated_echo_puts_its_truth_on_the_centre_columns_only` —— 生成契约
* `test_direct_path_truth_is_centre_only_too` —— 同一规则对直连场也成立
* `test_oracle_cfar_level_is_met_when_the_echo_is_the_generated_one` —— oracle 闭合
* `test_a_tangent_contaminated_echo_breaks_the_oracle_cfar_level` —— **把已修的缺陷留作回归**：
  故意写回被污染的真值，断言 H0 统计量必须被顶起来；若将来这条不再拒绝，说明生成器与检测器
  被一起改动、把错误藏起来了
* `test_echo_survival_is_a_pure_ratio_below_the_eps_guard` —— η 守卫
* `test_joint_support_is_the_protection_set_by_default` —— 声明式支撑 + 旧口径仍可选
* `test_protected_target_ids_agrees_with_the_protection_basis` —— 重构安全性

新增单测：`tests/test_cancellation_production_wiring.py`（6 条，生产接线）：

* `test_the_constant_passed_as_an_array_reproduces_the_frozen_path` —— **把常数当数组传回去必须
  逐位复现冻结路径**（否则"假设接收机 vs 实测接收机"的比较是在测的东西上又叠了一层算术差）
* `test_a_worse_fraction_raises_the_residual_ratio` —— 方向 + 参数确实生效
* `test_reuse_cannot_hand_back_a_table_built_under_another_receiver_model` —— 上面那条守卫
* `test_measure_mode_is_refused_with_a_pointer_to_the_real_interface` —— 拒绝并给正确入口
* `test_a_misspelt_mode_raises_instead_of_using_the_constant` —— 字符串开关不静默回落
* `test_measured_fraction_agrees_with_the_prediction_by_receiver_shape` —— 测量接口的形状/尺度/
  dB 一致性，且实测深度量级正确（`G_hw = 1`）

**CI 缺口（审计 P2 的真实障碍）**：两条逐位门禁**在 CI 里跑不了** —— 基线位于被 `.gitignore`
排除的 `.workbuddy/`（`git check-ignore` 实查），runner 没有可比对象。已把 CI 改成跑三道
**环境无关**门禁（发布身份 / 契约引用 / TP-UIC 接收机契约，python 钉 3.11 + numpy 2.2.6），
并加了一个显式步骤声明逐位门禁**不在 CI 覆盖内**，避免绿色 CI 被误读成「发布数字已钉住」。

**V1 / V1.1 的归档数字不可从当前树复现**：生成器与支撑口径都变了。想只还原支撑维度，
用 `--candidate-policy statistic`（`tools/run_tp_uic_v1.py` 也加了同一开关）；生成维度无法还原。

---

## 七、本版**没有**做（按审计清单）

| 审计项 | 状态 | 说明 |
| --- | --- | --- |
| P0-1 / P0-2 / P0-3(接口+工具) / P0-4 / provenance | **已做** | 见上 |
| P0-3 的「生产 MC 内自动生效」 | **未做** | 见 4.4：需贯穿 ~10 个调用点 |
| P1-1 `N/L/B/CPI` 分辨率扫描 | **未做（但前置条件已查）** | `tools/probe_waveform_mapping.py` 已量出：延迟轴量化挂在 `L`、噪声带宽挂在 `N`，**`N ≠ L` 时模型里有两个带宽**，而默认值 `N=L=64` 恰好掩盖它 ⇒ 不先裁定映射，扫描不可解释。见 `ANGULAR_IDENTIFIABILITY_PROBE.md` §七 |
| P1-2 angle / array manifold | **未做** | 这是把「DD-only 不可辨识」升级成「物理可辨识」的关键一步，需要给目标流形加空间维 |
| P1-3 选择器引入 `rho/xi/ncp` | **未做** | 依赖 P1-2 的判决；`Xi` 目标函数的定义已写在审计里 |
| P1-4 连续 DD / off-grid refinement | **未做** | |
| P2 trial 数 500–1000 + CI | **部分** | CI 已强化（见六）；V1.2 仍是 16/10 trials，`AUC` 的小样本偏差照旧 |
| 论文正文改判（仍 4 km / RCS 50） | **未做** | 与本版无关，属待改判项 |

一句话总结：

> **本版把「数据是怎么生成的」和「stage-2 的支撑是怎么来的」两件事从注释承诺变成了可断言、
> 可回归、可消融的事实，并第一次让消除器真的进入链路表。**
> 代价是三组 V1.1 数字被推翻（CFAR 缺口归因、单目标 P_D、Q1 量级），收益是这些结论现在
> 站在一个 oracle 下 `P_FA = 0.053` 的检测器上；而几何不可辨识的结论**没有动**，
> 它本来就与生成器无关 —— 这正是审计预判的结果。
