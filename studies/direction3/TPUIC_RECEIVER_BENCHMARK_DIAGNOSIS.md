# TP-UIC receiver-only benchmark：接口修复 + 性能诊断

日期：2026-09-22　|　脚本：`tools/run_tpuic_receiver_benchmark.py`
配套：分片驱动 `tools/run_tpuic_receiver_shards.py`、分析器 `tools/analyze_tpuic_receiver.py`
数据：`studies/direction3/data/tpuic_receiver_smoke/`

> 状态：**只做诊断，未出性能结论。** 正式轮未跑（规模见 §4，用户裁决暂缓）。
> 烟测 n=16，只能判定"机制接线正确"，不能判定"创新成立"。

---

## 1. 裁决行

| 项 | 判定 | 依据 |
|---|---|---|
| 脚本能否跑通 | ✅ 已跑通（修 2 个 bug 后） | smoke 448 条记录、EXIT=0 |
| 文档里的"正式"网格 | ❌ **不可行** | 57.6 万条 × 8 s ≈ 53 天 |
| direct-INR 压力轴 | ✅ 有效 | −3.018 → +16.982 dB（精确 +20.000） |
| `perfect_channel` oracle 上界 | ✅ 有效 | 门限 9.36 且不随 INR 变化 |
| 冲突指标 ξ 是否退化 | ⚠️ **不退化，但需富集采样** | median 0.022 / p90 0.461 / max 0.9997 |
| 烟测 P_D / P_FA | ❌ **不可用** | 每键仅 16 个 H0 标定样本 |
| 主终点该用哪个 | ⚠️ **必须换 AUC** | 统计量非枢纽，见 §5.3 |

---

## 2. 修掉的两个接口 bug

### 2.1 `only=` 逐臂剪枝必然 KeyError（根因级）

原写法对每个臂单独调用：

```python
results = cx.cancellation_arms(cfg, obs, weak_target=..., only=arm)
```

`arms_basic.py:96` 的装配条件是

```python
need_stage1 = wanted is None or ctx.candidate_policy == "statistic"
```

默认 `candidate_policy="protected_only"`，所以传 `only="tp_uic_stage1"` 时
`need_stage1=False`，该臂**根本不装配**，直接 `KeyError`。
`perfect_channel` 也只在 `only is None` 时才装配。

**修法**：每个观测一次性 `cancellation_arms(cfg, obs, weak_target=...)`（`only=None`），
7 个臂复用同一个 results。副作用是顺带把装配次数从 2×7 降到 2×1。
`only=None` 同时是被回归测试钉住的**数值参考路径**，所以这不但不冒险，反而更安全。

### 2.2 修 2.1 时漏行 → NameError

删掉了 `result = results[arm]`。已在 `_run_arm` 里补回并加了显式
`if arm not in results: raise KeyError(...)`（带 available 列表，便于下次定位）。

---

## 3. 运行方式（环境坑，下次直接照抄）

```bash
cd /d/Desktop/conference && PYTHONPATH='D:/Desktop/conference' \
  "E:/anaconda/3_11_python/python.exe" -u tools/run_tpuic_receiver_benchmark.py \
  --out <dir> ...
```

* **必须显式给 `PYTHONPATH`**：直接 `python tools/xxx.py` 时 `sys.path[0]` 是
  `tools/` 而不是 cwd，`import isac_sim` 会失败。且必须写 Windows 风格
  `D:/Desktop/conference`，Git Bash 的 `/d/...` 不会转换给 Windows Python。
* **bash coreutils 时有时无**（`ls`/`head`/`tail`/`dirname` 会 not found）→ 用
  Python / Glob 替代，输出重定向到文件再 Read。
* **前台 120 s 会被 SIGTERM** → 一律 `run_in_background`。

---

## 4. 性能账本（本次核心结论）

### 4.1 单条记录成本

一次 `_record_one` = 场景 × boost × 接收机 × 目标 × 实现，实测 **8.06 s**：

| 阶段 | 单次 | 次数 | 小计 | 占比 |
|---|---:|---:|---:|---:|
| `build_observation_pair` | 0.24 s | ×1 | 0.24 s | 3% |
| `cancellation_arms` | ~2.0 s | ×2（H1/H0 各一次） | 3.95 s | **51%** |
| `residual_model` + `target_conditioned_glrt` | 0.25 s | ×14（7 臂 × 2 观测） | 3.56 s | **46%** |

贵的根因是**维度**：`y` 为 16384 维（`N·L·m_rx = 64×64×4`），每个臂都在这个尺度上做
SVD / 协方差白化。

### 4.2 文档"正式"网格的规模

```
(50 cal + 50 test) × 64 实现 × 6 接收机 × 3 目标 × 5 boost
  = 576,000 条 × 8 s ≈ 4.6e6 s ≈ 53 天（串行）
                ≈ 3 天（18 路并行）
```

### 4.3 缩放律（分片后）

每个分片 = 1 接收机 × 1 目标，因此

$$\text{墙钟} \approx (\text{cal}+\text{test 场景数}) \times \text{boost 数} \times \text{实现数} \times 7.5\ \text{s}$$

**与采样多少 (接收机,目标) 对无关**（16 核可容 18 路）。所以 (接收机,目标) 是唯一
"免费"的维度——而它恰恰是覆盖高 ξ 尾部所必需的，不要砍。

### 4.4 一个真实的结构性浪费（未动）

`only=None` 实际装配 **10 个臂**，我们只报 7 个：

```
要报的 7 个：no_ic plain_ls ridge_ls protected_ls tp_uic_stage1 tp_uic_full perfect_channel
白算的 3 个：soft_tpuic  targeted_tpuic_stage1  targeted_tpuic_full
```

约 15% 总时长浪费。**没有去剪**：`only=` 剪枝会连 `tp_uic_stage1` 一起剪掉（§2.1），
且剪枝的逐位一致性不想在这种规模的运行里赌。

### 4.5 候选网格（16 核 / 18 路）

| 方案 | 网格 | 每键 cal/test 样本 | 墙钟 |
|---|---|---:|---:|
| A 快速体检 | 30 场景 × 2 boost × 3 实现 | 60 / 60 | ~23 min |
| B 可辩护 pilot | 40 场景 × 2 boost × 5 实现 | 100 / 100 | ~50 min |
| C 三档扫描 | 40 场景 × 3 boost × 5 实现 | 100 / 100 | ~75 min |

cal 样本不应低于 ~100：`P_FA=0.05` 在 60 个 H0 上只有 3 个超限点，门限本身不稳。

---

## 5. 烟测暴露的五个实质问题

### 5.1 κ 对 boost 不变 —— 这是对的，不是 bug

`plain_ls` κ = 12.93（boost 0）→ 12.99（boost 20）；`protected_ls` 8.13 → 8.14。
κ 是**比值** `i_in / i_res`，干扰放大 100 倍时分子分母同倍放大，比值不变。
⇒ 推论：**残余干扰的绝对电平随 INR 线性上升**，所以高 boost 才会压垮检测。

### 5.2 `perfect_channel` 的 κ 是 nan —— oracle 记账特性，不是 bug

`_depth_db` 在 `i_in <= 0` 时返回 nan。oracle 臂不"估计"直达路径，记账上 `i_in=0`。
可忽略，或在报告里标注为"不适用"。

### 5.3 ⚠️ 统计量非枢纽 —— 主终点必须换成 AUC

| 臂 | 门限 @INR −3 dB | 门限 @INR +17 dB | 倍数 |
|---|---:|---:|---:|
| `no_ic` | 126.4 | 12877.1 | ×102 |
| `plain_ls` | 126.9 | 12877.6 | ×102 |
| `tp_uic_full` | 129.2 | 13175.8 | ×102 |
| `protected_ls` | 140.5 | 13565.1 | ×97 |
| **`perfect_channel`** | **9.36** | **9.36** | **×1.00** |

`perfect_channel` 门限恒定 9.36（≈ χ² 分位）且 INR 免疫，说明**检测器本身是正确归一化的**；
其余臂门限随 INR 涨 100 倍，是**直达泄漏**进入统计量的结果，不是归一化 bug。

⇒ **后果：固定门限下的 P_D 不能跨 boost 比较。** 主终点必须是**门限无关的 AUC**，
P_D 只作为"在各自经验标定门限下"的次级终点报告。

### 5.4 ξ 不退化，但中位数很低 —— 必须富集采样

144 个 (场景, 接收机, 目标) 单元上：

```
median 0.0216   mean 0.130   p90 0.461   max 0.9997
frac(ξ>0.1) = 0.250      frac(ξ>0.3) = 0.146
```

烟测恰好抽到 ξ=0.003 的单元，所以看起来像"退化"。真实情况是**重尾**：
多数单元几乎无冲突，少数单元接近完全重合（ξ→1）。
⇒ `P_D vs ξ` 曲线画得出来，但必须采足够多单元才能填满高 ξ 区间。

### 5.5 η_q 在低 ξ 处饱和，在高 ξ 处会动 —— 指标可用但有条件

* 烟测（ξ=0.003）：全部臂 η_q = 0.9998 ~ 1.0000，`identifiable_fraction ≈ 0.985`（饱和，无分辨力）。
* 驱动自检（ξ=0.977）：`plain_ls` η_q = **0.922**，`tp_uic_full` 0.983，`protected_ls` 0.999。

⇒ η_q 只在**高冲突单元**上才分辨得出差异。低 ξ 单元上"η=1.000"不能解读为"保护无损"，
只能解读为"没有可损的东西"。

---

## 6. 烟测数字（n=16，**不可用于结论**，仅存档）

AUC（门限无关，是这组里唯一勉强可信的量；n0=n1=16 ⇒ se ≈ 0.11）：

| 臂 | boost 0（INR −3 dB） | boost 20（INR +17 dB） |
|---|---:|---:|
| `no_ic` | 0.660 | 0.562 |
| `plain_ls` | 0.660 | 0.562 |
| `ridge_ls` | 0.660 | 0.562 |
| `tp_uic_full` | 0.660 | 0.559 |
| `protected_ls` | 0.637 | 0.512 |
| `tp_uic_stage1` | 0.637 | 0.512 |
| `perfect_channel`（oracle） | 0.652 | 0.652 |

可读出的**两个方向性提示**（不是结论）：

1. 直达干扰确实伤检测：INR 从 −3 提到 +17 dB，AUC 掉约 0.10。
2. **13 dB 对消深度没有换回 AUC**：`plain_ls`(0.562) ≈ `no_ic`(0.562)，而 oracle 是 0.652。
   也就是说可争取的空间有 0.09，当前对消只兑现了 0。

这与 `README.md` §12.3 一致（主工作点上 TP-UIC 不是活动瓶颈：0.737859 / 0.738018 /
0.738023 / 0.737993）。**本基准存在的意义就是把 INR 拉到足够高，去定位 TP-UIC 的生效区间。**

P_D / P_FA 全为 0：每键只有 16 个 H0 标定样本，门限 = 最大值，故无一超限。
**这是样本量问题，不是算法问题。**

---

## 7. 下一步（待裁决后执行）

1. 按 §4.5 选网格重跑（推荐 B）。
2. 主终点固定为 **AUC**；P_D / P_FA 作为次级终点，并显式报告每键标定样本数。
3. 结果按 ξ 分箱出表（`tools/analyze_tpuic_receiver.py` 的 D 节），回答
   Q1（Plain LS 是否在高冲突下误删目标）/ Q2（TP-UIC 是否以更少 κ 损失换更高 η_q）。
4. 若高 ξ 单元仍然不足，考虑**构造性场景**（让某条 UAV–UAV 直达路径与目标回波在
   DD 上重合），而不是继续随机采样。

---

## 8. 无用信息审计（数据驱动，448 条烟测记录 + 源码 grep）

### 8.1 七个臂实际只有四种行为

按 (split, scene, realisation, boost) 配对的统计量相对差：

| 配对 | 中位相对差 | 判定 |
|---|---:|---|
| `plain_ls` vs `ridge_ls` | 9.8e-07 | 数值同一 |
| `protected_ls` vs `tp_uic_stage1` | 4.5e-07 | 数值同一 |
| `no_ic` vs `plain_ls` | 7.2e-05 | 近同一（偶发 1.4e-2） |
| `tp_uic_full` vs 其余 | ≥8.8e-02 | 独立 |
| `perfect_channel` vs 其余 | ≥3.3e-01 | 独立 |

⇒ 有效臂 = `{no_ic, plain_ls, protected_ls, tp_uic_full, perfect_channel}`，
建议 `--arms` 用这 5 个。可省 4/14 次 GLRT ≈ **13% 总时长**。

⚠️ `no_ic ≈ plain_ls` 本身是**发现不是故障**：13 dB 对消深度只让统计量动了 7e-5。
正确白化的检测器本就该如此——落在 nuisance 子空间里的干扰被白化掉了。

**白算的臂**：`soft_tpuic` / `targeted_tpuic_stage1` / `targeted_tpuic_full`
（`only=None` 必装，我们从不报告）。

### 8.2 恒定 / 饱和列（记录纯占空间）

| 列 | 观测 | 判定 |
|---|---|---|
| `dof_real` | 恒为 10 | 常量 |
| `n_coefficients` | 恒为 5（= X 的列数） | 常量 |
| `i_res_estimate` | std 1.1e-13，max 5e-13 | 恒为 0 |
| `identifiable_fraction` | ∈[0.966, 0.999]，std 0.008 | 饱和，无分辨力 |

### 8.3 冗余指标

`i_res_structural ≡ i_res`（34 个唯一值、min/max/std 全部相同）
⇒ `kappa_structural_db` 与 `kappa_accounted_db` 仅差约 0.002 dB，
在 `residual_accounting=measured`（默认）下 structural 分支不活跃，**二者留一个即可**。

η 家族中只有 `eta_survive_risk_q` 有大动态范围（0.18–1.00，std 0.24）；
`eta_survive_q` / `eta_survive_field` / `eta_survive_pred_q` / `eta_protect`
全部挤在 0.98–1.00 附近。

### 8.4 🔴 `protect_dim = 45` = 整个目标字典

`A` 的形状是 `(16384, 45)`，而 `protect_dim` 只取两个值：**0** 或 **45**。
`max_protected_targets=3` 配上 `Q=3` ⇒ **保护了全部 3 个目标的全部字典列**，
不是只保护被测的弱目标。

这直接解释了机制账：保护使 κ 从 13.00 dB 掉到 8.14 dB（少 5 dB 深度），
而 AUC 反而降约 0.05 —— 在本工作点上保护是**净亏损**。

⇒ **当前 `--max-protected-targets` 是一个"全保护开关"，不是逐目标门控。**
`README.md` §12.3 要求"后续必须采用逐目标收益门控"，而当前口径**无法回答该问题**。
要研究门控，必须先把保护集收窄到单个目标（`max_protected_targets=1` 且按需指定）。

### 8.5 `calibration_error_db`：不进检测链路，但揭示解析侧高估 36 dB

| 臂 | `i_res`（实测） | `i_res_pred`（解析） | `calibration_error_db` |
|---|---:|---:|---:|
| `no_ic` | 2.62e-08 | 2.62e-08 | 0.00 |
| `plain_ls` | 7.35e-10 | 1.92e-13 | **35.84** |
| `tp_uic_full` | 7.93e-10 | 3.57e-13 | **33.46** |
| `protected_ls` | 8.76e-09 | 1.03e-08 | 0.68 |

已验证 `isac_sim/receiver/cancellation_glrt/` 全目录**不读**
`i_res_pred` / `residual_accounting` / `retained`，`residual_model` 只用
`obs.sigma2` + 直达先验 + 系数误差项 ⇒ **不影响 AUC / P_D**，只是旁证。

但含义要记住：κ_pred ≈ 51 dB 而 κ 实测 15.5 dB，
**任何引用 predicted κ 的结论都会高估约 36 dB。**

### 8.6 死字段核查（含一处记忆修正）

脚本设的 config 键**全部是活的**：`protect_targets`、`max_protected_targets`、
`n_cpi`、`covariance_protection`、`adaptive_soft_enable`、`belief_error_in_cres`、
`direct_estimation_sigma_*`（`build_direct.py:61-62`）。

⚠️ **grep 死字段必须同时匹配 `c.X` 与 `getattr(c, "X")` 两种访问模式**；
只搜 `cancellation.X` 会把 `max_protected_targets`、`direct_estimation_sigma_*`
误判为死字段。唯一确认真死的仍是 `cancellation.search_half_width`（0 读取）。

⚠️ **项目记忆已修正**：旧记录称 `n_cpi` 是死字段，实为过时——
`cancellation_glrt/residual_model.py:79` 有 `est_factor / sqrt(n_cpi)`，
并有 `tests/test_cancellation_tp_uic_arms.py:231` 钉住。

### 8.7 产物冗余

* 每个分片都写一份 `scenes/*.npz`：场景只按 `scene_id` 播种，18 个分片的快照**完全相同**
  ⇒ 18 倍存储浪费。
* `scene_summary.csv` 与每分片 `manifest.json` 当前分析器未消费。

---

---

## 10. 逐目标保护预算实验（`mpt` probe，2026-09-22）

网格：boost **+40 dB** 单点，2 cal + 4 test 场景，3 实现，6 接收机 × 3 目标 = **18 个键**，
5 个臂，6 路并行（每轮 9–13 min）。数据：

* `data/mpt_probe_mpt1/`（`--max-protected-targets 1`）
* `data/mpt_probe_mpt3/`（`--max-protected-targets 3`）

⚠️ 每键测试样本仅 12 个 ⇒ AUC 标准误约 0.13。
**本节所有 AUC 结论只是方向性的，不作定论。**

### 10.1 保护预算确实是按目标起作用的（推翻 §8.4 的"全保护开关"读法）

| `--max-protected-targets` | `protect_dim` | 说明 |
|---:|---:|---|
| 0 | 45 | 保护一切（与 3 数值完全相同） |
| 1 | **15** | 保护 1 个目标（A 的 45 列 = 3 目标 × 15 列） |
| 2 | 30 | 保护 2 个目标 |
| 3 | 45 | 保护 3 个目标 ≡ 0 |

所以它**是**逐目标预算，不是全有全无开关。但当预算 ≥ Q 时退化成"保护整个目标字典"。

### 10.2 同键配对的代价

κ_structural（中位 [P10, P90]，18 键）：

| 臂 | mpt=1 | mpt=3 |
|---|---|---|
| `plain_ls` | 12.64 [11.23, 13.42] | 12.64 [11.23, 13.42] |
| `protected_ls` | **12.58** [11.21, 13.40] | **10.41** [8.29, 12.64] |
| `tp_uic_full` | 12.64 [11.23, 13.42] | 12.37 [11.13, 13.41] |

配对差（同一键内相减）：

| 量 | mpt=1 | mpt=3 |
|---|---|---|
| Δκ(`protected_ls` − `plain_ls`) | **−0.03 dB** [−0.21, −0.00] | **−1.67 dB** [−3.87, −0.54] |
| Δκ(`tp_uic_full` − `plain_ls`) | −0.00 dB [−0.00, −0.00] | −0.01 dB [−0.29, −0.00] |
| ΔAUC(`protected_ls` − `plain_ls`) | −0.003 | −0.003 |

⇒ **预算 1 的保护几乎免费（0.03 dB），预算 3 要付中位 1.67 dB（最坏 3.9 dB），
而 AUC 收益两边都是 −0.003。** 基准脚本的默认已从 3 改为 **1**。

### 10.3 🔴 保护预算按"最弱回波功率"选目标，与被测目标无关

`protection.py:34` 按目标回波总功率排序取最弱的前 `budget` 个。
实测：`tested_protected = 1` 的记录占比 = **0.333**（恰好 1/3）。

⇒ 在预算 1 下，**被测目标只有 1/3 的时间真的被保护**。
要回答"保护目标 q 是否保住了目标 q"，要么用保护一切（q 必被保护，但退化），
要么给 `build_observation_pair` 加一个 `protected_targets` 直通参数（改生产代码）。
当前脚本选择**记录事实**（新增 `tested_protected` / `n_protected_targets` 两列），
分析时按该列分层。

### 10.4 🔴 在 INR ≈ +37 dB，除 oracle 外全部臂都在随机猜测水平

AUC（中位 [P10, P90]，18 键）：

| 臂 | AUC |
|---|---|
| `no_ic` / `plain_ls` / `tp_uic_full` | 0.51 [0.49, 0.52] |
| `protected_ls` | 0.50 [0.48, 0.52] |
| `perfect_channel`（oracle） | **0.61** [0.53, 0.82] |

配对 ΔAUC(`perfect_channel` − `plain_ls`) = **+0.111** [+0.030, +0.328]。

⇒ **12–13 dB 的对消深度在 +37 dB 直达干扰下换不来任何检测能力**（ΔAUC = +0.000）；
可争取的全部空间（+0.111）只有完美信道知识才能兑现。

⚠️ 这说明 **boost = 40 dB 已经越过了悬崖**，所有臂都压到 AUC ≈ 0.50。
正式轮的扫描区间应放在 **0–20 dB**（烟测：boost 0 → AUC 0.66，boost 20 → 0.56，
oracle 恒 0.65），40 dB 只适合作为"失效区"的锚点，不适合作为主扫描点。

---

---

## 11. 裁决：目前没有测到检测性能优势

### 11.1 三次独立测量一致

| 来源 | `no_ic` | `plain_ls` | `tp_uic_full` | oracle |
|---|---:|---:|---:|---:|
| 烟测 boost 0（n=16） | 0.660 | 0.660 | 0.660 | 0.652 |
| 烟测 boost 20（n=16） | 0.562 | 0.562 | 0.559 | 0.652 |
| mpt probe boost 40（18 键） | 0.51 | 0.51 | 0.51 | **0.61** |
| `README.md` §12.3 主工作点 worst P_D | 0.737859 | 0.738018 | 0.738023 | 0.737993 |

配对 ΔAUC(`tp_uic_full` − `plain_ls`) 在 mpt probe 上是 **+0.000**。

### 11.2 三条机制根因（按可能性排序）

1. **基线检测器已经白化直达干扰。** `no_ic ≈ plain_ls`（中位相对差 **7.2e-5**）。
   协方差感知 GLRT 把直达先验当作 nuisance 处理掉了，显式对消没有空间可占。
   这与 `README.md` §12.3 的判断一致：TP-UIC 是"强统计干扰处理上的小修正"。
2. **η（目标存活）本来就 ≈ 1。** 实测最大偏离仅 **0.008**（ξ=0.138 处 `plain_ls` η=0.992）。
   TP-UIC 的全部卖点是防止"对消误删目标"，而**根本没有自消可防**（≤0.8%）。
3. **高 INR 时瓶颈是深度，不是保护。** +37 dB 下残余仍约 +25 dB INR，只有零残余的
   oracle 拿到 +0.111。TP-UIC 用深度换保护 = **给瓶颈加税去买已饱和的收益**。

### 11.3 ⚠️ 但前提尚未证伪：高 ξ 区间没采到

mpt probe 的 18 个键里 **ξ 最大只有 0.138**；而 144 单元样本的 ξ 分布是
median 0.022 / p90 0.461 / **max 0.9997**。

⇒ Q1「Plain LS 是否在高冲突下误删目标」是**未回答**，不是"回答为否"。
高 ξ 处白化才会真正伤到目标——那才是保护该生效的地方。
唯一见过的一次信号是 ξ=0.977 时 `plain_ls` η=0.922 vs `protected_ls` 0.999（但 n=1）。

### 11.4 ★ ξ 富集两阶段设计

已验证 **ξ 对 realisation / boost 的 spread ≤ 1.5e-2** ⇒ ξ 是 (场景, 接收机, 目标) 的属性，
不是噪声实现的属性。而算一次 ξ 只需建一次观测 = **0.24 s**，完整流水线 7.5 s，**便宜 31 倍**。

| 阶段 | 做法 | 成本 |
|---|---|---|
| 1. 筛 | 对 ~1000–2000 个 (场景, rx, 目标) 单元算 ξ，挑 ξ ≥ 0.5 + 低 ξ 对照 | 4–8 min |
| 2. 跑 | 只对入选单元跑完整流水线（约 30 个高 ξ 单元 × 8 场景 × 3 实现） | ~90 min |

配对 ΔAUC 分辨力可达约 0.02–0.03。

**预注册决策规则**：若高 ξ 单元上 ΔAUC(`tp_uic_full` − `plain_ls`) < 0.02
（且 n 足以分辨 0.02），则结论转为 **null**，按 §11.5 处理。

### 11.5 若仍为 null

三次一致 + 机制解释本身是可报的结果：**在当前模型与工作点上，配合协方差感知 GLRT，
Innovation I 换不来检测增益。** 可报的正面量是 κ = 12.64 dB [11.23, 13.42]（真实、可复现的
对消能力），但它不转化为检测。

---

## 9. 产出文件

| 文件 | 说明 |
|---|---|
| `tools/run_tpuic_receiver_benchmark.py` | 考场本体（已修 2 个 bug，协议未改） |
| `tools/run_tpuic_receiver_shards.py` | 分片并行驱动 + 合并（正确性论证见其 docstring） |
| `tools/analyze_tpuic_receiver.py` | 汇总分析：AUC/P_D 分臂、配对差、机制指标、ξ 分箱 |
| `studies/direction3/data/tpuic_receiver_smoke/` | 烟测输出（records/summary/scene_summary/manifest/scenes） |
| `studies/direction3/data/_driver_check/` | 分片驱动自检输出 |
