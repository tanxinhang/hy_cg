# 接收机闭环 V1.3 —— 四层共享同一份物理状态

日期：2026-09-19　基线：`589469ec`　未完成发布快照

## 0. 这版做什么，不做什么

上一轮审计的结论是：

> 现在最大的瓶颈已经不是缺一个新算法，而是 receiver–model–selector–detector
> 四层仍没有共享同一个物理状态。

本版**不加任何 P1-5/P1-6 新模块**，只做六件事：把 P0 代码 bug 修掉、让远端
CI 真正能跑、把接收机测量接口改成显式传 belief、把回波存活因子接进
numerator、把 TP-UIC 接进**正式** `proposed_c2f_adaptive_pd` 的同一个 trial、
把 detector 的证据口径从单一 pooled AUC 拆成三类。

一句话结论：

$$
\boxed{\text{半闭环（只接分母）会把 }P_D\text{ 高估约 }0.07,\ \text{且这是统计显著的。}}
$$

---

## 1. P0：`_protection_basis()` 两处缺 `return`

`isac_sim/cancellation.py` 的两个早退分支都**构造了零列数组却没有返回**：

```python
if not c.protect_targets or not sources:
    np.zeros((_n_obs(cfg), 0), dtype=complex)      # 少了 return
...
if not sources:
    np.zeros((_n_obs(cfg), 0), dtype=complex)      # 少了 return
```

执行会继续走进 `np.concatenate(blocks, axis=1)`，而 `blocks` 此时为空，必抛
`ValueError: need at least one array to concatenate`（已实测确认）。

标准场景（15 UAV / 10 目标 / `protect_targets=True`）永远不触发这两个分支，所以
全场景测试全绿；以下情况都会触发：`protect_targets=False`、无 target source、
active mask 过滤后 source 为空、以及将来的空场景单元测试。

修复：两处都改成 `return np.zeros(...)`。

新增三个测试（`tests/test_cancellation_tp_uic.py`）：

* `test_protect_targets_false_returns_empty_basis`
* `test_empty_protection_sources_returns_empty_basis`
* `test_disabled_protection_runs_the_full_arm_pipeline` —— 空基必须能穿过整条
  `cancellation_arms`，不只是穿过构造函数。

顺带记录一个不一致：`_protection_basis` 的 `n_bins` 参数**不被使用**，行数由
`_n_obs(cfg)` 决定。测试改为钉真正生效的那个量，而不是钉参数。

---

## 2. CI：远端一直是红的，而且是被跳过的

`589469ec` → `simulator-ci` run #14 → `conclusion = failure`，失败在 `Unit tests`：

```
ModuleNotFoundError: No module named 'scipy'
22 test module collection errors
```

原因是 `ci.yml` 用 `python-version: "3.12"` 且只装 `numpy matplotlib pytest`，
而 `scipy` 是 `isac_sim` 的**模块级硬依赖**（`mixture_probability` 的
`fftconvolve`/`gamma`/`norm`、`bundle_master` 与 `active_system` 的
`scipy.optimize.milp`）。收集阶段就死，因此 `frozen-gates` 与 `release-gate`
全部 `skipped`——提交说明里“CI carries …”从未被远端执行过。

修复：

* 新建 `requirements-ci.txt`（`numpy==2.2.6`、`scipy`、`matplotlib`、`pytest`）；
* 三个 job 统一 `python-version: "3.11"`，统一 `pip install -r requirements-ci.txt`；
* 加一步打印 `python / numpy / scipy` 版本，让以后任何一次运行都能自证环境。

**provenance 统一**：`TP_UIC_V12.md` 写 228 passed、commit message 写 271
passing。本版实测口径是 **285 passed**（py3.11.0 / numpy 2.2.6，274 个既有/改造
测试 + 本版新增的 11 个：3 个保护基 + 7 个接收机上下文 + 4 个分子桥接）。以后以
`pytest -q` 的实际 collected 数为准。

---

## 3. 接收机测量接口：belief 必须显式传

原 `measure_residual_fraction` 内部调用 `build_observation(cfg, geom, geom, ...)`，
即 **belief = truth**。测出来的 κ 是 perfect-tracking 字典下的深度，而配置同时
声明了 `sigma_p = 150 m`、`sigma_v = 15 m/s`，调度器跑的也是扰动后的几何。

新增 `ReceiverContext` / `ReceiverMeasurement` / `measure_receiver_context`：

```python
ctx = ReceiverContext.from_trial(cfg, geom_true, geom_belief, base,
                                 sense_power=..., radiated_power=...,
                                 processing_gain=..., hw_gain=...,
                                 active_mask=...)
meas = measure_receiver_context(ctx, rng=...)
meas.fraction        # 分母桥：I_res / I_in
meas.eta_survive     # 分子桥（整场口径）
meas.eta_survive_q   # 分子桥（被测目标口径）
meas.eta_protect     # 保护基对真实回波的覆盖率
```

`measure_residual_fraction` 保留 2 元组返回值（逐位兼容），新增
`geom_belief=None` 参数，`None` 明确定义为 **oracle-belief 上界**并在 docstring
里警告：`37.14 dB` 是这一口径，不能直接当系统深度引用。

### 实测：belief 误差落在分子，不在分母

600 m / RCS 0.1，M=6 / Q=3，`tp_uic_full`：

| 量 | oracle-belief | actual-belief |
|---|---|---|
| κ（中位） | 35.89 dB | 35.94 dB |
| `eta_protect` | 1.000 | **0.264** |
| `eta_survive` | 0.978 | 0.918 |
| `eta_survive_q` | ~1.002 | ~0.997 |

这修正了审计里“37.14 dB 是 oracle 因此偏乐观”的预期：

1. **κ 几乎与 belief 无关**（差 0.06 dB）。原因在 `build_observation` 的
   结构里：直连路径**从不依赖 belief**，只有目标字典和保护基依赖。所以
   “oracle belief 捧高了 κ”这个担心的方向不成立——κ 这一列可以放心引用。
2. **belief 的代价全在回波侧**：保护基对真实回波的覆盖率从 1.000 崩到
   **0.264**（约 4 倍），因为接收机保护的是它*以为*目标所在的位置。
3. 回波存活只掉 6%（0.978 → 0.918），因为 stage 1 的保护是**保守**的：它只是
   “不去减”，而“错的不减”比“错的乱减”便宜。这正是**不能用整场
   `eta_survive` 反推保护是否有效**的原因。

测试 `test_the_belief_error_is_paid_on_the_echo_side_not_on_kappa` 把上面这张表
钉住（覆盖率必须塌、κ 位移必须 < 1 dB）。

---

## 4. 分子桥接：`eta_surv` 进 production numerator

`compute_link_tables` 新增 `target_retention_by_receiver`（形状 `(M,)` 或
`(M, Q)`，默认 `None` 保持逐位不变）：

$$
\gamma_{ijq}=\frac{\eta^{\rm surv}_{j}\,S_{ijq}}{N_0+I_{{\rm res},j}}
$$

* 只作用于 `gamma_sense`；`raw_gamma_sense` 是“无任何接收处理”的基线，不动。
* 传入 `> 1` 直接 `ValueError`，而不是静默 clamp——实测存活比**确实会 > 1**
  （600 m 场景读到 1.001），那是联合 stage 把别的东西投到了目标块上，属于记账
  假象，不是白捡的增益。
* 与分母参数一样，会禁用 `reuse_from` 快路径（否则返回的是常数建的表）。

### 两个存活口径差 1.5 dB，不能任选一个假装精确

| 口径 | 600 m 中位 | 折算 |
|---|---|---|
| `eta_survive`（整场） | 0.697 | −1.57 dB |
| `eta_survive_q`（被测目标） | 0.981 | −0.08 dB |

`as_retention(source="field"|"q")` 显式二选一，默认保守的 `field`，CSV 同时落
另一个（`eta_survive_alt_median`）。审计估的“漏掉这项约 −0.08 dB”对应的是
`q` 口径；**整场口径下这项是 −1.6 dB，不是可忽略量**。

---

## 5. 正式 `proposed_c2f_adaptive_pd` 闭环（同一 trial）

新工具 `tools/run_receiver_closed_loop.py`。它**不是** production-like，而是
生产管线本身，逐项对齐 `run_one_trial`：

* `BeliefState.from_truth` → `geom_belief`（选择器与接收机同一份 belief）；
* `truth_captured_links` 过滤（belief 模式下，未被搜索窗捕获的链路不带证据）；
* `rng_for_detection`（共享 detector 流）；
* 评估表是 `c2f_tables`（`dd_gain=base_belief.eta_fine`），与正式一致；
* **fine stage 的表必须注入** `refined_table_builder` —— `select_c2f_adaptive`
  内部会用默认 builder 重建 fine 表，那张表会退回冻结常数，于是 coarse 在算法
  世界、fine 在假设世界。这个失败不报错、数字也好看。

**口径核对**：本工具 `constant` 臂在 20 trials 上给 **P_D = 0.5650**，而直接调
`run_one_trial(cfg, t, ["proposed_c2f_adaptive_pd"])` 在 paper-canonical 下给
**0.5667（3 trials）**——吻合，说明工具确实跑的是生产管线。⚠️ 但论文的 0.7169
属于 **`target-local-v1`** 口径（同法同配置实测 0.6667 @ 3 trials），**不是
paper-canonical**。两者差 0.10，和“地图族 / 主线 selector 口径差”是**另一个**
独立口径维度，不要再混。

两个 preset 各 20 trials，三臂（同一几何、同一 belief、同一选择器、同一
detector draw）：

*paper-canonical*

| 臂 | κ（中位） | `eta_surv` | `eta_protect` | P_D | P_FA |
|---|---|---|---|---|---|
| `constant`（冻结 40 dB） | 40.00 | 1.000 | — | 0.5650 | 0.0535 |
| `measured_denominator`（半闭环） | 37.25 | 1.000 | 0.242 | 0.4950 | 0.0483 |
| `measured_full`（全闭环） | 37.25 | 0.722 | 0.242 | 0.4250 | 0.0499 |

*target-local-v1（论文主结果口径）*

| 臂 | P_D | P_FA |
|---|---|---|
| `constant` | 0.6400 | 0.0504 |
| `measured_denominator` | 0.5200 | 0.0513 |
| `measured_full` | 0.4600 | 0.0527 |

配对差（同 trial，20 trials）：

| 对比 | paper-canonical | target-local-v1 |
|---|---|---|
| constant → 半闭环 | −0.0700 ± 0.0291 | −0.1200 ± 0.0313 |
| constant → 全闭环 | −0.1400 ± 0.0373 | **−0.1800 ± 0.0374** |
| 半闭环 → 全闭环 | −0.0700 ± 0.0252 | −0.0600 ± 0.0134 |

最后一行是本版最该记住的数字：**只接分母的“power-equivalent bridge”把 P_D
高估 0.06–0.07，约 2.8–4.5 个标准误**。审计估计的“暂时不大（−0.08 dB）”在整场
口径下不成立。20 个 trial 里 14 个下降、5 个持平、1 个上升。

可引用的说法（论文口径）：

> 把 40 dB 常数假设换成可执行 TP-UIC 接收机，配上真实信念几何，
> `proposed_c2f_adaptive_pd` 的 P_D 从 0.640 降到 0.460（配对
> −0.180 ± 0.037，20 trials，P_FA 基本不变 0.050 → 0.053）。其中约 0.06 是
> 只接分母的半闭环口径会漏掉的部分。

---

## 6. Detector evidence protocol：三类口径分开报

新工具 `tools/run_detector_evidence.py`。旧口径只有一个数：

```python
for x in H1_all_trials:
    for y in H0_all_trials:
        compare(x, y)
```

这是 **marginal** AUC：从整个部署分布各抽一个 H1 和 H0 比大小。它把几何、目标
强弱、belief 误差、残差尺度全部交叉混合，因此难场景的 H1 会和易场景的 H0 比，
混合物可以停在 0.5 而每个单独场景都能分开。

新工具报三个，且**不允许只报一个**：

* **conditional** —— 固定场景内（同一几何、同一接收机、同一目标）跨噪声实现的
  AUC，回答“这个统计量在这里分不分得开”，并给出跨场景的 spread；
* **paired** —— 配对实现上的 `P(T_H1 > T_H0)`，几何/直连/目标 strength 精确抵消，
  配 Wilson 区间；
* **marginal** —— 旧 pooled 口径，保留，好让读者看清之前的 0.5 有多少是混合物。

判读规则写进了工具输出：marginal ≈ 0.5 但 conditional > 0.5 ⇒ 统计量**能**分开，
旧数字量到的是场景混合；两者都 ≈ 0.5 ⇒ 真的没信息；**Wilson 区间跨过 0.5 只能
说“未确立”，不能说“路线已关闭”**。

这直接撤回 `P1_4_OFFGRID_AUDIT.md` 里“两个 oracle 都停在 AUC≈0.5，因此这两个
轴彻底关闭”的强表述，改成：

> 当前小样本未发现 oracle placement 带来明显收益，因此不支持继续优先投入
> off-grid refinement。

（同时撤回“oracle 是上界所以 20 trials 也够”的论证：oracle 是**信息假设**的上界，
不是它的 Monte-Carlo 估计不需要样本量；而且 max-over-search 与 oracle template
的 H0 分布并不相同，存在多重检验/选择效应，`AUC_oracle` 不是所有 composite
detector 的严格 ROC 上界。）

### 实测（m_rx = 8，8 trials × 5 realisations，40 对）

| arm | conditional AUC | paired P | Wilson 95% | marginal AUC |
|---|---|---|---|---|
| `baseline` | 0.600（scene spread 0.166） | 0.600 | [0.45, 0.74] | 0.544 |
| `oracle` | 0.760（scene spread 0.137） | 0.725 | [0.57, 0.84] | 0.705 |

两点：

1. `baseline` 的 marginal 0.544 与审计引用自旧 CSV 的 **pooled 0.540 几乎完全一
   致**，所以两套代码看的是同一个东西——差异纯粹来自口径，不是实现。
2. 但 `oracle` 的 conditional 0.760 / paired 0.725，与“oracle 停在 0.5”相差很远。
   同 draw 上的**配对**比较：

   * 配对 `ΔT_oracle − ΔT_baseline` 均值 `+3.429 ± 1.809`（约 1.9 s.e.）；
   * McNemar：discordant 对中 11/17 偏向 oracle；
   * 8 个场景里 5 个 oracle 平均占优。

   ⇒ **oracle placement 有正向趋势，但当前样本不足以确立。** 正确的表述是
   “未确立，且趋势为正，不应宣布关闭”，而不是“路线已关闭”。

**这是本版与上一轮审计结论的一处实质分歧**，请以后者为准：旧结论建立在 pooled
cross-AUC 上，而 pooled 把每个场景的合成难度也平均掉了。

---

## 7. CFAR：真正的 calibration test，以及一个必须归因的大发现

新工具 `tools/run_cfar_calibration.py`。旧的证据是 `E[T] = dof/2`——零假设均值对
得上。这是必要但不近乎空洞的条件：分布可以均值对、尾部错，而检测器只吃尾部。

新工具测尾部，且分三层以便归因：

* `oracle` —— 完美信道 + truth 字典，无 belief 误差；
* `truth` —— 几何带 belief 误差，但字典仍用 truth（隔离残差/保护侧）；
* `belief` —— 生产配置。

每层输出：经验 P_FA + Wilson 区间、经验 95 分位与名义阈值之比
（`q95_emp / thr`）、以及 `E[T]/(dof/2)` 作对照。

### 实测（m_rx=1，3 trials × 50 nulls，名义 P_FA = 0.05）

| level | emp. P_FA | Wilson 95% | `E[T]/(dof/2)` | `q95/thr` |
|---|---|---|---|---|
| `oracle` | 0.053 | [0.027, 0.102] | 0.986 | 1.004 |
| `truth` | 0.053 | [0.027, 0.102] | 0.991 | 1.008 |
| **`belief`** | **0.660** | [0.581, 0.731] | **2.072** | **1.954** |

这个 0.66 太大，不能凭直觉接受，所以加了归因对照 `--belief-sigma-pos 0`：把跟踪
位置误差设为 0，但**代码路径仍走 belief 那一支**。结果（2 trials × 30 nulls）：

| level | emp. P_FA | `E[T]/(dof/2)` |
|---|---|---|
| `oracle` / `truth` / `belief` | 0.067（三者相同） | 1.012 |

⇒ **失配 100% 来自声明的 tracker 位置误差（σ_p = 150 m），不是 belief 代码实现缺
陷。**

结论，以及它为什么重要：

> 在 150 m 信念误差下，`target_conditioned_glrt` 的名义 P_FA = 0.05 实际是
> 0.66，零假设统计量均值膨胀 2.07 倍。**这个检测器在 belief 口径下不具备 CFAR
> 性质。**

后果是硬的：**任何用固定阈值报 P_FA / P_D 的结果，在 belief 口径下都不可信**；
这也解释了为什么这一系列探针只能报 AUC——只能报 AUC。下一步要么逐一场景重新
标定阈值，要么把阈值改成对信念失配稳健的形式。在此之前，§5 里那些 P_FA 数字
只能在“它们是同一阈值下的相对比较”这个意义上使用。

---

## 8. 未关闭 / 下一步（按优先级）

0. **CFAR 失配（§7）优先级最高**。150 m 信念误差下名义 P_FA 0.05 实测 0.66。在这
   件事解决之前，任何“用固定阈值报 P_D”的数字都只能在“同一阈值下的相对比较”
   意义上使用。两条路：逐场景重新标定阈值，或改阈值使其对信念失配稳健。
1. **`eta_surv` 目前只有 per-receiver 粒度**。真正的量是 `η_{j,q}`；整场口径
   是保守方向，`q` 口径是乐观方向，两者差 1.5 dB。要收窄只能逐 `(j,q)` 测
   （M×Q 次 `build_observation`），代价大。在此之前引用任何“TP-UIC 的目标损伤”
   都必须声明口径。
2. **P1-3（ρ-aware selector）先做最便宜的 oracle 实验**：SINR selector vs
   SINR + oracle-ρ，同预算、同链路数、同 detector。已有结论是 ρ spread ≈ 0.78
   且 `|Spearman(ρ, γ)| ≈ 0.02`（ρ 不是 SINR 的重复表达），但**这推不出**
   “加 ρ 能提 P_D”。oracle-ρ 都买不到性能再关路线，比继续建优化器便宜。
3. **off-grid 不是“已关闭”，是“未确立且趋势为正”**（§6）。样本加到 ~50 scene 再
   裁；在此之前不要用“oracle 停在 0.5”这句话。
3.5 **论文暂不吸收新数字**：37.14/37.25 dB、0.4250/0.4600、双口径 η、阵列
   AUC 0.54/0.60 都还是旁路口径，且受 §7 的阈值问题影响。先把 0–1 关掉再谈进
   主文。
5. `TP_UIC_V12.md` 的 228 passed 应读作 274（见 §2）；V1.2 里“37.14 dB”应加
   限定语 **oracle-belief / perfect-tracking**，不是系统深度——不过 §3 说明 κ 实
   际对 belief 不敏感，所以限定语是口径说明，不是打折理由。

## 10. 本版对上一轮审计的两处修正

| 审计的判断 | 本版实测 | 处理 |
|---|---|---|
| “37.14 dB 是 oracle belief 的，所以偏乐观” | κ 对 belief 几乎不敏感（35.89 → 35.94 dB）；代价全在 `eta_protect` 1.000 → 0.264 | **方向不成立**，κ 可引用；信念代价改在回波侧记账 |
| “分子漏接只有 −0.08 dB，暂时不大” | 整场口径下 −1.57 dB，半闭环**高估 P_D 0.06–0.07（2.8–4.5 s.e.）** | 必须接，且必须声明 `field` / `q` 口径 |
| “oracle 停在 AUC≈0.5，off-grid 路线关闭” | oracle conditional 0.760 / paired 0.725，配对 `ΔΔT = +3.43 ± 1.81` | **未确立且趋势为正**，不应宣布关闭 |

---

## 9. 门禁状态（本版）

| 门禁 | 结果 |
|---|---|
| `pytest -q`（py3.11.0 / numpy 2.2.6） | **285 passed**（+ 6 subtests） |
| `check_release_identity.py --check` | **CLEAN**（96 冻结键，0 违约） |
| `check_contract_refs.py` | **CLEAN（32/32）**——本版手改了 6 条漂移的 model.py 行区间 |
| `check_contract_refs.py` 漂移原因 | `model.py` 新增 docstring + retention 校验，插入约 45 行 |
