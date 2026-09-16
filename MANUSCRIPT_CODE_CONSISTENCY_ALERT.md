# ⚠️ 稿件数字与当前代码一致性告警

日期：2026-09-16
状态：**✅ 已收口（2026-09-16 晚）**——主实验已按当前冻结代码完整重跑，
正文、补充材料与 fig2/3/4 已同步。见 `SUBMISSION_TRACEABILITY.md` §7。
严重级别：历史告警，保留存档价值（"先归因后重冻"的过程可复用；文中两处**已证伪**的归因不要重犯）

> **🔴 归因更正（2026-09-16 晚，冻结 V1 时复核）**
>
> 本文第 3 节把漂移根因归给"判决门限被替换"，**该归因已被代码级比对证伪**。
>
> | 部件 | 位置 | `03f9612^` vs 当前 HEAD |
> |---|---|---|
> | `threshold_from_pfa` | `model.py:112` | **逐字相同** |
> | `fused_h0_variance` | `fusion.py:252` | **逐字相同** |
> | `fused_h0_skewness` | `fusion.py:281` | 仅 `corr.enable or` → `corr.enable and len(links)>1`；`corr.enable=False` 时行为相同 |
> | 门限公式 | 旧 `simulate.py:158-159` vs 新 `fusion.py:350` | 同为 $(z_0+\frac{\kappa_0}{6}(z_0^2-1))\sqrt{v_{q,0}}$ |
>
> 且 `calibrated_fused_threshold` 在 `comm_error_model != "erasure"` 时**精确返回该式**
> （`fusion.py:351-356`）。归档 V1 走的正是 `gaussian_replacement` 路径 ⇒ **门限替换在归档
> 路径上不改变任何数值**。
>
> **真正的变化是检测阶段的随机数流**：旧代码用共享顺序流（`draw_h1_soft_stat(..., rng, ...)`），
> 新代码改用逐链路键控流（`simulate.py:209` `keyed_rngs(q, ordered_links, True, 0)`）。
> `git show 03f9612^:isac_sim/simulate.py | grep -c keyed_rngs` = **0**。
> 这也正是 `BASELINE_DRIFT_ATTRIBUTION.md` 中"机制 B 触发点未隔离"的答案。
>
> **不改结论的部分**：漂移真实存在，选路面量逐位一致、
> 只有判决计数变——与"换 RNG 流"完全吻合。**主实验 MC=1000 仍必须重跑。**
> 详见 `V1_STABLE_RELEASE.md` §5.1。
>
> **🔴 量级更正（2026-09-16 晚，MC=1000 重跑后）**：第 4 节那个 MC=100 的
> $-0.007$ 读数**本身落在 MC=100 的噪声内**（$P_D$ 95% 半宽约 $\pm0.009$），
> 高估了漂移量级。MC=1000 实测只有 **$+0.0002$**（0.9762 → 0.9764），
> 三个方法均在 $\pm0.0003$ 内。**"描述与数据不符"的定性判断成立，
> 但"数字被改动"的说法不成立**——旧数字与当前代码的差异不超过重采样噪声。
> 重跑仍然必须做（要的是可复现性，不是"数值近似正确"），只是危害等级从
> "数字错误"降为"口径不一致"。

---

## 1. 结论摘要

论文的**检测性能数字**（$P_D$、worst-target $P_D$、配对差值）来自 2026-09-14
生成的 CSV，而**检测阶段的随机数流已在后续修订中改变**。因此：

- 用**当前代码**跑同样配置，得到的 $P_D$ 与论文数字**不一致**；
- 论文**通信效率**相关数字（报告数、payload、串行时延、观测数）**不受影响**，
  实测与旧结果**逐位一致**；
- 论文正文已在声明新模型，**但数字来自旧实现**——这是"描述与数据不符"，
  而非"数据错误"。

---

## 2. 时间线

| 时间 | 事件 |
|---|---|
| 2026-09-14 17:46 | `84ae01c` release target-local fusion V1 and rebuild manuscript |
| **2026-09-14 18:39** | **`results_target_local_v1/main/main.csv` 生成**（主实验 MC=1000） |
| 2026-09-14 19:19 | `overview/fusion-rule/fusion-rule.csv` 生成（MC=100） |
| 2026-09-14 19:23 | `69f3300` 提交这批结果（即文档所称"生成主结果的源版本 69f3300"） |
| **2026-09-15 01:26** | **`03f9612` 判决阈值更换为 `calibrated_fused_threshold`** |
| 2026-09-15 17:59 | `efffc26` 后续相关改动 |
| 2026-09-15 23:29 | `62ceed2`（本次审计开始前的 HEAD） |

---

## 3. 根因（❌ 本节的"门限替换"归因已被证伪，见文首更正）

判决函数 `isac_sim/simulate.py: evaluate_detection` 的阈值计算被替换。

**源版本 69f3300**（结果所用）：

```python
base_thr = threshold_from_pfa(cfg)
var0 = fused_h0_variance(cfg, tables, q, links, weights, plan=plan, base=base)
skew0 = fused_h0_skewness(cfg, tables, q, links, weights, plan=plan, base=base)
z_cf = base_thr + (skew0 / 6.0) * (base_thr * base_thr - 1.0)   # Cornish–Fisher 近似
thr = z_cf * math.sqrt(max(var0, EPS))
```

**当前 HEAD**：

```python
weight_mode = fusion_weight_mode_for_method(method)
# Threshold calibration belongs to the detector and selected set, not
# to a proposed method label.  This gives every arm the same realised
# P_FA semantics under the true-erasure mixture.
thr = calibrated_fused_threshold(
    cfg, tables, q, links, weights, plan=plan, base=base,
    statistic_mode=weight_mode,
)
```

即：判决阈值从 **Cornish–Fisher 近似的正态阈值** 换成 **有限-look LLR 混合分布的
精确分位数校准**（`fusion.py: calibrated_fused_threshold`）。后者更精确，但**结果不同**。

> 注意方向：该函数在 `fusion.py` 的**预测**分支（`predicted_pd_for_links`）只在
> `weight_mode == "exact_llr_sum"` 时调用，而 `proposed_c2f_adaptive_pd` 的
> `weight_mode` 是 `"deflection"`。但**判决**分支（`evaluate_detection`）**无条件**
> 调用它。因此只看 `fusion.py` 会误判为"不影响 V1"——必须看 `simulate.py`。

---

## 4. 实测证据（MC=100，seed 2026，preset `target-local-v1`）

| 配置 | $P_D$ | actual worst-target $P_D$ | worst-target $D$ | 观测 | 报告 | 时延 (ms) |
|---|---|---|---|---|---|---|
| **9/14 代码** + `erasure`（当时语义 = 高斯 $a{=}9$） | **0.982** | 0.95 | 2310.1863613680944 | 12.24 | 0.74 | 0.7893333333333334 |
| **当前代码** + `erasure`（真零擦除） | 0.975 | 0.96 | 2310.1863613680944 | 12.24 | 0.75 | 0.8000000000000002 |
| **当前代码** + `gaussian_replacement` | **0.975** | 0.96 | 2310.1863613680944 | 12.24 | 0.74 | 0.7893333333333334 |

**读法**：

1. 第 1 行与第 3 行的**选择路径量完全一致**（$D$=2310.1863613680944、观测 12.24、
   报告 0.74、时延 0.7893333333333334），**只有 $P_D$ 不同**（0.982 → 0.975）。
   → 差异**只**来自判决阈值，与选择算法、信道、几何无关。
2. 第 2 行（真零擦除）与第 3 行 $P_D$ 相同但报告数/时延不同，说明 `erasure` 与
   `gaussian_replacement` 在**选择**上有细微差异（预测 $P_D$ 不同 → 选择略变），
   但恰好不改变检测结果。**两者不是同一个模型**。

产物：`results_v1_failure_model_check/erasure/`、`results_v1_utility_ablation/full/`。

---

## 5. 影响评估

| 论文主张 | 是否受影响 | 说明 |
|---|---|---|
| 通信效率：83.7% payload/时延下降、0.751 vs 4.595 报告 | ✅ 不受影响 | 选择路径未变，报告数/时延逐位一致（重跑后仍逐位一致） |
| 精细评估 61.278 / 860.244（92.88%） | ✅ 不受影响 | 属选择算法的计数，与判决阈值无关（重跑后仍逐位一致） |
| 主 $P_D$ 0.9762、worst-target 0.967 | ✅ 已重跑更新 | → **0.9764 / 0.966**（MC=1000，漂移 +0.0002） |
| 配对差值 +0.0351 / −0.0021 及其区间 | ✅ 已重跑更新 | → **+0.0356 [0.0311, 0.0401] / −0.0020 [−0.0050, 0.0010]**，显著性方向未变 |
| 融合位置消融的 $P_D$ 0.919 / 0.969 / 0.982 | ✅ 已重跑更新 | → **0.932 / 0.967 / 0.975**（MC=100，顺序未变） |
| 全精细对照 $P_D$ 0.980 vs 0.973 | ✅ 已重跑更新 | → **0.9775 vs 0.9715**，配对 0.0060 [0.0005, 0.0115]（下界贴近 0，但仍排除 0） |
| "两效用项不宣称必要" | 已由新消融补上 | 见 `V1_UTILITY_ABLATION.md` |

---

## 6. 建议行动（✅ 全部已完成）

1. ~~用当前代码重跑主实验 MC=1000~~ → 已完成，`--suite all --mc 1000 --workers 8`。
2. ~~同步重跑 `overview/fusion-rule`、`prediction-stress`、`geometry`~~ → 已完成（overview 全部 6 个扫描 + prediction-stress + full-refinement）。
3. ~~重建论文图~~ → 已完成，`tools/make_target_local_v1_paper_figs.py`（fig2/3 主稿 + fig4 补充材料）。
   ⚠️ 注意本文原稿写的 `tools/make_paper_figs.py` **是错的**：那读的是另一套
   `results_release` 协议（$P_D$ 量级 0.87），与 V1 释放线无关。
4. ~~更新 `SUBMISSION_TRACEABILITY.md` 的 commit 边界~~ → 已完成，同日重建。
5. 不需要走"声明数字对应旧实现"的降级路线。

> 时间压力：ICC 2027 Symposium 投稿截止 **2026-10-02**，剩余约 16 天。

---

## 7. 复现命令（已更新为收口后的入口）

```bash
# 完整 V1 释放树（main MC=1000 + full-refinement MC=200 + overview + prediction-stress）
E:/anaconda/3_11_python/python.exe tools/rerun_target_local_v1.py \
    --suite all --workers 8 --out results_target_local_v1

# 核对"是否只动了检测采样"：选路面必须逐位一致
E:/anaconda/3_11_python/python.exe tools/report_v1_rerun_drift.py \
    --old archive/results_target_local_v1_pre_rngfix_2026-09-16 \
    --new results_target_local_v1

# 重出图件
E:/anaconda/3_11_python/python.exe tools/make_target_local_v1_paper_figs.py
```

---

## 8. 附：另一个易混点（已澄清，非缺陷）

现有结果目录的 `config.json` 里 `detect.comm_error_model` 字段值均为 **`erasure`**，
而当前 `erasure` 的语义已变为"真零擦除"。**不要据此判定论文失败模型写错**：

源版本 `69f3300` 的 `soft_channel.py:142` 中，`erasure` 分支返回的正是
$\mathcal N(0,\sigma_{\rm scale}\sqrt{v_0})$，$\sigma_{\rm scale}=3.0$ → 方差 $9v_0$
→ **$a=9$ 高斯替代**。即当时的 `erasure` ≡ 当前的 `gaussian_replacement`。

所以：论文声明"Gaussian replacement, $a=9$"**与当时执行一致**；
`tools/rerun_target_local_v1.py` 显式固定 `gaussian_replacement` 也是**正确的复现入口**。
这只是历史命名债，不是模型错误。
